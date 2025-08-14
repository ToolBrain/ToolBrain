from copy import deepcopy
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ...core_types import Trace, Turn, ParsedCompletion


@dataclass(frozen=True)
class GRPOBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    advantages: torch.Tensor

    def to(self, device: torch.device) -> "GRPOBatch":
        return GRPOBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            completion_mask=self.completion_mask.to(device),
            advantages=self.advantages.to(device),
        )

    def shifted(self) -> "GRPOBatch":
        """Return a shifted batch with the first token removed along time dimension."""
        return GRPOBatch(
            input_ids=self.input_ids[:, 1:],
            attention_mask=self.attention_mask[:, 1:],
            completion_mask=self.completion_mask[:, 1:],
            advantages=self.advantages[:, 1:],
        )

def copy_model(model: nn.Module) -> nn.Module:
    return model.copy() if hasattr(model, 'copy') else deepcopy(model)

def get_llm_and_tokenizer_from_smolagent(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_id)
    return llm, tokenizer

def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Implements the reward normalization described in Section 4.1.2 (Outcome Supervision) of the DeepSeekMath GRPO paper.
    Each reward value corresponds to a whole trace, and we normalize these per-trace rewards across the batch.
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    normalized_r = (rewards - mean) / (std + 1e-8)
    return normalized_r

def build_inputs(
    traces: List[Trace],
    tokenizer: PreTrainedTokenizerBase,
    rewards: List[float],
) -> GRPOBatch:
    """
    Prepare a batch for GRPO from a batch of traces.

    Args:
        traces: Batch of traces. Each trace is a list of `Turn` dicts. A `Turn` contains:
            - prompt_for_model: The exact context the LLM saw
            - model_completion: The exact raw string the LLM generated
            - parsed_completion: Structured breakdown of the completion
            - tool_output (optional): Result from executing the tool code
        tokenizer: A HuggingFace tokenizer (already loaded). `pad_token` should be set.
        rewards: A list of reward-per-trace (final reward). The same scalar is expanded along the time dimension of that trace.

    Returns:
        GRPOBatch(input_ids, attention_mask, completion_mask, advantages):
            - input_ids: (B, L_max)
            - attention_mask: (B, L_max), 1 for real tokens, 0 for pad
            - completion_mask: (B, L_max), 1 only on tokens from model_completion across all turns; 0 for prompt_for_model & tool_output
            - advantages: (B, L_max) per-trace **normalized** rewards (computed over the unpadded B traces) expanded to token level; padded positions are 0.0
    """
    all_input_ids: List[List[int]] = []
    all_attention_masks: List[List[int]] = []
    all_completion_mask: List[List[int]] = []
    all_advantages: List[List[float]] = []

    # Normalize per-trace rewards across the batch (DeepSeekMath §4.1.2)
    normalized_rewards = compute_advantages(torch.tensor(rewards, dtype=torch.float32))  # shape: (B,)

    for idx, (trace, reward) in enumerate(zip(traces, rewards)):
        seq_ids: List[int] = []
        seq_attn: List[int] = []
        seq_comp_mask: List[int] = []
        seq_advs: List[float] = []

        for turn in trace:
            prompt = turn.get("prompt_for_model", "")
            model_text = turn.get("model_completion", "")
            tool_text = turn.get("tool_output", "")

            # Tokenize all three segments for this turn
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            model_ids = tokenizer.encode(model_text, add_special_tokens=False)
            tool_ids = tokenizer.encode(tool_text, add_special_tokens=False)

            # Append in the order actually seen by the model/logging
            # [prompt_for_model][model_completion][tool_output]
            turn_ids = prompt_ids + model_ids + tool_ids
            seq_ids.extend(turn_ids)

            # Attention mask: 1 for every real token
            seq_attn.extend([1] * len(turn_ids))

            # Completion mask: 1 only for model_completion tokens
            seq_comp_mask.extend([0] * len(prompt_ids) + [1] * len(model_ids) + [0] * len(tool_ids))

            # Expand the per-trace normalized reward along this turn's tokens
            seq_advs.extend([float(normalized_rewards[idx].item())] * len(turn_ids))

        # Accumulate per-trace sequences
        all_input_ids.append(seq_ids)
        all_attention_masks.append(seq_attn)
        all_completion_mask.append(seq_comp_mask)
        all_advantages.append(seq_advs)

    # Pad to batch-first tensors
    input_ids = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in all_input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = pad_sequence(
        [torch.tensor(seq, dtype=torch.int) for seq in all_attention_masks],
        batch_first=True,
        padding_value=0,
    )
    completion_mask = pad_sequence(
        [torch.tensor(seq, dtype=torch.int) for seq in all_completion_mask],
        batch_first=True,
        padding_value=0,
    )
    advantages = pad_sequence(
        [torch.tensor(seq, dtype=torch.float) for seq in all_advantages],
        batch_first=True,
        padding_value=0.0,
    )

    return GRPOBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        advantages=advantages
    )

def get_per_token_logps(
    logits: torch.Tensor,          # (B, L-1, V)
    input_ids: torch.Tensor,       # (B, L-1)
    chunk_len: int | None = None,  # if set, compute along L in chunks of C tokens to reduce peak memory
) -> torch.Tensor:
    """
    Return per-token log-probs aligned with targets.
    - logits: (B, L-1, V)
    - input_ids: (B, L-1)
    - chunk_len: optional chunk size along L for memory-friendly computation
    Returns: (B, L-1)
    """
    if chunk_len is None:
        lp = logits.log_softmax(dim=-1)  # (B, L-1, V)
        return lp.gather(dim=-1, index=input_ids.long().unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    # Chunked path along sequence length
    chunk_outputs: list[torch.Tensor] = []
    target_len = input_ids.size(1)
    for start_idx in range(0, target_len, chunk_len):
        end_idx = min(start_idx + chunk_len, target_len)
        lp = logits[:, start_idx:end_idx, :].log_softmax(dim=-1)  # (B, C, V)
        chunk_outputs.append(
            lp.gather(
                dim=-1,
                index=input_ids[:, start_idx:end_idx].long().unsqueeze(-1)).squeeze(-1)
        )  # (B, C)
    return torch.cat(chunk_outputs, dim=1)  # (B, L-1)

class Policy(nn.Module):
    def __init__(self, llm, tokenizer) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer

    def get_log_probs(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      chunk_len: int | None = None) -> torch.Tensor:
        """Return per-token log-probs aligned for causal LM loss
        (predict token t from context up to t-1).
        Output shape: (B, L-1).
        """
        logits = self.llm(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, L, V)
        
        # Exclude the last logit because it has no corresponding target token
        # (there is no "next token" after the last one)
        logits = logits[:, :-1, :]   # shape: (B, L-1, V)
        # Exclude the first token in input_ids because its prediction is based on
        # the context before it (no preceding token)
        input_ids = input_ids[:, 1:] # shape: (B, L-1)

        per_token_logps = get_per_token_logps(logits, input_ids, chunk_len=chunk_len)  # (B, L-1)
        return per_token_logps


if __name__ == "__main__":
    model_id = "gpt2"
    llm, tokenizer = get_llm_and_tokenizer_from_smolagent(model_id=model_id)
    policy_model = Policy(llm=llm, tokenizer=tokenizer)

    traces: List[List[Turn]] = [
        [
            Turn(
                prompt_for_model="You are a Python assistant. Compute the sum of 1..10 and explain briefly.",
                model_completion="Thought: I'll write a short Python loop.\n```python\ns=sum(range(1,11)); print(s)\n```\n",
                parsed_completion=ParsedCompletion(thought="thought", code="some code"),
                tool_output="Execution logs:\n55\nLast output from code snippet:\n55",
            ),
            Turn(
                prompt_for_model="Given the tool output above, provide the final answer.",
                model_completion="Final Answer: 55",
                parsed_completion=ParsedCompletion(),
                tool_output="",
            ),
        ],
        [
            Turn(
                prompt_for_model="You are a math helper. Sum 1..10.",
                model_completion="I can compute it mentally: 55.",
                parsed_completion=ParsedCompletion(),
            )
        ],
    ]

    rewards = [1.0] * len(traces)
    batch = build_inputs(traces=traces, rewards=rewards, tokenizer=policy_model.tokenizer)
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    completion_mask = batch.completion_mask
    advantages = batch.advantages

    print("Input IDs:", input_ids)
    print("Shape of input_ids:", input_ids.shape)
    print("Attention Mask:", attention_mask)
    print("Shape of attention_mask:", attention_mask.shape)
    print("Completion Mask:", completion_mask)
    print("Shape of completion_mask:", completion_mask.shape)
    print("Advantages:", advantages)
    print("Shape of advantages:", advantages.shape)
    print("Decoded first trace:", tokenizer.decode(input_ids[0]))

    log_probs = policy_model.get_log_probs(input_ids, attention_mask)  # (B, L-1)
    log_probs_chunked = policy_model.get_log_probs(input_ids, attention_mask, chunk_len=128)  # (B, L-1)
    print("Log probabilities shape:", log_probs.shape)
    print("Log probabilities chunked shape:", log_probs_chunked.shape)

    # Align masks/advantages for causal loss (match L-1 length)
    shifted_batch = batch.shifted()
    input_ids_loss = shifted_batch.input_ids
    attention_mask_loss = shifted_batch.attention_mask
    completion_mask_loss = shifted_batch.completion_mask
    advantages_loss = shifted_batch.advantages

    # Sanity checks
    assert log_probs.shape == input_ids_loss.shape, f"log_probs {log_probs.shape} vs labels {input_ids_loss.shape}"
    assert completion_mask_loss.shape == log_probs.shape, "completion_mask not aligned with shifted logits"
    assert advantages_loss.shape == log_probs.shape, "advantages not aligned with shifted logits"

    print("Completion mask (row 0, shifted):", completion_mask_loss[0])
    print("#model tokens row0 (shifted):", int(completion_mask_loss[0].sum().item()))
    print("#total tokens row0 (shifted):", int(attention_mask_loss[0].sum().item()))