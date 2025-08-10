from copy import deepcopy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
from typing import NamedTuple, List, Tuple
from toolbrain.core_types import Example



def copy_model(model: nn.Module) -> nn.Module:
    return model.copy() if hasattr(model, 'copy') else deepcopy(model)

def get_llm_and_tokenizer_from_smolagent(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_id)
    return llm, tokenizer

def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    normalized_r = (rewards - mean) / (std + 1e-8)
    return normalized_r

def build_inputs(
    traces: list[list[Example]],
    tokenizer: PreTrainedTokenizerBase,
    rewards: list[float],
) -> dict:
    """
    Args:
        traces: List of traces, each trace is a list of (prompt: str, completion: str) tuples.
        tokenizer: A HuggingFace tokenizer (already loaded).
        rewards: A list of reward-per-trace.
    
    Returns:
        dict: A dict containing input_ids, attention_mask, completion_mask, and advantages tensors.
    """
    all_input_ids = []
    all_completion_mask = []
    all_rewards = []
    all_attention_masks = []

    for trace, reward in zip(traces, rewards):
        input_ids_seq = []
        completion_mask_seq = []
        rewards_seq = []

        for prompt, completion in trace:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)

            input_ids_seq.extend(prompt_ids + completion_ids)
            completion_mask_seq.extend([0] * len(prompt_ids) + [1] * len(completion_ids))
            rewards_seq.extend([reward] * (len(prompt_ids) + len(completion_ids)))

        all_input_ids.append(input_ids_seq)
        all_completion_mask.append(completion_mask_seq)
        all_rewards.append(rewards_seq)
        all_attention_masks.append([1] * len(input_ids_seq))
        
    input_ids = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in all_input_ids],
        batch_first=True
    )
    attention_mask = pad_sequence(
        [torch.tensor(seq, dtype=torch.int) for seq in all_attention_masks],
        batch_first=True
    )
    completion_mask = pad_sequence(
        [torch.tensor(seq, dtype=torch.int) for seq in all_completion_mask],
        batch_first=True
    )
    rewards = pad_sequence(
        [torch.tensor(seq, dtype=torch.float) for seq in all_rewards],
        batch_first=True
    )
    advantages = compute_advantages(rewards)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "advantages": advantages,
    }

def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

class Policy(nn.Module):
    def __init__(self, llm, tokenizer) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer

    def get_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.llm(input_ids=input_ids, attention_mask=attention_mask).logits  # (N, T, V)
        
        #logits = logits[:, :-1, :]  # (N, T-1, V), exclude the last logit: it corresponds to the next token pred
        #input_ids = input_ids[:, 1:]  # (N, T-1), exclude the first input ID since we don't have logits for it
        
        per_token_logps = get_per_token_logps(logits, input_ids) # Shape: (N,T)
        return per_token_logps


if __name__ == "__main__":
    model_id = "gpt2"
    llm, tokenizer = get_llm_and_tokenizer_from_smolagent(model_id=model_id)
    policy_model = Policy(llm=llm, tokenizer=tokenizer)

    traces = [
        [
            (
                "Thought: I can use a simple Python loop to iterate over the numbers from 1 to 10 and sum them up.",
                "Observation: Execution logs:\n55\nLast output from code snippet:\nNone"
            ),
            (
                "Thought: The code snippet has successfully calculated the sum of numbers from 1 to 10, which is 55. Now, I can use the `final_answer` tool to provide the final answer.",
                "Observation: Execution logs:\nLast output from code snippet:\n55"
            )
        ],
        [
            (
                "Thought: To sum all numbers from 1 to 10, I can use a simple loop in Python to iterate over the range of numbers and add them up. I will use the built-in `range` function to generate the numbers from 1 to 10, and a variable to keep track of the sum.",
                "Observation: Execution logs:\n55\nLast output from code snippet:\nNone"
            ),
            (
                "Thought: The code snippet has successfully calculated the sum of all numbers from 1 to 10, which is 55. Now, I can use the `final_answer` tool to provide the final answer.",
                "Observation: Execution logs:\nLast output from code snippet:\n55"
            )
        ],
    ]

    rewards = [1.0] * len(traces)
    batch = build_inputs(traces=traces, rewards=rewards, tokenizer=policy_model.tokenizer)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    completion_mask = batch["completion_mask"]
    advantages = batch["advantages"]

    print("Input IDs:", input_ids)
    print("Shape of input_ids:", input_ids.shape)
    print("Attention Mask:", attention_mask)
    print("Shape of attention_mask:", attention_mask.shape)
    print("Completion Mask:", completion_mask)
    print("Shape of completion_mask:", completion_mask.shape)
    print("Advantages:", advantages)
    print("Shape of advantages:", advantages.shape)
    print("Decoded first trace:", tokenizer.decode(input_ids[0]))

    log_probs = policy_model.get_log_probs(input_ids, attention_mask)
    print("Log probabilities shape:", log_probs.shape)
    print("Log probabilities:", log_probs)