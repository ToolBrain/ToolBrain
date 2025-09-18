from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import random
from toolbrain.core_types import ChatSegment


@dataclass(frozen=True)
class DPOBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor

    def to(self, device: torch.device) -> "DPOBatch":
        return DPOBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            completion_mask=self.completion_mask.to(device),
        )

    def shifted(self) -> "DPOBatch":
        """Return a shifted batch with the first token removed along sequence."""
        return DPOBatch(
            input_ids=self.input_ids[:, 1:],
            attention_mask=self.attention_mask[:, 1:],
            completion_mask=self.completion_mask[:, 1:],
        )


def build_inputs(
        segments: List[List[ChatSegment]],
        tokenizer: PreTrainedTokenizerBase,
) -> DPOBatch:
    """
    Prepare a batch for GRPO from a batch of traces.

    Args:
        segments: Batch of chat segments. Each chat segment is a list of `ChatSegment` dicts. A `ChatSegment` contains:
            - role: role of tje segment, either assistant or other
            - text: text of the chat segment in the chat history
        tokenizer: A HuggingFace tokenizer (already loaded). `pad_token` should be set.

    Returns:
        GRPOBatch(input_ids, attention_mask, completion_mask, advantages):
            - input_ids: (B, L_max)
            - attention_mask: (B, L_max), 1 for real tokens, 0 for pad
            - completion_mask: (B, L_max), 1 only on tokens from model_completion across all turns; 0 for prompt_for_model & tool_output
    """
    all_input_ids: List[List[int]] = []
    all_attention_masks: List[List[int]] = []
    all_completion_mask: List[List[int]] = []

    for idx, trace in enumerate(segments):
        seq_ids: List[int] = []
        seq_attn: List[int] = []
        seq_comp_mask: List[int] = []

        for i, segment in enumerate(trace):
            segment_ids = tokenizer.encode(segment["text"], add_special_tokens=False)
            seq_ids.extend(segment_ids)
            # Attention mask: 1 for every real token
            seq_attn.extend([1] * len(segment_ids))
            # Completion mask: 1 only for model_completion tokens
            if segment["role"] != "assistant":
                seq_comp_mask.extend([0] * len(segment_ids))
            else:
                seq_comp_mask.extend([1] * len(segment_ids))

        # Accumulate per-trace sequences
        all_input_ids.append(seq_ids)
        all_attention_masks.append(seq_attn)
        all_completion_mask.append(seq_comp_mask)

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

    return DPOBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
    )


def make_dpo_pairs(
    rl_inputs: List[List[ChatSegment]],
    rewards: List[float]
) -> Tuple[List[List[ChatSegment]], List[List[ChatSegment]]]:
    """
    Create chosen and rejected batches for DPO training using sort+pair.

    Args:
        rl_inputs: list of list of ChatSegments (the completions)
        rewards: list of scalar rewards, same length as rl_inputs

    Returns:
        chosen_segments: top half according to reward
        rejected_segments: bottom half according to reward
    """
    assert len(rl_inputs) == len(rewards), "rl_inputs and rewards must have same length"

    # Sort indices by reward descending
    sorted_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)

    mid = len(sorted_indices) // 2
    chosen_indices = sorted_indices[:mid]
    rejected_indices = sorted_indices[mid:mid*2]  # same length as chosen

    # Select segments
    chosen_segments = [rl_inputs[i] for i in chosen_indices]
    rejected_segments = [rl_inputs[i] for i in rejected_indices]

    # Optional: shuffle within chosen/rejected to avoid ordering bias
    random.shuffle(chosen_segments)
    random.shuffle(rejected_segments)

    return chosen_segments, rejected_segments
