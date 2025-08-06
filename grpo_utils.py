from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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
    trace: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    reward_function: callable
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        trace: List of (prompt: str, completion: str) tuples.
        tokenizer: A HuggingFace tokenizer (already loaded).
    
    Returns:
        input_ids: torch.LongTensor [total_seq_len]
        completion_mask: torch.BoolTensor [total_seq_len]
    """
    all_input_ids = []
    all_completion_mask = []
    all_rewards = []
    token_counts = []

    for prompt, completion in trace:
        # Get tokenized input
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        all_input_ids.extend(prompt_ids + completion_ids)

        # Count total tokens
        token_counts.append(len(prompt_ids) + len(completion_ids))
        
        # Create completion mask
        completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        all_completion_mask.extend(completion_mask)

        # Compute reward for each completion
        reward_per_completion = reward_function(prompt, completion)
        all_rewards.extend([reward_per_completion] * (len(prompt_ids) + len(completion_ids)))
        

    input_ids = torch.tensor(all_input_ids, dtype=torch.long) # Shape: (T,)
    completion_mask = torch.tensor(all_completion_mask, dtype=torch.int) # Shape: (T,)
    rewards = torch.tensor(all_rewards, dtype=torch.float32) # Shape: (T,)
    advantages = compute_advantages(rewards) # Shape: (T,)

    assert input_ids.shape[0] == completion_mask.shape[0] == rewards.shape[0] == advantages.shape[0], \
        "Input IDs, completion mask, rewards, and advantages must have the same length."

    return input_ids, completion_mask, rewards, advantages

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

    def get_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.llm(input_ids.unsqueeze(0)).logits  # (N, T, V)
        #logits = logits[:, :-1, :]  # (N, T-1, V), exclude the last logit: it corresponds to the next token pred
        
        input_ids = input_ids.unsqueeze(0)  # (N, T), add batch dimension
        #input_ids = input_ids[:, 1:]  # (N, T-1), exclude the first input ID since we don't have logits for it
        
        per_token_logps = get_per_token_logps(logits, input_ids).squeeze(0) # Shape: (T)
        # per_token_logps = per_token_logps[completion_mask==1.0] # Only keep log probabilities for completion tokens
        return per_token_logps


if __name__ == "__main__":
    model_id = "gpt2"

    llm, tokenizer = get_llm_and_tokenizer_from_smolagent(model_id=model_id)
    policy_model = Policy(llm=llm, tokenizer=tokenizer)

    trace = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen wrote 'Pride and Prejudice'."),
    ]
    input_ids, completion_mask, rewards, advantages = build_inputs(
        trace=trace,
        tokenizer=tokenizer,
        reward_function=lambda p, c: 1.0)
    log_probs = policy_model.get_log_probs(input_ids)

    print("Log probabilities shape:", log_probs.shape)
    print("Log probabilities:", log_probs)
    print("Input IDs:", input_ids)
    print("Shape of input_ids:", input_ids.shape)
    print("Completion Mask:", completion_mask)
    print("Shape of completion_mask:", completion_mask.shape)
    print("Rewards:", rewards)
    print("Shape of rewards:", rewards.shape)
    print("Advantages:", advantages)
    print("Shape of advantages:", advantages.shape)
    print("Decoded:", tokenizer.decode(input_ids))