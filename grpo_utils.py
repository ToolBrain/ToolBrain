import random
from copy import deepcopy
import torch
from transformers import AutoTokenizer


def copy_model(model):
    return model.copy() if hasattr(model, 'copy') else deepcopy(model)

def sample_batch(dataset, batch_size=4):
    return random.sample(dataset, batch_size)

def compute_advantages(rewards):
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    normalized_r = (rewards - mean) / (std + 1e-8)
    return normalized_r

def build_inputs(trace, tokenizer, reward_function):
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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    trace = [
        ("What is the capital of France?", " Paris."),
        ("2 + 2 equals", " 4."),
    ]

    input_ids, completion_mask, rewards, advantages = build_inputs(
        trace=trace,
        tokenizer=tokenizer,
        reward_function=lambda p, c: torch.scalar_tensor(1.0)
    )

    print("Input IDs:", input_ids)
    print("Shape of input_ids:", input_ids.shape)
    print("Completion Mask:", completion_mask)
    print("Shape of completion_mask:", completion_mask.shape)
    print("Rewards:", rewards)
    print("Shape of rewards:", rewards.shape)
    print("Advantages:", advantages)
    print("Shape of advantages:", advantages.shape)
    print("Decoded:", tokenizer.decode(input_ids))