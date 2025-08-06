# Paper DeepSeekMath: https://arxiv.org/pdf/2402.03300


import torch
from wrapper import CodeAgentWrapper
from grpo_utils import *


# Outcome supervision
# Calculate advantages for the entire output as described in section 4.1.2
def compute_advantanges(rewards, completion_lengths):
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std(unbiased=False)
    normalized_r = (r - mean) / (std + 1e-8)

    advantages = []
    for norm_r_i, length in zip(normalized_r, completion_lengths):
        advantages.extend([norm_r_i] * length)
    return advantages


def build_input_and_completion_mask(pairs, tokenizer):
    """
    Args:
        pairs: List of (prompt: str, completion: str) tuples.
        tokenizer: A HuggingFace tokenizer (already loaded).
    
    Returns:
        input_ids: torch.LongTensor [total_seq_len]
        completion_mask: torch.BoolTensor [total_seq_len]
    """
    all_input_ids = []
    all_completion_mask = []

    for prompt, completion in pairs:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)

        all_input_ids.extend(prompt_ids + completion_ids)
        completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        all_completion_mask.extend(completion_mask)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    completion_mask = torch.tensor(all_completion_mask, dtype=torch.bool)

    return input_ids, completion_mask



def grpo_loss(pi_theta_log_probs, pi_ref_log_probs, advantages, epsilon, beta):
    """
    GRPO objective function as described in Equation 3 but return the loss instead of the gain.
    """
    # Clipped surrogate gain
    log_ratio = pi_theta_log_probs - pi_ref_log_probs  # log(pi_theta) - log(pi_ref) = log(pi_theta / pi_ref)
    ratio = torch.exp(log_ratio)  # exp(log(pi_theta / pi_ref)) = pi_theta / pi_ref
    unclipped = ratio * advantages  # shape: (N, T)

    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped_ratio * advantages  # shape: (N, T)

    policy_gain = torch.min(unclipped, clipped)  # shape: (N, T)

    # KL divergence (Equation 4)
    # Equation 4: (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    log_kl_ratio = pi_ref_log_probs - pi_theta_log_probs  # log(pi_ref) - log(pi_theta) = log(pi_ref / pi_theta)
    kl_ratio = torch.exp(log_kl_ratio)  # exp(log(pi_ref / pi_theta)) = pi_ref / pi_theta
    kl_term = kl_ratio - log_kl_ratio - 1.0  # shape: (N, T)

    # Token-level loss
    loss = -(policy_gain - beta * kl_term)  # shape: (N, T)
    return loss.mean() # scalar loss value


def update_policy(pi_theta, loss):
    pass



def train_grpo(
        initial_policy,
        traces,
        reward_function,
        config,
):
    pi_theta = copy_model(initial_policy)

    for _ in range(config["I"]):
        rewards = [reward_function(trace) for trace in traces]
        advantages = compute_advantanges(rewards)

        for _ in range(config["mu"]):
            prompts = [p['prompt'] for p in traces]
            completions = [p['completion'] for p in traces]

            # Get log-probabilities from models
            pi_theta_log_probs, pi_ref_log_probs = pi_theta.get_log_probs(
                initial_policy, prompts, completions
            )  # shape: (N, T)

            loss = grpo_loss(
                pi_theta_log_probs=pi_theta_log_probs,
                pi_ref_log_probs=pi_ref_log_probs,
                advantages=advantages,
                epsilon=config["epsilon"],
                beta=config["beta"]
            )
            pi_theta = update_policy(pi_theta, loss)

    return pi_theta


if __name__ == "__main__":
    def get_groundtruth(prompt):
        return "ground_truth_answer_for_" + prompt

    def compute_reward(prompt, completion):
        ground_truth = get_groundtruth(prompt)
        return 1.0 if completion == ground_truth else 0.0

    initial_policy = CodeAgentWrapper("gpt2")
    traces = [
        ("What is 7 + 6?", "13"),
        ("Which number is greater: 12 or 9?", "12"),
        ("Sum all number from 1 to 10", "55"),
        ("What is the capital of France?", "Paris"),
    ]
    config = {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "mu": 2, # Number of GRPO optimization steps per batch
        "I": 3, # Number of policy updates
        "M": 4, # Number of batches per iteration
    }
    new_policy = train_grpo(
        initial_policy=initial_policy,
        traces=traces,
        reward_function=compute_reward,
        config=config
    )