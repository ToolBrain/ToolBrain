# Paper DeepSeekMath: https://arxiv.org/pdf/2402.03300


import torch
from wrapper import CodeAgentWrapper
from grpo_utils import *


# Outcome supervision
# Calculate advantages for the entire output as described in section 4.1.2
def compute_advantanges(traces, rewards, tokenizer):
    """
    Computes advantages from traces.
    Args:
        traces: List of (prompt: str, completion: str) tuples.
        rewards: List of rewards for each trace.
        tokenizer: A HuggingFace tokenizer (already loaded).
    Returns:
        advantages: A torch tensor of shape (N,) where N is the total number of completion tokens.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completion_lengths = [len(tokenizer.encode(completion, add_special_tokens=False)) for _, completion in traces]

    r = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean = r.mean()
    std = r.std(unbiased=False)
    normalized_r = (r - mean) / (std + 1e-8)

    advantages = torch.cat([
        normalized_r[i].repeat(length)
        for i, length in enumerate(completion_lengths)
    ])
    return advantages


def grpo_loss(pi_theta_log_probs, pi_ref_log_probs, advantages, epsilon, beta, completion_mask):
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
    per_token_loss = -(policy_gain - beta * kl_term)  # shape: (N, T)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss # scalar loss value


def update_policy(pi_theta, loss):
    return pi_theta


def train_grpo(
        initial_policy,
        traces,
        reward_function,
        config,
):
    pi_theta = copy_model(initial_policy)

    for _ in range(config["I"]):
        rewards = reward_function(traces)
        advantages = compute_advantanges(traces, rewards, pi_theta.tokenizer)
        print("Advantages:", advantages.shape)

        # for _ in range(config["mu"]):
        #     input_ids, completion_mask = build_input_and_completion_mask(traces, pi_theta.tokenizer)
            
        #     # Get log-probabilities from models
        #     pi_theta_log_probs = pi_theta.get_log_probs(input_ids)  # shape: (B, L-1)
        #     pi_ref_log_probs = initial_policy.get_log_probs(input_ids)  # shape: (B, L-1)

        #     loss = grpo_loss(
        #         pi_theta_log_probs=pi_theta_log_probs,
        #         pi_ref_log_probs=pi_ref_log_probs,
        #         advantages=advantages,
        #         epsilon=config["epsilon"],
        #         beta=config["beta"],
        #         completion_mask=completion_mask
        #     )
        #     pi_theta = update_policy(pi_theta, loss)

    return pi_theta


if __name__ == "__main__":
    import random
    def compute_reward(traces):
        return [random.uniform(0, 1) for _ in traces]

    initial_policy = CodeAgentWrapper("gpt2")
    traces = [
        ("What is 7 + 6?", "13"),
        ("Which number is greater: 12 or 9?", "12"),
    ]
    config = {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "mu": 2, # Number of GRPO optimization steps per batch
        "I": 1, # Number of policy updates
        "M": 1, # Number of batches per iteration
    }
    new_policy = train_grpo(
        initial_policy=initial_policy,
        traces=traces,
        reward_function=compute_reward,
        config=config
    )