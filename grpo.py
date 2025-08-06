# Paper DeepSeekMath: https://arxiv.org/pdf/2402.03300

import torch
from wrapper import CodeAgentWrapper
from grpo_utils import *

def grpo_loss(
        pi_theta_log_probs: torch.Tensor,
        pi_ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        epsilon: float,
        beta: float,
        completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes the GRPO loss function as defined in Equation (3) of the DeepSeekMath paper.
    This implementation returns the negative of the original gain (objective) for optimization via gradient descent.
    """

    # Clipped surrogate gain
    log_ratio = pi_theta_log_probs - pi_ref_log_probs  # log(pi_theta) - log(pi_ref) = log(pi_theta / pi_ref)
    ratio = torch.exp(log_ratio)  # exp(log(pi_theta / pi_ref)) = pi_theta / pi_ref
    unclipped = ratio * advantages  # shape: (T,)

    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped_ratio * advantages  # shape: (T,)

    policy_gain = torch.min(unclipped, clipped)  # shape: (T,)

    # KL divergence (Equation 4)
    # Equation 4: (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    log_kl_ratio = pi_ref_log_probs - pi_theta_log_probs  # log(pi_ref) - log(pi_theta) = log(pi_ref / pi_theta)
    kl_ratio = torch.exp(log_kl_ratio)  # exp(log(pi_ref / pi_theta)) = pi_ref / pi_theta
    kl_term = kl_ratio - log_kl_ratio - 1.0  # shape: (T,)

    # Token-level loss
    per_token_loss = -(policy_gain - beta * kl_term)  # shape: (T,)
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    return loss # scalar loss value


def update_policy(pi_theta, loss):
    return pi_theta


def grpo_step(
        initial_policy: CodeAgentWrapper,
        trace: list[tuple[str, str]],
        reward_function: callable,
        config: dict
) -> CodeAgentWrapper:
    
    pi_theta = copy_model(initial_policy)

    input_ids, completion_mask, rewards, advantages = build_inputs(
        trace=trace,
        tokenizer=pi_theta.tokenizer,
        reward_function=reward_function,
    ) # All have shape (T,)

    for _ in range(config["mu"]):

        # Get log-probabilities from models
        pi_theta_log_probs = pi_theta.get_log_probs(input_ids)  # shape: (T,)
        pi_ref_log_probs = initial_policy.get_log_probs(input_ids)  # shape: (T,)

        loss = grpo_loss(
            pi_theta_log_probs=pi_theta_log_probs,
            pi_ref_log_probs=pi_ref_log_probs,
            advantages=advantages,
            epsilon=config["epsilon"],
            beta=config["beta"],
            completion_mask=completion_mask
        )
        pi_theta = update_policy(pi_theta, loss)

    return pi_theta


if __name__ == "__main__":
    def compute_reward(prompt, completion):
        return torch.rand(1, dtype=torch.float32).item()

    initial_policy = CodeAgentWrapper("gpt2")
    trace = [
        ("What is 7 + 6?", "13"),
        ("Which number is greater: 12 or 9?", "12"),
    ]
    config = {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "mu": 1, # Number of GRPO optimization steps per batch
    }
    new_policy = grpo_step(
        initial_policy=initial_policy,
        trace=trace,
        reward_function=compute_reward,
        config=config
    )