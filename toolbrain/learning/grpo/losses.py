import torch

def grpo_loss(
        pi_theta_logps: torch.Tensor,
        pi_theta_old_logps: torch.Tensor,
        pi_ref_logps: torch.Tensor,
        advantages: torch.Tensor,
        epsilon: float,
        beta: float,
        completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes the GRPO loss function as defined in Equation (3) of the DeepSeekMath paper at https://arxiv.org/pdf/2402.03300
    This implementation returns the negative of the original gain (objective) for optimization via gradient descent.
    """

    # Clipped surrogate gain
    log_ratio = pi_theta_logps - pi_theta_old_logps  # log(pi_theta) - log(pi_old) = log(pi_theta / pi_old) 
    ratio = torch.exp(log_ratio)  # exp(log(pi_theta / pi_old)) = pi_theta / pi_old
    unclipped = ratio * advantages  # shape: (B, L)

    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped_ratio * advantages  # shape: (B, L)

    policy_gain = torch.min(unclipped, clipped)  # shape: (B, L)

    # KL divergence (Equation 4)
    # Equation 4: (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    log_kl_ratio = pi_ref_logps - pi_theta_logps  # log(pi_ref) - log(pi_theta) = log(pi_ref / pi_theta)
    kl_ratio = torch.exp(log_kl_ratio)  # exp(log(pi_ref / pi_theta)) = pi_ref / pi_theta
    kl_divergence = kl_ratio - log_kl_ratio - 1.0  # shape: (B, L)

    # Token-level loss
    per_token_loss = -(policy_gain - beta * kl_divergence)  # shape: (B, L)

    # Average loss
    # 1) Average over non-masked tokens for each completion
    # 2) Average equally across completions in the group
    mask = completion_mask.to(per_token_loss.dtype)
    valid_counts = mask.sum(dim=1).clamp_min(1.0)
    per_completion_mean = ((per_token_loss * mask).sum(dim=1)) / valid_counts  # shape: (B,)

    loss = per_completion_mean.mean()  # scalar
    return loss


if __name__ == "__main__":
    pi_theta_logps = torch.log(torch.tensor([
        [0.2, 0.3, 0.5, 0.0, 0.0],
        [0.1, 0.4, 0.0, 0.0, 0.0],
        [0.3, 0.3, 0.2, 0.1, 0.1]
    ]) + 1e-6)

    pi_theta_old_logps = pi_theta_logps - 0.05
    pi_ref_logps = pi_theta_logps - 0.02

    advantages = torch.tensor([
        [1.0, 0.5, -0.5, 0.0, 0.0],
        [0.3, -0.2, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5, 0.5]
    ])

    epsilon = 0.2
    beta = 0.01

    completion_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ])

    loss = grpo_loss(
        pi_theta_logps,
        pi_theta_old_logps,
        pi_ref_logps,
        advantages,
        epsilon,
        beta,
        completion_mask
    )
