import torch

def grpo_loss(
        pi_theta_log_probs: torch.Tensor,
        pi_theta_old_log_probs: torch.Tensor,
        pi_ref_log_probs: torch.Tensor,
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
    log_ratio = pi_theta_log_probs - pi_theta_old_log_probs  # log(pi_theta) - log(pi_old) = log(pi_theta / pi_old) 
    ratio = torch.exp(log_ratio)  # exp(log(pi_theta / pi_old)) = pi_theta / pi_old
    unclipped = ratio * advantages  # shape: (N,T)

    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped_ratio * advantages  # shape: (N,T)

    policy_gain = torch.min(unclipped, clipped)  # shape: (N,T)

    # KL divergence (Equation 4)
    # Equation 4: (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    log_kl_ratio = pi_ref_log_probs - pi_theta_log_probs  # log(pi_ref) - log(pi_theta) = log(pi_ref / pi_theta)
    kl_ratio = torch.exp(log_kl_ratio)  # exp(log(pi_ref / pi_theta)) = pi_ref / pi_theta
    kl_divergence = kl_ratio - log_kl_ratio - 1.0  # shape: (N,T)

    # Token-level loss
    per_token_loss = -(policy_gain - beta * kl_divergence)  # shape: (N,T)

    # Average loss
    # 1) Average over non-masked tokens for each completion
    # 2) Average equally across completions in the group
    mask = completion_mask.to(per_token_loss.dtype)
    valid_counts = mask.sum(dim=1).clamp_min(1.0)
    per_completion_mean = ((per_token_loss * mask).sum(dim=1)) / valid_counts  # shape: (N,)

    loss = per_completion_mean.mean()  # scalar
    return loss


if __name__ == "__main__":
    pi_theta_log_probs = torch.log(torch.tensor([
        [0.2, 0.3, 0.5, 0.0, 0.0],
        [0.1, 0.4, 0.0, 0.0, 0.0],
        [0.3, 0.3, 0.2, 0.1, 0.1]
    ]) + 1e-6)

    pi_theta_old_log_probs = pi_theta_log_probs - 0.05
    pi_ref_log_probs = pi_theta_log_probs - 0.02

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
        pi_theta_log_probs,
        pi_theta_old_log_probs,
        pi_ref_log_probs,
        advantages,
        epsilon,
        beta,
        completion_mask
    )

    print("Test loss:", loss.item()) # -0.30953896045684814