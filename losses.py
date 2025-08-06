import torch

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