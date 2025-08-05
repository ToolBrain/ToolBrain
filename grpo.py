import random
from copy import deepcopy
import torch


def copy_model(model):
    return model.copy() if hasattr(model, 'copy') else deepcopy(model)

def sample_batch(dataset, batch_size=4):
    return random.sample(dataset, batch_size)

def compute_advantanges(rewards):
    pass

def grpo_loss(pi_theta, pi_old, traces, advantages, epsilon, beta):

    pairs = traces
    device = advantages.device
    trace_map = torch.arange(len(traces), device=device)


    all_prompts = [p['prompt'] for p in pairs]
    all_completions = [p['completion'] for p in pairs]

    # Get log-probabilities from models
    current_log_p_batch, old_log_p_batch, completion_lengths = pi_theta.get_log_probs_for_batch(
        pi_old, all_prompts, all_completions
    )  # shape: (N, T)

    device = current_log_p_batch.device
    trace_map = torch.arange(len(traces), device=device)
    advantages = advantages.to(device)

    # === GRPO loss ===

    # Compute sum of log-probs per completion
    current_log_p_sum = current_log_p_batch.sum(dim=-1)
    old_log_p_sum = old_log_p_batch.sum(dim=-1)

    # Probability ratios
    ratio = torch.exp(current_log_p_sum - old_log_p_sum)
    A_per_completion = advantages  # map back to each completion

    # Clipped surrogate gain
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_gain = torch.min(ratio * A_per_completion, clipped_ratio * A_per_completion)

    # === KL regularization (Equation 4, KL[pi_theta || pi_old]) ===
    # log(pi_theta) - log(pi_old)
    log_ratio = current_log_p_batch - old_log_p_batch
    kl_term = (torch.exp(log_ratio) - log_ratio - 1.0).sum(dim=-1)  # sum over tokens

    # Completion-level loss
    completion_loss = -(policy_gain - beta * kl_term)

    # === Aggregate per-trace ===
    trace_losses = torch.zeros(len(traces), device=device)
    trace_counts = torch.zeros(len(traces), device=device)

    for i, loss in enumerate(completion_loss):
        trace_idx = trace_map[i]
        trace_losses[trace_idx] += loss
        trace_counts[trace_idx] += 1

    trace_losses = trace_losses / trace_counts.clamp(min=1.0)
    return trace_losses.mean()

def update_policy(pi_theta, loss):
    pass



def train_grpo(
        initial_policy,
        traces,
        reward_function,
        config,
):
    # pi_theta is the updated policy model
    pi_theta = copy_model(initial_policy)

    for _ in range(config["I"]):
        rewards = [reward_function(trace) for trace in traces]
        advantages = compute_advantanges(rewards)

        for _ in range(config["mu"]):
            loss = grpo_loss(pi_theta, initial_policy, traces, advantages, config["epsilon"], config["beta"])
            pi_theta = update_policy(pi_theta, loss)

    return pi_theta


if __name__ == "__main__":
    initial_policy = None
    traces = []
    reward_function = None
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
        reward_function=reward_function,
        config=config
    )