import random
from copy import deepcopy
import torch
from wrapper import CodeAgentWrapper


def copy_model(model):
    return model.copy() if hasattr(model, 'copy') else deepcopy(model)

def sample_batch(dataset, batch_size=4):
    return random.sample(dataset, batch_size)

# Outcome supervision (for the entire output)
def compute_advantanges(rewards, completion_lengths):
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std(unbiased=False)
    normalized_r = (r - mean) / (std + 1e-8)

    advantages = []
    for norm_r_i, length in zip(normalized_r, completion_lengths):
        advantages.extend([norm_r_i] * length)
    return advantages

def grpo_loss(pi_theta_log_probs, pi_old_log_probs, traces, advantages, epsilon, beta):
    """
    Implement Equation 3 for a batch of traces, but use pi_old for pi_ref for simplicity.
    """
    device = pi_theta_log_probs.device
    trace_map = torch.arange(len(traces), device=device)

    # Compute sum of log-probs per completion
    pi_theta_log_probs_sum = pi_theta_log_probs.sum(dim=-1)
    pi_old_log_probs_sum = pi_old_log_probs.sum(dim=-1)

    # Probability ratios
    ratio = torch.exp(pi_theta_log_probs_sum - pi_old_log_probs_sum)
    A_per_completion = advantages  # map back to each completion

    # Clipped surrogate gain
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_gain = torch.min(ratio * A_per_completion, clipped_ratio * A_per_completion)

    # KL regularization (Equation 4)
    # Equation 4: (pi_old / pi_theta) - log(pi_old / pi_theta) - 1
    kl_ratio = torch.exp(log_ratio) # exp(log(pi_old / pi_theta)) = pi_old / pi_theta
    log_ratio = pi_old_log_probs - pi_theta_log_probs  # log(pi_old / pi_theta) = log(pi_old) - log(pi_theta)
    kl_term = (kl_ratio - log_ratio - 1.0).sum(dim=-1)  # shape: (N,)

    # Completion-level loss
    completion_loss = -(policy_gain - beta * kl_term)

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
    # pi_theta is the updated policy model, pi_old is the initial policy model.
    pi_theta = copy_model(initial_policy)

    for _ in range(config["I"]):
        rewards = [reward_function(trace) for trace in traces]
        advantages = compute_advantanges(rewards)

        for _ in range(config["mu"]):
            prompts = [p['prompt'] for p in traces]
            completions = [p['completion'] for p in traces]

            # Get log-probabilities from models
            pi_theta_log_probs, pi_old_log_probs = pi_theta.get_log_probs(
                initial_policy, prompts, completions
            )  # shape: (N, T)
            loss = grpo_loss(
                pi_theta_log_probs=pi_theta_log_probs,
                pi_old_log_probs=pi_old_log_probs,
                traces=traces,
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
        {"prompt": "What is 7 + 6?", "completion": "13"},
        {"prompt": "Which number is greater: 12 or 9?", "completion": "12"},
        {"prompt": "Is 10 an even number?", "completion": "Yes"},
        {"prompt": "What is the square of 5?", "completion": "25"},
        {"prompt": "What is the next number after 99?", "completion": "100"}
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