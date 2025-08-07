# Paper DeepSeekMath: https://arxiv.org/pdf/2402.03300

import torch
from grpo_utils import *
from losses import grpo_loss


def update_policy(pi_theta, loss):
    return pi_theta


def grpo_step(
        initial_policy: Policy,
        traces: list[list[tuple[str, str]]],
        reward_function: callable,
        config: dict
) -> Policy:
    
    pi_theta = copy_model(initial_policy)

    input_ids, completion_mask, advantages = build_inputs(
        traces=traces,
        tokenizer=pi_theta.tokenizer,
        reward_function=reward_function,
    ) # All have shape (N,T)

    for _ in range(config["mu"]):
        # Get log-probabilities from models
        pi_theta_log_probs = pi_theta.get_log_probs(input_ids)  # shape: (N,T)
        pi_ref_log_probs = initial_policy.get_log_probs(input_ids)  # shape: (N,T)

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
    def compute_reward(trace):
        return torch.rand(1, dtype=torch.float32).item()

    llm, tokenizer = get_llm_and_tokenizer_from_smolagent("gpt2")
    initial_policy = Policy(llm=llm, tokenizer=tokenizer)
    traces = [
        [
            ("What is 7 + 6?", "13"),
            ("Which number is greater: 12 or 9?", "12"),
        ],
        [
            ("What is 7 + 6?", "13"),
            ("Which number is greater: 12 or 9?", "12"),
        ],

    ]
    
    config = {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "mu": 1, # Number of GRPO optimization steps per batch
    }
    new_policy = grpo_step(
        initial_policy=initial_policy,
        traces=traces,
        reward_function=compute_reward,
        config=config
    )