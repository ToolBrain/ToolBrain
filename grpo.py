# Paper DeepSeekMath: https://arxiv.org/pdf/2402.03300

import torch
from torch.nn.utils import clip_grad_norm_
from grpo_utils import *
from losses import grpo_loss
from typing import Callable, List, Tuple


def _policy_device(policy) -> torch.device:
    model = getattr(policy, "llm", None) or getattr(policy, "model", None)
    if model is None:
        raise AttributeError("Could not find underlying model on policy (tried 'llm' and 'model').")
    return next(model.parameters()).device


def update_policy(pi_theta, loss):
    """
    Perform a single optimization step on the policy model using the provided loss.
    Handles locating the underlying model and optimizer, gradient clipping, and optimizer stepping.
    """
    # Locate underlying model
    model = getattr(pi_theta, "llm", None)
    if model is None:
        model = getattr(pi_theta, "model", None)
    if model is None:
        raise AttributeError("Could not find model on pi_theta (tried 'llm' and 'model').")

    # Ensure an optimizer exists
    if not hasattr(pi_theta, "optimizer") or pi_theta.optimizer is None:
        lr = getattr(pi_theta, "lr", 1e-5)
        pi_theta.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = pi_theta.optimizer

    # Training step
    model.train()
    optimizer.zero_grad()
    loss.backward()
    max_grad_norm = getattr(pi_theta, "max_grad_norm", 1.0)
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return pi_theta


def grpo_step(
        initial_policy: Policy,
        traces: List[List[Tuple[str, str]]],
        reward_function: Callable[[List[Tuple[str, str]]], float] | Callable[[Tuple[str, str]], float],
        config: dict
) -> Policy:
    """
    Perform GRPO optimization starting from the initial_policy using given traces and reward function.
    Args:
        initial_policy: The starting policy to be optimized.
        traces: A batch of traces, each a list of (prompt, completion) tuples.
        reward_function: A callable to compute reward from a trace .
        config: Configuration dictionary with keys 'epsilon', 'beta', and 'mu'.
    Returns:
        An updated policy after performing GRPO steps.
    """
    device = _policy_device(initial_policy)  # Determine device of the policy model

    pi_theta = copy_model(initial_policy)

    # Build inputs and move tensors to the policy device
    input_ids, attention_mask, completion_mask, advantages = build_inputs(
        traces=traces,
        tokenizer=pi_theta.tokenizer,
        reward_function=reward_function,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    completion_mask = completion_mask.to(device)
    advantages = advantages.to(device)

    # Compute reference log probabilities once under no grad
    with torch.no_grad():
        pi_ref_log_probs = initial_policy.get_log_probs(input_ids, attention_mask)

    # Perform mu optimization steps
    for _ in range(config["mu"]):
        # Compute current policy log probabilities
        pi_theta_log_probs = pi_theta.get_log_probs(input_ids, attention_mask)

        # Compute GRPO loss
        loss = grpo_loss(
            pi_theta_log_probs=pi_theta_log_probs,
            pi_ref_log_probs=pi_ref_log_probs,
            advantages=advantages,
            epsilon=config["epsilon"],
            beta=config["beta"],
            completion_mask=completion_mask
        )

        # Update policy parameters
        pi_theta = update_policy(pi_theta, loss)

    return pi_theta


if __name__ == "__main__":
    def compute_reward(trace):
        return torch.rand(1, dtype=torch.float32).item()

    llm, tokenizer = get_llm_and_tokenizer_from_smolagent("gpt2")
    initial_policy = Policy(llm=llm, tokenizer=tokenizer)
    traces = [
        [
            (
                "Thought: I can use a simple Python loop to iterate over the numbers from 1 to 10 and sum them up.",
                "Observation: Execution logs:\n55\nLast output from code snippet:\nNone"
            ),
            (
                "Thought: The code snippet has successfully calculated the sum of numbers from 1 to 10, which is 55. Now, I can use the `final_answer` tool to provide the final answer.",
                "Observation: Execution logs:\nLast output from code snippet:\n55"
            )
        ],
        [
            (
                "Thought: To sum all numbers from 1 to 10, I can use a simple loop in Python to iterate over the range of numbers and add them up. I will use the built-in `range` function to generate the numbers from 1 to 10, and a variable to keep track of the sum.",
                "Observation: Execution logs:\n55\nLast output from code snippet:\nNone"
            ),
            (
                "Thought: The code snippet has successfully calculated the sum of all numbers from 1 to 10, which is 55. Now, I can use the `final_answer` tool to provide the final answer.",
                "Observation: Execution logs:\nLast output from code snippet:\n55"
            )
        ],
        [
            (
                "Thought: To sum all numbers from 1 to 10, I can use a simple Python loop to iterate over the range of numbers and add them up. I will use the built-in `range` function to generate the numbers from 1 to 10, and then use a `for` loop to iterate over the range and add each number to a running total.",
                "Observation: Execution logs:\n55\nLast output from code snippet:\nNone"
            ),
            (
                "Thought: The code snippet has successfully calculated the sum of numbers from 1 to 10, which is 55. Now, I can use the `final_answer` tool to provide the final answer.",
                "Observation: Execution logs:\nLast output from code snippet:\n55"
            )
        ]
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