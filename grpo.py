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