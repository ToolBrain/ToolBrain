# Paper DeepSeekMath: https://arxiv.org/pdf/2402.03300

from typing import Callable, List, Tuple, Optional
import torch
from torch.nn.utils import clip_grad_norm_
from grpo_utils import Policy, copy_model, get_llm_and_tokenizer_from_smolagent
from losses import grpo_loss



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




class GRPOAlgorithm:
    """Lightweight trainer that wraps GRPO optimization around a Policy.

    Usage:
        algo = GRPOAlgorithm(policy)
        algo.train_step(traces, rewards)
    """
    def __init__(
        self,
        policy: Policy,
        config: dict = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.training_steps = 0

    @property
    def device(self) -> torch.device:
        return _policy_device(self.policy)

    def train_step(
        self,
        traces: List[List[Tuple[str, str]]],
        rewards: List[float],
    ) -> Policy:
        """Run one GRPO update over a batch of traces.

        Args:
            traces: batch of traces; each trace is a list of (prompt, completion) pairs.
            rewards: list of scalar rewards, one per trace.
        Returns:
            Updated Policy (also stored in self.policy).
        """
        device = _policy_device(self.policy)
        pi_theta = copy_model(self.policy)
        assert len(traces) == len(rewards)

        batch = pi_theta.encode_traces(traces=traces, rewards=rewards)
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        completion_mask = batch.completion_mask.to(device)
        advantages = batch.advantages.to(device)

        with torch.no_grad():
            pi_ref_logps = self.policy.get_log_probs(input_ids, attention_mask)

        for _ in range(self.config["mu"]):
            pi_theta_logps = pi_theta.get_log_probs(input_ids, attention_mask)
            loss = grpo_loss(
                pi_theta_log_probs=pi_theta_logps,
                pi_ref_log_probs=pi_ref_logps,
                advantages=advantages,
                epsilon=self.config["epsilon"],
                beta=self.config["beta"],
                completion_mask=completion_mask
            )
            pi_theta = update_policy(pi_theta, loss)
        self.policy = pi_theta
        self.training_steps += 1
        return pi_theta

    def __repr__(self) -> str:
        return (
            f"GRPOAlgorithm(epsilon={self.config.get('epsilon')}, beta={self.config.get('beta')}, "
            f"mu={self.config.get('mu')}, steps={self.training_steps})"
        )


if __name__ == "__main__":
    def compute_reward(trace):
        return torch.rand(1, dtype=torch.float32).item()

    llm, tokenizer = get_llm_and_tokenizer_from_smolagent("gpt2")
    initial_policy = Policy(llm=llm, tokenizer=tokenizer)
    ref_policy = copy_model(initial_policy)
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
    rewards = [compute_reward(trace) for trace in traces]
    config = {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "mu": 1, # Number of GRPO optimization steps per batch
    }

    algo = GRPOAlgorithm(
        policy=initial_policy,
        ref_policy=ref_policy,
        config=config
    )

    algo.train_step(traces=traces, rewards=rewards)