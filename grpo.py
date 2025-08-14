# Paper DeepSeekMath: https://arxiv.org/pdf/2402.03300

from typing import Callable, List, Tuple, Optional
import torch
from torch.nn.utils import clip_grad_norm_
from grpo_utils import Policy, copy_model, get_llm_and_tokenizer_from_smolagent, build_inputs, Turn
from losses import grpo_loss


class GRPOAlgorithm:
    """Lightweight trainer that wraps GRPO optimization around a Policy.

    Usage:
        algo = GRPOAlgorithm(policy)
        algo.train_step(traces, rewards)
    """
    def __init__(
        self,
        initial_policy: Policy,
        ref_policy: Policy = None,
        config: dict = None,
    ) -> None:
        self.policy = initial_policy
        # the reference model, usually the initial Supervised Fine-Tuning model
        self.pi_ref = ref_policy if ref_policy else copy_model(initial_policy) 
        self.config = config
        self.training_steps = 0
        self.device = next(initial_policy.llm.parameters()).device
        self.optimizer = torch.optim.AdamW(self.policy.llm.parameters(), lr=self.config["lr"])

    def _update_policy(self, pi_theta, loss):
        """Apply one optimizer step using the algorithm's optimizer and gradient clipping."""
        model = getattr(pi_theta, "llm", None) or getattr(pi_theta, "model", None)
        if model is None:
            raise AttributeError("No model found in pi_theta")
        
        model.train()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()
        return pi_theta

    def train_step(
        self,
        traces: List[List[Turn]],
        rewards: List[float],
    ) -> Policy:
        """Run one GRPO update over a batch of traces.

        Args:
            traces: batch of traces; each trace is a list TraceStep.
            rewards: list of scalar rewards, one per trace.
        Returns:
            Updated Policy (also stored in self.policy).
        """
        device = self.device
        pi_theta = copy_model(self.policy).to(self.device)
        assert len(traces) == len(rewards)

        batch = build_inputs(
            traces=traces,
            rewards=rewards,
            tokenizer=pi_theta.tokenizer
        )
 
        input_ids = batch.input_ids.to(device) # shape: (N, T)
        attention_mask = batch.attention_mask.to(device) # shape: (N, T)
        completion_mask = batch.completion_mask.to(device) # shape: (N, T)
        advantages = batch.advantages.to(device) # shape: (N, T)

        # Prepare old-policy (for ratio) and a fixed reference (for KL) log-probs.
        #   - pi_old_logps: starts as current pre-update policy; will be refreshed each grpo loss step.
        #   - pi_ref_logps: fixed reference for KL across the grpo iteration (use pre-update self.policy).
        with torch.no_grad():
            pi_old_logps = pi_theta.get_log_probs(input_ids, attention_mask) # shape: (N, T-1)
            pi_ref_logps = self.pi_ref.get_log_probs(input_ids, attention_mask) # shape: (N, T-1)

        for _ in range(self.config["mu"]):
            # Current policy log-probs
            pi_theta_logps = pi_theta.get_log_probs(input_ids, attention_mask) # shape: (N, T-1)

            loss = grpo_loss(
                pi_theta_log_probs=pi_theta_logps,
                pi_theta_old_log_probs=pi_old_logps,
                pi_ref_log_probs=pi_ref_logps,
                advantages=advantages[:,1:], # shape: (N, T-1)
                epsilon=self.config["epsilon"],
                beta=self.config["beta"],
                completion_mask=completion_mask[:,1:], # shape: (N, T-1)
            )

            # Apply update
            pi_theta = self._update_policy(pi_theta, loss)

            # Cache current log-probs as next step's old-policy (detach from graph)
            pi_old_logps = pi_theta_logps.detach()

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
        # Placeholder: assign a random scalar per-trace reward
        return torch.rand(1, dtype=torch.float32).item()

    llm, tokenizer = get_llm_and_tokenizer_from_smolagent("gpt2")
    initial_policy = Policy(llm=llm, tokenizer=tokenizer)
    traces: List[List[Turn]] = [
        [
            Turn(
                prompt_for_model="You are a Python assistant. Compute the sum of 1..10 and explain briefly.",
                model_completion="Thought: I'll write a short Python loop.\n```python\ns=sum(range(1,11)); print(s)\n```\n",
                parsed_completion={"thought": True, "has_code": True},
                tool_output="Execution logs:\n55\nLast output from code snippet:\n55",
            ),
            Turn(
                prompt_for_model="Given the tool output above, provide the final answer.",
                model_completion="Final Answer: 55",
                parsed_completion={"final_answer": True},
                tool_output="",
            ),
        ],
        [
            Turn(
                prompt_for_model="You are a math helper. Sum 1..10.",
                model_completion="I can compute it mentally: 55.",
                parsed_completion={"final_answer": True},
                tool_output="",
            )
        ],
        [
            Turn(
                prompt_for_model="Explain quickly and give the result for 1..10.",
                model_completion="Summing 1 through 10 gives 55.",
                parsed_completion={"final_answer": True},
                tool_output="",
            )
        ],
    ]
    rewards = [compute_reward(trace) for trace in traces]
    config = {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "mu": 3, # Number of GRPO optimization steps per batch
        "lr": 1e-5, # Learning rate for optimizer
        "max_grad_norm" :1.0, 
    }

    algo = GRPOAlgorithm(
        initial_policy=initial_policy,
        config=config
    )

    algo.train_step(traces=traces, rewards=rewards)