# Paper DeepSeekMath: https://arxiv.org/pdf/2402.03300
import copy
from typing import List

import torch
from torch.nn.utils import clip_grad_norm_

from .utils import Policy, build_inputs
from .losses import grpo_loss
from ...core_types import Trace, Turn, ParsedCompletion, ChatSegment


def validate_config(config: dict) -> None:
    """
    Validate that the config dict contains all the required keys.
    Raises:
        ValueError: If any required keys are missing.
    """
    required_keys = {
        "epsilon",
        "beta",
        "opt_steps",
        "lr",
        "max_grad_norm",
        "chunk_len"
    }
    
    for key in required_keys:
        if key not in config.keys():
            raise ValueError(f"Invalid config: missing '{key}'")

class GRPOAlgorithm:
    """Lightweight trainer that wraps GRPO optimization around a Policy.

    Usage:
        algo = GRPOAlgorithm(policy)
        algo.train_step(traces, rewards)
    """
    def __init__(
        self,
        initial_policy: Policy,
        config: dict,
        ref_policy: Policy = None,
    ) -> None:
        self.device = next(initial_policy.llm.parameters()).device

        self.policy = initial_policy
        self.policy = self.policy.to(self.device)

        self.pi_ref = ref_policy if ref_policy else copy.deepcopy(initial_policy)
        self.pi_ref = self.pi_ref.to(self.device)

        validate_config(config)
        self.config = config

        self.training_steps = 0        
        self.optimizer = torch.optim.AdamW(
            self.policy.llm.parameters(),
            lr=self.config["lr"]
        )

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
        segments: List[List[ChatSegment]],
        rewards: List[float],
    ) -> None:
        """Run one GRPO update over a batch of traces.

        Args:
            traces: batch of traces; each trace is a list of Trace.
            rewards: list of scalar rewards, one per trace.
        """
        device = self.device
        pi_theta = self.policy  # train the main policy in-place to keep optimizer params in sync
        assert len(segments) == len(rewards), f"Length of traces and rewards must be the same. Received {len(traces)} traces, {len(rewards)} rewards."

        batch = build_inputs(
            segments=segments,
            rewards=rewards,
            tokenizer=pi_theta.tokenizer
        )
 
        input_ids = batch.input_ids.to(device) # shape: (B, L)
        attention_mask = batch.attention_mask.to(device) # shape: (B, L)
        completion_mask = batch.completion_mask.to(device) # shape: (B, L)
        advantages = batch.advantages.to(device) # shape: (B, L)

        # Prepare old-policy (for ratio) and a fixed reference (for KL) log-probs.
        #   - pi_theta_old_logps: starts as current pre-update policy; will be refreshed each grpo loss step.
        #   - pi_ref_logps: fixed reference for KL across the grpo iteration (use pre-update self.policy).
        chunk_len = self.config["chunk_len"]
        with torch.no_grad():
            pi_theta_old_logps = pi_theta.get_per_token_logps(
                input_ids=input_ids,                  # shape: (B, L)
                attention_mask=attention_mask,        # shape: (B, L)
                chunk_len=chunk_len
            )                                         # shape: (B, L-1)
            pi_ref_logps = self.pi_ref.get_per_token_logps(
                input_ids=input_ids,                  # shape: (B, L)
                attention_mask=attention_mask,        # shape: (B, L)
                chunk_len=chunk_len
            )                                         # shape: (B, L-1)

        for _ in range(self.config["opt_steps"]):
            # Current policy log-probs
            # get_per_token_logps drops the first token after logits computation, before per-token logprobs.
            pi_theta_logps = pi_theta.get_per_token_logps(input_ids, attention_mask, chunk_len=chunk_len) # shape: (B, L-1)

            # Must shift advantages and completion_mask by 1 token
            # so their shapes match the (B, L-1) log-probs tensors
            loss = grpo_loss(
                pi_theta_logps=pi_theta_logps,         # shape: (B, L-1)
                pi_theta_old_logps=pi_theta_old_logps, # shape: (B, L-1)
                pi_ref_logps=pi_ref_logps,             # shape: (B, L-1)
                advantages=advantages[:,1:],           # shape: (B, L-1)
                completion_mask=completion_mask[:,1:], # shape: (B, L-1)
                epsilon=self.config["epsilon"],
                beta=self.config["beta"],
            )

            # Apply update
            pi_theta = self._update_policy(pi_theta, loss)

            # Cache current log-probs as next step's old-policy (detach from graph)
            pi_theta_old_logps = pi_theta_logps.detach()

        self.policy = pi_theta
        self.training_steps += 1

    def __repr__(self) -> str:
        return (
            f"GRPOAlgorithm(epsilon={self.config.get('epsilon')}, beta={self.config.get('beta')}, "
            f"opt_steps={self.config.get('opt_steps')}, steps={self.training_steps})"
        )


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Traces
    traces: List[Trace] = [
        [
            Turn(
                prompt_for_model="You are a Python assistant. Compute the sum of 1..10 and explain briefly.",
                model_completion="Thought: I'll write a short Python loop.\n```python\ns=sum(range(1,11)); print(s)\n```\n",
                parsed_completion=ParsedCompletion(thought="thought", code="some code"),
                tool_output="Execution logs:\n55\nLast output from code snippet:\n55",
            ),
            Turn(
                prompt_for_model="Given the tool output above, provide the final answer.",
                model_completion="Final Answer: 55",
                parsed_completion=ParsedCompletion(),
                tool_output="",
            ),
        ],
        [
            Turn(
                prompt_for_model="You are a math helper. Sum 1..10.",
                model_completion="I can compute it mentally: 55.",
                parsed_completion=ParsedCompletion(),
            )
        ],
    ]

    # Policy
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_id)
    initial_policy = Policy(llm=llm, tokenizer=tokenizer)

    # Rewards
    rewards = torch.rand(len(traces))

    # Config
    config = {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "opt_steps": 3, # Number of GRPO optimization steps per batch
        "lr": 1e-5, # Learning rate for optimizer
        "max_grad_norm" :1.0, 
        "chunk_len": 128, # If not None, get_per_token_logps will process in chunks
    }

    algo = GRPOAlgorithm(
        initial_policy=initial_policy,
        config=config
    )

    algo.train_step(traces=traces, rewards=rewards)