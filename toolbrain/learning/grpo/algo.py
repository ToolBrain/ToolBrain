"""GRPO (Group Relative Policy Optimization) implementation.

This module implements the GRPO algorithm, which performs ratio-based policy updates
relative to a frozen “old” policy, while regularizing with a KL divergence penalty
to a fixed reference policy. The implementation is based on the paper
DeepSeekMath at https://arxiv.org/pdf/2402.03300

Objective terms:
- Policy ratio with advantage (encourages improvement over old policy)
- Clipping via epsilon (to limit policy update magnitude)
- KL penalty weighted by beta (to keep policy close to reference)
"""

import copy
import logging
from typing import List, Optional

import torch
from torch.nn.utils import clip_grad_norm_

from .utils import Policy, build_inputs
from .losses import grpo_loss
from ...core_types import ChatSegment

# bitsandbytes only work for non-MacOS
import platform
if platform.system() != "Darwin":  # Darwin = macOS
    try:
        import bitsandbytes as bnb
    except ImportError:
        bnb = None
else:
    bnb = None


def validate_config(config: dict) -> None:
    """
    Validate that the config dict contains all the required keys.

    Required keys:
        - epsilon
        - beta
        - opt_steps
        - learning_rate
        - max_grad_norm
        - chunk_len

    Raises:
        ValueError: If any required keys are missing, listing all missing keys.
    """
    required_keys = {"epsilon", "beta", "opt_steps", "learning_rate", "max_grad_norm", "chunk_len"}
    missing_keys = required_keys - config.keys()
    if missing_keys:
        raise ValueError(f"Invalid config: missing keys {sorted(missing_keys)}")


class GRPOAlgorithm:
    """
    Lightweight trainer that wraps GRPO optimization around a Policy.

    The training loop performs multiple optimization steps per batch,
    computing policy ratio and KL penalty terms to update the policy.

    Args:
        initial_policy (Policy): The policy to be trained/updated in-place.
        config (dict): Configuration dictionary with required keys.
        ref_policy (Optional[Policy]): Fixed reference policy used for KL penalty.
            If None, a deep copy of initial_policy is used as reference.

    During training, the initial_policy parameters are updated, while ref_policy remains fixed.
    """

    def __init__(
        self,
        initial_policy: Policy,
        config: dict,
        ref_policy: Optional[Policy] = None,
    ) -> None:
        """
        Initialize GRPOAlgorithm with policies and configuration.

        Args:
            initial_policy (Policy): The policy to train.
            config (dict): Configuration dictionary.
            ref_policy (Optional[Policy]): Reference policy for KL penalty. Defaults to a copy of initial_policy.

        Raises:
            ImportError: If use_bitsandbytes=True in config but bitsandbytes is not installed.
        """
        self.device = next(initial_policy.llm.parameters()).device

        # The policy to be trained/updated in-place
        self.policy = initial_policy

        # The fixed reference policy used for KL penalty (not updated)
        self.pi_ref = ref_policy if ref_policy else copy.deepcopy(initial_policy)

        validate_config(config)
        self.config = config

        # ----- FP16 or Bitsandbytes setup -----
        self.fp16 = config.get("fp16", False)

        self.training_steps = 0
        use_bitandbytes = config.get("use_bitsandbytes", False)
        if use_bitandbytes:
            self.fp16 = False  # if bitsandbytes is enabled, disable fp16
            if bnb is None:
                raise ImportError(
                    "bitsandbytes is not installed but 'use_bitsandbytes=True' was set in config. "
                    "Please install bitsandbytes to use 8-bit optimizer."
                )
            self.optimizer = bnb.optim.AdamW8bit(
                self.policy.llm.parameters(),
                lr=self.config["learning_rate"],
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.policy.llm.parameters(),
                lr=self.config["learning_rate"]
            )

    def _update_policy(self, pi_theta: Policy, loss: torch.Tensor) -> Policy:
        """
        Perform one optimizer step: zero_grad, backward, gradient clipping, and optimizer step.

        Args:
            pi_theta (Policy): The policy to update.
            loss (torch.Tensor): The computed loss tensor to backpropagate.

        Returns:
            Policy: The updated policy (same instance as input).
        """
        model = getattr(pi_theta, "llm", None) or getattr(pi_theta, "model", None)
        if model is None:
            raise AttributeError("No model found in pi_theta")

        model.train()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()

        torch.cuda.empty_cache()
        return pi_theta

    def train_step(
        self,
        segments: List[List[ChatSegment]],
        rewards: List[float],
    ) -> None:
        """
        Run one GRPO update over a batch of traces.

        Args:
            segments (List[List[ChatSegment]]): Batch of traces; each trace is a list of ChatSegment.
            rewards (List[float]): List of scalar rewards, one per trace.

        Batch returned by build_inputs contains:
            - input_ids: (B, L) token ids
            - attention_mask: (B, L) mask for tokens
            - completion_mask: (B, L) mask indicating completion tokens
            - advantages: (B, L) advantage values per token

        Note:
            The per-token log-probs computed by get_per_token_logps drop the first token,
            so advantages and completion_mask are sliced as [:, 1:] to align shapes with log-probs (B, L-1).
        """
        # torch.cuda.reset_peak_memory_stats()
        device = self.device
        pi_theta = self.policy  # train the main policy in-place to keep optimizer params in sync
        assert len(segments) == len(
            rewards
        ), f"Length of traces and rewards must be the same. Received {len(segments)} traces, {len(rewards)} rewards."

        batch = build_inputs(
            segments=segments, rewards=rewards, tokenizer=pi_theta.tokenizer
        )

        input_ids = batch.input_ids.to(device)  # shape: (B, L)
        attention_mask = batch.attention_mask.to(device)  # shape: (B, L)
        completion_mask = batch.completion_mask.to(device)  # shape: (B, L)
        advantages = batch.advantages.to(device)  # shape: (B, L)

        # Prepare old-policy (for ratio) and a fixed reference (for KL) log-probs.
        #   - pi_theta_old_logps: starts as current pre-update policy; will be refreshed each grpo loss step.
        #   - pi_ref_logps: fixed reference for KL across the grpo iteration (use pre-update self.policy).
        chunk_len = self.config.get("chunk_len", None)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                pi_theta_old_logps = pi_theta.get_per_token_logps(
                    input_ids=input_ids,  # shape: (B, L)
                    attention_mask=attention_mask,  # shape: (B, L)
                    chunk_len=chunk_len,
                )  # shape: (B, L-1)
                pi_ref_logps = self.pi_ref.get_per_token_logps(
                    input_ids=input_ids,  # shape: (B, L)
                    attention_mask=attention_mask,  # shape: (B, L)
                    chunk_len=chunk_len,
                )  # shape: (B, L-1)

        for _ in range(self.config["opt_steps"]):
            # Current policy log-probs
            # get_per_token_logps drops the first token after logits computation, before per-token logprobs.
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                pi_theta_logps = pi_theta.get_per_token_logps(
                    input_ids, attention_mask, chunk_len=chunk_len
                )  # shape: (B, L-1)

                # Must shift advantages and completion_mask by 1 token
                # so their shapes match the (B, L-1) log-probs tensors
                loss = grpo_loss(
                    pi_theta_logps=pi_theta_logps,  # shape: (B, L-1)
                    pi_theta_old_logps=pi_theta_old_logps,  # shape: (B, L-1)
                    pi_ref_logps=pi_ref_logps,  # shape: (B, L-1)
                    advantages=advantages[:, 1:],  # shape: (B, L-1)
                    completion_mask=completion_mask[:, 1:],  # shape: (B, L-1)
                    epsilon=self.config["epsilon"],
                    beta=self.config["beta"],
                )
            # peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # print(f"Peak GPU memory usage: {peak_memory:.2f} GB")
            # Apply update
            pi_theta = self._update_policy(pi_theta, loss)

            # Cache current log-probs as next step's old-policy (detach from graph)
            pi_theta_old_logps = pi_theta_logps.detach()
            del pi_theta_logps, loss
            torch.cuda.empty_cache()

        self.policy = pi_theta
        self.training_steps += 1

    def __repr__(self) -> str:
        return (
            f"GRPOAlgorithm(epsilon={self.config.get('epsilon')}, beta={self.config.get('beta')}, "
            f"opt_steps={self.config.get('opt_steps')}, steps={self.training_steps})"
        )
