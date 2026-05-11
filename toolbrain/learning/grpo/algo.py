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
from ...logging.base_logger import Logger
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
        logger: Optional[Logger] = None
    ) -> None:
        """
        Initialize GRPOAlgorithm with policies and configuration.

        Args:
            initial_policy (Policy): The policy to train.
            config (dict): Configuration dictionary.
            ref_policy (Optional[Policy]): Reference policy for KL penalty. Defaults to a copy of initial_policy.
            logger (Optional[Logger]): An abstract implementation instance of the Logger which will be used for logging GRPO related metrics.

        Raises:
            ImportError: If use_bitsandbytes=True in config but bitsandbytes is not installed.
        """
        self.logger = logger
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
            grad_norm (float): The norm of the gradients before clipping, for logging purposes.
        """
        model = getattr(pi_theta, "llm", None) or getattr(pi_theta, "model", None)
        if model is None:
            raise AttributeError("No model found in pi_theta")

        model.train()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()

        torch.cuda.empty_cache()
        return pi_theta, grad_norm.item()

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
                pi_theta_old_logps, pi_theta_old_entropies = pi_theta.get_per_token_logps(
                    input_ids=input_ids,  # shape: (B, L)
                    attention_mask=attention_mask,  # shape: (B, L)
                    chunk_len=chunk_len,
                )  # shape: (B, L-1)
                pi_ref_logps, pi_ref_entropies = self.pi_ref.get_per_token_logps(
                    input_ids=input_ids,  # shape: (B, L)
                    attention_mask=attention_mask,  # shape: (B, L)
                    chunk_len=chunk_len,
                )  # shape: (B, L-1)

        losses = []
        grad_norms = []
        kl_values = []
        policy_ratio_means = []
        policy_ratio_stds = []
        clip_fracs = []
        entropy_means = []
        entropy_stds = []
        for _ in range(self.config["opt_steps"]):
            # Current policy log-probs
            # get_per_token_logps drops the first token after logits computation, before per-token logprobs.
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                pi_theta_logps, pi_theta_entropies = pi_theta.get_per_token_logps(
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
            pi_theta, grad_norm = self._update_policy(pi_theta, loss)

            # -- logging:
            stats = GRPOAlgorithm._compute_grpo_stats_for_logging(
                pi_theta_logps=pi_theta_logps,
                pi_theta_old_logps=pi_theta_old_logps,
                pi_ref_logps=pi_ref_logps,
                advantages=advantages[:, 1:],
                completion_mask=completion_mask[:, 1:],
                epsilon=self.config["epsilon"],
                pi_theta_entropies=pi_theta_entropies,
            )

            self.logger.log_scalars({
                "opt_step/loss": loss.item(),
                "opt_step/grad_norm": grad_norm,
                "opt_step/kl_mean": stats["kl_mean"],
                "opt_step/policy_logprob_mean": stats["policy_logprob_mean"],
                "opt_step/policy_logprob_std": stats["policy_logprob_std"],
                "opt_step/policy_ratio_mean": stats["policy_ratio_mean"],
                "opt_step/policy_ratio_std": stats["policy_ratio_std"],
                "opt_step/clip_frac": stats["clip_frac"],
                "opt_step/entropy_mean": stats["entropy_mean"],
                "opt_step/entropy_std": stats["entropy_std"],
            })

            losses.append(loss.detach())
            grad_norms.append(grad_norm)
            kl_values.append(stats["kl_mean"])
            policy_ratio_means.append(stats["policy_ratio_mean"])
            policy_ratio_stds.append(stats["policy_ratio_std"])
            clip_fracs.append(stats["clip_frac"])
            entropy_means.append(stats["entropy_mean"])
            entropy_stds.append(stats["entropy_std"])
            # --

            # Cache current log-probs as next step's old-policy (detach from graph)
            pi_theta_old_logps = pi_theta_logps.detach()
            del pi_theta_logps, loss
            torch.cuda.empty_cache()

        # -- logging --
        loss_mean = torch.stack(losses).mean()
        loss_max = torch.stack(losses).max()

        grad_mean = sum(grad_norms) / len(grad_norms)
        grad_max = max(grad_norms)

        kl_mean = sum(kl_values) / len(kl_values)
        policy_ratio_mean = sum(policy_ratio_means) / len(policy_ratio_means)
        policy_ratio_std = sum(policy_ratio_stds) / len(policy_ratio_stds)
        clip_frac_mean = sum(clip_fracs) / len(clip_fracs)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        self.logger.log_scalars({
            "training_step/step": self.training_steps,

            "training_step/loss_mean": loss_mean.item(),
            "training_step/loss_max": loss_max.item(),

            "training_step/reward_mean": reward_tensor.mean().item(),
            "training_step/reward_std": reward_tensor.std().item(),

            "training_step/advantage_mean": advantages.mean().item(),
            "training_step/advantage_std": advantages.std().item(),

            "training_step/kl_mean": kl_mean,
            "training_step/policy_ratio_mean": policy_ratio_mean,
            "training_step/policy_ratio_std": policy_ratio_std,
            "training_step/clip_frac": clip_frac_mean,

            "training_step/grad_norm_mean": grad_mean,
            "training_step/grad_norm_max": grad_max,

            "training_step/entropy_mean": sum(entropy_means) / len(entropy_means),
            "training_step/entropy_std": sum(entropy_stds) / len(entropy_stds),

            "training_step/max_batch_input_ids_len": input_ids.shape[1],
            "training_step/max_batch_completion_mask_len": completion_mask.sum(dim=1).max().item(),
        })
        # --

        self.policy = pi_theta
        self.training_steps += 1
    
    @staticmethod
    def _compute_grpo_stats_for_logging(
        pi_theta_logps,
        pi_theta_old_logps,
        pi_ref_logps,
        advantages,
        completion_mask,
        epsilon,
        pi_theta_entropies=None,
    ):
        with torch.no_grad():
            policy_log_ratio = pi_theta_logps - pi_theta_old_logps
            policy_ratio = torch.exp(policy_log_ratio)

            # Mask valid tokens
            mask = completion_mask.bool()

            policy_logprobs_masked = pi_theta_logps[mask]
            policy_ratio_masked = policy_ratio[mask]
            adv_masked = advantages[mask]

            # Clip fraction
            clipped = (policy_ratio_masked > (1 + epsilon)) | (policy_ratio_masked < (1 - epsilon))
            clip_frac = clipped.float().mean().item() if clipped.numel() > 0 else 0.0

            # KL (per token)
            kl = (pi_theta_logps - pi_ref_logps)
            kl_mean = (kl * completion_mask).sum() / completion_mask.sum() if completion_mask.sum() > 0 else 0.0

            if pi_theta_entropies is not None:
                entropy_masked = pi_theta_entropies[mask]
                entropy_mean = entropy_masked.mean().item() if entropy_masked.numel() > 0 else 0.0
                entropy_std = entropy_masked.std().item() if entropy_masked.numel() > 0 else 0.0
            else:
                entropy_mean = 0.0
                entropy_std = 0.0

            return {
                "policy_logprob_mean": policy_logprobs_masked.mean().item() if policy_logprobs_masked.numel() > 0 else 0.0,
                "policy_logprob_std": policy_logprobs_masked.std().item() if policy_logprobs_masked.numel() > 0 else 0.0,
                "policy_ratio_mean": policy_ratio_masked.mean().item() if policy_ratio_masked.numel() > 0 else 0.0,
                "policy_ratio_std": policy_ratio_masked.std().item() if policy_ratio_masked.numel() > 0 else 0.0,
                "adv_mean": adv_masked.mean().item() if adv_masked.numel() > 0 else 0.0,
                "adv_std": adv_masked.std().item() if adv_masked.numel() > 0 else 0.0,
                "clip_frac": clip_frac,
                "kl_mean": kl_mean,
                "entropy_mean": entropy_mean,
                "entropy_std": entropy_std,
            }

    def __repr__(self) -> str:
        return (
            f"GRPOAlgorithm(epsilon={self.config.get('epsilon')}, beta={self.config.get('beta')}, "
            f"opt_steps={self.config.get('opt_steps')}, steps={self.training_steps})"
        )
