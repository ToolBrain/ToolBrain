import copy
import torch
from torch.nn.utils import clip_grad_norm_
import platform

# bitsandbytes only work for non-MacOS
if platform.system() != "Darwin":  # Darwin = macOS
    try:
        import bitsandbytes as bnb
    except ImportError:
        bnb = None
else:
    bnb = None
from typing import List

from toolbrain.core_types import ChatSegment
from toolbrain.learning import Policy
from toolbrain.learning.dpo.utils import build_inputs


class DPOAlgorithm:
    """Trainer that implements Direct Preference Optimization (DPO).

    Usage:
        algo = DPOAlgorithm(policy, config, ref_policy)
        algo.train_step(chosen_segments, rejected_segments)
    """

    def __init__(
        self,
        initial_policy: Policy,
        config: dict,
        ref_policy: Policy = None,
    ) -> None:
        self.device = next(initial_policy.llm.parameters()).device

        self.policy = initial_policy.to(self.device)
        self.pi_ref = ref_policy if ref_policy else copy.deepcopy(initial_policy)
        self.pi_ref = self.pi_ref.to(self.device)

        self.config = config
        self.training_steps = 0

        # ----- FP16 or Bitsandbytes setup -----
        self.fp16 = config.get("fp16", False)

        use_bitandbytes = config.get("use_bitsandbytes", False)
        if use_bitandbytes:
            self.fp16 = False  # if bitsandbytes is enabled, disable fp16
            self.optimizer = bnb.optim.AdamW8bit(
                self.policy.llm.parameters(),
                lr=self.config["learning_rate"],
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.policy.llm.parameters(),
                lr=self.config["learning_rate"]
            )

    def _update_policy(self, pi_theta, loss):
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
        chosen_segments: List[List[ChatSegment]],
        rejected_segments: List[List[ChatSegment]],
    ) -> None:
        """
        Run one DPO update over a batch.

        Args:
            chosen_segments: list of ChatSegment lists (preferred completions).
            rejected_segments: list of ChatSegment lists (less-preferred completions).
        """
        assert len(chosen_segments) == len(rejected_segments), \
            f"Batch mismatch: {len(chosen_segments)} chosen vs {len(rejected_segments)} rejected"

        pi_theta = self.policy  # train in-place

        # Build inputs for chosen and rejected
        chosen_batch = build_inputs(segments=chosen_segments, tokenizer=pi_theta.tokenizer)
        rejected_batch = build_inputs(segments=rejected_segments, tokenizer=pi_theta.tokenizer)

        chunk_len = self.config.get("chunk_len", None)
        # Mask of completion tokens (B, L-1)
        chosen_completion_mask = chosen_batch.completion_mask[:, 1:].to(self.device)
        rejected_completion_mask = rejected_batch.completion_mask[:, 1:].to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            # Per-token log-probs from current policy
            pi_chosen_logps_per_token = pi_theta.get_per_token_logps(
                input_ids=chosen_batch.input_ids.to(self.device),
                attention_mask=chosen_batch.attention_mask.to(self.device),
                chunk_len=chunk_len,
            )  # (B, L-1)

            pi_rejected_logps_per_token = pi_theta.get_per_token_logps(
                input_ids=rejected_batch.input_ids.to(self.device),
                attention_mask=rejected_batch.attention_mask.to(self.device),
                chunk_len=chunk_len,
            )  # (B, L-1)

        # Per-token log-probs from reference policy
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                pi_ref_chosen_logps_per_token = self.pi_ref.get_per_token_logps(
                    input_ids=chosen_batch.input_ids.to(self.device),
                    attention_mask=chosen_batch.attention_mask.to(self.device),
                    chunk_len=chunk_len,
                )  # (B, L-1)

                pi_ref_rejected_logps_per_token = self.pi_ref.get_per_token_logps(
                    input_ids=rejected_batch.input_ids.to(self.device),
                    attention_mask=rejected_batch.attention_mask.to(self.device),
                    chunk_len=chunk_len,
                )  # (B, L-1)

        # === Sequence-level mean logps over completion tokens only ===
        def masked_mean_logps(logps_per_token, mask):
            # (logps * mask).sum() / mask.sum() per sequence
            masked_sum = (logps_per_token * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            return masked_sum / lengths  # shape: (B,)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            pi_chosen_logps = masked_mean_logps(pi_chosen_logps_per_token, chosen_completion_mask)
            pi_rejected_logps = masked_mean_logps(pi_rejected_logps_per_token, rejected_completion_mask)
            pi_ref_chosen_logps = masked_mean_logps(pi_ref_chosen_logps_per_token, chosen_completion_mask)
            pi_ref_rejected_logps = masked_mean_logps(pi_ref_rejected_logps_per_token, rejected_completion_mask)

            # === DPO loss ===
            beta = self.config.get("beta", 0.1)
            logits = beta * (
                    (pi_chosen_logps - pi_rejected_logps)
                    - (pi_ref_chosen_logps - pi_ref_rejected_logps)
            )
            loss = -torch.nn.functional.logsigmoid(logits).mean()

        # Update step
        pi_theta = self._update_policy(pi_theta, loss)

        self.policy = pi_theta
        self.training_steps += 1

    def __repr__(self) -> str:
        return (
            f"DPOAlgorithm(beta={self.config.get('beta')}, "
            f"steps={self.training_steps})"
        )


