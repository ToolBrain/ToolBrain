import copy
import torch
from torch.nn.utils import clip_grad_norm_
import bitsandbytes as bnb
from typing import List

from toolbrain.core_types import ChatSegment
from toolbrain.learning import Policy
from toolbrain.learning.dpo.utils import build_inputs


class SupervisedAlgorithm:
    """Trainer that implements supervised learning.

    Usage:
        algo = SupervisedAlgorithm(policy, config)
        algo.train_step(segments)
    """

    def __init__(
        self,
        initial_policy: Policy,
        config: dict,
    ) -> None:
        self.device = next(initial_policy.llm.parameters()).device
        self.policy = initial_policy.to(self.device)
        self.config = config
        self.training_steps = 0

        use_bitandbytes = config.get("use_bitsandbytes", False)
        if use_bitandbytes:
            self.optimizer = bnb.optim.AdamW8bit(
                self.policy.llm.parameters(),
                lr=self.config["lr"],
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.policy.llm.parameters(),
                lr=self.config["lr"]
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
        print(f"Loss at step {self.training_steps}: {loss.item():.4f}")
        return pi_theta

    def train_step(
        self,
        segments: List[List[ChatSegment]],
    ) -> None:
        """
        Run one supervised training update over a batch.

        Args:
            segments: list of ChatSegment lists
        """
        pi_theta = self.policy  # train in-place

        # Build inputs
        batch = build_inputs(segments=segments, tokenizer=pi_theta.tokenizer)

        chunk_len = self.config.get("chunk_len", None)
        # Mask of completion tokens (B, L-1) shifted to ignore the first token
        completion_mask = batch.completion_mask[:, 1:].to(self.device)

        # Per-token log-probs from current policy
        pi_logps_per_token = pi_theta.get_per_token_logps(
            input_ids=batch.input_ids.to(self.device),
            attention_mask=batch.attention_mask.to(self.device),
            chunk_len=chunk_len,
        )  # (B, L-1)

        # cross entropy is negative of log probs:
        pi_ce_per_token = - pi_logps_per_token  # (B, L-1)

        # === Sequence-level mean cross-entropy over completion tokens only ===
        def masked_mean_ce(pi_ce_per_token, mask):
            # (pi_ce_per_token * mask).sum() / mask.sum() per sequence
            masked_sum = (pi_ce_per_token * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            return masked_sum / lengths  # shape: (B,)
        loss = masked_mean_ce(pi_ce_per_token, completion_mask).mean()
        # Update step
        pi_theta = self._update_policy(pi_theta, loss)

        self.policy = pi_theta
        self.training_steps += 1



if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # === Batch of size 2 ===
    segments: List[List[ChatSegment]] = [
        [
            ChatSegment(
                role="other",
                text="You are a Python assistant. Compute the sum of 1..10 and explain briefly.",
            ),
            ChatSegment(
                role="assistant",
                text="The sum of numbers from 1 to 10 is 55, using the formula n(n+1)/2."
            ),
        ],
        [
            ChatSegment(
                role="other",
                text="Translate 'Good morning' into French.",
            ),
            ChatSegment(
                role="assistant",
                text="In French, it is 'Bonjour'."
            ),
        ],
    ]

    # === Policy ===
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_id)
    initial_policy = Policy(llm=llm, tokenizer=tokenizer)

    # === Config ===
    config = {
        "lr": 1e-5,
        "max_grad_norm": 1.0,
        "chunk_len": 128,
        "beta": 0.1,
    }

    algo = SupervisedAlgorithm(initial_policy=initial_policy, config=config)

    # === One train step with batch size 2 ===
    algo.train_step(
        segments=segments
    )
