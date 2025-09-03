import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
import copy


def compute_per_token_logps(
    logits: torch.Tensor,  # (B, L-1, V)
    input_ids: torch.Tensor,  # (B, L-1)
    chunk_len: (
        int | None
    ) = None,  # if set, compute along L in chunks of C tokens to reduce peak memory
) -> torch.Tensor:
    """
    Return per-token log-probs aligned with targets.
    - logits: (B, L-1, V)
    - input_ids: (B, L-1)
    - chunk_len: optional chunk size along L for memory-friendly computation
    Returns: (B, L-1)
    """
    if chunk_len is None:
        lp = logits.log_softmax(dim=-1)  # (B, L-1, V)
        return lp.gather(dim=-1, index=input_ids.long().unsqueeze(-1)).squeeze(
            -1
        )  # (B, L-1)

    # Chunked path along sequence length
    chunk_outputs: list[torch.Tensor] = []
    target_len = input_ids.size(1)
    for start_idx in range(0, target_len, chunk_len):
        end_idx = min(start_idx + chunk_len, target_len)
        lp = logits[:, start_idx:end_idx, :].log_softmax(dim=-1)  # (B, C, V)
        chunk_outputs.append(
            lp.gather(
                dim=-1, index=input_ids[:, start_idx:end_idx].long().unsqueeze(-1)
            ).squeeze(-1)
        )  # (B, C)
    return torch.cat(chunk_outputs, dim=1)  # (B, L-1)


class Policy(nn.Module):
    def __init__(
        self, llm: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def to(self, device):
        self.llm.to(device)
        return self

    def get_per_token_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_len: int | None = None,
    ) -> torch.Tensor:
        """Return per-token log-probs aligned for causal LM loss
        (predict token t from context up to t-1).
        Output shape: (B, L-1).
        """
        logits = self.llm(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits  # shape: (B, L, V)

        # Exclude the last logit because it has no corresponding target token
        # (there is no "next token" after the last one)
        logits = logits[:, :-1, :]  # shape: (B, L-1, V)
        # Exclude the first token in input_ids because its prediction is based on
        # the context before it (no preceding token)
        input_ids = input_ids[:, 1:]  # shape: (B, L-1)

        per_token_logps = compute_per_token_logps(
            logits, input_ids, chunk_len=chunk_len
        )  # shape: (B, L-1)
        return per_token_logps

    def copy(self) -> "Policy":
        """
        Create a deep copy of this Policy model to the same device.
        The copy is independent from the original: llm's parameters update on one will not affect the other.
        """
        llm_copy = AutoModelForCausalLM.from_config(self.llm.config)
        llm_copy.load_state_dict(self.llm.state_dict())
        llm_copy.to(next(self.llm.parameters()).device)

        tokenizer_copy = copy.deepcopy(self.tokenizer)
        return Policy(llm=llm_copy, tokenizer=tokenizer_copy)
