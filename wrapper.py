import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from grpo_utils import build_input_and_completion_mask


def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

class CodeAgentWrapper(nn.Module):
    def __init__(self, model_id):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def get_log_probs(self, input_ids: torch.Tensor):
        logits = self.model(input_ids.unsqueeze(0)).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        
        input_ids = input_ids.unsqueeze(0)  # (B, L), add batch dimension
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        
        per_token_logps = get_per_token_logps(logits, input_ids) # Shape: (B, L-1)
        return per_token_logps.squeeze(0)  # Remove batch dimension, shape: (L-1,)

if __name__ == "__main__":
    model = CodeAgentWrapper("gpt2")

    traces = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen wrote 'Pride and Prejudice'."),
    ]
    input_ids, completion_mask = build_input_and_completion_mask(traces, model.tokenizer)
    log_probs = model.get_log_probs(input_ids)

    print("Log probabilities shape:", log_probs.shape)
    print("Log probabilities:", log_probs)
