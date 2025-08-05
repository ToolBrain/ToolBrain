import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class CodeAgentWrapper(nn.Module):
    def __init__(self, model_id):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def get_log_probs(self, pi_old, prompts: list[str], completions: list[str]):
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        completion_inputs = self.tokenizer(
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {
            "input_ids": torch.cat(
                [prompt_inputs["input_ids"], completion_inputs["input_ids"]],
                dim=1
            ),
            "attention_mask": torch.cat(
                [prompt_inputs["attention_mask"], completion_inputs["attention_mask"]],
                dim=1
            )
        }

        current_logits = self.model(**inputs).logits # shape [batch_size, seq_len, vocab_size]
        with torch.no_grad():
            old_logits = pi_old.model(**inputs).logits # shape [batch_size, seq_len, vocab_size]

        start_idx = prompt_inputs['input_ids'].shape[1]
        end_idx = inputs['input_ids'].shape[1] - 1
        

        sliced_current_logits = current_logits[:, start_idx - 1:end_idx]
        sliced_old_logits = old_logits[:, start_idx - 1:end_idx]
        sliced_labels = inputs['input_ids'][:, start_idx:end_idx + 1]
        
        current_log_probs = F.log_softmax(sliced_current_logits, dim=-1)\
                                .gather(2, sliced_labels.unsqueeze(-1))\
                                .squeeze(-1)
        old_log_probs = F.log_softmax(sliced_old_logits, dim=-1)\
                                .gather(2, sliced_labels\
                                .unsqueeze(-1))\
                                .squeeze(-1)
        
        completion_lengths = (sliced_labels != self.tokenizer.pad_token_id).sum(dim=-1)
        
        return current_log_probs, old_log_probs

if __name__ == "__main__":
    model_id = "gpt2"
    wrapper = CodeAgentWrapper(model_id)
    
    prompts = [
        "What is the capital of France?",
        "Who wrote 'Pride and Prejudice'?",
        "Explain the theory of relativity."
    ]
    completions = [
        "The capital of France is Paris.",
        "Jane Austen wrote 'Pride and Prejudice'.",
        "The theory of relativity was developed by Albert Einstein."
    ]
    
    pi_old = wrapper 
    log_probs, old_log_probs = wrapper.get_log_probs(pi_old, prompts, completions)
    
    print("Log probabilities:", log_probs)
    print("Old log probabilities:", old_log_probs)