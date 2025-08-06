from transformers import AutoTokenizer
import torch

def build_input_and_completion_mask(pairs, tokenizer):
    """
    Args:
        pairs: List of (prompt: str, completion: str) tuples.
        tokenizer: A HuggingFace tokenizer (already loaded).
    
    Returns:
        input_ids: torch.LongTensor [total_seq_len]
        completion_mask: torch.BoolTensor [total_seq_len]
    """
    all_input_ids = []
    all_completion_mask = []

    for prompt, completion in pairs:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)

        all_input_ids.extend(prompt_ids + completion_ids)
        completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        all_completion_mask.extend(completion_mask)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    completion_mask = torch.tensor(all_completion_mask, dtype=torch.bool)

    return input_ids, completion_mask

if __name__ == "__main__":
    # Example usage
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    pairs = [
        ("What is the capital of France?", " Paris."),
        ("2 + 2 equals", " 4."),
    ]

    input_ids, completion_mask = build_input_and_completion_mask(pairs, tokenizer)

    print("Input IDs:", input_ids)
    print("Completion Mask:", completion_mask)
    print("Decoded:", tokenizer.decode(input_ids))