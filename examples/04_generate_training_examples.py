import json

from smolagents import TransformersModel, CodeAgent, tool
from toolbrain import Brain, get_default_config

@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): First addend.
        b (int): Second addend.

    Returns:
        int: Sum of a and b.
    """
    return a + b


def main() -> None:
    description = "Create realistic scenarios for testing AI agents that interact with MCP (Model Context Protocol) servers"
    model = TransformersModel("Qwen/Qwen2.5-0.5B-Instruct")

    # Ensure pad token is distinct from EOS so attention_mask can be inferred
    tok = model.tokenizer
    mdl = getattr(model, "model", None)
    if tok.pad_token_id is None or (tok.eos_token_id is not None and tok.pad_token_id == tok.eos_token_id):
        tok.add_special_tokens({"pad_token": "<|pad|>"})
        if mdl is not None:
            mdl.resize_token_embeddings(len(tok))
            mdl.config.pad_token_id = tok.pad_token_id
    tok.padding_side = "left"

    # If the tokenizer does not yet have a chat template, set a default one
    tokenizer = model.tokenizer
    if tokenizer.chat_template is None:
        print("Setting a default chat template for the tokenizer.")
        # This template supports dict-based content parts
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% set _text = '' %}"
            "{% if message['content'] is string %}"
            "{% set _text = message['content'] %}"
            "{% else %}"
            "{% for part in message['content'] %}"
            "{% if part['type'] == 'text' %}{{ _text ~ part['text'] }}{% set _text = _text + part['text'] %}{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            "{% if message['role'] == 'system' %}{{ '<|system|>\n' ~ _text ~ '<|end|>\n' }}"
            "{% elif message['role'] == 'user' %}{{ '<|user|>\n' ~ _text ~ '<|end|>\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  ~ _text ~ '<|end|>\n' }}{% endif %}"
            "{% endfor %}"
        )
    my_agent = CodeAgent(
        tools=[add],
        model=model,
    )

    brain = Brain(
        agent=my_agent,
        reward_func=lambda x: 1.0,
        learning_algorithm="GRPO",
        config=get_default_config(),
    )
    examples = brain.generate_training_examples(
        task_description=description,
        num_examples=1,
        external_model=None,
    )
    for example in examples:
        print(example["prompt"].content)


if __name__ == "__main__":
    main()