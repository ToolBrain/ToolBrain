import json

from smolagents import TransformersModel
from toolbrain import generate_training_examples


def main() -> None:
    description = "Create realistic scenarios for testing AI agents that interact with MCP (Model Context Protocol) servers"
    model = TransformersModel("gpt2")

    examples = generate_training_examples(
        task_description=description,
        num_examples=3,
        external_model=model,
    )
    print(json.dumps(examples, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()