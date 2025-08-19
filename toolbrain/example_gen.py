# toolbrain/example_gen.py
from typing import Any, Dict, List
from smolagents import ChatMessage

def generate_training_examples(
    task_description: str,
    num_examples: int = 5,
    external_model: Any = None,
) -> List[Dict[str, Any]]:
    """
    Generate training examples.

    Args:
        task_description: High-level description guiding example creation.
        num_examples: Number of examples to return.
        external_model: LLM provider (callable, or exposes .propose/.generate).
    Returns:
        List of dicts with keys:
            - 'prompt' (str)
            - 'required_tools' (List[str])
            - 'min_tool_calls' (int)
            - 'acceptance_tests' (List[{"type","spec"}])
    """

    examples: List[Dict[str, Any]] = []

    for _ in range(num_examples):
        prompt = task_description
        required_tools = ["tool"]
        min_tool_calls = 1
        acceptance_tests = [{"type": "must_call", "spec": {"tool": "tool"}}]

        examples.append({
            "prompt": prompt,
            "required_tools": required_tools,
            "min_tool_calls": min_tool_calls,
            "acceptance_tests": acceptance_tests,
        })

    return examples