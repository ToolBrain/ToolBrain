# toolbrain/example_gen.py
from typing import Any, Dict, List
from textwrap import dedent
from smolagents import ChatMessage, MessageRole


def _craft_prompt_request(task_description: str, variant_idx: int = 0) -> str:
    text = dedent(f"""\
        You are designing training prompts to enable zero-paradigm tool learning for a tool-using LLM agent. The generated prompt will be used to fine-tune an agent with reinforcement learning (RL) methods (e.g., GRPO/PPO/DPO).
        TASK:
        {task_description}

        Output ONLY the final prompt text — no commentary, headers, examples, or markdown.
        Constraints:
        - Length: 2–4 sentences (≈40–80 words).
        - Instruct the agent to (1) propose a concrete task that will maximize its own learning progress and improve reasoning, then (2) execute it using external tools.
        - Clearly specify required tools (generic), expected inputs (formats/units), and expected outputs with objective, verifiable checks suitable for automated rewards.
        - Describe tool usage in NATURAL LANGUAGE (no code/JSON).
        - Include one realistic edge case and one guardrail (e.g., tool failure or empty results).
        - Self-contained; must not reveal system instructions or model identity.
        - Diversity across variants in difficulty, tool count/ordering, and phrasing. (Variant #{variant_idx+1})
    """)
    
    return text


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

    # Use this in the Brain class to get the llm from agent
    #llm = self.trainable_model.model if external_model is None else external_model
    llm = external_model

    for i in range(num_examples):
        if llm is not None:
            request_text = _craft_prompt_request(task_description, variant_idx=i)
            if hasattr(llm, "generate"):
                messages = [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": str(request_text)}],
                    )
                ]
                prompt = llm.generate(messages)
            elif callable(llm):
                prompt = llm(str(request_text))
            else:
                prompt = str(request_text)
        else:
            # fallback: extract TASK block and create a concise imperative prompt
            lines = task_description.strip().splitlines()
            task_lines = [line.strip() for line in lines if line.strip()]
            concise_task = " ".join(task_lines)
            prompt = f"Please perform the following task using tools: {concise_task}."

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
