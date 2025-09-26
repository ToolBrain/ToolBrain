from __future__ import annotations

import inspect
from textwrap import dedent
from typing import Any, Dict, Sequence
from smolagents import Tool, tool, ChatMessage


def _extract_tool_meta(tool: Tool) -> Dict[str, Any]:
    """
    Extracts and returns a uniform metadata dictionary for a smolagents Tool object.

    The returned metadata dictionary provides key information about the Tool, including its name,
    description, and optional attributes such as input/output types and initialization status.

    Args:
        tool (Tool): The smolagents Tool object from which to extract metadata.

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - name (str): Name of the tool.
            - description (str): Description of the tool.
            - inputs (Optional[Any]): Additional inputs attribute if present.
            - output_type (Optional[Any]): Additional output_type attribute if present.
            - is_initialized (Optional[bool]): Additional is_initialized attribute if present.
            - parameters (Dict[str, Any]): Parameters dict extracted from tool.spec if available.
            - returns (Optional[str]): Short returns hint parsed from docstring.
            - examples (Optional[str]): Examples snippet parsed from docstring.
    """
    if not isinstance(tool, Tool):
        raise TypeError(f"Expected a smolagents.Tool instance, got {type(tool).__name__}")
    
    meta: Dict[str, Any] = {
        "name": getattr(tool, "name", None) or getattr(tool, "__name__", "tool"),
        "description": getattr(tool, "description", None),
        "inputs": getattr(tool, "inputs", None),
        "output_type": getattr(tool, "output_type", None),
        "is_initialized": getattr(tool, "is_initialized", None),
        "parameters": {},
        "returns": None,
        "examples": None,
    }

    # Extract parameters from tool.spec if present and is dict
    spec = getattr(tool, "spec", None)
    if isinstance(spec, dict):
        parameters = spec.get("parameters")
        if isinstance(parameters, dict):
            meta["parameters"] = parameters

    # Attempt to parse docstring for returns and examples
    doc = inspect.getdoc(tool)
    if not doc and hasattr(tool, "forward"):
        doc = inspect.getdoc(tool.forward)
    if doc:
        lines = doc.splitlines()
        returns_lines = []
        examples_lines = []
        in_returns = False
        in_examples = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("returns:") or stripped.lower().startswith("return:"):
                in_returns = True
                in_examples = False
                returns_lines.append(stripped.partition(":")[2].strip())
            elif stripped.lower().startswith("examples:") or stripped.lower().startswith("example:"):
                in_examples = True
                in_returns = False
            elif in_returns:
                if stripped == "":
                    in_returns = False
                else:
                    returns_lines.append(stripped)
            elif in_examples:
                if stripped == "":
                    in_examples = False
                else:
                    examples_lines.append(line)
        if returns_lines:
            meta["returns"] = " ".join(returns_lines).strip()
        if examples_lines:
            meta["examples"] = "\n".join(examples_lines).strip()

    return meta

def tool_to_card(tool: Tool) -> str:
    """
    Converts a smolagents Tool object into a formatted string card ready for prompt insertion.

    This function extracts key metadata from the Tool and organizes it into a human-readable
    multiline string that can be used for display or documentation purposes.

    Args:
        tool (Tool): The smolagents Tool object to convert.

    Returns:
        str: A formatted string representing the tool, including its name, description,
             inputs, output_type, and is_initialized status.
    """
    meta = _extract_tool_meta(tool)
    lines = []
    lines.append(f"Tool: {meta['name']}")
    lines.append(f"Purpose: {meta['description'] if meta['description'] else 'No description provided.'}")
    lines.append("Args (JSON):")

    if meta["parameters"]:
        for pname, pinfo in meta["parameters"].items():
            ptype = pinfo.get("type", "unknown") if isinstance(pinfo, dict) else "unknown"
            pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
            pdefault = pinfo.get("default") if isinstance(pinfo, dict) else None
            default_str = f" (default={pdefault})" if pdefault is not None else ""
            lines.append(f"- {pname}: {ptype}, {pdesc}{default_str}".rstrip(", "))
    elif meta["inputs"]:
        # inputs fallback, show as string repr
        lines.append(f"- {meta['inputs']}")
    else:
        lines.append("- None")

    if meta["returns"]:
        lines.append(f"Returns: {meta['returns']}")
    elif meta["output_type"]:
        lines.append(f"Output type: {meta['output_type']}")

    if meta["examples"]:
        lines.append("\nExample:")
        lines.append(meta["examples"])

    card = "\n".join(lines)
    return card


def tools_to_card(tools: Sequence[Any]) -> str:
    """
    Render multiple Tool objects into a prompt-ready string.

    The output begins with a header 'Available tools (N):', where N is the number of tools.
    Each tool is enumerated (1-based) with its card, prefixed by its index (e.g., '1) ').
    Tool cards are separated by a clear separator line '\n---\n', with no trailing separator after the last.
    If the input list is empty, returns 'No tools provided.'

    Args:
        tools (Sequence[Tool]): Sequence of smolagent Tool to render as tool cards.

    Returns:
        str: A formatted string listing all provided tools, or 'No tools provided.' if empty.

    Raises:
        TypeError: If any item in the input is not an instance of Tool.
    """
    if not tools:
        return "No tools provided."
    cards = []
    for idx, t in enumerate(tools, start=1):
        if not isinstance(t, Tool):
            raise TypeError(f"Item at index {idx-1} is not a Tool: got {type(t).__name__}")
        card = tool_to_card(t)
        cards.append(f"{idx}) {card}")
    header = f"Available tools ({len(cards)}):"
    return header + "\n" + ("\n---\n".join(cards))



def validate_model(model: Any) -> None:
    """
    Validate the provided LLM model.

    Requirements:
    - model must not be None
    - model must expose a callable `.generate` method
    - `.generate` should accept at least one positional argument (e.g., a list[ChatMessage])
    """
    if model is None:
        raise RuntimeError(
            "No LLM provided. Please supply `external_model` or ensure the agent adapter returns a model."
        )

    gen = getattr(model, "generate", None)
    if gen is None or not callable(gen):
        raise NotImplementedError(
            "LLM must implement a callable `.generate(messages: list[ChatMessage], ...)` method."
        )

    # Best-effort signature check: ensure at least one parameter
    try:
        sig = inspect.signature(gen)
        # exclude 'self' for bound methods
        params = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        ]
        if len(params) < 1:
            raise TypeError
    except (ValueError, TypeError):
        raise TypeError(
            "LLM `.generate` must accept at least one positional argument (messages)."
        ) from None


def validate_tools(tools: Sequence[Tool]) -> None:
    """
    Validate the sequence of tools.

    Requirements:
    - `tools` must be a non-empty sequence (not a string/bytes)
    - every item must be an instance of `smolagents.Tool`
    - each tool must be callable and have a non-empty string `name`
    - tool names must be unique (case-insensitive)
    """
    # Basic container checks
    if isinstance(tools, (str, bytes)):
        raise TypeError("`tools` must be a sequence of Tool objects, not a string/bytes.")
    if not isinstance(tools, Sequence):
        raise TypeError("`tools` must be a sequence of smolagents Tool objects.")
    if len(tools) == 0:
        raise ValueError("`tools` must be non-empty.")

    seen = set()
    for idx, t in enumerate(tools):
        if not isinstance(t, Tool):
            print(t)
            print(type(t), "+"*30)
            raise TypeError(f"Item at index {idx} is not a smolagents Tool: got {type(t).__name__}")
        if not callable(t):
            raise TypeError(f"Tool at index {idx} ('{getattr(t, 'name', None)}') is not callable.")
        name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Tool at index {idx} has no valid `name` string.")
        key = name.strip().lower()
        if key in seen:
            raise ValueError(f"Duplicate tool name detected: '{name}'. Tool names must be unique (case-insensitive).")
        seen.add(key)


def build_prompt_to_generate_training_examples(
    tools_description: str,
    task_description: str | None = None,
    min_tool_calls: int = 1,
    max_words: int = 80,
    guidance_example: str | None = None
) -> str:
    """
    Constructs a prompt string for generating training examples using the provided tools.

    The prompt includes a task description, a list of available tools with their details,
    and specific instructions for generating examples. It can enforce the use of multiple
    tools in each example if required.

    Args:
        task_description (str): A brief description of the task for which examples are to be generated.
        tools_description (str): A detailed description of tools.
        min_tool_calls (int, optional): Minimum number of distinct tools that must be used in each example. Defaults to 1.
        max_words (int, optional): Maximum word count for the generated example. Defaults to 80.
        guidance_example (Optional[str], optional): A few-shot, non-binding hint shown above the instructions to steer style and difficulty. This is NOT the output to return.

    Returns:
        str: A formatted prompt string ready for use in example generation.

    Raises:
        TypeError: If any item in the tools sequence is not an instance of smolagent Tool.
    """
    
    max_words = max(10, max_words)  # enforce a sensible minimum
    
    prompt = dedent(f"""\
    You are an AI tasked that generate realistic tasks for that can only solve using the provided tools.
    The task you generate must call required the use of provided tools in the Tool desription.
    """)

    if task_description and task_description.strip():
        prompt += f"Task description: {task_description}\n\n"

    prompt += f"Tool description: {tools_description}\n\n"

    if guidance_example:
        prompt += dedent(f"""\
        Few-shot guidance. Do not reuse numbers, names, or phrasings from the few-shot guidance. Create a new, realistic scenario:
        {guidance_example}
        """)
    prompt += dedent(f"""\
    Instructions:
    1) Output only a single plain query string (no JSON, no schema, no backticks, no greetings, no explanations, no emojis), maximum {max_words} words.
    2) The task must explicitly include all necessary concrete values required for the tool’s parameters (read from the Tool description). If the parameter is numeric, include realistic values. The task should be specific and unambiguous.
    4) The task should be complex enough to require the use minimum {min_tool_calls} tools in the Tool description, and should not be answerable without them.
    5) Do not ask question about the formulas, the tool definition, only generate queries that can only solve using the tools in the Tool description
    6) Return a gold answer (the correct answer for the generated query) starts with '[Gold answer]' after the query.
    """)

    return prompt
    


if __name__ == "__main__":
    @tool
    def add(x: int, y: int) -> int:
        """Add two numbers.

        Args:
            x (int): First number.
            y (int, optional): Second number.

        Returns:
            int: The sum of x and y.
        """
        return x + y
    
    @tool
    def multiply(a: int, b: int) -> int:
        """
        Multiply two integers.

        Args:
            a (int): First factor.
            b (int): Second factor.

        Returns:
            int: Product of a and b.
        """
        return a * b

    meta = _extract_tool_meta(add)
    card = tool_to_card(add)
    full_card = tools_to_card([add, multiply])
    prompt = build_prompt_to_generate_training_examples(
        task_description="Perform basic arithmetic operations.",
        tools_description=full_card,
        guidance_example=None,
    )
    print(prompt)
