"""
Flexible reward functions for ToolBrain.

These functions follow the RewardFunction protocol and accept arbitrary
keyword arguments so they can be composed in many settings, including
cases without a single gold answer.
"""

from typing import Any, Optional
from .core_types import Trace


def reward_exact_match(trace: Trace, **kwargs: Any) -> float:
    """
    Reward 1.0 if the final answer exactly matches the provided gold_answer, else 0.0.
    This version is robust against None values.
    """
    gold_answer = kwargs.get("gold_answer")
    if gold_answer is None:
        return 0.0
    
    for turn in trace:
        parsed = turn.get("parsed_completion", {})
        if parsed:
            final_answer = parsed.get("final_answer")
            
            if isinstance(final_answer, str):
                if final_answer.strip() == str(gold_answer).strip():
                    return 1.0
    
    return 0.0


def reward_tool_execution_success(trace: Trace, **kwargs: Any) -> float:
    """
    Reward 1.0 if there is at least one tool_output and none contain errors; else 0.0.
    
    Updated for new Trace = List[Turn] structure.
    """
    for turn in trace:
        if turn["tool_output"] is not None:
            if "error" in turn["tool_output"].lower():
                return 0.0
            return 1.0
    return 0.0


def reward_step_efficiency(trace: Trace, **kwargs: Any) -> float:
    """
    Reward higher for shorter traces.
    max_turns (int, default=5) can be passed via kwargs.
    
    Updated for new Trace = List[Turn] structure.
    """
    max_turns = int(kwargs.get("max_turns", 5))
    num_turns = len(trace)
    if num_turns <= max_turns:
        return 1.0
    return max(0.0, 1.0 - (num_turns - max_turns) * 0.1)


def reward_behavior_uses_search_first(trace: Trace, **kwargs: Any) -> float:
    """
    Example behavior-based reward:
    Return 1.0 if the first tool_code mentions 'search', else 0.0.
    
    Updated for new Trace = List[Turn] structure.
    """
    for turn in trace:
        tool_code = turn["parsed_completion"].get("tool_code")
        if tool_code:
            return 1.0 if "search" in tool_code.lower() else 0.0
    return 0.0


def reward_safety_no_os_system(trace: Trace, **kwargs: Any) -> float:
    """
    Safety reward: return 1.0 if no tool_code contains os.system, else 0.0.
    
    Updated for new Trace = List[Turn] structure.
    """
    for turn in trace:
        tool_code = turn["parsed_completion"].get("tool_code")
        if tool_code and "os.system" in tool_code.lower():
            return 0.0
    return 1.0


def reward_combined(trace: Trace, **kwargs: Any) -> float:
    """
    Combine multiple rewards with weights.
    Provide weights via kwargs (default: exact_match=0.7, tool_success=0.2, efficiency=0.1).
    
    Updated for new Trace = List[Turn] structure.
    """
    weights = kwargs.get(
        "weights",
        {"exact_match": 0.7, "tool_success": 0.2, "efficiency": 0.1},
    )
    r_exact = reward_exact_match(trace, **kwargs)
    r_tool = reward_tool_execution_success(trace, **kwargs)
    r_eff = reward_step_efficiency(trace, **kwargs)
    total = (
        weights.get("exact_match", 0.0) * r_exact
        + weights.get("tool_success", 0.0) * r_tool
        + weights.get("efficiency", 0.0) * r_eff
    )
    return max(0.0, min(1.0, total))


# --------------------------
# LLM-as-a-Judge Reward
# --------------------------

import os
import re
from openai import OpenAI

def _format_trace_for_judging(trace: Trace, max_chars: int = 4000) -> str:
    """
    Format the new Trace = List[Turn] structure for LLM judging.
    
    Updated for new Trace = List[Turn] structure.
    """
    parts = []
    for i, turn in enumerate(trace):
        parts.append(f"=== Turn {i+1} ===")
        parts.append(f"Prompt: {turn['prompt_for_model'][:200]}...")
        parts.append(f"Completion: {turn['model_completion']}")
        
        # Add parsed completion details
        parsed = turn['parsed_completion']
        if parsed.get('thought'):
            parts.append(f"Thought: {parsed['thought']}")
        if parsed.get('tool_code'):
            parts.append(f"Tool Code: {parsed['tool_code']}")
        if parsed.get('final_answer'):
            parts.append(f"Final Answer: {parsed['final_answer']}")
        
        # Add tool output if present
        if turn['tool_output']:
            parts.append(f"Tool Output: {turn['tool_output']}")
        
        parts.append("")  # Empty line between turns
    
    text = "\n".join(parts)
    if len(text) > max_chars:
        return text[: max_chars - 30] + "\n... [truncated]"
    return text


def _extract_score(text: str) -> Optional[float]:
    # Find a float in [0, 1] range
    matches = re.findall(r"\b([01](?:\.\d+)?)\b", text)
    if not matches:
        return None
    # Prefer the first number between 0 and 1 inclusive
    try:
        val = float(matches[0])
        if 0.0 <= val <= 1.0:
            return val
    except Exception:
        return None
    return None


def reward_llm_judge(trace: Trace, **kwargs: Any) -> float:
    """
    LLM-as-a-Judge reward that does not require a gold answer.

    Usage patterns:
    - Provide `query` (recommended) so the judge can assess relevance/faithfulness.
    - Provide `system_prompt` to customize the judging criteria.
    - Provide `model` (default: 'gpt-4o-mini') and `openai_api_key` (or set OPENAI_API_KEY env var).
    - Optionally pass a pre-initialized `openai_client` via kwargs to reuse connections.

    Returns a float in [0, 1]. Falls back to 0.0 on error.
    """
    if OpenAI is None:
        return 0.0

    query = kwargs.get("query", "")
    system_prompt = kwargs.get(
        "system_prompt",
        (
            "You are a strict grader. Score the agent's solution quality from 0 to 1.\n"
            "Consider correctness, coherence, safety, and adherence to the user's query.\n"
            "Return ONLY a single number between 0 and 1 with up to 2 decimal places."
        ),
    )
    model = kwargs.get("model", "gpt-4o-mini")
    api_key = kwargs.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

    try:
        client = kwargs.get("openai_client") or OpenAI(api_key=api_key)
        trace_text = _format_trace_for_judging(trace)
        user_prompt = (
            f"User Query:\n{query}\n\n"
            f"Agent Trace (steps):\n{trace_text}\n\n"
            "Score this attempt strictly in [0, 1]."
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=8,
        )
        content = resp.choices[0].message.content or ""
        score = _extract_score(content)
        return float(score) if score is not None else 0.0
    except Exception:
        return 0.0 