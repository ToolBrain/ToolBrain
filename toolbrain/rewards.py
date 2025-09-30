"""
Flexible reward functions for ToolBrain.

These functions follow the RewardFunction protocol and accept arbitrary
keyword arguments so they can be composed in many settings, including
cases without a single gold answer.
"""

from typing import Any, Optional, List, Union, Callable
from .core_types import Trace, RewardFunction, BatchRewardFunction
import re
from pydantic import BaseModel

def reward_exact_match(trace: Trace, **kwargs: Any) -> float:
    """
    Reward 1.0 if the final answer exactly matches the provided gold_answer, else 0.0.
    Handles multiple sources for final answers:
    1. parsed_completion["final_answer"] - direct final answer field
    2. tool_output - when final_answer() function call returns the result
    3. model_completion - parsing "Final answer: X" from raw output
    """
    gold_answer = kwargs.get("gold_answer")
    if gold_answer is None:
        return 0.0
    
    gold_str = str(gold_answer).strip()
    
    for turn in reversed(trace):
        parsed = turn.get("parsed_completion", {})
        
        # Method 1: Check parsed final_answer field (standard case)
        if parsed:
            final_answer = parsed.get("final_answer")
            if isinstance(final_answer, str) and final_answer.strip() == gold_str:
                return 1.0
        
        # Method 2: Check tool_output if final_answer() was called
        action_output = turn.get("action_output")
        if action_output is not None:
            tool_output_str = str(action_output).strip()
            if tool_output_str == gold_str:
                return 1.0
        
        # Method 3: Parse "Final answer: X" from model_completion
        model_completion = turn.get("model_completion", "")
        if "Final answer:" in model_completion:
            parts = model_completion.split("Final answer:")
            if len(parts) > 1:
                extracted_answer = parts[-1].strip()
                if extracted_answer == gold_str:
                    return 1.0
        
        # Method 4: Check if tool_code contains final_answer() call and tool_output matches
        if parsed:
            tool_code = parsed.get("tool_code", "")
            if "final_answer(" in tool_code and action_output is not None:
                if str(action_output).strip() == gold_str:
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


# --------------------------------------------------
# LLM-as-a-Judge Reward (GRPO-Compatible via Ranking)
# --------------------------------------------------
try:
    import litellm
    litellm.set_verbose=False
except ImportError:
    litellm = None

# --- PYDANTIC MODELS FOR STRUCTURED OUTPUT ---

class JudgeRankingResponse(BaseModel):
    ranking: List[int]  # e.g., [1, 3, 2] 
    reasoning: str  # Brief explanation of ranking decision

# --- TRACE SUMMARIZATION FUNCTIONS ---

def _summarize_trace_for_judge(trace: Trace, query: str) -> str:
    """
    Summarizes a trace into a concise format for LLM judge evaluation.
    Only includes essential information: query, reasoning flow, final answer.
    """
    if not trace:
        return "Empty trace - no actions taken"
    
    summary_parts = []
    
    # 1. Add the query
    summary_parts.append(f"Query: {query}")
    summary_parts.append("")
    
    # 2. Extract reasoning and action flow (without long tool outputs)
    summary_parts.append("Reasoning & Action Flow:")
    for i, turn in enumerate(trace, 1):
        thought = turn.get("parsed_completion", {}).get("thought", "").strip()
        tool_call = turn.get("parsed_completion", {}).get("tool_call", {})
        
        if thought:
            summary_parts.append(f"  {i}. Thought: {thought[:200]}{'...' if len(thought) > 200 else ''}")
        
        if tool_call:
            tool_name = tool_call.get("tool_name", "unknown")
            # Summarize tool arguments without full content
            args = tool_call.get("arguments", {})
            if args:
                # Only show key argument names, not values
                arg_summary = ", ".join(f"{k}: {str(v)[:50]}{'...' if len(str(v)) > 50 else ''}" for k, v in args.items())
                summary_parts.append(f"     Action: {tool_name}({arg_summary})")
            else:
                summary_parts.append(f"     Action: {tool_name}()")
    
    summary_parts.append("")
    
    # 3. Extract final answer
    final_answer = "No final answer provided"
    for turn in reversed(trace):
        parsed = turn.get("parsed_completion", {})
        if parsed.get("final_answer"):
            final_answer = parsed["final_answer"]
            break
    
    summary_parts.append(f"Final Answer: {final_answer}")
    
    return "\n".join(summary_parts)

def _format_traces_for_ranking(traces: List[Trace], query: str = "") -> str:
    """
    Formats a list of traces into a summarized string for an LLM judge.
    Uses trace summarization to reduce noise and bias from long tool outputs.
    """
    parts = []
    for i, trace in enumerate(traces):
        if trace and len(trace) > 0:
            # Use summarization instead of raw trace data
            if query:
                summary = _summarize_trace_for_judge(trace, query)
                parts.append(f"<trajectory id='{i+1}'>\n{summary}\n</trajectory>")
            else:
                # Fallback to formatted conversation if no query provided
                formatted_text = trace[0].get("formatted_conversation", "")
                
                # If no formatted text available, fall back to simple representation
                if not formatted_text:
                    trace_parts = []
                    for turn in trace:
                        if turn.get('model_completion'):
                            trace_parts.append(turn['model_completion'])
                        if turn.get('tool_output'):
                            trace_parts.append(f"Tool Output: {turn['tool_output'][:200]}...")
                    formatted_text = "\n".join(trace_parts)
                
                # Truncate if too long
                if len(formatted_text) > 2000:
                    formatted_text = formatted_text[:1970] + "\n... [truncated]"
                    
                parts.append(f"<trajectory id='{i+1}'>\n{formatted_text}\n</trajectory>")
        else:
            parts.append(f"<trajectory id='{i+1}'>\n[Empty trace]\n</trajectory>")
    
    return "\n\n".join(parts)

def _parse_ranking_from_llm(response_text: str, num_traces: int) -> Optional[List[int]]:
    """
    Extracts a ranked list of trajectory IDs from the judge's response.
    This version is more robust and looks for a list-like structure.
    """
    try:
        # 1. Find a list-like structure, e.g., [3, 1, 2] or (3, 1, 2)
        list_match = re.search(r'[\(\[]\s*(\d+(?:\s*,\s*\d+)*)\s*[\)\]]', response_text)
        
        if list_match:
            # 2. If found, only process the numbers inside
            numbers_str = list_match.group(1)
            ranked_ids = [int(n.strip()) for n in numbers_str.split(',')]
        else:
            # 3. If not found, revert to the old method of finding all numbers
            numbers = re.findall(r'\d+', response_text)
            ranked_ids = [int(n) for n in numbers]

        # 4. Validate the result
        #    Add check for duplicate numbers
        if (len(ranked_ids) == num_traces and 
            all(1 <= i <= num_traces for i in ranked_ids) and
            len(set(ranked_ids)) == num_traces):
            return ranked_ids
        else:
            return None
    except Exception:
        return None

def _convert_ranking_to_scores(ranking: List[int], num_traces: int) -> List[float]:
    """Converts a ranked list of IDs into a list of scores using linear spacing."""
    scores = [0.0] * num_traces
    step = 1.0 / (num_traces - 1) if num_traces > 1 else 0.0
    
    for rank, trace_id in enumerate(ranking):
        score = 1.0 - (rank * step)
        original_index = trace_id - 1
        scores[original_index] = score
        
    return scores


def reward_llm_judge_via_ranking(traces: List[Trace], **kwargs: Any) -> List[float]:
    """
    An LLM-as-a-Judge reward function compatible with GRPO.
    
    It asks an LLM judge to RANK the given traces, then converts that
    ranking into a list of numerical scores. This function operates on a BATCH of traces.
    
    Uses pre-formatted conversation text from smolagents utilities for consistency
    with training data format.

    Args:
        traces: The list of traces to be judged.
        **kwargs: Must include:
            - query: The original user query.
            - judge_model: The model ID for the judge (e.g., "gemini/gemini-1.5-flash").

    Returns:
        A list of float scores, one for each trace, in the original order.
    """
    if litellm is None:
        return [0.5] * len(traces)

    if not traces:
        return []

    num_traces = len(traces)
    query = kwargs.get("original_question")
    judge_model = kwargs.get("judge_model")

    if not all([query, judge_model]):
        raise ValueError("`reward_llm_judge_via_ranking` requires 'query' and 'judge_model' in kwargs.")

    # 1. Prepare prompt for judge with query context
    traces_str = _format_traces_for_ranking(traces, query)
    system_prompt = """You are a fair and impartial AI performance evaluator. Your task is to rank multiple agent trajectories based on their quality.
    
    Return your response in JSON format with:
    - ranking: A list of trajectory IDs ranked from BEST to WORST (e.g., [3, 1, 2])
    - reasoning: Brief explanation of your ranking decision"""
    
    user_prompt = f"""User Query:
{query}

Agent Trajectories:
{traces_str}

---
INSTRUCTION:
Review all trajectories above. Rank them from BEST to WORST based on correctness, efficiency, and adherence to the query."""

    # 2. Call LLM Judge
    try:
        response = litellm.completion(
            model=judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=JudgeRankingResponse,
            temperature=0.0
        )
        
        # Parse structured response
        try:
            # Try structured parsing first
            result_obj = response.choices[0].message.parsed
            ranking = result_obj.ranking
            reasoning = result_obj.reasoning
        except (AttributeError, KeyError):
            # Fallback: parse JSON from content
            import json
            content = response.choices[0].message.content
            result_dict = json.loads(content)
            ranking = result_dict.get("ranking", [])
            reasoning = result_dict.get("reasoning", "No reasoning provided")
        

        
        # 3. Validate ranking
        
        if ranking:
            # 4. Convert ranking to scores
            scores = _convert_ranking_to_scores(ranking, num_traces)
            return scores
        else:
            return [0.5] * num_traces

    except Exception as e:
        return [0.5] * num_traces


# --------------------------------------------------
# Dynamic Reward Wrapper for Supporting Both Types
# --------------------------------------------------

class RewardFunctionWrapper:
    """
    Dynamic wrapper that can handle both single-trace and batch reward functions.
    
    This allows Brain to use a unified interface regardless of reward function type.
    """
    
    def __init__(self, reward_func: Union[RewardFunction, BatchRewardFunction]):
        """
        Initialize wrapper with either single-trace or batch reward function.
        
        Args:
            reward_func: Either a RewardFunction (trace -> float) or 
                        BatchRewardFunction (traces -> List[float])
        """
        self.reward_func = reward_func
        self._is_batch_function = self._detect_batch_function(reward_func)
        
    def _detect_batch_function(self, func: Callable) -> bool:
        """
        Detect if function is a batch function by checking its signature.
        
        Batch functions typically have 'traces' as first parameter.
        Single functions have 'trace' as first parameter.
        """
        import inspect
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if params:
                first_param = params[0]
                # Batch functions typically use 'traces' (plural)
                # Single functions use 'trace' (singular)  
                return first_param == 'traces'
            return False
        except Exception:
            # Assume single-trace function
            return False
    
    def __call__(self, trace: Trace, **kwargs: Any) -> float:
        """
        Unified interface that always accepts single trace and returns single score.
        
        For batch functions, we need to collect traces and call in batches.
        This method should not be called directly - use in Brain context.
        """
        if self._is_batch_function:
            raise ValueError("Batch reward function cannot be called with single trace. Use get_batch_scores instead.")
        else:
            return self.reward_func(trace=trace, **kwargs)
    
    def get_batch_scores(self, traces: List[Trace], **kwargs: Any) -> List[float]:
        """
        Get scores for a batch of traces.
        
        For single-trace functions, applies function to each trace individually.
        For batch functions, calls function once with all traces.
        """
        if self._is_batch_function:
            return self.reward_func(traces=traces, **kwargs)
        else:
            # Apply single-trace function to each trace
            scores = []
            for trace in traces:
                try:
                    score = self.reward_func(trace=trace, **kwargs)
                    scores.append(float(score))
                except Exception as e:
                    scores.append(0.0)
            return scores
    
    @property
    def is_batch_function(self) -> bool:
        """Returns True if wrapped function is a batch function."""
        return self._is_batch_function


# --------------------------------------------------
# Convenience Functions for Easy Usage
# --------------------------------------------------

def create_reward_function(func: Union[RewardFunction, BatchRewardFunction]) -> RewardFunctionWrapper:
    """
    Convenience function to create a RewardFunctionWrapper.
    
    Usage:
        # Single trace function
        reward_func = create_reward_function(reward_exact_match)
        
        # Batch function  
        reward_func = create_reward_function(reward_llm_judge_via_ranking)
    """
    return RewardFunctionWrapper(func)