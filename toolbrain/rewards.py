"""
Reward functions for evaluating agent performance.

This module provides pre-built reward functions that can be used
to evaluate the quality of agent execution traces.
"""

from typing import Optional
from .types import Trace


def reward_exact_match(trace: Trace, gold_answer: str) -> float:
    """
    Reward function based on exact string matching of the final answer.
    
    Args:
        trace: The execution trace from the agent
        gold_answer: The expected correct answer
        
    Returns:
        1.0 if final answer matches gold_answer exactly, 0.0 otherwise
    """
    final_answer_steps = [step for step in trace if step["type"] == "final_answer"]
    
    if not final_answer_steps:
        return 0.0
    
    final_answer = final_answer_steps[-1]["content"].strip()
    return 1.0 if final_answer == gold_answer else 0.0


def reward_tool_execution_success(trace: Trace) -> float:
    """
    Reward function based on successful tool execution.
    
    Args:
        trace: The execution trace from the agent
        
    Returns:
        1.0 if tools were executed without errors, 0.0 otherwise
    """
    tool_output_steps = [step for step in trace if step["type"] == "tool_output"]
    
    if not tool_output_steps:
        return 0.0
    
    # Check if any tool output contains an error
    for step in tool_output_steps:
        if "error" in step["content"].lower():
            return 0.0
    
    return 1.0


def reward_step_efficiency(trace: Trace, max_steps: int = 10) -> float:
    """
    Reward function that penalizes traces with too many steps.
    
    Args:
        trace: The execution trace from the agent
        max_steps: Maximum number of steps considered efficient
        
    Returns:
        Score between 0.0 and 1.0 based on step efficiency
    """
    num_steps = len(trace)
    
    if num_steps <= max_steps:
        return 1.0
    else:
        # Linear penalty for exceeding max_steps
        return max(0.0, 1.0 - (num_steps - max_steps) * 0.1)


def reward_combined(trace: Trace, gold_answer: str, weights: Optional[dict] = None) -> float:
    """
    Combined reward function using multiple criteria.
    
    Args:
        trace: The execution trace from the agent
        gold_answer: The expected correct answer
        weights: Dictionary of weights for each reward component
        
    Returns:
        Weighted combination of multiple reward functions
    """
    if weights is None:
        weights = {"exact_match": 0.7, "tool_success": 0.2, "efficiency": 0.1}
    
    exact_match_reward = reward_exact_match(trace, gold_answer)
    tool_success_reward = reward_tool_execution_success(trace)
    efficiency_reward = reward_step_efficiency(trace)
    
    total_reward = (
        weights.get("exact_match", 0.0) * exact_match_reward +
        weights.get("tool_success", 0.0) * tool_success_reward +
        weights.get("efficiency", 0.0) * efficiency_reward
    )
    
    return min(1.0, max(0.0, total_reward)) 