"""
Core type definitions for ToolBrain.

This module defines the fundamental data structures used throughout
the ToolBrain framework for representing agent execution traces.

The new structure ensures 100% data fidelity for RL training by preserving
the exact context the LLM saw and what it generated.
"""

from typing import List, TypedDict, Protocol, Any, Optional

class ParsedCompletion(TypedDict):
    """
    Structured breakdown of the LLM's completion.
    
    All fields are required to ensure consistent structure.
    """
    thought: Optional[str]
    tool_code: Optional[str]
    final_answer: Optional[str]


class Turn(TypedDict):
    """
    A single, complete interaction turn that preserves data consistency.
    
    This structure ensures we can reconstruct exactly what the LLM saw
    and generated for accurate RL training signals.
    """
    prompt_for_model: str  # The exact context the LLM saw
    model_completion: str  # The exact raw string the LLM generated
    parsed_completion: ParsedCompletion  # Structured breakdown of the completion
    tool_output: Optional[str]  # Result from executing the tool code (if any)
    action_output: Optional[str]  # Result from executing the tool code (if any)
    formatted_conversation: Optional[str]  # Formatted text using smolagents utilities


# Trace structure: List of Turns for data consistency
Trace = List[Turn]

class ChatSegment(TypedDict):
    role: str
    text: str

class RewardFunction(Protocol):
    """
    Protocol for reward functions.

    Implementations should accept a Trace (List[Turn]) and arbitrary keyword arguments,
    returning a float score in [0, 1] (or any real-valued score, depending on use-case).
    
    The Trace structure provides access to:
    - turn.prompt_for_model: What the LLM saw
    - turn.model_completion: What the LLM generated
    - turn.parsed_completion: Structured breakdown
    - turn.tool_output: Environment feedback
    - turn.formatted_conversation: Formatted text using smolagents utilities
    
    Advanced reward functions can also access raw memory data via kwargs:
    - kwargs.get("raw_memory_collection"): List of raw agent memory steps for detailed analysis
    """
    def __call__(self, trace: Trace, **kwargs: Any) -> float: ...


class BatchRewardFunction(Protocol):
    """
    Protocol for batch reward functions that process multiple traces at once.
    
    This is useful for LLM-as-a-Judge functions that need to compare traces
    against each other for ranking or relative scoring.
    
    Advanced batch reward functions can access raw memory data via kwargs:
    - kwargs.get("raw_memory_collection"): List of List[raw_memory_steps] for cross-trace analysis
    """
    def __call__(self, traces: List[Trace], **kwargs: Any) -> List[float]: ...



