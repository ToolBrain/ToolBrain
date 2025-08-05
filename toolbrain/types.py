"""
Core type definitions for ToolBrain.

This module defines the fundamental data structures used throughout
the ToolBrain framework for representing agent execution traces.
"""

from typing import List, Literal, TypedDict


class TraceStep(TypedDict):
    """A single step in an agent's execution trace."""
    type: Literal["thought", "tool_code", "tool_output", "final_answer"]
    content: str


# Type alias for a complete execution trace
Trace = List[TraceStep] 