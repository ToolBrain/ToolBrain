"""
ToolBrain - A framework for training LLM-powered agents to use tools effectively.

ToolBrain provides a structured approach to improving agent performance through
reinforcement learning, with built-in support for execution trace capture,
reward function evaluation, and RL algorithm integration.
"""

from .brain import Brain
from .core_types import Trace, TraceStep

__version__ = "0.1.0"
__all__ = ["Brain", "Trace", "TraceStep"] 