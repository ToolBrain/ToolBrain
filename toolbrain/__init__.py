"""
ToolBrain - A framework for training LLM-powered agents to use tools effectively.

ToolBrain provides a structured approach to improving agent performance through
reinforcement learning, with built-in support for execution trace capture,
reward function evaluation, and RL algorithm integration.

The framework uses the Adapter pattern for clean separation of concerns.
"""

from .brain import Brain
from .core_types import Trace, TraceStep
from .adapters import BaseAgentAdapter, SmolAgentAdapter

__version__ = "0.1.0"
__all__ = [
    "Brain", 
    "Trace", 
    "TraceStep",
    "BaseAgentAdapter",
    "SmolAgentAdapter"
] 