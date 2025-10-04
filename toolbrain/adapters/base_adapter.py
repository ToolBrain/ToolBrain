"""
Base adapter interface for ToolBrain.

This module provides the abstract base class that all agent adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
from ..core_types import Trace


class BaseAgentAdapter(ABC):
    """Abstract base class for agent adapters."""

    @abstractmethod
    def run(self, query: str) -> Tuple[Trace, Any, Any]:
        """
        Execute a query and return a structured execution trace.
        
        Args:
            query: The input query/prompt for the agent
            
        Returns:
            Tuple of (trace, rl_input, raw_memory_steps)
        """
        pass

    @abstractmethod
    def get_trainable_model(self) -> Any:
        """Return the underlying trainable model from the agent."""
        pass