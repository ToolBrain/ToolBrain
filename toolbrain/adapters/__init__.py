"""
ToolBrain adapters module.

This module provides adapters for different agent frameworks,
organized into separate subdirectories for better code organization.
"""

from .base_adapter import BaseAgentAdapter
from .smolagent import SmolAgentAdapter
from .langchain import LangChainAdapter, create_huggingface_chat_model

__all__ = [
    "BaseAgentAdapter",
    "SmolAgentAdapter", 
    "LangChainAdapter",
    "create_huggingface_chat_model",  # Helper function
]