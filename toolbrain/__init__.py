"""
ToolBrain - A framework for training LLM-powered agents to use tools effectively.

Clean and simple API with separated concerns:

# === SIMPLIFIED API === 
from toolbrain import create_agent, Brain

# 1. Create agent with tools
agent = create_agent("microsoft/DialoGPT-medium", tools)

# 2. Create brain with explicit parameters  
brain = Brain(agent, algorithm="GRPO", learning_rate=1e-4, epsilon=0.2)
brain.train(dataset)

# 3. Save and load models
brain.save("./my_model")
loaded_agent = Brain.load_agent("./my_model", "microsoft/DialoGPT-medium", tools)

# Clean separation: Agent handles execution, Brain handles training strategy!
"""

# Main factory functions - primary API
from .factory import (
    create_agent,
    create_optimized_model,
)

# Core classes 
from .brain import Brain
from .config import get_config, GRPOConfig, DPOConfig, SupervisedConfig, BaseConfig
from .rewards import (
    # Built-in single-trace reward functions
    reward_exact_match,
    reward_tool_execution_success,
    reward_step_efficiency,
    reward_behavior_uses_search_first,
    reward_safety_no_os_system,
    reward_combined,
    
    # LLM-as-a-Judge batch reward function
    reward_llm_judge_via_ranking,
    
    # Utility classes and functions
    RewardFunctionWrapper,
    create_reward_function,
)

# Legacy/Advanced imports for backward compatibility
from .core_types import Trace
from .adapters import BaseAgentAdapter, SmolAgentAdapter, LangChainAdapter, create_huggingface_chat_model

__version__ = "0.1.0"

# Public API - what most users should import
__all__ = [
    # === PRIMARY API ===
    "Brain",               
    
    # === FACTORY FUNCTIONS ===
    "create_agent",        # Create agent with tools
    "create_optimized_model",  # Create UnslothModel or TransformersModel
    
    # === CONFIGURATION ===
    "get_config",
    "GRPOConfig", 
    "DPOConfig",
    "SupervisedConfig",
    "BaseConfig",
    
    # === REWARD FUNCTIONS ===
    # Built-in single-trace rewards
    "reward_exact_match",
    "reward_tool_execution_success", 
    "reward_step_efficiency",
    "reward_behavior_uses_search_first",
    "reward_safety_no_os_system",
    "reward_combined",
    
    # LLM-as-a-Judge batch reward
    "reward_llm_judge_via_ranking",
    
    # Utility classes
    "RewardFunctionWrapper",
    "create_reward_function",
    
    # === ADVANCED/LEGACY ===
    "Trace",
    "BaseAgentAdapter", 
    "SmolAgentAdapter",
    "LangChainAdapter",
    "create_huggingface_chat_model",  # Helper function
]
