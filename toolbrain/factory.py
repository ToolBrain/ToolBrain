"""
Factory functions for creating ToolBrain components with clean separation of concerns.

Philosophy:
- Agent: Pure execution engine. Handles model + tools, nothing else.  
- Brain: Strategy and training logic. Handles tool retrieval, reward functions, algorithms.

This design eliminates complexity and provides a single, clear API:
1. create_agent() - Creates standard agents (Athlete with full toolbox)
2. Brain() - Applies training strategies (Coach decides tactics)

Clean separation: Execution vs Strategy.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import platform
from .config import BaseConfig, get_config
from .rewards import RewardFunctionWrapper, create_reward_function, reward_exact_match
from .core_types import RewardFunction, BatchRewardFunction

def create_optimized_model(
    model_id: str,
    use_unsloth: Optional[bool] = None,
    max_seq_length: int = 17408,  # Match examples 08: 16896 + buffer for safety
    max_new_tokens: int = 512,    # Match examples 08
    **model_kwargs
):
    """
    Factory function to create optimized model automatically.
    
    Args:
        model_id: HuggingFace model ID
        use_unsloth: Force use/not use Unsloth (auto-detect if None)
        max_seq_length: Maximum sequence length
        max_new_tokens: Maximum tokens to generate during inference
        **model_kwargs: Additional model arguments
    
    Returns:
        Optimized TransformersModel or UnslothModel
    """
    try:
        from .models import UnslothModel, UNSLOTH_AVAILABLE
        from smolagents import TransformersModel
    except ImportError:
        from smolagents import TransformersModel
        UNSLOTH_AVAILABLE = False
    
    # Auto-detect whether to use Unsloth
    if use_unsloth is None:
        # Use Unsloth if available and not on macOS
        use_unsloth = (UNSLOTH_AVAILABLE and platform.system() != "Darwin")
    
    if use_unsloth and UNSLOTH_AVAILABLE:
        return UnslothModel(
            model_id=model_id,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            **model_kwargs
        )
    else:
        return TransformersModel(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            **model_kwargs
        )

def create_agent(
    model_id: str,
    tools: List[Any],
    *,  # Force keyword-only arguments
    use_unsloth: Optional[bool] = None,
    max_seq_length: int = 17408,  
    max_new_tokens: int = 512,
    max_steps: int = 10,    
    **model_kwargs
):
    """
    Create a standard CodeAgent - pure execution engine.
    
    This creates a "standard athlete" - an agent with a model and full access
    to all provided tools. The agent has no special behaviors, retrieval logic,
    or training strategies. It's purely an execution engine.
    
    Strategy decisions (tool retrieval, reward functions, algorithms) are handled 
    by Brain, maintaining clean separation of concerns.
    
    Args:
        model_id: HuggingFace model ID
        tools: Complete list of tools the agent can use
        use_unsloth: Force use/not use Unsloth (auto-detect if None)
        max_seq_length: Maximum sequence length for the model
        max_new_tokens: Maximum tokens to generate during inference
        max_steps: Maximum reasoning steps per query
        **model_kwargs: Additional arguments passed to model creation
    
    Returns:
        Standard CodeAgent ready for Brain-directed training
        
    Example:
        >>> # Create agent with full toolbox
        >>> tools = [add_tool, search_tool, email_tool, db_tool]
        >>> agent = create_agent("microsoft/DialoGPT-medium", tools)
        >>> 
        >>> # Brain decides strategy (e.g., which tools to use per query)
        >>> brain = Brain(agent, algorithm="GRPO", enable_tool_retrieval=True)
        >>> # Now Brain intelligently filters tools per query
    """
    from smolagents import CodeAgent
    
    # Create optimized model
    model = create_optimized_model(
        model_id=model_id,
        use_unsloth=use_unsloth,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        **model_kwargs
    )
    
    # Ensure tokenizer has chat template
    tokenizer = model.tokenizer
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}"

    # Create agent with model and tools
    agent = CodeAgent(
        model=model,
        tools=tools,
        max_steps=max_steps  
    )
    
    return agent
