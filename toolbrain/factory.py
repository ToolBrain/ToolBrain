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

from typing import List, Any, Optional
import platform

def create_optimized_model(
    model_id: str,
    use_unsloth: Optional[bool] = None,
    max_seq_length: int = 4096,  
    max_new_tokens: int = 512,
    load_in_4bit: bool = True,   # Important Unsloth optimization
    **model_kwargs
):
    """
    Create optimized model with clear model-level parameters.
    
    These parameters are for model creation only and define model capacity.
    Training-specific optimizations should be configured via Brain config:
    - config.unsloth_config_overrides = {...} for training optimizations
    - config.lora_config_overrides = {...} for LoRA training settings
    
    Args:
        model_id: HuggingFace model ID
        use_unsloth: Force use/not use Unsloth (auto-detect if None)
        max_seq_length: Maximum sequence length for model (model capacity)
        max_new_tokens: Maximum tokens to generate during inference
        load_in_4bit: Enable 4-bit quantization for memory efficiency
        **model_kwargs: Additional model arguments
    
    Returns:
        Optimized TransformersModel or UnslothModel
        
    Example:
        # Model configuration (creation time)
        model = create_optimized_model(
            "Qwen/Qwen2.5-0.5B-Instruct",
            max_seq_length=4096,      # Model capacity
            load_in_4bit=True         # Model optimization
        )
        
        # Training configuration (separate)
        config = GRPOConfig()
        config.unsloth_config_overrides = {"use_rslora": True}  # Training-only
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
            load_in_4bit=load_in_4bit, 
            **model_kwargs
        )
    else:
        # Standard model
        model_kwargs_filtered = {k: v for k, v in model_kwargs.items() 
                               if k not in ['load_in_4bit']}  # Filter Unsloth-specific
        return TransformersModel(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            **model_kwargs_filtered
        )

def create_agent(
    model_id: str,
    tools: List[Any],
    *,  # Force keyword-only arguments
    # Model creation parameters 
    use_unsloth: Optional[bool] = None,
    max_seq_length: int = 4096,   
    max_new_tokens: int = 512,
    load_in_4bit: bool = True,    
    
    # Agent parameters
    max_steps: int = 10,    
    **model_kwargs
):
    """
    Create a standard CodeAgent with optimized model - pure execution engine.
    
    This creates a "standard athlete" - an agent with a model and full access
    to all provided tools. Model parameters set here define model capacity.
    
    Training optimizations should be configured separately via Brain config:
    - config.unsloth_config_overrides = {...} for training-specific settings
    - config.lora_config_overrides = {...} for LoRA training configuration
    
    Strategy decisions (tool retrieval, reward functions, algorithms) are handled 
    by Brain, maintaining clean separation of concerns.
    
    Args:
        model_id: HuggingFace model ID
        tools: Complete list of tools the agent can use
        use_unsloth: Force use/not use Unsloth (auto-detect if None)
        max_seq_length: Maximum sequence length for model capacity
        max_new_tokens: Maximum tokens to generate during inference
        load_in_4bit: Enable 4-bit quantization for memory efficiency
        max_steps: Maximum reasoning steps per query
        **model_kwargs: Additional arguments passed to model creation
    
    Returns:
        Standard CodeAgent ready for Brain-directed training
        
    Example:
        # Model configuration (creation time)
        agent = create_agent(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            tools=tools,
            max_seq_length=4096,      # Model capacity
            load_in_4bit=True         # Model optimization
        )
        
        # Training configuration (separate from model)
        config = GRPOConfig()
        config.unsloth_config_overrides = {"use_rslora": True}
        config.lora_config_overrides = {"r": 32}
        
        # Brain applies training strategy
        brain = Brain(agent, config=config)
    """
    from smolagents import CodeAgent
    
    # Create optimized model 
    model = create_optimized_model(
        model_id=model_id,
        use_unsloth=use_unsloth,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        load_in_4bit=load_in_4bit,  
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
