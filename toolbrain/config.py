from peft import LoraConfig
from typing import Dict, Any, Optional
from abc import ABC

class BaseConfig(ABC):
    """Base configuration class with common parameters."""
    
    def __init__(self):
        # Common parameters for all algorithms
        self.learning_rate = 1e-5
        self.max_grad_norm = 1.0
        self.chunk_len = 128
        self.use_bitsandbytes = False
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, LoraConfig):
                # Convert LoraConfig to dict
                result[key] = {
                    'r': value.r,
                    'lora_alpha': value.lora_alpha,
                    'target_modules': value.target_modules,
                    'lora_dropout': value.lora_dropout,
                    'bias': value.bias,
                    'task_type': value.task_type
                }
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default=None):
        """Get attribute value."""
        return getattr(self, key, default)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if key == 'lora_config' and isinstance(value, dict):
                # Reconstruct LoraConfig
                setattr(config, key, LoraConfig(**value))
            else:
                setattr(config, key, value)
        return config

class GRPOConfig(BaseConfig):
    """GRPO-specific configuration."""
    
    def __init__(self):
        super().__init__()
        self.epsilon = 0.2  # Clipping parameter for GRPO
        self.beta = 0.04    # KL divergence penalty coefficient
        self.opt_steps = 3  # Number of optimization steps per batch
        self.num_group_members = 2  # Group members for GRPO

class DPOConfig(BaseConfig):
    """DPO-specific configuration."""
    
    def __init__(self):
        super().__init__()
        self.beta = 0.1  # Temperature parameter for DPO
        self.label_smoothing = 0.0
        self.loss_type = "sigmoid"  # Loss type for DPO

class SupervisedConfig(BaseConfig):
    """Supervised learning configuration."""
    
    def __init__(self):
        super().__init__()
        self.learning_rate = 2e-5  # Higher learning rate for supervised learning

def get_config(algorithm: str = "GRPO") -> BaseConfig:
    """
    Get default configuration for specified algorithm.
    
    Args:
        algorithm: Algorithm name ("GRPO", "DPO", "Supervised")
        
    Returns:
        Algorithm-specific configuration object
        
    Raises:
        ValueError: If algorithm is not supported
    """
    configs = {
        "GRPO": GRPOConfig,
        "DPO": DPOConfig, 
        "Supervised": SupervisedConfig,
    }
    
    if algorithm not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown algorithm: {algorithm}. Available algorithms: {available}")
    
    return configs[algorithm]()

# Legacy function for backward compatibility
def get_default_config():
    """Legacy function - returns GRPO config for backward compatibility."""
    return get_config("GRPO").to_dict()