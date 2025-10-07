"""
ToolBrain Training Example

This script is similar to the standard hello-world example but running training and inference with fp16 to save GPU memory.
This example only run on machines with GPU supported.

This script demonstrates the new, ultra-simplified ToolBrain API:
1. Create a smolagent CodeAgent
2. Create brain with Brain() constructor (all parameters as keywords)
3. Train with explicit, self-documenting parameters

"""

from smolagents import tool, TransformersModel, CodeAgent
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match
import torch

# --- 1. Define Tools and Reward Function (User-defined) ---
@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): First addend.
        b (int): Second addend.

    Returns:
        int: Sum of a and b.
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): First factor.
        b (int): Second factor.

    Returns:
        int: Product of a and b.
    """
    return a * b


# --- 2. Prepare Training Data ---
training_dataset = [
    {
        "query": "Use the add tool to calculate 5 + 7",
        "gold_answer": "12"
    },
    {
        "query": "What is 8 multiplied by 6?",
        "gold_answer": "48"
    },
    # Add more examples here
]


print("🧠 ToolBrain Training Example with Reinforcement Learning")
print("=" * 60)

# 1. Create agent
model = TransformersModel(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",  # use a bigger model for better results
    max_new_tokens=128,
    torch_dtype=torch.float16  # We must set this to run with FP16 precision
)

agent = CodeAgent(
    model=model,
    tools=[add, multiply],
    max_steps=1
)

print("✅ Agent created.")

# 2. Create Brain

# This is a simplified version of Brain with default parameters settings, for advanced parameter settings please
# refer to the documentation.
brain = Brain(
    agent,                          # Agent instance
    algorithm="GRPO",                # Algorithm choice
    # Customised reward function is defined here, we use a mocking reward function with value 1.0
    # for an exact gold_answer match and 0 otherwise, llm as judge can be used for automatic reward
    reward_func=reward_exact_match,
    fp16= True  # Enable fp16 training in Brain
)

# 3. Train the agent with RL for 10 training GRPO steps
brain.train(training_dataset, num_iterations=10)