"""
ToolBrain Training Example - Flexible Rewards

This script demonstrates different reward functions for training agents:
1. Exact match (standard accuracy)
2. Tool execution success 
3. Combined rewards with weights
4. LLM-as-a-Judge with Gemini

"""

from dotenv import load_dotenv
load_dotenv()

from smolagents import tool, TransformersModel, CodeAgent
from toolbrain import Brain
from toolbrain.rewards import (
    reward_exact_match,
    reward_tool_execution_success,
    reward_combined,
)
from toolbrain.core_types import Trace
from typing import Any

# --- 1. Define Tools ---
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

# --- 2. Training Data ---
training_dataset = [
    {
        "query": "Use the add tool to calculate 5 + 7",
        "gold_answer": "12"
    },
    {
        "query": "What is 8 multiplied by 6?",
        "gold_answer": "48"
    },
]


# Custom combined reward function
def custom_combined_reward(trace: Trace, **kwargs: Any) -> float:
    weights = {
        "exact_match": 0.7,    # 70% weight on correctness
        "tool_success": 0.3,   # 30% weight on successful tool execution
    }
    kwargs["weights"] = weights
    return reward_combined(trace, **kwargs)

print("🧠 ToolBrain Flexible Rewards Example")
print("=" * 60)

# 1. Create agent 
model = TransformersModel(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=128
)

agent = CodeAgent(
    model=model,
    tools=[add, multiply],
    max_steps=1
)

print("✅ Agent created.")

# 2. Train with exact match reward (standard)
# print("\n🎯 Training with exact match reward...")
# brain_exact = Brain(
#     agent,
#     algorithm="GRPO",
#     reward_func=reward_exact_match
# )
# brain_exact.train(training_dataset[:1], num_iterations=1)

# # 3. Train with tool execution success reward
# print("\n🛠️ Training with tool execution success reward...")
# brain_tool = Brain(
#     agent,
#     algorithm="GRPO",
#     reward_func=reward_tool_execution_success
# )
# brain_tool.train(training_dataset[:1], num_iterations=1)

# 4. Train with combined reward (exact match + tool success)
print("\n🎭 Training with combined reward (70% accuracy + 30% tool success)...")
brain_combined = Brain(
    agent,
    algorithm="GRPO",
    reward_func=custom_combined_reward
)
brain_combined.train(training_dataset[:1], num_iterations=1)



