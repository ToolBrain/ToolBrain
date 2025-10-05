"""
ToolBrain Training Example - Flexible Rewards

This script demonstrates different reward functions for training agents:
1. Exact match (standard accuracy)
2. Tool execution success
3. Combined rewards with weights
4. LLM-as-a-Judge with Gemini

"""

import os

from dotenv import load_dotenv
load_dotenv()

from smolagents import tool, TransformersModel, CodeAgent
from toolbrain import Brain
from toolbrain.rewards import (
    reward_llm_judge_via_ranking,
)

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


# Dataset for LLM judge testing
def get_judge_model():
    """Auto-detect available LLM judge model based on API keys."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-flash"
    elif os.getenv("OPENAI_API_KEY"):
        return "gpt-4o"
    else:
        return None


judge_model_id = get_judge_model()

judge_dataset = [
    {
        "query": "Please use the provided tool add(a: int, b: int) to calculate sum of 5 and 7. Do not use anything else than that function.",
        "gold_answer": "12"
    }
]

print("🧠 ToolBrain LLM as a Judge Example")
print("=" * 60)

model = TransformersModel(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=128
)

agent = CodeAgent(
    model=model,
    tools=[add],
    max_steps=1
)

print("✅ Agent created.")
if judge_model_id:
    print(f"\n🤖 Training with LLM-as-a-Judge ({judge_model_id})...")
    brain_judge = Brain(
        agent,
        algorithm="GRPO",
        reward_func=reward_llm_judge_via_ranking,
        judge_model_id=judge_model_id,
        num_group_members=3
    )
    brain_judge.train(judge_dataset, num_iterations=1)
else:
    print("\n🤖 LLM Judge skipped (requires GEMINI_API_KEY or OPENAI_API_KEY)")
    print("Set one of these environment variables:")
    print("  export GEMINI_API_KEY='your-gemini-key'")
    print("  export OPENAI_API_KEY='your-openai-key'")

