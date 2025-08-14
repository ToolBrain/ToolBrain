"""
ToolBrain Training Example - Adapter Pattern Demonstration

This script demonstrates the new Adapter pattern approach:
1. Create a standard CodeAgent
2. Wrap it with SmolAgentAdapter (explicit adapter creation)
3. Pass the adapter to Brain
4. Run training

The Adapter pattern makes the process explicit and testable.
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import CodeAgent, InferenceClientModel, tool, LiteLLMModel
from toolbrain import Brain
from toolbrain.adapters import SmolAgentAdapter
from toolbrain.rewards import reward_exact_match


@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        The sum of a and b
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers together.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
        
    Returns:
        The product of a and b
    """
    return a * b


def main():
    """Demonstrate the Adapter pattern approach."""
    
    print("🧠 ToolBrain Training Example - Adapter Pattern")
    print("=" * 60)
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "❌ HF_TOKEN environment variable is required!\n"
            "Create a .env file with: HF_TOKEN=your_token_here\n"
            "Get a token at: https://huggingface.co/settings/tokens"
        )
    
    print("✅ HF_TOKEN found")
    
    # 1. Config variables
    model_id = os.getenv("MODEL_ID")
    api_base = os.getenv("API_BASE")

    tools = [add, multiply]

    # 2. Init model for SmolAgent (placeholder for API calls)
    model_for_agent = LiteLLMModel(model_id=model_id)
    
    # 3. Create standard CodeAgent
    print("\n🤖 Creating standard CodeAgent...")
    original_agent = CodeAgent(tools=tools, model=model_for_agent)
    print("✅ Standard CodeAgent created")

    # 4. Wrap agent with SmolAgentAdapter (Adapter Pattern)
    print("\n🔧 Creating SmolAgentAdapter...")
    agent_adapter = SmolAgentAdapter(
        agent=original_agent, 
        model_id=model_id, 
        api_base=api_base,
        api_key=hf_token
    )
    print(f"✅ Adapter created and ready.")
    
    # 5. Initialize Brain with adapter
    print("\n🧠 Initializing Brain with adapter...")
    brain = Brain(
        agent_adapter=agent_adapter,
        reward_func=reward_exact_match,
        learning_algorithm="MockRL"
    )
    
    # 6. Run training step
    print("\n🚀 Running training step...")
    
    query = "Use the add tool to calculate 5 + 7"
    gold_answer = "12"
    
    # With a gold answer (optional)
    brain.train_step(
        query=query,
        num_group_members=3,
        gold_answer=gold_answer,
    )

    # Example without a gold answer (creative task) – will work with behavior/efficiency rewards
    # brain.train_step(
    #     query="Write a poem about autumn.",
    #     num_group_members=3,
    # )
    
    # 7. Show training statistics
    stats = brain.get_training_stats()
    print(f"\n📊 Training Statistics:")
    print(f"  Algorithm: {stats['algorithm']}")
    print(f"  Training Steps: {stats['training_steps']}")
    print(f"  Adapter Type: {stats['adapter_type']}")
    
    print("\n✅ Training step completed successfully!")


if __name__ == "__main__":
    main() 