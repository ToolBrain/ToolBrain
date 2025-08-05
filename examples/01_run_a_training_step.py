"""
Simple ToolBrain example - Ideal user experience.

This script demonstrates how easy it is to use ToolBrain:
1. Create a normal CodeAgent
2. Pass it to Brain
3. Run training

"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import CodeAgent, InferenceClientModel, tool
from toolbrain import Brain
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


def main() -> None:
    """Demonstrate the ideal ToolBrain user experience."""
    
    print("ToolBrain Training Example")
    print("=" * 50)
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "❌ HF_TOKEN environment variable is required!\n"
            "Create a .env file with: HF_TOKEN=your_token_here\n"
            "Get a token at: https://huggingface.co/settings/tokens"
        )
    
    print("✅ HF_TOKEN found")
    
    # Initialize model
    print("🔗 Connecting to Hugging Face Inference API...")
    model = InferenceClientModel(
        model_id="HuggingFaceTB/SmolLM3-3B",
        token=hf_token
    )
    print("✅ Model initialized")
    
    # Create tools
    tools = [add, multiply]
    print(f"🛠️  Created {len(tools)} tools: {[tool.name for tool in tools]}")
    
    # --- USER EXPERIENCE ---
    # User creates a completely normal CodeAgent
    print("🤖 Creating CodeAgent...")
    agent = CodeAgent(tools=tools, model=model)
    print("✅ Agent created")
    
    # User simply passes their agent to Brain
    # Brain automatically handles all trace instrumentation
    print("🧠 Initializing Brain...")
    brain = Brain(
        agent=agent,  
        reward_func=reward_exact_match,
        learning_algorithm="MockDPO"
    )
    
    # User runs training 
    print("\n🚀 Running training step...")
    
    query = "Use the add tool to calculate 5 + 7"
    gold_answer = "12"
    
    try:
        brain.train_step(
            query=query,
            gold_answer=gold_answer,
            num_group_members=3
        )
        
        # Show training statistics
        stats = brain.get_training_stats()
        print(f"\n📊 Training Statistics:")
        print(f"  Algorithm: {stats['algorithm']}")
        print(f"  Training Steps: {stats['training_steps']}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    print("\n✅ Training step completed successfully!")


if __name__ == "__main__":
    main() 