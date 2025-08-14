"""
ToolBrain Training Example - Simplified API

This script demonstrates the new, simplified ToolBrain API.
1. Define a configuration dictionary.
2. Pass the config to the Brain.
3. Call brain.train().
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import tool
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match, reward_combined

# --- 1. Define Tools and Reward Function (User-defined) ---
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
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

def main():
    print("🧠 ToolBrain Simplified Training Example")
    print("=" * 60)

    # --- 3. Define Brain Configuration ---
    # User defines a single dictionary
    toolbrain_config = {
        # Trainable model (downloaded locally)
        "model_id": "HuggingFaceTB/SmolLM-135M-Instruct",
        "tools": [add, multiply],
        "reward_func": reward_exact_match, # Select a reward function
        "learning_algorithm": "GRPO",
        
        # Optional parameters
        "num_group_members": 4, # Traces per query
        "rl_config": {
            "lr": 1e-5,
            "epsilon": 0.2,
            "beta": 0.01,
            "mu": 1,
            "max_grad_norm": 1.0,
        }
    }

    # --- 4. Initialize and Train ---
    # User only needs to know the Brain class
    try:
        # Brain handles everything internally
        brain = Brain(toolbrain_config)
        
        # Train on the full dataset
        brain.train(training_dataset, num_iterations=1)
        
        # Get the trained agent for use
        trained_agent = brain.get_agent()
        print("\n🤖 Agent after training:")
        
        # Test the trained agent
        result_trace = trained_agent.run("What is 100 + 50?")
        print(result_trace)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()