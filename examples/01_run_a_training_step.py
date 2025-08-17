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

from smolagents import CodeAgent, TransformersModel, tool
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match

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
    print("🧠 ToolBrain Flexible Training Example")
    print("=" * 60)

    # --- 3. User creates their agent freely ---
    print("🤖 User is creating their own agent...")
    # User must use TransformersModel to train
    trainable_model = TransformersModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")
    
    my_agent = CodeAgent(
        tools=[add, multiply],
        model=trainable_model
    )
    print("✅ Agent created.")

    # --- 4. Initialize Brain and Train ---
    # User only needs to pass their agent to Brain
    try:
        brain = Brain(
            agent=my_agent,
            reward_func=reward_exact_match,
            learning_algorithm="GRPO",
            rl_config={"lr": 1e-5} # RL configuration
        )
        
        brain.train(training_dataset)
        
        # Get the trained agent
        trained_agent = brain.get_agent()
        print("\n🤖 Agent after training is ready to use.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()