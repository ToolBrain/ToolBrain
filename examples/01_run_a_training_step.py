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
            rl_config={
                "epsilon": 0.2, # clipping parameter
                "beta": 0.04, # KL divergence penalty coefficient
                "opt_steps": 3, # Number of GRPO optimization steps per batch
                "lr": 1e-5, # Learning rate for optimizer
                "max_grad_norm" :1.0, 
                "chunk_len": 128, # If not None, get_per_token_logps will process in chunks

            },
            brain_config={
                "num_group_members": 2,
            }
        )
        
        brain.train(training_dataset)
        
        # Get the trained agent
        trained_agent = brain.get_agent()
        print("\n🤖 Agent after training is ready to use.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()