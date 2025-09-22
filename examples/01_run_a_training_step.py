"""
ToolBrain Training Example

This script demonstrates the new, ultra-simplified ToolBrain API:
1. Create agent with create_agent() or manually
2. Create brain with Brain() constructor (all parameters as keywords)
3. Train with explicit, self-documenting parameters

"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import tool
from toolbrain import create_agent, Brain
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
    print("🧠 ToolBrain Training Example")
    print("=" * 60)
   
    # 1. Create agent 
    agent = create_agent(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tools=[add, multiply]  
        # use_unsloth=True (optional, default False)
    )
    
    print("✅ Agent created.")

    # 2. Create Brain 

    # === TYPICAL USAGE: Only 2 key parameters ===
    brain = Brain(
        agent,                          # Agent instance  
        algorithm="GRPO"                # Algorithm choice
        # learning_rate=3e-5 (default)
        # epsilon=0.2 (default)  
        # num_group_members=10 (default)
        # batch_size=1 (default)
        # max_grad_norm=1.0 (default)
        # use_bitsandbytes=False (default)
        # reward_func=None -> exact_match (default)
        # enable_tool_retrieval=False (default)
    )

    # User only needs to specify what they want to change

    # === DEMO: Override just 1-2 key parameters ===
    brain_custom = Brain(
        agent,
        algorithm="GRPO",
        learning_rate=1e-5,             # Override learning rate
        num_group_members=2             # And group size for faster
    )

    # 3. Demo other algorithms

    # DPO example - only override algorithm-specific parameters
    dpo_brain = Brain(
        agent,
        algorithm="DPO",                # Different algorithm
        beta=0.04                       # Only override DPO-specific parameter
        # All others use defaults: learning_rate=3e-5, num_group_members=10, etc.
    )
    
    # Supervised example
    supervised_brain = Brain(
        agent,
        algorithm="Supervised",         # Supervised learning
        learning_rate=5e-5
    )

    # === DEMO: Tool Retrieval Feature ===
    retrieval_brain = Brain(
        agent,
        algorithm="GRPO",
        learning_rate=1e-5,
        num_group_members=2,
        enable_tool_retrieval=True,      # Enable intelligent tool filtering
        retrieval_topic="mathematics",   # Domain for tool selection
        retrieval_guidelines="Select only tools needed for mathematical calculations"
    )

    # 4. Train with the GRPO brain (using brain_custom for faster)
    print("\n🚀 Starting training...")
    
    brain_custom.train(training_dataset, num_iterations=1)
    
    # 5. Get trained agent
    print("\n🎉 Training completed!")
    trained_agent = brain_custom.get_agent()
    print("✅ Trained agent is ready to use!")

    # === DEMO: Save/Load Functionality ===
    print("\n💾 Save/Load Demo")
    print("-" * 30)
    
    # Save the trained model
    model_save_path = "./saved_math_model"
    print(f"Saving model to: {model_save_path}")
    brain_custom.save(model_save_path)
    
    # Load the saved model back
    print(f"Loading model from: {model_save_path}")
    loaded_agent = Brain.load_agent(
        model_dir=model_save_path,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",  # Same as original
        tools=[add, multiply]  # Same tools as original
    )
    
    print("✅ Model saved and loaded successfully!")
    
    # === DEMO: Continue Training ===
    print("\n🔄 Continue Training Demo")  
    print("-" * 30)
    
    # Load model and continue training
    brain_continued = Brain.load_and_continue_training(
        model_dir=model_save_path,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tools=[add, multiply],
        algorithm="GRPO",
        learning_rate=1e-6  # Lower learning rate for fine-tuning
    )
    
    # Additional training data
    more_training_data = [
        {
            "query": "Calculate 15 plus 25", 
            "gold_answer": "40"
        },
        {
            "query": "What's 9 times 7?",
            "gold_answer": "63" 
        }
    ]
    
    print("Training more steps...")
    brain_continued.train(more_training_data, num_iterations=1)
    print("✅ Continued training completed!")

if __name__ == "__main__":
    main()