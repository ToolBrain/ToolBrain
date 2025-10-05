"""
ToolBrain Training Example with LangChain Agent

This example demonstrates how to train a LangChain agent using ToolBrain.
It uses HuggingFace models with custom tool calling for local training and fine-tuning.
"""

from toolbrain import Brain, create_huggingface_chat_model
from toolbrain.rewards import reward_exact_match


try:
    from langchain.agents import create_agent
    from langchain_core.tools import tool
except ImportError:
    print("Please install required packages: pip install 'langchain>=1.0.0a10' 'langgraph' 'langchain-huggingface' 'transformers'")
    exit()

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


def main():
    """Main training function."""
    
    # Training dataset
    training_dataset = [
        {"query": "What is 8 multiplied by 6?", "gold_answer": "48"},
    ]

    print("🧠 ToolBrain Training Example with LangChain Agent")
    print("=" * 60)

    # Initialize HuggingFace model
    print("📥 Initializing HuggingFace model...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Use helper function to create the model
    trainable_llm = create_huggingface_chat_model(
        model_id=model_id,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=True,
        device_map="cpu",
    )
    print("✅ HuggingFace model initialized with custom tool calling support.")

    # Create LangChain agent
    print("\n🔧 Creating LangChain agent...")

    langchain_agent = create_agent(
        model=trainable_llm,
        tools=[multiply],
        prompt="You are a helpful assistant. You must call a tool to answer the user.",
    )
    print("✅ LangChain agent created.")

    # Initialize Brain for training
    print("\n🧠 Initializing Brain for the LangChain agent...")
    brain = Brain(
        agent=langchain_agent,
        trainable_model=trainable_llm,
        algorithm="GRPO",
        reward_func=reward_exact_match
    )
    
    brain.train(training_dataset, num_iterations=5)

if __name__ == "__main__":
    main()