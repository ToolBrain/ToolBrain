"""
ToolBrain Distillation Example - Knowledge Transfer from Teacher to Student

This script demonstrates the new, simplified ToolBrain distillation API.
1. Create a student agent with a small model.
2. Use brain.distill() to transfer knowledge from a larger teacher model.
3. Continue with regular RL training on the pre-trained student.

The distillation process:
- Teacher model generates high-quality execution traces
- Student model learns from these traces via supervised learning
- Student is then ready for efficient RL training
"""


from smolagents import tool, TransformersModel, CodeAgent
from toolbrain import create_agent, Brain, reward_tool_execution_success

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

@tool
def calculate_power(base: float, exponent: int) -> float:
    """
    Calculate base raised to the power of exponent.

    Args:
        base (float): The base number.
        exponent (int): The exponent.

    Returns:
        float: Result of base^exponent.
    """
    return base ** exponent


# --- 2. Prepare Training Data ---
training_dataset = [
    {
        "query": "Use the add tool to calculate 15 + 27",
        "gold_answer": "42"
    },
    {
        "query": "What is 8 multiplied by 7? Use the multiply tool",
        "gold_answer": "56"
    },
    {
        "query": "Calculate 2 to the power of 5 using the calculate_power tool",
        "gold_answer": "32"
    },
    {
        "query": "Add 100 and 200 using the add tool",
        "gold_answer": "300"
    }
]


def main():
    print("🎓 ToolBrain Distillation Example")
    print("=" * 60)
    print("This example demonstrates knowledge distillation:")
    print("• Teacher: Large, capable model (Qwen2.5-7B-Instruct)")
    print("• Student: Small, efficient model (Qwen2.5-0.5B-Instruct)")
    print("• Process: Teacher uses same tools → High-quality traces → Student learning")
    print("• Reward: Tool execution success (tool works without errors)")
    print("=" * 60)

    # --- 1. Create Student Agent ---
    print("\nCreating student agent (small model)...")
    print("📥 Initializing TransformersModel...")
    trainable_model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B-Instruct",
                                        max_new_tokens=128)

    print("✅ TransformersModel initialized.")

    print("🔧 Creating CodeAgent...")
    student_agent = CodeAgent(
        tools=[add, multiply, calculate_power],
        model=trainable_model,
        max_steps=1
    )
    print("✅ Student agent created.")

    # --- 2. Initialize Brain ---
    print("\n🧠 Initializing Brain for the student agent...")

    brain = Brain(
        student_agent,                          # Agent instance
        algorithm="GRPO",               # Algorithm choice
        reward_func=reward_tool_execution_success  # Use tool success
    )
    print("✅ Brain initialized.")

    # --- 3. Set Dataset and Run Distillation ---
    print("\n" + "=" * 60)
    print("🎓 PHASE 1: KNOWLEDGE DISTILLATION")
    print("=" * 60)
    print("The student will now learn from a larger teacher model...")

    # Distill knowledge from teacher to student
    # Teacher automatically uses the same tools as the student
    brain.distill(
        dataset=training_dataset,
        teacher_model_id="Qwen/Qwen2.5-7B-Instruct"
    )

    # --- 4. Continue with Regular Training ---
    print("\n" + "=" * 60)
    print("PHASE 2: REINFORCEMENT LEARNING TRAINING")
    print("=" * 60)
    print("Now continuing with regular RL training on the pre-trained student...")

    brain.train(training_dataset, num_iterations=2)

    # --- 5. Get the Trained Agent ---
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")


if __name__ == "__main__":
    main()