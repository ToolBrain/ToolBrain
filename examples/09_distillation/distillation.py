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

import os
import sys
from typing import Any

from peft import LoraConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smolagents import CodeAgent, TransformersModel, tool
from toolbrain import Brain, Trace
from transformers import BitsAndBytesConfig

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
        "query": "What is 8 multiplied by 7?",
        "gold_answer": "56"
    },
    {
        "query": "Calculate 2 to the power of 5",
        "gold_answer": "32.0"
    },
    {
        "query": "Add 100 and 200, then show the result",
        "gold_answer": "300"
    }
]

def reward_simple_exact_match(trace: Trace, **kwargs: Any) -> float:
    """
    Enhanced exact match reward function.
    
    Checks multiple sources for the correct answer:
    1. parsed_completion["final_answer"] (preferred)
    2. action_output (numeric values from final_answer calls)
    3. tool_output "Last output from code snippet" (fallback)
    
    Returns 1.0 for exact match, 0.0 otherwise.
    """
    gold_answer = kwargs.get("gold_answer")
    if gold_answer is None:
        return 0.0
    
    gold_str = str(gold_answer).strip()
    
    for turn in reversed(trace):
        # Method 1: Check parsed final_answer (most reliable)
        parsed = turn.get("parsed_completion", {})
        if parsed and parsed.get("final_answer"):
            final_answer = str(parsed["final_answer"]).strip()
            if final_answer == gold_str:
                return 1.0
        
        # Method 2: Check action_output (for successful final_answer calls)
        action_output = turn.get("action_output")
        if action_output is not None and str(action_output).strip() == gold_str:
            return 1.0
            
        # Method 3: Check tool_output for "Last output from code snippet"
        tool_output = turn.get("tool_output", "")
        if "Last output from code snippet:" in tool_output:
            # Extract the value after "Last output from code snippet:"
            import re
            match = re.search(r"Last output from code snippet:\n(.+?)(?:\n|$)", tool_output)
            if match:
                output_value = match.group(1).strip()
                if output_value == gold_str:
                    return 1.0
    
    return 0.0

def main():
    print("🎓 ToolBrain Distillation Example")
    print("=" * 60)
    print("This example demonstrates knowledge distillation:")
    print("• Teacher: Large, capable model (Qwen2.5-7B-Instruct)")
    print("• Student: Small, efficient model (Qwen2.5-0.5B-Instruct)")
    print("• Process: Teacher uses same tools → High-quality traces → Student learning")
    print("=" * 60)

    print("\n Creating student agent (small model)...")

    print(" Initializing student TransformersModel...")
    
    # Use a small model as the student
    use_bitsandbytes = False
    if use_bitsandbytes:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )
        student_model = TransformersModel(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            model_kwargs={"quantization_config": nf4_config},
        )
    else:
        student_model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    
    print("✅ Student TransformersModel initialized.")

    # Get the tokenizer from inside the model
    tokenizer = student_model.tokenizer

    # If the tokenizer does not yet have a chat template, set a default one
    if tokenizer.chat_template is None:
        print(" Setting a default chat template for the tokenizer.")
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}"

    print(" Creating student CodeAgent...")
    student_agent = CodeAgent(
        tools=[add, multiply, calculate_power], 
        model=student_model, 
        max_steps=3  # Allow more steps for proper final_answer calls
    )
    print("✅ Student agent created.")

    # --- 3. Initialize Brain ---
    print("\n Initializing Brain for the student agent...")
    
    brain = Brain(
        agent=student_agent,
        reward_func=reward_simple_exact_match,
        learning_algorithm="GRPO",
        config={
            "epsilon": 0.2,  # clipping parameter
            "beta": 0.04,  # KL divergence penalty coefficient
            "opt_steps": 3,  # Number of GRPO optimization steps per batch
            "lr": 1e-5,  # Learning rate for optimizer
            "max_grad_norm": 1.0,
            "chunk_len": 128,  # Token processing chunk size
            "num_group_members": 2,  # Group members for GRPO
            "batch_size": 4,  # Batch size for distillation
            "use_bitsandbytes": use_bitsandbytes,
            "lora_config": LoraConfig(  # LoRA for memory-efficient training
                r=8,  # LoRA rank
                lora_alpha=16,  # scaling
                target_modules=["q_proj", "v_proj"],  # modules to apply LoRA
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        },
    )
    print("✅ Brain initialized.")

    # --- 4. Set Dataset and Run Distillation ---
    print("\n" + "=" * 60)
    print(" PHASE 1: KNOWLEDGE DISTILLATION")
    print("=" * 60)
    print("The student will now learn from a larger teacher model...")
    
    # Distill knowledge from teacher to student
    # Teacher automatically uses the same tools as the student
    brain.distill(
        training_dataset=training_dataset, 
        teacher_model_id="Qwen/Qwen2.5-7B-Instruct"
    )

    # --- 5. Continue with Regular Training ---
    print("\n" + "=" * 60)
    print(" PHASE 2: REINFORCEMENT LEARNING TRAINING")
    print("=" * 60)
    print("Now continuing with regular RL training on the pre-trained student...")
    
    brain.train(training_dataset, num_iterations=2)

    # --- 6. Get the Trained Agent ---
    trained_agent = brain.get_agent()
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE!")

if __name__ == "__main__":
    main()


