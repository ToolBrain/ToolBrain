"""
ToolBrain Training Example - Simplified API

This script demonstrates the new, simplified ToolBrain API.
1. Define a configuration dictionary.
2. Pass the config to the Brain.
3. Call brain.train().
"""
import os
import sys
from dotenv import load_dotenv

from peft import LoraConfig

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import CodeAgent, TransformersModel, tool, ChatMessage, MessageRole
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match, reward_llm_judge_via_ranking

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

    print("🤖 User is creating their own agent...")
   
    print("📥 Initializing TransformersModel...")
    trainable_model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    print("✅ TransformersModel initialized.")
    
    # Get the tokenizer from inside the model
    tokenizer = trainable_model.tokenizer
    
    # If the tokenizer does not yet have a chat template, set a default one
    if tokenizer.chat_template is None:
        print("🔧 Setting a default chat template for the tokenizer.")
        # This is a very common and safe template
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}"

    print("🔧 Creating CodeAgent...")
    my_agent = CodeAgent(
        tools=[add, multiply],
        model=trainable_model
    )
    print("✅ Agent created.")

    # --- 4. Initialize Brain and Train ---
    # User only needs to pass their agent to Brain
    brain = Brain(
        agent=my_agent,
        reward_func=reward_exact_match,
        learning_algorithm="GRPO",
        config={
            "epsilon": 0.2, # clipping parameter
            "beta": 0.04, # KL divergence penalty coefficient
            "opt_steps": 3, # Number of GRPO optimization steps per batch
            "lr": 1e-5, # Learning rate for optimizer
            "max_grad_norm": 1.0,
            "chunk_len": 128, # If not None, get_per_token_logps will process in chunks
            "num_group_members": 2,  # The number of group members used for GRPO training steps
            "lora_config": LoraConfig( # If set, the RL will perform LoRA finetuning instead of full fine-tuning
                                    r=8,  # LoRA rank
                                    lora_alpha=16,  # scaling
                                    target_modules=["q_proj", "v_proj"],  # the modules to apply LoRA
                                    lora_dropout=0.1,
                                    bias="none",
                                    task_type="CAUSAL_LM",  # or "SEQ_2_SEQ_LM" for encoder-decoder
                                )
        }
    )

    brain.train(training_dataset)

    # Get the trained agent
    trained_agent = brain.get_agent()
    print("\n🤖 Agent after training is ready to use.")


# def main():
#     print("🧠 ToolBrain Training Example with LLM-as-a-Judge")
#     print("=" * 60)

#     print("🤖 Initializing agent to be trained...")
#     trainable_model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    
#     tokenizer = trainable_model.tokenizer
#     if tokenizer.chat_template is None:
#         print("🔧 Setting a default chat template for the tokenizer.")
#         tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}"

#     my_agent = CodeAgent(
#         tools=[add, multiply],
#         model=trainable_model
#     )
#     print("✅ Agent created.")

#     try:
#         brain = Brain(
#             agent=my_agent,
#             reward_func=reward_llm_judge_via_ranking, 
#             learning_algorithm="GRPO",
#             rl_config={"lr": 1e-5},
#             brain_config={"num_group_members": 3}
#         )
        
#         brain.train(
#             dataset=training_dataset,
#             judge_model="gemini/gemini-2.5-pro" 
#         )
        
#         trained_agent = brain.get_agent()
#         print("\n🤖 Agent after training is ready to use.")

#     except Exception as e:
#         print(f"\n❌ An error occurred: {e}")
#         import traceback
#         traceback.print_exc()

if __name__ == "__main__":
    main()