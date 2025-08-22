"""
ToolBrain Training Example - Simplified API with supervised learning

This script demonstrates the new, simplified ToolBrain API.
1. Define a configuration dictionary.
2. Pass the config to the Brain.
3. Call brain.train().
"""
import os
import sys
from typing import List

from dotenv import load_dotenv

from peft import LoraConfig

from toolbrain.core_types import ChatSegment

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smolagents import CodeAgent, TransformersModel
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match


# for supervised learning each training example is a list of ChatSegments with text and role, assistant's text is
# considered as gold answers for supervised learning, the learning can be multi-turns
training_datasets: List[List[ChatSegment]] = [
        [
            ChatSegment(
                role="other",
                text="You are a Python assistant. Compute the sum of 1..10 and explain briefly.",
            ),
            ChatSegment(
                role="assistant",
                text="The sum of numbers from 1 to 10 is 55, using the formula n(n+1)/2."
            ),
        ],
        [
            ChatSegment(
                role="other",
                text="Translate 'Good morning' into French.",
            ),
            ChatSegment(
                role="assistant",
                text="In French, it is 'Bonjour'."
            ),
        ],
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
        tools=[],
        model=trainable_model
    )
    print("✅ Agent created.")

    # --- 4. Initialize Brain and Train ---
    # User only needs to pass their agent to Brain
    brain = Brain(
        agent=my_agent,
        reward_func=reward_exact_match,
        learning_algorithm="Supervised",
        config={
            "epsilon": 0.2,  # clipping parameter
            "beta": 0.04,  # KL divergence penalty coefficient
            "opt_steps": 3,  # Number of GRPO optimization steps per batch
            "lr": 1e-5,  # Learning rate for optimizer
            "max_grad_norm": 1.0,
            "chunk_len": 128,  # If not None, get_per_token_logps will process in chunks
            "lora_config": LoraConfig(  # If set, the RL will perform LoRA finetuning instead of full fine-tuning
                r=8,  # LoRA rank
                lora_alpha=16,  # scaling
                target_modules=["q_proj", "v_proj"],  # the modules to apply LoRA
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",  # or "SEQ_2_SEQ_LM" for encoder-decoder
            )
        }
    )

    brain.train(training_datasets, num_iterations=5)

    # Get the trained agent
    trained_agent = brain.get_agent()
    print("\n🤖 Agent after training is ready to use.")


if __name__ == "__main__":
    main()