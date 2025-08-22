"""
ToolBrain Training Example - Use RL of ToolBrain for HPO. Running with DPO instead of GRPO

"""

import os
import sys
from typing import Any

from peft import LoraConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from smolagents import CodeAgent, TransformersModel, tool
from toolbrain import Brain, Trace
from lightgbm_model import run_lightgbm
from transformers import BitsAndBytesConfig


# --- 2. Prepare Training Data ---
training_dataset = [
    {
        "query": "Use the run_lightgbm tool with a value of feature_fraction (must be between 0.0 and 1.0)."
    }
]


def reward_accuracy(trace: Trace, **kwargs: Any) -> float:
    for turn in trace:
        try:
            reward = float(turn["action_output"]["Accuracy"])
            return reward
        except:
            reward = 0.0
        return reward


def main():
    print("🧠 ToolBrain Flexible Training Example")
    print("=" * 60)

    print("🤖 User is creating their own agent...")

    print("📥 Initializing TransformersModel...")

    use_bitsandbytes = False
    if use_bitsandbytes:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )
        trainable_model = TransformersModel(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            model_kwargs={"quantization_config": nf4_config},
        )
    else:
        trainable_model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    # trainable_model = TransformersModel(model_id="ibm-granite/granite-3.0-2b-instruct")

    print("✅ TransformersModel initialized.")

    # Get the tokenizer from inside the model
    tokenizer = trainable_model.tokenizer

    # If the tokenizer does not yet have a chat template, set a default one
    if tokenizer.chat_template is None:
        print("🔧 Setting a default chat template for the tokenizer.")
        # This is a very common and safe template
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}"

    print("🔧 Creating CodeAgent...")
    my_agent = CodeAgent(tools=[run_lightgbm], model=trainable_model, max_steps=1)
    print("✅ Agent created.")

    # User only needs to pass their agent to Brain
    brain = Brain(
        agent=my_agent,
        reward_func=reward_accuracy,
        learning_algorithm="DPO",
        config={
            "epsilon": 0.2,  # clipping parameter
            "beta": 0.1,  # KL divergence penalty coefficient
            "opt_steps": 3,  # Number of GRPO optimization steps per batch
            "lr": 1e-5,  # Learning rate for optimizer
            "max_grad_norm": 1.0,
            "chunk_len": 128,  # If not None, get_per_token_logps will process in chunks
            "num_group_members": 2,  # The number of group members used for DPO at least 2
            "use_bitsandbytes": use_bitsandbytes, # Whether to use bitandbytes for training
            "lora_config": LoraConfig(  # If set, the RL will perform LoRA finetuning instead of full fine-tuning
                r=8,  # LoRA rank
                lora_alpha=16,  # scaling
                target_modules=["q_proj", "v_proj"],  # the modules to apply LoRA
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",  # or "SEQ_2_SEQ_LM" for encoder-decoder
            ),
        },
    )

    brain.train(training_dataset, num_iterations=1)

    # Get the trained agent
    trained_agent = brain.get_agent()
    print("\n🤖 Agent after training is ready to use.")


if __name__ == "__main__":
    main()
