"""
ToolBrain Training Example - Use RL of ToolBrain for HPO.
This script now supports two modes:
- 'train': Runs the full RL training pipeline.
- 'demo': Runs the untrained agent multiple times, collects the results,
          and generates a plot for the demo video.

python examples/02_run_hpo_training/run_hpo_training.py --mode demo

python examples/02_run_hpo_training/run_hpo_training.py --mode train --output_dir ./models/my_trained_hpo_agent
"""

import os
import sys
from typing import Any, List, Dict
import argparse
import re

from peft import LoraConfig

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from smolagents import CodeAgent, TransformersModel, tool
from toolbrain import Brain, Trace, SmolAgentAdapter 
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


def generate_plot(results: List[Dict[str, float]], filename: str):
    """Generates and saves a scatter plot of HPO results."""
    if not results:
        print("No results to plot.")
        return

    x_vals = [res['feature_fraction'] for res in results]
    y_vals = [res['accuracy'] for res in results]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, alpha=0.7, edgecolors='w', s=100)
    
    plt.title("Baseline Agent Performance: Random Search", fontsize=16)
    plt.xlabel("Feature Fraction", fontsize=12)
    plt.ylabel("Model Accuracy", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(min(y_vals) - 0.01, max(y_vals) + 0.01) # Dynamic Y-axis
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(filename)
    print(f"✅ Plot saved to '{filename}'")
    plt.close()

# --- MODIFIED: Function for Demo Mode now collects and returns data ---
def run_untrained_demo(agent: CodeAgent, num_runs: int = 20) -> List[Dict[str, float]]:
    """
    Runs the untrained agent multiple times, collects results, and returns them.
    """
    print("🤖 Running in DEMO mode. Collecting data for baseline plot...")
    print("=" * 60)
    
    query = "Find the best value for feature_fraction (between 0.0 and 1.0) using the run_lightgbm tool."
    adapter = SmolAgentAdapter(agent=agent, config={})
    results = []
    
    for i in range(num_runs):
        print(f"\n--- DEMO RUN {i+1}/{num_runs} ---")
        try:
            trace, _, _ = adapter.run(query)
            
            feature_fraction = None
            accuracy = 0.0
            
            if trace and len(trace) > 0:
                turn = trace[0]
                # Extract feature_fraction from the generated code
                tool_code = turn.get("parsed_completion", {}).get("tool_code", "")
                match = re.search(r"[-+]?\d*\.\d+", tool_code)
                if match:
                    feature_fraction = float(match.group(0))
                
                # Extract accuracy from the structured action_output
                action_output = turn.get("action_output")
                if isinstance(action_output, dict) and "Accuracy" in action_output:
                    accuracy = float(action_output["Accuracy"])
            
            if feature_fraction is not None:
                results.append({"feature_fraction": feature_fraction, "accuracy": accuracy})
                print(f"    => Tried: {feature_fraction:.4f}, Got Accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"An error occurred during demo run: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Demo data collection completed.")
    return results


def main(args):
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

    # --- Mode Selection ---
    if args.mode == 'demo':
        # Run the demo, get the results, and generate the plot
        demo_results = run_untrained_demo(my_agent, num_runs=20)
        
        # Create an output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, "baseline_hpo_plot.png")
        
        generate_plot(demo_results, plot_filename)
        return

    # --- The rest of the code is for 'train' mode ---
    print("\n🧠 Starting TRAINING mode...")
    print(f"   - Output directory for trained model: {args.output_dir}")

    # User only needs to pass their agent to Brain
    brain = Brain(
        agent=my_agent,
        reward_func=reward_accuracy,
        learning_algorithm="GRPO",
        config={
            "epsilon": 0.2,  # clipping parameter
            "beta": 0.04,  # KL divergence penalty coefficient
            "opt_steps": 3,  # Number of GRPO optimization steps per batch
            "lr": 1e-5,  # Learning rate for optimizer
            "max_grad_norm": 1.0,
            "chunk_len": 128,  # If not None, get_per_token_logps will process in chunks
            "num_group_members": 5,  # The number of group members used for GRPO training steps
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

    brain.train(training_dataset, num_iterations=5)

    print(f"\n💾 Saving fine-tuned model to '{args.output_dir}'...")

    os.makedirs(args.output_dir, exist_ok=True)


    # Get the trained agent
    trained_agent = brain.get_agent()

    trained_agent.model.model.save_pretrained(args.output_dir)
    trained_agent.model.tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Model successfully saved to '{args.output_dir}'.")
    print("\n🤖 Agent after training is ready to use and saved for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ToolBrain HPO example.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "demo"],
        help="The mode to run the script in."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/hpo_agent_v1", 
        help="The directory where the fine-tuned model will be saved (used in 'train' mode)."
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train' and args.output_dir is None:
        parser.error("--output_dir is required when running in 'train' mode.")

    main(args)