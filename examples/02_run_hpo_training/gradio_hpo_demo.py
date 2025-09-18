"""
ToolBrain HPO Live Agent Demonstration with Gradio.

This script launches a Gradio web interface to provide a live, interactive
demonstration of a fine-tuned HPO agent that optimizes LightGBM hyperparameters.

How to run from the command line (from the project root TOOLBRAIN/):
# First, ensure you have a trained model saved in a directory.
# Then, run the demo:
python -m examples.02_run_hpo_training.gradio_hpo_demo \
    --model_dir ./models/my_trained_hpo_agent
"""

import argparse
import logging
import gradio as gr
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from smolagents import CodeAgent, TransformersModel, stream_to_gradio
from toolbrain.models import UnslothModel 
from peft import PeftModel
from lightgbm_model import run_lightgbm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# --- Global variable to hold the loaded agent ---
AGENT = None

# Configuration constants
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 512
MAX_AGENT_TURNS = 5

def load_trained_agent(model_dir: str) -> CodeAgent:
    """
    Loads a fine-tuned agent from a specified directory, using the `load_adapter` method.
    """
    logging.info(f"Loading fine-tuned agent from: {model_dir} using Unsloth.")

    # base_model = TransformersModel(
    #     model_id=config.BASE_MODEL_ID,
    #     max_seq_length=config.MAX_TOOL_OUTPUT_CHARS + config.MAX_NEW_TOKENS,
    #     max_new_tokens=config.MAX_NEW_TOKENS,
    # )

    base_model = UnslothModel(
        model_id=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    base_model.model.load_adapter(model_dir)

    agent = CodeAgent(
        tools=[run_lightgbm],
        model=base_model,
        max_steps=MAX_AGENT_TURNS,
    )
    
    logging.info("HPO Agent loaded successfully.")
    return agent
    

def chat_interface(message, history):
    """
    This is the core function that Gradio calls for each user interaction.
    It runs the agent with the user's message and streams the output.
    """
    if AGENT is None:
        yield "Error: Agent has not been loaded. Please check the model directory and restart."
        return

    logging.info(f"Received user query: '{message}'")
    
    # Create a focused prompt for HPO tasks
    full_prompt = f"""You are a hyperparameter optimization (HPO) agent. You have access to a LightGBM training tool that takes a feature_fraction parameter.

Your task is to help optimize machine learning model performance by selecting appropriate hyperparameters. You can use the run_lightgbm tool to train models with different feature_fraction values (must be between 0.0 and 1.0) and observe the accuracy results.

When asked to optimize or find the best parameters, try different values systematically and recommend the best performing configuration.

User question: {message}
"""
    
    try:
        yield from stream_to_gradio(AGENT.run(full_prompt, stream=True))
    except Exception as e:
        logging.error(f"An error occurred during agent execution: {e}")
        yield f"Sorry, an error occurred: {e}"

def simple_hpo_interface(feature_fraction):
    """
    Simple interface for direct HPO testing without the agent.
    """
    try:
        if not (0.0 <= feature_fraction <= 1.0):
            return "Error: feature_fraction must be between 0.0 and 1.0"
        
        result = run_lightgbm(feature_fraction)
        accuracy = result["Accuracy"]
        return f"Feature Fraction: {feature_fraction:.3f}\nAccuracy: {accuracy:.4f}"
    except Exception as e:
        return f"Error running LightGBM: {e}"

def main(args):
    """Launches the Gradio web interface."""
    global AGENT
    
    # Load agent if model directory is provided
    if args.model_dir:
        AGENT = load_trained_agent(args.model_dir)
        model_info = f"Model loaded from: {args.model_dir}"
    else:
        logging.warning("No model directory provided. Creating agent with base model.")
        base_model = TransformersModel(
            model_id=BASE_MODEL_ID,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        AGENT = CodeAgent(
            tools=[run_lightgbm],
            model=base_model,
            max_steps=MAX_AGENT_TURNS,
        )
        model_info = f"Using base model: {BASE_MODEL_ID}"

    # Create tabs for different interfaces
    with gr.Blocks(title="ToolBrain HPO Demo") as demo:
        gr.Markdown("# 🧠 ToolBrain - HPO Agent Demo")
        gr.Markdown(f"**{model_info}**")
        
        with gr.Tab("Chat with HPO Agent"):
            gr.Markdown("### Interact with the trained HPO agent")
            chat_interface_component = gr.ChatInterface(
                fn=chat_interface,
                title="HPO Agent Chat",
                description="Ask the agent to optimize LightGBM hyperparameters or find the best feature_fraction value.",
                examples=[
                    "Find the best feature_fraction value for LightGBM",
                    "Test feature_fraction=0.8 and tell me the accuracy",
                    "Compare different feature_fraction values and recommend the best one",
                    "Run hyperparameter optimization to maximize accuracy"
                ]
            )
        
        with gr.Tab("Direct HPO Testing"):
            gr.Markdown("### Test LightGBM directly with different feature_fraction values")
            with gr.Row():
                with gr.Column():
                    feature_fraction_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        label="Feature Fraction"
                    )
                    run_button = gr.Button("Run LightGBM", variant="primary")
                
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Results",
                        lines=5,
                        placeholder="Results will appear here..."
                    )
            
            run_button.click(
                fn=simple_hpo_interface,
                inputs=feature_fraction_slider,
                outputs=output_text
            )
            
            # Example buttons for quick testing
            gr.Markdown("### Quick Test Examples")
            with gr.Row():
                test_buttons = []
                test_values = [0.3, 0.5, 0.7, 0.9]
                for val in test_values:
                    btn = gr.Button(f"Test {val}")
                    btn.click(
                        fn=lambda x=val: simple_hpo_interface(x),
                        outputs=output_text
                    )

    # Launch the web server
    logging.info("Launching Gradio HPO demo... Access it at the URL provided below.")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for a trained ToolBrain HPO agent.")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="The directory where the fine-tuned model (LoRA adapters) is saved. If not provided, uses base model."
    )

    args = parser.parse_args()
    main(args)
