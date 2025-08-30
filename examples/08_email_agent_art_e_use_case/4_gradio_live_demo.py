"""
Step 4: Live Agent Demonstration with Gradio.

This script launches a Gradio web interface to provide a live, interactive
demonstration of a fine-tuned email agent.

How to run from the command line (from the project root TOOLBRAIN/):
# First, ensure you have a trained model saved in a directory.
# Then, run the demo:
python -m examples.08_email_agent_art_e_use_case.4_gradio_live_demo \
    --model_dir ./models/art_e_grpo_toolbrain_judge
"""

import argparse
import logging
import gradio as gr

from . import config
from . import email_tools
from smolagents import CodeAgent, TransformersModel, stream_to_gradio
from toolbrain.models import UnslothModel 
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# --- Global variable to hold the loaded agent ---
AGENT = None

def load_trained_agent(model_dir: str) -> CodeAgent:
    """
    Loads a fine-tuned agent from a specified directory, ensuring that Unsloth
    optimizations are applied for inference, consistent with training.
    """
    logging.info(f"Loading fine-tuned agent from: {model_dir} using Unsloth.")

    base_model = UnslothModel(
        model_id=config.BASE_MODEL_ID,
        max_seq_length=config.MAX_TOOL_OUTPUT_CHARS + config.MAX_NEW_TOKENS,
        max_new_tokens=config.MAX_NEW_TOKENS
    )

    peft_model = PeftModel.from_pretrained(base_model.model, model_dir)
    base_model.model = peft_model
    
    agent = CodeAgent(
        tools=[email_tools.search_emails, email_tools.read_email],
        model=base_model
    )
    logging.info("✅ Agent loaded successfully with Unsloth optimizations.")
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
    
    # For a live demo, we can use placeholder context.
    full_prompt = f"""You are an email search agent.
User's email address is jane.doe@enron.com
Today's date is 2001-09-01

User question: {message}
"""
    
    try:
        yield from stream_to_gradio(AGENT.run(full_prompt, stream=True))
    except Exception as e:
        logging.error(f"An error occurred during agent execution: {e}")
        yield f"Sorry, an error occurred: {e}"


def main(args):
    """Launches the Gradio web interface."""
    global AGENT
    AGENT = load_trained_agent(args.model_dir)

    # Create the Gradio interface
    demo = gr.ChatInterface(
        fn=chat_interface,
        title="🤖 ToolBrain - Live Email Agent Demo",
        description=f"An interactive demo of an agent fine-tuned on the Enron email dataset. Model loaded from: {args.model_dir}",
        examples=[
            "When is Shari's move to Portland targeted for?",
            "What is my confirmation number for my Continental Airlines flight to Colorado Springs?",
            "What issues will be discussed at the Tuesday afternoon meeting with EES?"
        ]
    )

    # Launch the web server
    logging.info("Launching Gradio demo... Access it at the URL provided below.")
    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for a trained ToolBrain agent.")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The directory where the fine-tuned model (LoRA adapters) is saved."
    )

    args = parser.parse_args()
    main(args)