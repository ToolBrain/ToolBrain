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
    Loads a fine-tuned agent from a specified directory, using the `load_adapter` method.
    """
    logging.info(f"Loading fine-tuned agent from: {model_dir} using Unsloth.")

    # base_model = TransformersModel(
    #     model_id=config.BASE_MODEL_ID,
    #     max_seq_length=config.MAX_TOOL_OUTPUT_CHARS + config.MAX_NEW_TOKENS,
    #     max_new_tokens=config.MAX_NEW_TOKENS,
    # )

    base_model = UnslothModel(
        model_id=config.BASE_MODEL_ID,
        max_seq_length=config.MAX_TOOL_OUTPUT_CHARS + config.MAX_NEW_TOKENS,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )

    base_model.model.load_adapter(model_dir)
    
    agent = CodeAgent(
        tools=[email_tools.search_emails, email_tools.read_email],
        model=base_model,
        max_steps=config.MAX_AGENT_TURNS,
    )
    
    logging.info("Agent loaded successfully.")
    return agent

def chat_interface(message, history):
    """
    Core function for Gradio interaction. It wraps the user's message in the
    exact, consistent context the agent was trained on.
    """
    if AGENT is None:
        yield "Error: Agent not loaded."
        return

    logging.info(f"Received user query: '{message}'")
    
    # --- Use the exact context from the successful trace ---
    # This ensures maximum reliability for the demo.
    simulated_inbox = "gerald.nemec@enron.com"
    simulated_date = "2000-04-30"

    full_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(
        max_turns=config.MAX_AGENT_TURNS,
        inbox_address=simulated_inbox,
        query_date=simulated_date,
        question=message # The user's question from the Gradio interface
    )
    
    logging.info(f"Constructed full prompt for agent:\n{full_prompt}")
    
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