"""
Step 3: Live Agent Demonstration with Gradio.

This script launches a Gradio web interface to provide a live, interactive
demonstration of a fine-tuned email agent.

How to run from the command line (from the project root TOOLBRAIN/):
# First, ensure you have a trained model saved in a directory.
# Then, run the demo:
python -m examples.07_email_search_agent.4_gradio_live_demo \
    --model_dir ./models/art_e_grpo_art_judge
"""

import argparse
import logging
import gradio as gr

from . import config
from . import email_tools
from smolagents import stream_to_gradio
from toolbrain import Brain

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# --- Global variable to hold the loaded agent ---
AGENT = None

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

    system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(
        max_turns=config.MAX_AGENT_TURNS,
        inbox_address=simulated_inbox,
        query_date=simulated_date,
    )
    full_prompt = f"{system_prompt}\nUser question: {message}"
    
    logging.info(f"Constructed full prompt for agent:\n{full_prompt}")
    
    try:
        yield from stream_to_gradio(
            agent=AGENT,
            task=full_prompt,
            reset_agent_memory=True
        )
    except Exception as e:
        logging.error(f"An error occurred during agent execution: {e}")
        yield f"Sorry, an error occurred: {e}"


def main(args):
    """Launches the Gradio web interface."""
    global AGENT
    # Load trained agent directly
    logging.info(f"Loading fine-tuned agent from: {args.model_dir}")
    AGENT = Brain.load_agent(
        model_dir=args.model_dir,
        base_model_id=config.BASE_MODEL_ID,
        tools=[email_tools.search_emails, email_tools.read_email],
        max_steps=config.MAX_AGENT_TURNS,
        use_unsloth=True
    )

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