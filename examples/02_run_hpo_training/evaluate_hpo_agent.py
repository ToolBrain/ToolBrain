# examples/02_run_hpo_training/evaluate_hpo_agent.py

"""
Evaluate a fine-tuned HPO agent.

This script loads a trained agent, runs it multiple times to see how it
explores the parameter space, and generates a plot of its performance.

Run `python examples/02_run_hpo_training/evaluate_hpo_agent.py --model_dir <model_directory>` to execute.
"""

import argparse
import logging
import os
import re
from typing import List, Dict

# Import the plotting function from the training script
from run_hpo_training import generate_plot, reward_accuracy
# Import the agent loading function from the demo script
from gradio_hpo_demo import load_trained_agent

from smolagents import CodeAgent
from toolbrain import SmolAgentAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def evaluate_agent(agent: CodeAgent, num_runs: int = 20) -> List[Dict[str, float]]:
    """
    Runs the TRAINED agent multiple times to collect its exploration data.
    """
    print(f"🤖 Evaluating trained agent for {num_runs} runs...")
    print("=" * 60)
    
    query = "Find the best value for feature_fraction (between 0.0 and 1.0) using the run_lightgbm tool."
    adapter = SmolAgentAdapter(agent=agent, config={})
    results = []
    
    for i in range(num_runs):
        print(f"\n--- EVALUATION RUN {i+1}/{num_runs} ---")
        try:
            trace, _, _ = adapter.run(query)
            
            feature_fraction = None
            accuracy = 0.0
            
            if trace and len(trace) > 0:
                turn = trace[0]
                tool_code = turn.get("parsed_completion", {}).get("tool_code", "")
                match = re.search(r"[-+]?\d*\.\d+", tool_code)
                if match:
                    feature_fraction = float(match.group(0))
                
                action_output = turn.get("action_output")
                if isinstance(action_output, dict) and "Accuracy" in action_output:
                    accuracy = float(action_output["Accuracy"])
            
            if feature_fraction is not None:
                results.append({"feature_fraction": feature_fraction, "accuracy": accuracy})
                print(f"    => Agent tried: {feature_fraction:.4f}, Got Accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"An error occurred during evaluation run: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation data collection completed.")
    return results

def main(args):
    """Main evaluation function."""
    agent_to_evaluate = load_trained_agent(args.model_dir)
    
    evaluation_results = evaluate_agent(agent_to_evaluate, num_runs=20)
    
    # Create an output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, "after_training_hpo_plot.png")
    
    # Generate and save the plot
    generate_plot(evaluation_results, plot_filename)
    
    # Find and print the best result
    if evaluation_results:
        best_run = max(evaluation_results, key=lambda x: x['accuracy'])
        print("\n--- Best Result Found ---")
        print(f"Optimal Feature Fraction: {best_run['feature_fraction']:.4f}")
        print(f"Highest Accuracy: {best_run['accuracy']:.4f}")
        print("="*27)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ToolBrain HPO Agent.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The directory where the fine-tuned model is saved."
    )
    args = parser.parse_args()
    main(args)