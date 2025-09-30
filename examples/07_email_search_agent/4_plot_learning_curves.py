"""
Step 4: Plot Learning Curves from Evaluation History.

This script reads one or more 'validation_history.json' files and plots the
learning curves for comparison. It's designed to be flexible, allowing for
comparison between different training runs (e.g., GRPO vs. DPO) or different
evaluation methodologies from a single run.

How to run from the command line (from the project root TOOLBRAIN/):

# Example 1: Compare two different training runs (GRPO vs DPO)
python -m examples.07_email_search_agent.4_plot_learning_curves \
    --results_files \
        ./models/run_grpo/validation_history_art.json \
        ./models/run_dpo/validation_history_art.json \
    --labels "GRPO Run" "DPO Run" \
    --output_image grpo_vs_dpo.png

"""

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(results_files: list[str], labels: list[str], output_image: str):
    """
    Reads validation history files, plots the learning curves for correctness rate,
    and saves the plot to a file.
    """
    if len(results_files) != len(labels):
        raise ValueError("The number of results files must match the number of labels.")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    all_data = []
    for i, file_path in enumerate(results_files):
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        
        with open(file_path, 'r') as f:
            history = json.load(f)
        
        if not history:
            print(f"Warning: File is empty, skipping: {file_path}")
            continue

        df = pd.DataFrame(history)
        df['label'] = labels[i]
        all_data.append(df)

    if not all_data:
        print("No valid data found to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    sns.lineplot(
        data=combined_df,
        x='step',
        y='correctness_rate',
        hue='label',
        marker='o',
        ax=ax
    )

    ax.set_title('Learning Curve Comparison: Correctness Rate vs. Training Steps', fontsize=16, weight='bold')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Validation Correctness Rate (%)', fontsize=12)
    ax.legend(title='Experiment / Evaluation Method', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"✅ Learning curve plot saved to '{output_image}'")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from training evaluation history.")
    
    parser.add_argument(
        "--results_files",
        type=str,
        nargs='+',
        required=True,
        help="Paths to validation_history.json files (e.g., from different runs or eval methods)."
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        required=True,
        help="Labels for the legend, one for each file."
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="learning_curve.png",
        help="The name of the output image file to save the plot."
    )

    args = parser.parse_args()
    plot_learning_curves(args.results_files, args.labels, args.output_image)