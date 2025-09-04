"""
Step 2: Run Training Experiments for the ART-E Email Agent.

This is the main script to conduct the training experiments. It allows for
training the email agent using different RL algorithms (GRPO, DPO)
and THREE different reward functions to compare their effectiveness:
1.  'f1': A simple, metric-based F1 score.
2.  'art_judge': A direct-assessment LLM judge mimicking the ART-E project's logic.
3.  'toolbrain_judge': The native ranking-based LLM judge from the ToolBrain library.

How to run from the command line (from the project root TOOLBRAIN/):
1. Train with GRPO and F1 score reward:
   python -m examples.08_email_agent_art_e_use_case.2_run_training_experiments \
       --algorithm GRPO --reward_function f1 --output_dir ./models/art_e_grpo_f1

2. Train with DPO and ART-E's judge style:
   python -m examples.08_email_agent_art_e_use_case.2_run_training_experiments \
       --algorithm DPO --reward_function art_judge --output_dir ./models/art_e_dpo_art_judge

3. Train with GRPO and ToolBrain's ranking judge:
   python -m examples.08_email_agent_art_e_use_case.2_run_training_experiments \
       --algorithm GRPO --reward_function toolbrain_judge --output_dir ./models/art_e_grpo_toolbrain_judge
"""

import unsloth
from toolbrain.models import UnslothModel

import argparse
import os
import logging
from datasets import load_dataset

# from . import config
# from . import custom_rewards
# from . import email_tools
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))
import config
import custom_rewards
import email_tools

from toolbrain import Brain
from toolbrain import rewards as core_rewards
from smolagents import CodeAgent, TransformersModel
from transformers import BitsAndBytesConfig
import torch
import json
from .5_run_evaluation_art_style import run_evaluation as run_evaluation_art_style
from .3_evaluate_agents import run_evaluation_toolbrain_style, load_validation_dataset_for_eval as load_validation_dataset


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s"
)

def load_and_prepare_dataset():
    """
    Downloads the questions dataset, splits it into train/test sets,
    and prepares the training portion for the Brain's loop.
    """
    logging.info(f"Loading questions dataset from '{config.QUESTIONS_DATASET_ID}'...")

    full_dataset = load_dataset(
        config.QUESTIONS_DATASET_ID, split=config.QUESTIONS_DATASET_SPLIT
    )

    initial_size = len(full_dataset)
    full_dataset = full_dataset.filter(lambda x: x["how_realistic"] > 0.7)
    logging.info(
        f"Filtered dataset from {initial_size} to {len(full_dataset)} samples."
    )

    # Split the dataset into training and testing sets using the configured ratio.
    # Using a fixed seed ensures the split is the same every time.
    split_dataset = full_dataset.train_test_split(
        test_size=1.0 - config.TRAIN_TEST_SPLIT_RATIO, seed=config.DATASET_SPLIT_SEED
    )
    train_dataset = split_dataset["train"]
    logging.info(
        f"Split dataset into {len(train_dataset)} training samples and {len(split_dataset['test'])} test samples."
    )

    # Optional: Override for quick development runs
    if config.MAX_TRAIN_SAMPLES is not None and config.MAX_TRAIN_SAMPLES < len(
        train_dataset
    ):
        train_dataset = train_dataset.select(range(config.MAX_TRAIN_SAMPLES))
        logging.warning(
            f"OVERRIDE: Limiting training to the first {config.MAX_TRAIN_SAMPLES} samples for development."
        )

    formatted_data = []
    for item in train_dataset:
        system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(
            max_turns=config.MAX_AGENT_TURNS,
            inbox_address=item['inbox_address'],
            query_date=item['query_date']
        )
        full_query = f"{system_prompt}\nUser question: {item['question']}"

        formatted_data.append({
            "query": full_query,
            "gold_answer": item["answer"],
            "original_question": item["question"],
        })

    logging.info(
        f"Dataset prepared with {len(formatted_data)} samples using the new detailed prompt."
    )
    return formatted_data


def initialize_agent():
    """
    Initializes the TransformersModel and the CodeAgent with the email tools.
    """
    logging.info(f"Initializing base model: '{config.BASE_MODEL_ID}'")

    trainable_model = UnslothModel(
        model_id=config.BASE_MODEL_ID,
        max_new_tokens=config.MAX_NEW_TOKENS,
        max_seq_length=config.MAX_TOOL_OUTPUT_CHARS + config.MAX_NEW_TOKENS,
    )

    logging.info("Initializing CodeAgent with email tools...")

    agent = CodeAgent(
        tools=[email_tools.search_emails, email_tools.read_email],
        model=trainable_model,
        max_steps=config.MAX_AGENT_TURNS,
    )   
    return agent


def main(args):
    """Main function to orchestrate the training experiment."""

    logging.info("--- Starting Email Agent Training Experiment ---")
    logging.info(f"Algorithm: {args.algorithm}")
    logging.info(f"Reward Function: {args.reward_function}")
    logging.info(f"Output Directory: {args.output_dir}")

    training_data = load_and_prepare_dataset()

    agent = initialize_agent()

    # --- Select Reward Function and Configuration ---
    if args.reward_function == "f1":
        reward_func = custom_rewards.reward_f1_score
        logging.info("Using F1 score as the reward function.")

    elif args.reward_function == "art_judge":
        reward_func = custom_rewards.reward_art_style_judge
        for item in training_data:
            # The ART-style judge needs the original question for its prompt
            item["query"] = item["original_question"]
            item["judge_model"] = config.JUDGE_MODEL_ID
        logging.info(
            f"Using ART-E style direct-assessment judge ('{config.JUDGE_MODEL_ID}')."
        )

    # --- NEW OPTION: Use ToolBrain's native ranking judge ---
    elif args.reward_function == "toolbrain_judge":
        reward_func = core_rewards.reward_llm_judge_via_ranking
        for item in training_data:
            # The ranking judge also needs the original question and judge model
            item["query"] = item["original_question"]
            item["judge_model"] = config.JUDGE_MODEL_ID
        logging.info(
            f"Using ToolBrain's native ranking-based judge ('{config.JUDGE_MODEL_ID}')."
        )

    else:
        raise ValueError(f"Invalid reward function: {args.reward_function}")


    # Load validation data once at the beginning
    logging.info("Loading validation dataset for periodic evaluation...")
    validation_dataset = load_validation_dataset()
    validation_history_art = []
    validation_history_toolbrain = []

    # --- Select Algorithm and Run Training ---
    brain = None
    if args.algorithm == "GRPO":
        logging.info("Configuring Brain for GRPO training...")
        training_config = config.GRPO_CONFIG
        brain = Brain(
            agent=agent,
            reward_func=reward_func,
            config=training_config,
            learning_algorithm="GRPO",
        )
    elif args.algorithm == "DPO":
        logging.info("Configuring Brain for DPO training...")
        training_config = config.DPO_CONFIG
        brain = Brain(
            agent=agent,
            reward_func=reward_func,
            config=training_config,
            learning_algorithm="DPO",
        )
    else:
        raise ValueError(f"Invalid algorithm: {args.algorithm}")

    # --- UPDATED TRAINING LOOP WITH PERIODIC VALIDATION ---
    logging.info(f"Starting training for {config.NUM_TRAIN_EPOCHS} epoch(s)...")
    
    training_step_counter = 0
    
    # Run initial evaluation at step 0 for a baseline
    logging.info("--- Running initial validation at step 0 ---")
    # Run evaluation using ART-E's style
    metrics_art = run_evaluation_art_style(brain.get_agent(), validation_dataset, args.output_dir)
    validation_history_art.append({'step': 0, **metrics_art})
    print(f"Validation (ART-Style) at step 0: {metrics_art}")

    # Run evaluation using ToolBrain's style
    metrics_toolbrain = run_evaluation_toolbrain_style(brain.get_agent(), validation_dataset, args.output_dir)
    validation_history_toolbrain.append({'step': 0, **metrics_toolbrain})
    print(f"Validation (ToolBrain-Style) at step 0: {metrics_toolbrain}")

    for i in range(config.NUM_TRAIN_EPOCHS):
        for example in tqdm(training_data, desc=f"Epoch {i+1}"):
            brain.train_step(query=example["query"], reward_kwargs=example)
            training_step_counter += 1

            if training_step_counter % config.VALIDATION_INTERVAL == 0:
                logging.info(f"--- Running validation at step {training_step_counter} ---")
                current_agent = brain.get_agent()
                
                # Run both evaluation methods
                metrics_art = run_evaluation_art_style(current_agent, validation_dataset, args.output_dir)
                validation_history_art.append({'step': training_step_counter, **metrics_art})
                print(f"Validation (ART-Style) at step {training_step_counter}: {metrics_art}")

                metrics_toolbrain = run_evaluation_toolbrain_style(current_agent, validation_dataset, args.output_dir)
                validation_history_toolbrain.append({'step': training_step_counter, **metrics_toolbrain})
                print(f"Validation (ToolBrain-Style) at step {training_step_counter}: {metrics_toolbrain}")

    logging.info("--- Training finished! ---")

    # Save both evaluation histories
    path_art = os.path.join(args.output_dir, "validation_history_art.json")
    with open(path_art, 'w') as f:
        json.dump(validation_history_art, f, indent=4)

    path_toolbrain = os.path.join(args.output_dir, "validation_history_toolbrain.json")
    with open(path_toolbrain, 'w') as f:
        json.dump(validation_history_toolbrain, f, indent=4)

    logging.info(f"Saving fine-tuned model to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)

    trained_agent = brain.get_agent()

    trained_agent.model.model.save_pretrained(args.output_dir)
    trained_agent.model.tokenizer.save_pretrained(args.output_dir)

    logging.info(f"Model successfully saved. You can now use it for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training experiments for the ToolBrain Email Agent."
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["GRPO", "DPO"],
        help="The reinforcement learning algorithm to use for training.",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        required=True,
        # UPDATED CHOICES
        choices=["f1", "art_judge", "toolbrain_judge"],
        help="The reward function to use: 'f1' (F1 score), 'art_judge' (ART-E's direct assessment style), 'toolbrain_judge' (ToolBrain's ranking style).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the fine-tuned model adapters will be saved.",
    )

    args = parser.parse_args()
    main(args)
