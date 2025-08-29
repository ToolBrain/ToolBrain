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
#CUDA_VISIBLE_DEVICES="2" python -m examples.08_email_agent_art_e_use_case.2_run_training_experiments        --algorithm GRPO --reward_function toolbrain_judge --output_dir ./models/art_e_grpo_toolbrain_judge_qwen14b_qlora; CUDA_VISIBLE_DEVICES="3" python -m examples.08_email_agent_art_e_use_case.2_run_training_experiments --algorithm DPO --reward_function art_judge --output_dir ./models/art_e_dpo_art_judge; CUDA_VISIBLE_DEVICES="2" python -m examples.08_email_agent_art_e_use_case.2_run_training_experiments --algorithm GRPO --reward_function f1 --output_dir ./models/art_e_grpo_f1

import argparse
import os
import logging
from datasets import load_dataset

from . import config
from . import custom_rewards
from . import email_tools

from toolbrain import Brain
from toolbrain import rewards as core_rewards
from toolbrain.models import UnslothModel 
from smolagents import CodeAgent, TransformersModel
from transformers import BitsAndBytesConfig
import torch


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s"
)

# This prompt acts as a "manual" for the agent.
SYSTEM_PROMPT_TEMPLATE = """You are a highly capable email search agent. Your goal is to answer the user's question by searching their email inbox.

**User Context:**
- User's email address: {inbox_address}
- Today's date: {query_date}

**Available Tools:**
You have two tools to help you:
1. `search_emails(keywords: List[str], ...)`: Searches for emails. It returns a LIST of search results. Each result is a DICTIONARY containing a 'message_id' and a 'snippet'.
2. `read_email(message_id: str)`: Reads the full content of a single email using its 'message_id'.

**CRITICAL WORKFLOW:**
1.  Start by using `search_emails` with relevant keywords from the user's question.
2.  Examine the `snippet` from the search results to see which email is most promising.
3.  **You MUST extract the 'message_id' string from the chosen search result dictionary.**
4.  Use this `message_id` string as the input for the `read_email` tool.
5.  After reading the email, analyze its content to find the final answer.
6.  When you have the answer, state it clearly using "Final Answer:".

**Example:**
search_result = search_emails(keywords=['Shari', 'Portland'])
# search_result might look like: [{'message_id': '<123@...>', 'snippet': '...move to Portland...'}]
message_id_to_read = search_result[0]['message_id']
email_content = read_email(message_id=message_id_to_read)
# ... analyze email_content ...
# Final Answer: The move is targeted for the end of February.

Now, begin.
User question: {question}
"""

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
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            inbox_address=item['inbox_address'],
            query_date=item['query_date'],
            question=item['question']
        )
        formatted_data.append(
            {
                "query": prompt,
                "gold_answer": item["answer"],
                "original_question": item["question"],
            }
        )

    logging.info(f"Dataset prepared with {len(formatted_data)} samples using the new detailed prompt.")
    return formatted_data


def initialize_agent():
    """
    Initializes the TransformersModel and the CodeAgent with the email tools.
    """
    logging.info(f"Initializing base model: '{config.BASE_MODEL_ID}'")
    # trainable_model = TransformersModel(model_id=config.BASE_MODEL_ID)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    trainable_model = UnslothModel(
        model_id=config.BASE_MODEL_ID,
        model_kwargs={"quantization_config": nf4_config, 'attn_implementation': "flash_attention_2", },
        max_new_tokens=512,
        max_seq_length=config.MAX_TOOL_OUTPUT_CHARS + 2048
    )

    logging.info("Initializing CodeAgent with email tools...")
    system_prompt = config.NEW_SYSTEM_PROMPT

    agent = CodeAgent(
        tools=[email_tools.search_emails, email_tools.read_email],
        model=trainable_model,
        max_steps=10,
        planning_interval=None,
    )
    agent.prompt_templates["system_prompt"] = system_prompt
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

    logging.info(f"Starting training for {config.NUM_TRAIN_EPOCHS} epoch(s)...")
    # We pass the full prompt to the agent via the 'query' key
    # and the original question to the judge via the same 'query' key
    # The reward_kwargs will correctly pass all necessary info to the reward function.
    brain.train(dataset=training_data, num_iterations=config.NUM_TRAIN_EPOCHS)
    logging.info("--- Training finished! ---")

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
