"""
Step 3: Evaluate Fine-Tuned Email Agents.

This script takes a trained agent model (saved from the training script) and
evaluates its performance on the test set of the Enron questions dataset.

It calculates and reports three key metrics:
1.  Correctness: The percentage of questions answered correctly.
2.  Efficiency: The average number of turns the agent takes to answer.
3.  Anti-Hallucination: The fraction of attempted answers that are incorrect.

How to run from the command line (from the project root TOOLBRAIN/):
# Evaluate the model trained with GRPO and ToolBrain's judge
python -m examples.08_email_agent_art_e_use_case.3_evaluate_agents \
    --model_dir ./models/art_e_grpo_toolbrain_judge
"""

import argparse
import logging
import os
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel

from . import config
from . import email_tools
from smolagents import CodeAgent, TransformersModel
from toolbrain.adapters import SmolAgentAdapter

try:
    import litellem
except ImportError:
    litellm = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")


def load_test_dataset():
    """
    Loads the questions dataset and extracts the 'test' split consistent
    with the split used during training.
    """
    logging.info(f"Loading TEST questions dataset from '{config.QUESTIONS_DATASET_ID}'...")
    
    full_dataset = load_dataset(config.QUESTIONS_DATASET_ID, split=config.QUESTIONS_DATASET_SPLIT)
    full_dataset = full_dataset.filter(lambda x: x['how_realistic'] > 0.7)

    # Recreate the exact same split as in the training script by using the same seed.
    split_dataset = full_dataset.train_test_split(
        test_size=1.0 - config.TRAIN_TEST_SPLIT_RATIO, 
        seed=config.DATASET_SPLIT_SEED
    )
    test_dataset = split_dataset['test']
    logging.info(f"Using the consistent test split with {len(test_dataset)} samples for evaluation.")

    # Optional: Override for quick development runs
    if config.MAX_TEST_SAMPLES is not None and config.MAX_TEST_SAMPLES < len(
        test_dataset
    ):
        test_dataset = test_dataset.select(range(config.MAX_TEST_SAMPLES))
        logging.warning(
            f"OVERRIDE: Limiting testing to the first {config.MAX_TEST_SAMPLES} samples for development."
        )

    # Format data similarly to the training script
    formatted_data = []
    for item in test_dataset:
        prompt = f"""You are an email search agent.
User's email address is {item['inbox_address']}
Today's date is {item['query_date']}

User question: {item['question']}
"""
        formatted_data.append({
            "prompt": prompt,
            "gold_answer": item["answer"],
            "original_question": item["question"]
        })
    return formatted_data


def load_trained_agent(model_dir: str) -> CodeAgent:
    """Loads a fine-tuned agent from a specified directory."""
    logging.info(f"Loading fine-tuned agent from: {model_dir}")
    
    # 1. Load the base model first
    base_model = TransformersModel(model_id=config.BASE_MODEL_ID)

    if model_dir:
        # 2. Apply the LoRA adapters on top of the base model
        # This merges the fine-tuned weights with the base model weights
        peft_model = PeftModel.from_pretrained(base_model.model, model_dir)
        
        # Update the model in our wrapper
        base_model.model = peft_model
        
    # 3. Create the agent instance with the tuned model
    agent = CodeAgent(
        tools=[email_tools.search_emails, email_tools.read_email],
        model=base_model,
        max_steps=1,
        planning_interval=None,
    )
    logging.info("Agent loaded successfully.")
    return agent


def judge_correctness(agent_answer: str, gold_answer: str, original_question: str) -> str:
    """
    Uses an LLM to judge if the agent's answer is correct.
    Returns 'CORRECT', 'INCORRECT', or 'NO_ANSWER'.
    """
    if litellm is None:
        raise ImportError("litellm is required for evaluation. Please install it.")

    # If agent refuses to answer, it's not a hallucination, just an inability to answer.
    if "don't know" in agent_answer.lower() or "unable to find" in agent_answer.lower():
        return "NO_ANSWER"

    try:
        system_prompt = "You are an AI evaluator. Your job is to determine if an AI's answer is correct based on a ground truth answer. Return only one word: CORRECT or INCORRECT."
        user_prompt = f"Question: {original_question}\nGround Truth Answer: {gold_answer}\nAI's Answer: {agent_answer}"
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        response = litellm.completion(
            model=config.JUDGE_MODEL_ID, messages=messages, temperature=0.0, max_tokens=5
        )
        result = response.choices[0].message.content.strip().upper()
        return "CORRECT" if "CORRECT" in result else "INCORRECT"
    except Exception as e:
        logging.error(f"Error calling evaluation judge: {e}")
        return "INCORRECT" # Assume incorrect on error


def main(args):
    """Main evaluation function."""
    logging.info(f"--- Evaluating Model from '{args.model_dir}' ---")

    # 1. Load resources
    test_data = load_test_dataset()
    agent = load_trained_agent(args.model_dir)

    adapter = SmolAgentAdapter(agent=agent, config={})
    logging.info("Created adapter to run the agent for evaluation.")

    # 2. Run agent on all test questions
    evaluation_results = []
    for item in tqdm(test_data, desc="Running evaluation"):
        # The adapter's run method returns the trace
        trace, _, _ = adapter.run(item["prompt"])
        
        # Find the agent's final answer from the trace
        agent_answer = "I don't know" # Default if no answer is found
        for turn in reversed(trace):
            if turn.get("parsed_completion", {}).get("final_answer"):
                agent_answer = turn["parsed_completion"]["final_answer"]
                break
        
        evaluation_results.append({
            "trace": trace,
            "agent_answer": agent_answer,
            "gold_answer": item["gold_answer"],
            "original_question": item["original_question"]
        })

    # 3. Calculate metrics from the results
    num_correct = 0
    num_incorrect = 0
    num_no_answer = 0
    total_turns = 0
    
    logging.info("Judging correctness of all answers...")
    for result in tqdm(evaluation_results, desc="Judging results"):
        judgement = judge_correctness(result["agent_answer"], result["gold_answer"], result["original_question"])
        if judgement == "CORRECT":
            num_correct += 1
        elif judgement == "INCORRECT":
            num_incorrect += 1
        else: # NO_ANSWER
            num_no_answer += 1
        
        total_turns += len(result["trace"])

    total_questions = len(evaluation_results)
    total_attempted_answers = num_correct + num_incorrect
    
    # Metric 1: Correctness
    correctness_rate = (num_correct / total_questions) * 100 if total_questions > 0 else 0
    
    # Metric 2: Efficiency
    avg_turns = total_turns / total_questions if total_questions > 0 else 0
    
    # Metric 3: Anti-Hallucination (Fraction of *attempted* answers that are wrong)
    hallucination_rate = (num_incorrect / total_attempted_answers) * 100 if total_attempted_answers > 0 else 0

    # 4. Report the final scores
    print("\n" + "="*50)
    print(f"EVALUATION REPORT FOR: {args.model_dir}")
    print("="*50)
    print(f"Correctness Rate:      {correctness_rate:.2f}% ({num_correct}/{total_questions} correct)")
    print(f"Efficiency (Avg Turns):  {avg_turns:.2f}")
    print(f"Hallucination Rate:    {hallucination_rate:.2f}% ({num_incorrect}/{total_attempted_answers} incorrect attempts)")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned ToolBrain Email Agent.")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        default=None,
        help="The directory where the fine-tuned model (LoRA adapters) is saved."
    )

    args = parser.parse_args()
    main(args)
