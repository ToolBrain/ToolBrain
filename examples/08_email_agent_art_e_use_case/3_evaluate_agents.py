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
    
    base_model = TransformersModel(model_id=config.BASE_MODEL_ID)
    peft_model = PeftModel.from_pretrained(base_model.model, model_dir)
    base_model.model = peft_model
    
    agent = CodeAgent(
        tools=[email_tools.search_emails, email_tools.read_email],
        model=base_model
    )
    logging.info("Agent loaded successfully.")
    return agent


def classify_answer_with_llm(agent_answer: str, gold_answer: str, original_question: str) -> str:
    """
    Uses a powerful LLM to classify the agent's answer into one of three categories.
    """
    if litellm is None:
        raise ImportError("litellm is required for evaluation. Please install it.")

    try:
        system_prompt = """You are a meticulous AI evaluator. Your task is to classify an agent's answer into one of three categories based on a ground truth answer.
The categories are:
1. CORRECT: The agent's answer is factually and semantically equivalent to the ground truth.
2. INCORRECT: The agent's answer provides factually wrong information. This is a hallucination.
3. NO_ANSWER: The agent explicitly states it cannot find the answer, does not know, or that no information is available (e.g., "no email found").

Respond with ONLY ONE WORD: CORRECT, INCORRECT, or NO_ANSWER."""

        user_prompt = f"""Question: {original_question}
Ground Truth Answer: {gold_answer}
Agent's Answer: {agent_answer}"""
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        response = litellm.completion(
            model=config.JUDGE_MODEL_ID, messages=messages, temperature=0.0, max_tokens=5
        )
        result = response.choices[0].message.content.strip().upper()

        if result in ["CORRECT", "INCORRECT", "NO_ANSWER"]:
            return result
        else:
            logging.warning(f"Judge returned an invalid classification: '{result}'. Defaulting to INCORRECT.")
            return "INCORRECT"
            
    except Exception as e:
        logging.error(f"Error calling evaluation judge: {e}")
        return "INCORRECT"


def main(args):
    """Main evaluation function."""
    logging.info(f"--- Evaluating Model from '{args.model_dir}' ---")

    test_data = load_test_dataset()
    agent = load_trained_agent(args.model_dir)
    adapter = SmolAgentAdapter(agent=agent, config={})

    evaluation_results = []
    for item in tqdm(test_data, desc="Running evaluation"):
        trace, _, _ = adapter.run(item["prompt"])
        
        agent_answer = "I don't know"
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

    num_correct, num_incorrect, num_no_answer, total_turns = 0, 0, 0, 0
    
    logging.info("Classifying all answers using the LLM Judge...")
    for result in tqdm(evaluation_results, desc="Judging results"):
        answer_type = classify_answer_with_llm(
            result["agent_answer"], result["gold_answer"], result["original_question"]
        )
        
        if answer_type == "CORRECT":
            num_correct += 1
        elif answer_type == "INCORRECT":
            num_incorrect += 1
        elif answer_type == "NO_ANSWER":
            num_no_answer += 1
        
        total_turns += len(result["trace"])

    total_questions = len(evaluation_results)
    total_attempted_answers = num_correct + num_incorrect
    
    correctness_rate = (num_correct / total_questions) * 100 if total_questions > 0 else 0
    avg_turns = total_turns / total_questions if total_questions > 0 else 0
    hallucination_rate = (num_incorrect / total_attempted_answers) * 100 if total_attempted_answers > 0 else 0

    print("\n" + "="*50)
    print(f"EVALUATION REPORT FOR: {args.model_dir}")
    print("="*50)
    print(f"Correctness Rate:      {correctness_rate:.2f}% ({num_correct}/{total_questions} correct)")
    print(f"Efficiency (Avg Turns):  {avg_turns:.2f}")
    print(f"Hallucination Rate:    {hallucination_rate:.2f}% ({num_incorrect}/{total_attempted_answers} incorrect attempts)")
    print(f"No Answer Rate:        {(num_no_answer / total_questions) * 100:.2f}% ({num_no_answer}/{total_questions} unanswered)")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned ToolBrain Email Agent.")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The directory where the fine-tuned model (LoRA adapters) is saved."
    )

    args = parser.parse_args()
    main(args)