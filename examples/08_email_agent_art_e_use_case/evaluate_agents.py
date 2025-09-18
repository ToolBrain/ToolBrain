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
import json
import logging
import os
import time
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel

# from . import config
# from . import email_tools
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))
import config
import email_tools
from smolagents import CodeAgent, TransformersModel
from toolbrain.adapters import SmolAgentAdapter
from toolbrain.models import UnslothModel
from pydantic import BaseModel

# try:
#     import litellm
# except ImportError:
#     litellm = None
import litellm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s"
)

# --- PYDANTIC MODELS FOR STRUCTURED OUTPUT ---

class JudgeResponse(BaseModel):
    verdict: str  # "CORRECT", "INCORRECT", or "NO_ANSWER"
    reasoning: str  # Brief explanation of the decision


def load_validation_dataset_for_eval() -> List[Dict[str, Any]]:
    """Loads the validation dataset for evaluation purposes."""
    full_dataset = load_dataset(config.QUESTIONS_DATASET_ID, split=config.QUESTIONS_DATASET_SPLIT)
    full_dataset = full_dataset.filter(lambda x: x['how_realistic'] > 0.7)
    split_dataset = full_dataset.train_test_split(
        test_size=1.0 - config.TRAIN_TEST_SPLIT_RATIO, seed=config.DATASET_SPLIT_SEED
    )
    validation_dataset = split_dataset['test']
    
    formatted_data = []
    for item in validation_dataset:
        system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(
            max_turns=config.MAX_AGENT_TURNS,
            inbox_address=item['inbox_address'],
            query_date=item['query_date']
        )
        full_query = f"{system_prompt}\nUser question: {item['question']}"
        formatted_data.append({
            "prompt": full_query,
            "gold_answer": item["answer"],
            "original_question": item["question"]
        })
    return formatted_data


def load_trained_agent(model_dir: str) -> CodeAgent:
    """
    Loads a fine-tuned agent from a specified directory, ensuring that Unsloth
    optimizations are applied for inference, consistent with training.
    """
    logging.info(f"Loading fine-tuned agent from: {model_dir} using Unsloth.")

    # base_model = TransformersModel(
    #     model_id=config.BASE_MODEL_ID,
    #     # max_seq_length=config.MAX_TOOL_OUTPUT_CHARS + config.MAX_NEW_TOKENS,
    #     max_seq_length=8192,
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


def classify_answer_with_llm(
    agent_answer: str, gold_answer: str, original_question: str, log_file_path: str
) -> str:
    """
    Uses a powerful LLM to classify the agent's answer into one of three categories using structured output.
    """

    system_prompt = """You are a meticulous AI evaluator. Your task is to classify an agent's answer into one of three categories based on a ground truth answer.

The categories are:
1. CORRECT: The agent's answer is factually and semantically equivalent to the ground truth.
2. INCORRECT: The agent's answer provides factually wrong information. This is a hallucination.
3. NO_ANSWER: The agent explicitly states it cannot find the answer, does not know, or that no information is available (e.g., "no email found").

Return your response in JSON format with:
- verdict: "CORRECT", "INCORRECT", or "NO_ANSWER"
- reasoning: Brief explanation of your decision"""

    user_prompt = f"""Question: {original_question}
Ground Truth Answer: {gold_answer}
Agent's Answer: {agent_answer}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = litellm.completion(
            model=config.JUDGE_MODEL_ID,
            messages=messages,
            response_format=JudgeResponse,
            temperature=0.0,
        )
        
        # Parse structured response
        result_obj = response.choices[0].message.parsed
        result = result_obj.verdict.upper()

        # Validate verdict
        if result in ["CORRECT", "INCORRECT", "NO_ANSWER"]:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                log_entry = {
                    "question": original_question, 
                    "agent_answer": agent_answer, 
                    "verdict": result,
                    "reasoning": result_obj.reasoning
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            return result
        else:
            logging.warning(
                f"Judge returned an invalid classification: '{result}'. Defaulting to INCORRECT."
            )
            with open(log_file_path, 'a', encoding='utf-8') as f:
                log_entry = {"question": original_question, "agent_answer": agent_answer, "verdict": "INCORRECT", "reasoning": "Invalid verdict"}
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            return "INCORRECT"

    except Exception as e:
        logging.error(f"Error calling evaluation judge: {e}")
        logging.error(f"--- FAILED PROMPT ---")
        logging.error(user_prompt)
        logging.error(f"--- END OF FAILED PROMPT ---")
        with open(log_file_path, 'a', encoding='utf-8') as f:
            log_entry = {"question": original_question, "agent_answer": agent_answer, "verdict": "INCORRECT", "reasoning": f"Error: {e}"}
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        return "INCORRECT"


def run_evaluation_toolbrain_style(agent: CodeAgent, validation_data: List[Dict[str, Any]], output_dir: str) -> Dict[str, float]:
    """
    Runs evaluation using ToolBrain's native 3-way classification judge and logs details.
    """
    adapter = SmolAgentAdapter(agent=agent, config={})
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, config.JUDGE_TOOL_BRAIN_LOG_FILENAME) 

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ToolBrain-Style Judge Log for Evaluation Run ---\n")

    num_correct, num_incorrect, num_no_answer, total_turns = 0, 0, 0, 0
    
    for item in tqdm(validation_data, desc="Running ToolBrain-Style Validation", leave=False):
        trace, _, _ = adapter.run(item["prompt"])
        
        agent_answer = "I don't know"
        for turn in reversed(trace):
            if turn.get("parsed_completion", {}).get("final_answer"):
                agent_answer = turn["parsed_completion"]["final_answer"]
                break
        
        answer_type = classify_answer_with_llm(agent_answer, item["gold_answer"], item["original_question"], log_file_path)
        
        if answer_type == "CORRECT":
            num_correct += 1
        elif answer_type == "INCORRECT":
            num_incorrect += 1
        elif answer_type == "NO_ANSWER":
            num_no_answer += 1
        
        total_turns += len(trace)
        time.sleep(2)

    total_questions = len(validation_data)
    total_attempted_answers = num_correct + num_incorrect
    
    metrics = {
        "correctness_rate": (num_correct / total_questions) * 100 if total_questions > 0 else 0,
        "avg_turns": total_turns / total_questions if total_questions > 0 else 0,
        "hallucination_rate": (num_incorrect / total_attempted_answers) * 100 if total_attempted_answers > 0 else 0,
        "no_answer_rate": (num_no_answer / total_questions) * 100 if total_questions > 0 else 0,
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned agent using ToolBrain's methodology.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the fine-tuned model.")
    args = parser.parse_args()

    agent_to_evaluate = load_trained_agent(args.model_dir)
    validation_dataset = load_validation_dataset_for_eval()
    
    final_metrics = run_evaluation_toolbrain_style(agent_to_evaluate, validation_dataset, args.model_dir)

    print("\n" + "="*50)
    print(f"EVALUATION REPORT (TOOLBRAIN-STYLE) FOR: {args.model_dir}")
    print("="*50)
    for key, value in final_metrics.items():
        print(f"{key.replace('_', ' ').title():<25}: {value:.2f}")
    print("="*50)
