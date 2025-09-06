"""
Step 5: A standalone evaluation script that faithfully replicates the
ART-E project's evaluation methodology and includes detailed logging.
"""

import argparse
import logging
import os
import json
import time
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel
from pydantic import BaseModel

# Import from our use case and core modules
from . import config
from . import email_tools
from toolbrain.models import UnslothModel 
from smolagents import CodeAgent, TransformersModel
from toolbrain.adapters import SmolAgentAdapter

try:
    import litellm
except ImportError:
    litellm = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# --- PYDANTIC MODELS FOR STRUCTURED OUTPUT ---

class ArtJudgeResponse(BaseModel):
    is_correct: bool
    explanation: str

# --- CORE FUNCTIONS ---

def load_validation_dataset() -> List[Dict[str, Any]]:
    """Loads the validation dataset based on the main config."""
    full_dataset = load_dataset(config.QUESTIONS_DATASET_ID, split=config.QUESTIONS_DATASET_SPLIT)
    full_dataset = full_dataset.filter(lambda x: x['how_realistic'] > 0.7)
    
    split_dataset = full_dataset.train_test_split(
        test_size=1.0 - config.TRAIN_TEST_SPLIT_RATIO, 
        seed=config.DATASET_SPLIT_SEED
    )
    validation_dataset = split_dataset['test']
    
    formatted_data = []
    for item in validation_dataset:
        # Use the exact prompt structure from ART-E
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

def determine_if_correct_art_style(agent_answer: str, gold_answer: str, original_question: str, log_file_path: str) -> bool:
    """ART-E's judge using structured output for reliability."""
    if litellm is None: return False
    
    system_prompt = """You will be given a question and two different answers to the question, the correct answer and the answer given by an AI. Your job is to determine if the answer given by the AI is correct.
    
    Return your response in JSON format with:
    - is_correct: true if the AI answer is semantically similar to the correct answer, false otherwise
    - explanation: Brief reasoning for your decision"""
    
    user_prompt = f"Question: {original_question}\nCorrect answer: {gold_answer}\nAI answer: {agent_answer}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    try:
        response = litellm.completion(
            model=config.JUDGE_MODEL_ID, 
            messages=messages, 
            response_format=ArtJudgeResponse,
            temperature=0.0
        )
        
        # Parse the structured response
        result_obj = response.choices[0].message.parsed
        is_correct = result_obj.is_correct
        judge_output = f"is_correct: {is_correct}, explanation: {result_obj.explanation}"
        
    except Exception as e:
        judge_output = f"ERROR: {e}"
        is_correct = False
        result_obj = None

    # Detailed Logging for debugging
    with open(log_file_path, 'a', encoding='utf-8') as f:
        log_entry = {
            "question": original_question,
            "agent_answer": agent_answer,
            "gold_answer": gold_answer,
            "judge_prompt": messages,
            "judge_output_structured": judge_output,
            "verdict": "CORRECT" if is_correct else "INCORRECT"
        }
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
    return is_correct

def run_evaluation(agent: CodeAgent, validation_data: List[Dict[str, Any]], output_dir: str) -> Dict[str, float]:
    """
    Runs the agent on a validation set and returns a dictionary of metrics,
    following the ART-E methodology.
    """
    adapter = SmolAgentAdapter(agent=agent, config={})
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, config.JUDGE_LOG_FILENAME)
    
    # Initialize the log file for this evaluation run
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Judge Log for Evaluation Run ---\n")

    num_correct, num_incorrect, num_no_answer, total_turns = 0, 0, 0, 0
    
    for item in tqdm(validation_data, desc="Running Validation", leave=False):
        trace, _, _ = adapter.run(item["prompt"])
        
        agent_answer = "I don't know" # Default value
        for turn in reversed(trace):
            if turn.get("parsed_completion", {}).get("final_answer"):
                agent_answer = turn["parsed_completion"]["final_answer"]
                break
        
        # Replicating ART-E's logic strictly
        # Step 1: Rule-based check for "I don't know".
        is_no_answer = (agent_answer.strip().lower() == "i don't know")
        
        if is_no_answer:
            num_no_answer += 1
            with open(log_file_path, 'a', encoding='utf-8') as f:
                log_entry = {"question": item["original_question"], "agent_answer": agent_answer, "verdict": "NO_ANSWER"}
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        else:
            # Step 2: LLM-based check if the agent attempted an answer.
            if determine_if_correct_art_style(agent_answer, item["gold_answer"], item["original_question"], log_file_path):
                num_correct += 1
            else:
                num_incorrect += 1
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned agent using ART-E's methodology.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the fine-tuned model.")
    args = parser.parse_args()

    agent_to_evaluate = load_trained_agent(args.model_dir)
    validation_dataset = load_validation_dataset()
    
    final_metrics = run_evaluation(agent_to_evaluate, validation_dataset, args.model_dir)

    print("\n" + "="*50)
    print(f"EVALUATION REPORT FOR: {args.model_dir}")
    print("="*50)
    for key, value in final_metrics.items():
        print(f"{key.replace('_', ' ').title():<25}: {value:.2f}")
    print("="*50)