"""
A standalone evaluation script that faithfully replicates the
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

from . import config
from . import email_tools
from smolagents import CodeAgent
from toolbrain.adapters import SmolAgentAdapter
from toolbrain import Brain

try:
    import litellm
except ImportError:
    litellm = None


class JudgeResponse(BaseModel):
    is_correct: bool  # True if answer is correct, False otherwise

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s"
)


def load_validation_dataset() -> List[Dict[str, Any]]:
    """Loads the validation dataset based on the main config."""
    full_dataset = load_dataset(
        config.QUESTIONS_DATASET_ID, split=config.QUESTIONS_DATASET_SPLIT
    )
    full_dataset = full_dataset.filter(lambda x: x["how_realistic"] > 0.7)

    split_dataset = full_dataset.train_test_split(
        test_size=1.0 - config.TRAIN_TEST_SPLIT_RATIO, seed=config.DATASET_SPLIT_SEED
    )
    validation_dataset = split_dataset["test"]

    if config.MAX_TEST_SAMPLES is not None and config.MAX_TEST_SAMPLES < len(
        validation_dataset
    ):
        validation_dataset = validation_dataset.select(range(config.MAX_TEST_SAMPLES))
        logging.warning(
            f"OVERRIDE: Limiting training to the first {config.MAX_TEST_SAMPLES} samples for development."
        )

    formatted_data = []
    for item in validation_dataset:
        system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(
            max_turns=config.MAX_AGENT_TURNS,
            inbox_address=item["inbox_address"],
            query_date=item["query_date"],
        )
        full_query = f"{system_prompt}\nUser question: {item['question']}"

        formatted_data.append(
            {
                "agent_prompt": full_query,
                "gold_answer": item["answer"],
                "query": item["question"],
            }
        )
    return formatted_data


def determine_if_correct_art_style(
    agent_answer: str, gold_answer: str, query: str
) -> bool:
    """ART-E's judge using the exact original prompt for consistency."""
    if litellm is None:
        return False

    # Use exact ART-E prompt for consistency
    system_prompt = "You will be given an question and two different answers to the question, the correct answer and the answer given by an AI. Your job is to determine if the answer given by the AI is correct. Return True if the answer is semantically similar to the correct answer, and False otherwise. Return only the word True or False, no other text."

    user_prompt = f"Question: {query}\nCorrect answer: {gold_answer}\nAI answer: {agent_answer}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    try:
        response = litellm.completion(
            model=config.JUDGE_MODEL_ID,
            messages=messages,
            response_format=JudgeResponse,
            temperature=0.0,
        )
        
        msg = response.choices[0].message

        try:
            result_obj = msg.parsed
            if result_obj:
                if isinstance(result_obj, dict):
                    is_correct = result_obj.get("is_correct", False)
                else:
                    is_correct = getattr(result_obj, "is_correct", False)
            else:
                is_correct = False
        except (AttributeError, KeyError):
            try:
                content = msg.content.strip() if msg.content else ""
                
                # Try to parse JSON from content
                if content.startswith('{') and content.endswith('}'):
                    import json
                    parsed_content = json.loads(content)
                    is_correct = parsed_content.get("is_correct", False)
                else:
                    # Original fallback for plain text
                    is_correct = content.lower() == "true"
            except (AttributeError, json.JSONDecodeError):
                is_correct = False
        
        return is_correct
        
    except Exception as e:
        return False

def run_evaluation(
    agent: CodeAgent, validation_data: List[Dict[str, Any]], output_dir: str
) -> Dict[str, float]:
    """
    Runs the agent on a validation set and returns a dictionary of metrics,
    following the ART-E methodology.
    """
    adapter = SmolAgentAdapter(agent=agent, config={})

    num_correct, num_incorrect, num_no_answer, total_turns = 0, 0, 0, 0

    for item in tqdm(validation_data, desc="Running Validation", leave=False):
        trace, _, _ = adapter.run(item["agent_prompt"])

        agent_answer = "I don't know"  # Default value
        for turn in reversed(trace):
            parsed = turn.get("parsed_completion", {})
            
            # Method 1: Check parsed final_answer field (standard case)
            if parsed and parsed.get("final_answer"):
                agent_answer = parsed["final_answer"]
                break
            
            # Method 2: Check both action_output (SmolAgent) and tool_output (LangChain)
            action_output = turn.get("action_output") or turn.get("tool_output")
            if action_output is not None:
                agent_answer = str(action_output)
                break
            
            # Method 3: Parse "Final answer: X" from model_completion
            model_completion = turn.get("model_completion", "")
            if "Final answer:" in model_completion:
                parts = model_completion.split("Final answer:")
                if len(parts) > 1:
                    agent_answer = parts[-1].strip()
                    break
            
            # Method 4: Check if tool_code contains final_answer() call and output matches
            if parsed:
                tool_code = parsed.get("tool_code", "")
                if "final_answer(" in tool_code and action_output is not None:
                    agent_answer = str(action_output)
                    break

        # Step 1: Rule-based check for "I don't know".
        is_no_answer = agent_answer.strip().lower() == "i don't know"

        if is_no_answer:
            num_no_answer += 1
        else:
            # Step 2: LLM-based check if the agent attempted an answer.
            if determine_if_correct_art_style(
                agent_answer,
                item["gold_answer"],
                item["query"],
            ):
                num_correct += 1
            else:
                num_incorrect += 1

        total_turns += len(trace)
        time.sleep(2)

    total_questions = len(validation_data)
    total_attempted_answers = num_correct + num_incorrect

    metrics = {
        "correctness_rate": (
            (num_correct / total_questions) * 100 if total_questions > 0 else 0
        ),
        "avg_turns": total_turns / total_questions if total_questions > 0 else 0,
        "hallucination_rate": (
            (num_incorrect / total_attempted_answers) * 100
            if total_attempted_answers > 0
            else 0
        ),
        "no_answer_rate": (
            (num_no_answer / total_questions) * 100 if total_questions > 0 else 0
        ),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned agent using ART-E's methodology."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory of the fine-tuned model.",
    )
    args = parser.parse_args()

    # Load trained agent directly
    logging.info(f"Loading fine-tuned agent from: {args.model_dir}")

    agent_to_evaluate = Brain.load_agent(
        model_dir=args.model_dir,
        base_model_id=config.BASE_MODEL_ID,
        tools=[email_tools.search_emails, email_tools.read_email],
        max_steps=config.MAX_AGENT_TURNS,
        use_unsloth=True,
    )
    validation_dataset = load_validation_dataset()

    final_metrics = run_evaluation(
        agent_to_evaluate, validation_dataset, args.model_dir
    )

    print("\n" + "=" * 50)
    print(f"EVALUATION REPORT FOR: {args.model_dir}")
    print("=" * 50)
    for key, value in final_metrics.items():
        print(f"{key.replace('_', ' ').title():<25}: {value:.2f}")
    print("=" * 50)