"""
Custom Reward Functions for the ART-E Email Agent Experiment.

This module contains reward functions specifically designed for the email search task.
They are kept separate from the main `toolbrain/rewards.py` to maintain modularity
and keep the core library clean of use-case-specific logic.
"""

import re
from collections import Counter
from typing import List, Dict, Any

from toolbrain.core_types import Trace
from pydantic import BaseModel
import logging
import config

try:
    import litellm
except ImportError:
    litellm = None

# --- PYDANTIC MODELS FOR STRUCTURED OUTPUT ---

class CustomJudgeResponse(BaseModel):
    is_correct: bool
    explanation: str

# --- Reward Function A: Metric-based (F1 Score) ---

def _get_agent_final_answer(trace: Trace) -> str | None:
    """Helper function to extract the final answer from a trace."""
    for turn in reversed(trace):
        parsed_completion = turn.get("parsed_completion", {})
        if parsed_completion and parsed_completion.get("final_answer"):
            return parsed_completion["final_answer"]
    return None

def reward_f1_score(trace: Trace, **kwargs: Any) -> float:
    """
    Calculates a reward based on the F1 score between the agent's final answer
    and the ground truth answer. This is a fast, objective, lexical-based reward.

    Args:
        trace: The execution trace of the agent.
        **kwargs: Must contain 'gold_answer' (str) for comparison.

    Returns:
        A float score between 0.0 and 1.0 representing the F1 score.
    """
    gold_answer = kwargs.get("gold_answer")
    agent_answer = _get_agent_final_answer(trace)

    if not gold_answer or not agent_answer:
        return 0.0

    # Normalize and tokenize the answers
    gold_tokens = re.findall(r'\w+', gold_answer.lower())
    agent_tokens = re.findall(r'\w+', agent_answer.lower())

    if not agent_tokens:
        return 0.0

    # Use collections.Counter to find common tokens
    common_tokens = Counter(gold_tokens) & Counter(agent_tokens)
    num_common = sum(common_tokens.values())

    # Calculate precision, recall, and F1 score
    precision = num_common / len(agent_tokens)
    recall = num_common / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# --- Reward Function B: ART-E Style Hybrid Judge ---

def reward_art_style_judge(trace: Trace, **kwargs: Any) -> float:
    """
    A hybrid reward function that mimics the logic from the ART-E project.
    It combines an LLM-based correctness check with rule-based penalties and bonuses.

    Args:
        trace: The execution trace of the agent.
        **kwargs: Must contain 'query', 'gold_answer', and 'judge_model'.

    Returns:
        A float score representing the assessed quality of the trace.
        - Positive values for correct answers (1.0 to ~1.1)
        - Negative values for incorrect answers (-0.5)
        - Neutral values for no answer (0.0)
    """
    if litellm is None:
        logging.warning("litellm is not installed. ART-style judge is disabled, returning 0.0.")
        return 0.0

    # 1. Extract necessary information
    query = kwargs.get("original_question")
    gold_answer = kwargs.get("gold_answer")
    judge_model = kwargs.get("judge_model", config.JUDGE_MODEL_ID)
    agent_answer = _get_agent_final_answer(trace)

    if not all([query, gold_answer, judge_model]):
        raise ValueError("ART-style judge requires 'query', 'gold_answer', and 'judge_model' in kwargs.")

    # 2. Handle the case where the agent gave no answer
    if not agent_answer:
        return 0.0 

    # 3. Call the LLM Judge to determine correctness
    is_correct = False
    try:
        system_prompt = """You are an AI evaluator. Your job is to determine if an AI's answer is correct.
        
        Return your response in JSON format with:
        - is_correct: true if the AI answer is semantically similar to the correct answer, false otherwise
        - explanation: Brief reasoning for your decision"""
        
        user_prompt = f"Question: {query}\nCorrect answer: {gold_answer}\nAI answer: {agent_answer}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = litellm.completion(
            model=judge_model,
            messages=messages,
            response_format=CustomJudgeResponse,
            temperature=0.0
        )
        
        # Parse structured response
        result_obj = response.choices[0].message.parsed
        is_correct = result_obj.is_correct

    except Exception as e:
        logging.error(f"Error calling LLM Judge for ART-style reward: {e}")
        return 0.0 

    # 4. Calculate final reward based on correctness
    if is_correct:
        reward = 1.0
        
        MAX_TURNS = 10 
        efficiency_bonus = 0.1 * (1 - len(trace) / MAX_TURNS)
        reward += max(0, efficiency_bonus) 
        
        return reward
    else:
        # Apply a significant penalty for hallucination (incorrect answer)
        return -0.5
