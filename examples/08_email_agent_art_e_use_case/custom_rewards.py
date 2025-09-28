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


class CustomJudgeResponse(BaseModel):
    is_correct: bool
    explanation: str


def _call_llm_judge_with_fallback(model: str, messages: list, response_format=None, max_retries: int = 3) -> CustomJudgeResponse:
    """
    Clean and reliable LLM judge caller with clear failure modes.
    
    Strategy:
    1. Single API call with response_format (providers ignore if not supported)
    2. Check for .parsed attribute (OpenAI structured output)
    3. Fallback to regex-based JSON parsing from content (Gemini, Claude, etc.)
    4. Fail clearly if parsing fails (no keyword guessing!)
    
    Benefits:
    - Single API call to avoid quota waste
    - Regex-based JSON extraction for accuracy
    - Retry mechanism for network/quota errors
    - Clear failure reporting instead of keyword guessing
    - Better debugging with raw content logging
    
    Returns:
        CustomJudgeResponse: Object with is_correct (bool) and explanation (str)
    """
    import time
    import random
    
    # Retry logic for network/quota errors
    for attempt in range(max_retries):
        try:
            # Single API call with response_format (providers ignore if not supported)
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0.0
            )
            
            msg = response.choices[0].message
            
            # Strategy 1: Check for structured output (.parsed) - OpenAI
            if hasattr(msg, 'parsed') and msg.parsed:
                logging.info("✅ Using structured output (.parsed)")
                return msg.parsed
            
            # Strategy 2: Parse JSON from content - Gemini, Claude, etc.
            content = msg.content.strip()
            logging.info(f"🔄 Falling back to JSON parsing from content: {content[:100]}...")
            
            # Clean content from markdown code blocks
            content = _clean_json_content(content)
            
            # Try regex-based JSON extraction
            json_result = _extract_json_with_regex(content)
            if json_result:
                logging.info("✅ Successfully parsed JSON with regex")
                return json_result
                
            # Strategy 3: Clear failure instead of keyword guessing
            logging.error(f"❌ Failed to parse response from {model}")
            logging.error(f"Raw content: {content[:300]}...")
            
            return CustomJudgeResponse(
                is_correct=False,
                explanation=f"PARSE_ERROR: Model returned unparseable response. Expected JSON format but got: {content[:200]}..."
            )
                    
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a retryable error
            retryable_errors = ['503', '429', 'rate limit', 'quota', 'timeout', 'network', 'connection']
            is_retryable = any(error_type in error_str for error_type in retryable_errors)
            
            if is_retryable and attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"⚠️ Retryable error ({e}), retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                # Non-retryable error or max retries reached
                logging.error(f"❌ Error calling LLM Judge: {e} (attempt {attempt + 1}/{max_retries})")
                return CustomJudgeResponse(is_correct=False, explanation=f"API_ERROR after {attempt + 1} attempts: {e}")
    
    return CustomJudgeResponse(is_correct=False, explanation="ERROR: Max retries exceeded")


def _clean_json_content(content: str) -> str:
    """
    Clean content to extract JSON from markdown code blocks or other wrappers.
    
    Handles patterns like:
    - ```json { ... } ```
    - ``` { ... } ```
    - **JSON:** { ... }
    """
    # Remove markdown code blocks
    if content.startswith("```"):
        # Find closing ```
        end_marker = content.find("```", 3)
        if end_marker != -1:
            content = content[3:end_marker].strip()
            # Remove language identifier (e.g., "json")
            if content.lower().startswith("json"):
                content = content[4:].strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "**JSON:**", "**json:**", "JSON:", "json:", 
        "Here's the JSON:", "Response:", "Result:"
    ]
    
    for prefix in prefixes_to_remove:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
            break
    
    return content


def _extract_json_with_regex(content: str) -> CustomJudgeResponse | None:
    """
    Extract JSON using regex for more accurate parsing.
    
    Handles multiple JSON objects and nested braces properly.
    
    Returns:
        CustomJudgeResponse if successful, None if failed
    """
    import json
    import re
    
    # Try multiple regex patterns to find JSON
    patterns = [
        # Pattern 1: Complete JSON object (with potential nested objects)
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        # Pattern 2: More permissive - any content between outermost braces
        r'\{.*?\}',
        # Pattern 3: Multiline JSON with DOTALL flag
        r'\{.*\}',
    ]
    
    for i, pattern in enumerate(patterns):
        try:
            flags = re.DOTALL if i == 2 else 0
            matches = re.findall(pattern, content, flags)
            
            for match in matches:
                try:
                    result_dict = json.loads(match)
                    
                    # Validate required fields
                    if isinstance(result_dict, dict) and 'is_correct' in result_dict:
                        is_correct = result_dict.get('is_correct', False)
                        explanation = result_dict.get('explanation', 'No explanation provided')
                        
                        logging.info(f"Regex pattern {i+1} successfully extracted JSON")
                        return CustomJudgeResponse(is_correct=is_correct, explanation=explanation)
                        
                except json.JSONDecodeError:
                    continue  # Try next match
                    
        except Exception as e:
            logging.debug(f"Regex pattern {i+1} failed: {e}")
            continue
    
    logging.warning("All regex patterns failed to extract valid JSON")
    return None

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

        # Use the universal LLM judge caller
        result = _call_llm_judge_with_fallback(
            model=judge_model,
            messages=messages,
            response_format=CustomJudgeResponse
        )
        is_correct = result.is_correct

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
