"""
Test script to verify LLM judge compatibility with Gemini models.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from custom_rewards import _call_llm_judge_with_fallback, CustomJudgeResponse
import logging

logging.basicConfig(level=logging.INFO)

def test_judge_compatibility():
    """Test the judge function with Gemini models."""
    
    # Check if Gemini API key is loaded
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found in environment variables!")
        print("   Please check your .env file contains: GEMINI_API_KEY=your_key_here")
        return
    
    print(f"✅ Gemini API Key loaded: {gemini_key[:10]}...")
    
    # Test data - simple case with improved prompt
    test_messages = [
        {
            "role": "system", 
            "content": """You are an AI evaluator. Your job is to determine if an AI's answer is correct.

You MUST respond with valid JSON in this exact format:
{
  "is_correct": true/false,
  "explanation": "your reasoning here"
}

Do not include any other text outside the JSON."""
        },
        {
            "role": "user",
            "content": "Question: What is 2+2?\nCorrect answer: 4\nAI answer: 4"
        }
    ]
    
    # Test only Gemini models
    test_models = [
        "gemini/gemini-2.5-flash-lite",
    ]
    
    for model in test_models:
        print(f"\n🧪 Testing model: {model}")
        print("-" * 50)
        
        try:
            result = _call_llm_judge_with_fallback(
                model=model,
                messages=test_messages,
                response_format=CustomJudgeResponse
            )
            
            print(f"✅ Result: is_correct={result.is_correct}")
            print(f"📝 Explanation: {result.explanation}")
            
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Gemini compatibility test completed!")
    
    # Test with multiple complex cases
    test_cases = [
        {
            "name": "Complex correct case",
            "messages": [
                {
                    "role": "system", 
                    "content": """You are an AI evaluator. 

Respond with ONLY valid JSON in this format:
{"is_correct": true/false, "explanation": "your reasoning"}"""
                },
                {
                    "role": "user",
                    "content": "Question: What is the capital of France?\nCorrect answer: Paris\nAI answer: The capital of France is Paris, which is located in the north-central part of the country."
                }
            ]
        },
        {
            "name": "Tricky negative case",
            "messages": [
                {
                    "role": "system", 
                    "content": """You are an AI evaluator. 

Respond with ONLY valid JSON in this format:
{"is_correct": true/false, "explanation": "your reasoning"}"""
                },
                {
                    "role": "user",
                    "content": "Question: What is 2+2?\nCorrect answer: 4\nAI answer: The answer is not correct, it should be 5."
                }
            ]
        },
        {
            "name": "Ambiguous case",
            "messages": [
                {
                    "role": "system", 
                    "content": """You are an AI evaluator. 

Respond with ONLY valid JSON in this format:
{"is_correct": true/false, "explanation": "your reasoning"}"""
                },
                {
                    "role": "user",
                    "content": "Question: Explain photosynthesis\nCorrect answer: Plants convert sunlight to energy\nAI answer: Photosynthesis is a process but I'm not sure about the details."
                }
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 50)
        
        try:
            result = _call_llm_judge_with_fallback(
                model="gemini/gemini-2.5-flash-lite",
                messages=test_case["messages"],
                response_format=CustomJudgeResponse,
                max_retries=2  # Test retry functionality
            )
            
            print(f"✅ Result: is_correct={result.is_correct}")
            print(f"📝 Explanation: {result.explanation}")
            
        except Exception as e:
            print(f"❌ Error with {test_case['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Advanced compatibility testing completed!")

if __name__ == "__main__":
    test_judge_compatibility()