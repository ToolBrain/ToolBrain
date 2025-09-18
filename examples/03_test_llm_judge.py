"""
A standalone test script to verify the LLM-as-a-Judge reward function.

This script does NOT run a real agent. Instead, it uses pre-defined mock traces
to test if the `reward_llm_judge_via_ranking` function can correctly:
1. Call the judge LLM (e.g., Gemini).
2. Parse the ranking response.
3. Convert the ranking into a list of scores.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from toolbrain.rewards import reward_llm_judge_via_ranking
from toolbrain.core_types import Trace, Turn, ParsedCompletion

# --- 1. Create mock traces ---
#    We will create 3 versions: a good one, a bad one, and an okay one.

# Good trace: Agent thinks logically, uses the tool correctly, and gives the correct answer.
good_trace: Trace = [
    {
        "prompt_for_model": "User Query: Calculate 5 + 7",
        "model_completion": "Thought: The user wants to add 5 and 7. I will use the `add` tool and print the result.\nCode:\nprint(add(5, 7))",
        "parsed_completion": {"thought": "The user wants to add 5 and 7...", "tool_code": "print(add(5, 7))", "final_answer": None},
        "tool_output": "12"
    },
    {
        "prompt_for_model": "User Query: Calculate 5 + 7\nASSISTANT RESPONSE:...\nTOOL OUTPUT: 12",
        "model_completion": "Thought: The tool returned 12, which is the correct answer.\nFinal Answer: 12",
        "parsed_completion": {"thought": "The tool returned 12...", "tool_code": None, "final_answer": "12"},
        "tool_output": None
    }
]

# Bad trace: Agent is confused, thinks randomly, and gives up.
bad_trace: Trace = [
    {
        "prompt_for_model": "User Query: Calculate 5 + 7",
        "model_completion": "Thought: I will try to multiply instead.\nCode:\nprint(multiply(5, 7))",
        "parsed_completion": {"thought": "I will try to multiply instead.", "tool_code": "print(multiply(5, 7))", "final_answer": None},
        "tool_output": "35"
    },
    {
        "prompt_for_model": "User Query: Calculate 5 + 7\nASSISTANT RESPONSE:...\nTOOL OUTPUT: 35",
        "model_completion": "Thought: The result is 35. This is wrong. I cannot solve this.\nFinal Answer: I am unable to calculate 5 + 7.",
        "parsed_completion": {"thought": "The result is 35. This is wrong...", "tool_code": None, "final_answer": "I am unable to calculate 5 + 7."},
        "tool_output": None
    }
]

# Okay trace: Agent does the right thing but inefficiently (e.g., doesn't use tool).
okay_trace: Trace = [
    {
        "prompt_for_model": "User Query: Calculate 5 + 7",
        "model_completion": "Thought: I can do this in my head. 5 + 7 is 12.\nFinal Answer: 12",
        "parsed_completion": {"thought": "I can do this in my head...", "tool_code": None, "final_answer": "12"},
        "tool_output": None
    }
]

def main():
    print("🧪 Testing the LLM-as-a-Judge Reward Function...")
    print("=" * 60)

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("❌ GEMINI_API_KEY environment variable is required for this test!")
    print("✅ GEMINI_API_KEY found.")

    # --- 2. Prepare data to send to judge ---
    query_for_judge = "Use the add tool to calculate 5 + 7"
    traces_to_judge = [good_trace, bad_trace, okay_trace]
    
    judge_model_id = "gemini/gemini-1.5-flash" 

    print(f"\n📚 We have 3 mock traces to judge for the query: '{query_for_judge}'")
    print("  - Trace 1: Good (uses tool correctly)")
    print("  - Trace 2: Bad (uses wrong tool, gives up)")
    print("  - Trace 3: Okay (correct answer, but doesn't use tool)")

    # --- 3. Call the reward function ---
    try:
        scores = reward_llm_judge_via_ranking(
            traces=traces_to_judge,
            query=query_for_judge,
            judge_model=judge_model_id
        )
        
        print("\n--- RESULTS ---")
        print(f"🏆 Final scores returned by the reward function: {scores}")

    except Exception as e:
        print(f"\n❌ An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()