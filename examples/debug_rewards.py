# Debug trace structure to see why rewards are zero

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

print("📥 Quick test setup...")
text_gen_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=256,
    temperature=0.1,
    return_full_text=False
)

hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
trainable_llm = ChatHuggingFace(llm=hf_pipeline)

langchain_agent = create_agent(
    model=trainable_llm,
    tools=[add],
    prompt="You are a helpful assistant. You must call a tool to answer the user.",
)

tools_list = [add]

brain = Brain(
    agent=langchain_agent,
    trainable_model=trainable_llm,
    algorithm="GRPO",
    reward_func=reward_exact_match
)

# Re-initialize adapter with tools
from toolbrain.langchain_adapter_v2 import LangChainAdapter
brain.agent_adapter = LangChainAdapter(
    agent=langchain_agent,
    trainable_model=trainable_llm,
    config=brain.config.to_dict() if hasattr(brain.config, 'to_dict') else vars(brain.config),
    tools=tools_list
)

print("\n🔍 Testing single trace...")
query = "Use the add tool to calculate 5 + 7"
trace, rl_input, raw_memory = brain.agent_adapter.run(query)

print(f"\n📊 TRACE ANALYSIS:")
print(f"Trace length: {len(trace)}")

for i, turn in enumerate(trace):
    print(f"\n--- Turn {i+1} ---")
    print(f"Keys: {list(turn.keys())}")
    
    if "parsed_completion" in turn:
        pc = turn["parsed_completion"]
        print(f"Parsed completion type: {type(pc)}")
        if hasattr(pc, 'thought'):
            print(f"  Thought: {pc.thought[:100]}...")
        if hasattr(pc, 'tool_code'):
            print(f"  Tool code: {pc.tool_code}")
        if hasattr(pc, 'final_answer'):
            print(f"  Final answer: {pc.final_answer}")
    
    if "tool_output" in turn:
        print(f"Tool output: {turn['tool_output']}")
    
    if "model_completion" in turn:
        print(f"Model completion: {turn['model_completion'][:100]}...")

print(f"\n🎯 REWARD TEST:")
try:
    reward = reward_exact_match(trace, query="Use the add tool to calculate 5 + 7", gold_answer="12")
    print(f"Reward: {reward}")
    
    # Manual debug reward function
    print(f"\n🔍 Manual reward debug:")
    for i, turn in enumerate(trace):
        action_output = turn.get("tool_output")
        print(f"Turn {i+1} tool_output: '{action_output}'")
        if action_output:
            print(f"  Stripped: '{str(action_output).strip()}'")
            print(f"  Matches '12': {str(action_output).strip() == '12'}")
    
except Exception as e:
    print(f"Reward error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n📋 RL INPUT:")
print(f"RL input length: {len(rl_input)}")
for i, segment in enumerate(rl_input):
    print(f"Segment {i+1}: Role={segment['role']}, Text={segment['text'][:50]}...")