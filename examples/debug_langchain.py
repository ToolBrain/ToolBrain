# Debug script để kiểm tra LangChain agent tracing

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

print("📥 Initializing simple HuggingFace model...")
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    return_full_text=False
)

hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
trainable_llm = ChatHuggingFace(llm=hf_pipeline)

print("🔧 Creating LangChain agent...")
langchain_agent = create_agent(
    model=trainable_llm,
    tools=[add],
    prompt="You are a helpful assistant. You must call a tool to answer the user.",
)

print("🧠 Initializing Brain...")
brain = Brain(
    agent=langchain_agent,
    trainable_model=trainable_llm,
    algorithm="GRPO",
    reward_func=reward_exact_match
)

print("\n🔍 Testing agent execution...")
query = "Use the add tool to calculate 5 + 7"
print(f"Query: {query}")

# Chạy một trace và xem kết quả
trace, rl_input, raw_memory = brain.agent_adapter.run(query)
print(f"\nTrace length: {len(trace)}")
for i, turn in enumerate(trace):
    print(f"Turn {i+1}:")
    print(f"  parsed_completion: {turn.get('parsed_completion')}")
    print(f"  tool_output: {turn.get('tool_output')}")
    print(f"  model_completion: {turn.get('model_completion')}")

print(f"\nRL Input segments: {len(rl_input)}")
for i, segment in enumerate(rl_input):
    print(f"Segment {i+1}: {type(segment)} -> Role: {segment['role']}, Text: {segment['text'][:100]}...")

# Test reward function
print(f"\n🎯 Testing reward function...")
try:
    reward = reward_exact_match(trace, query="Use the add tool to calculate 5 + 7", gold_answer="12")
    print(f"Reward for gold_answer='12': {reward}")
except Exception as e:
    print(f"Reward function error: {e}")