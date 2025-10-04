# Test trực tiếp xem agent có gọi tool không

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    result = a + b
    print(f"🔧 TOOL WAS CALLED! add({a}, {b}) = {result}")
    return result

print("📥 Initializing model...")
text_gen_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    return_full_text=False
)

hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
trainable_llm = ChatHuggingFace(llm=hf_pipeline)

print("🔧 Creating agent...")
agent = create_agent(
    model=trainable_llm,
    tools=[add],
    prompt="You are a helpful assistant. You must call the add tool to perform addition.",
)

print("🚀 Testing direct invocation...")
query = "Calculate 5 + 7"
print(f"Query: {query}")

# Test với invoke
result = agent.invoke({"messages": [("user", query)]})
print(f"\n📋 Messages received: {len(result['messages'])}")

for i, msg in enumerate(result['messages']):
    print(f"\nMessage {i+1}: {type(msg).__name__}")
    print(f"  Content: {msg.content}")
    
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"  🔧 Tool calls: {msg.tool_calls}")
    else:
        print(f"  ❌ No tool calls")

print("\n" + "="*60)
print("🔍 Testing with stream to see intermediate steps...")

# Test với stream để xem các bước trung gian
stream = agent.stream(
    {"messages": [("user", query)]},
    stream_mode="updates",
)

step = 0
for chunk in stream:
    step += 1
    print(f"\n--- Stream Step {step} ---")
    print(f"Chunk keys: {list(chunk.keys())}")
    
    if "agent" in chunk:
        msgs = chunk["agent"]["messages"]
        last_msg = msgs[-1]
        print(f"Agent message type: {type(last_msg).__name__}")
        print(f"Content: {last_msg.content}")
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            print(f"🔧 Tool calls detected: {last_msg.tool_calls}")
        else:
            print("❌ No tool calls in agent message")
            
    if "tools" in chunk:
        print("🔧 Tools step detected!")
        msgs = chunk["tools"]["messages"]
        tool_msg = msgs[-1]
        print(f"Tool result: {tool_msg.content}")
        
print(f"\nTotal stream steps: {step}")
print("If step count is 1 and no tool calls detected, the model is not calling tools properly.")