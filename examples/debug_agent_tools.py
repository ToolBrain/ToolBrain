# Debug script để xem tools trong LangChain agent

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

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
langchain_agent = create_agent(
    model=trainable_llm,
    tools=[add],
    prompt="You are a helpful assistant. You must call a tool to answer the user.",
)

print("\n🔍 Inspecting agent structure...")
print(f"Agent type: {type(langchain_agent)}")
print(f"Agent attributes: {[attr for attr in dir(langchain_agent) if not attr.startswith('_')]}")

# Check for tools
if hasattr(langchain_agent, 'tools'):
    print(f"✅ Agent.tools found: {langchain_agent.tools}")
    print(f"Tools type: {type(langchain_agent.tools)}")
    if isinstance(langchain_agent.tools, dict):
        for name, tool in langchain_agent.tools.items():
            print(f"  Tool: {name} -> {tool}")
    elif isinstance(langchain_agent.tools, list):
        for i, tool in enumerate(langchain_agent.tools):
            print(f"  Tool {i}: {tool}")
else:
    print("❌ No tools attribute found")

# Check graph structure
if hasattr(langchain_agent, 'get_graph'):
    print(f"\n📊 Graph structure:")
    graph = langchain_agent.get_graph()
    print(f"Graph nodes: {list(graph.nodes.keys())}")
    
    for node_id, node_data in graph.nodes.items():
        print(f"  Node {node_id}: {type(node_data)}")
        if hasattr(node_data, '__dict__'):
            attrs = [attr for attr in dir(node_data) if not attr.startswith('_')]
            print(f"    Attributes: {attrs}")
            
            # Look for tools in node
            if hasattr(node_data, 'bound'):
                print(f"    Bound: {node_data.bound}")
            if hasattr(node_data, 'tools'):
                print(f"    Tools: {node_data.tools}")

# Check edges
if hasattr(langchain_agent, 'get_graph'):
    edges = graph.edges
    print(f"  Graph edges: {list(edges)}")

print("\n🔧 Manual tools access test:")
print(f"Original tools passed: {[add]}")
print(f"Add tool name: {add.name}")
print(f"Add tool description: {add.description}")
print(f"Add tool type: {type(add)}")