# Test comprehensive để tìm ra vấn đề tool calling

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
import json

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

print("\n" + "="*60)
print("🧪 TEST 1: Direct model call without agent")
print("="*60)

# Test 1: Direct model call
try:
    response = trainable_llm.invoke([
        ("system", "You are a helpful assistant that can use tools."),
        ("user", "Calculate 5 + 7")
    ])
    print(f"Direct response: {response.content}")
except Exception as e:
    print(f"Direct call error: {e}")

print("\n" + "="*60)
print("🧪 TEST 2: Model with explicit tool binding")
print("="*60)

# Test 2: Bind tools explicitly
try:
    model_with_tools = trainable_llm.bind_tools([add])
    response = model_with_tools.invoke([
        ("system", "You have access to tools. Use the add tool for calculations."),
        ("user", "Use the add tool to calculate 5 + 7")
    ])
    print(f"Tool-bound response: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"🔧 Tool calls detected: {response.tool_calls}")
    else:
        print("❌ No tool calls in response")
except Exception as e:
    print(f"Tool binding error: {e}")

print("\n" + "="*60)
print("🧪 TEST 3: Standard create_agent approach")
print("="*60)

# Test 3: Standard create_agent approach
try:
    agent = create_agent(
        model=trainable_llm,
        tools=[add],
        prompt="You are a helpful assistant. Use the provided tools when needed."
    )
    
    result = agent.invoke({"messages": [("user", "Calculate 5 + 7")]})
    print(f"Standard agent result: {result}")
    
except Exception as e:
    print(f"Standard agent error: {e}")

print("\n" + "="*60)
print("🧪 TEST 4: Manual tool format")
print("="*60)

# Test 4: Manual tool calling format
try:
    # Format tool để model hiểu
    tool_prompt = """You are a helpful assistant with access to tools. 

Available tools:
- add(a: int, b: int): Adds two integers

To use a tool, respond in JSON format:
{"tool_call": {"name": "add", "arguments": {"a": 5, "b": 7}}}

User: Calculate 5 + 7"""
    
    response = trainable_llm.invoke([("user", tool_prompt)])
    print(f"Manual format response: {response.content}")
    
    # Try to parse JSON response
    try:
        if "{" in response.content and "}" in response.content:
            # Extract JSON part
            start = response.content.find("{")
            end = response.content.rfind("}") + 1
            json_part = response.content[start:end]
            parsed = json.loads(json_part)
            print(f"📋 Parsed JSON: {parsed}")
            
            if "tool_call" in parsed:
                tool_name = parsed["tool_call"]["name"]
                args = parsed["tool_call"]["arguments"]
                if tool_name == "add":
                    result = add(**args)
                    print(f"🔧 Tool executed: {result}")
        else:
            print("❌ No JSON format detected")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        
except Exception as e:
    print(f"Manual format error: {e}")

print("\n" + "="*60)
print("🧪 TEST 5: ReAct format")
print("="*60)

# Test 5: ReAct format (thinking step by step)
try:
    react_prompt = """You are a helpful assistant. Use the following format:

Thought: I need to calculate 5 + 7 using the add tool
Action: add
Action Input: {"a": 5, "b": 7}

User: Calculate 5 + 7"""
    
    response = trainable_llm.invoke([("user", react_prompt)])
    print(f"ReAct format response: {response.content}")
    
    # Check if it follows ReAct format
    if "Action:" in response.content and "add" in response.content:
        print("✅ ReAct format detected")
    else:
        print("❌ ReAct format not followed")
        
except Exception as e:
    print(f"ReAct format error: {e}")

print("\n" + "="*60)
print("🧪 TEST 6: Check model capabilities")
print("="*60)

# Test 6: Model capabilities
try:
    # Test if model understands tool format at all
    capability_test = """Can you understand this tool call format?
{"function_call": {"name": "add", "arguments": {"a": 5, "b": 7}}}

Please respond with 'YES' if you understand, or explain what you see."""
    
    response = trainable_llm.invoke([("user", capability_test)])
    print(f"Capability test: {response.content}")
    
except Exception as e:
    print(f"Capability test error: {e}")

print("\n" + "="*60)
print("🧪 TEST 7: Check if bind_tools works")
print("="*60)

# Test 7: Test bind_tools functionality
try:
    print("Testing if bind_tools method exists...")
    if hasattr(trainable_llm, 'bind_tools'):
        print("✅ bind_tools method exists")
        model_with_tools = trainable_llm.bind_tools([add])
        print("✅ Tools bound successfully")
        
        # Check what tools are available
        if hasattr(model_with_tools, 'kwargs') and 'tools' in model_with_tools.kwargs:
            print(f"📋 Tools in kwargs: {model_with_tools.kwargs['tools']}")
        else:
            print("❌ No tools found in model kwargs")
            
    else:
        print("❌ bind_tools method not available")
        
except Exception as e:
    print(f"bind_tools test error: {e}")

print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)
print("If all tests show no tool calling, the issue is likely:")
print("1. Model is not trained for tool calling")
print("2. ChatHuggingFace wrapper doesn't support tool format properly")
print("3. Need different model or approach")
print("4. HuggingFace models might need explicit tool calling training")