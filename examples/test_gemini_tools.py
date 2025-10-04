# Test LangChain với Gemini API để so sánh với HuggingFace models

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    result = a + b
    print(f"🔧 TOOL EXECUTED: add({a}, {b}) = {result}")
    return result

@tool  
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    result = a * b
    print(f"🔧 TOOL EXECUTED: multiply({a}, {b}) = {result}")
    return result

# Check for API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("❌ GEMINI_API_KEY not found in .env file!")
    print("Make sure you have GEMINI_API_KEY in your .env file")
    sys.exit(1)

print(f"✅ Found API key: {gemini_api_key[:10]}...{gemini_api_key[-4:]}")

print("📥 Initializing Gemini model...")
try:
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=gemini_api_key
    )
    print("✅ Gemini model initialized")
except Exception as e:
    print(f"❌ Error initializing Gemini: {e}")
    print("Make sure you have: pip install langchain-google-genai")
    sys.exit(1)

print("\n" + "="*60)
print("🧪 TEST 1: Direct Gemini call")
print("="*60)

try:
    response = gemini_model.invoke([
        ("system", "You are a helpful assistant."),
        ("user", "Calculate 5 + 7")
    ])
    print(f"Direct response: {response.content}")
except Exception as e:
    print(f"Direct call error: {e}")

print("\n" + "="*60)
print("🧪 TEST 2: Gemini with bind_tools")
print("="*60)

try:
    model_with_tools = gemini_model.bind_tools([add, multiply])
    response = model_with_tools.invoke([
        ("system", "You have access to tools. Use them when needed."),
        ("user", "Use the add tool to calculate 5 + 7")
    ])
    print(f"Tool-bound response: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"🔧 Tool calls detected: {response.tool_calls}")
        
        # Execute the tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name == "add":
                result = add.invoke(tool_args)
                print(f"🔧 Tool result: {result}")
    else:
        print("❌ No tool calls in response")
except Exception as e:
    print(f"Tool binding error: {e}")

print("\n" + "="*60)
print("🧪 TEST 3: Gemini with create_agent")
print("="*60)

try:
    gemini_agent = create_agent(
        model=gemini_model,
        tools=[add, multiply],
        prompt="You are a helpful assistant. Use the provided tools when needed."
    )
    
    result = gemini_agent.invoke({"messages": [("user", "Calculate 5 + 7")]})
    print(f"Agent result: {result}")
    
    # Check messages for tool calls
    for i, msg in enumerate(result['messages']):
        print(f"\nMessage {i+1}: {type(msg).__name__}")
        if hasattr(msg, 'content'):
            print(f"  Content: {msg.content}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  🔧 Tool calls: {msg.tool_calls}")
            
except Exception as e:
    print(f"Gemini agent error: {e}")

print("\n" + "="*60)
print("🧪 TEST 4: Gemini with stream")
print("="*60)

try:
    stream = gemini_agent.stream(
        {"messages": [("user", "Calculate 8 * 6")]},
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
            if hasattr(last_msg, 'content'):
                print(f"Content: {last_msg.content}")
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"🔧 Tool calls detected: {last_msg.tool_calls}")
                
        if "tools" in chunk:
            print("🔧 Tools step detected!")
            msgs = chunk["tools"]["messages"]
            tool_msg = msgs[-1]
            print(f"Tool result: {tool_msg.content}")
            
    print(f"\nTotal stream steps: {step}")
    
except Exception as e:
    print(f"Gemini stream error: {e}")

print("\n" + "="*60)
print("📋 COMPARISON SUMMARY")
print("="*60)
print("Compare Gemini results with HuggingFace results:")
print("- If Gemini works with tool calling → Issue is HuggingFace wrapper")
print("- If Gemini also doesn't work → Issue is LangChain setup")
print("- If Gemini stream shows multiple steps → Tool calling works properly")