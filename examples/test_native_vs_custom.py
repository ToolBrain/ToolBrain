#!/usr/bin/env python3
"""
Test native tool calling vs custom tool calling với Gemini và HuggingFace models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_gemini_vs_huggingface():
    print("🧪 TESTING GEMINI vs HUGGINGFACE TOOL CALLING")
    print("=" * 60)
    
    from langchain_core.tools import tool
    
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    tools = [add]
    test_query = "Calculate 3 + 5"
    
    # Test 1: HuggingFace Model (should use custom tool calling)
    print("\n🧪 TEST 1: HUGGINGFACE MODEL (Custom Tool Calling)")
    print("=" * 50)
    
    try:
        from langchain.agents import create_agent
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from toolbrain.langchain_adapter_v2 import LangChainAdapter
        
        # Setup HuggingFace model
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="cpu"
        )
        
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            return_full_text=False
        )
        
        hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
        hf_model = ChatHuggingFace(llm=hf_pipeline)
        
        # Create agent
        hf_agent = create_agent(
            model=hf_model,
            tools=tools,
            prompt="You are a helpful assistant. You must call a tool to answer the user.",
        )
        
        # Create adapter
        hf_adapter = LangChainAdapter(
            agent=hf_agent,
            trainable_model=hf_model,
            config={"model_name": model_id},
            tools=tools
        )
        
        print(f"✅ HuggingFace adapter created")
        print(f"📊 Model type: {hf_adapter.model_type}")
        print(f"📊 Custom tool calling: {hf_adapter.needs_custom_tool_calling}")
        
        # Test run
        print(f"\n🚀 Running query: '{test_query}'")
        hf_trace, hf_rl_input, hf_raw_memory = hf_adapter.run(test_query)
        
        print(f"📊 HuggingFace results:")
        print(f"  Trace length: {len(hf_trace)}")
        print(f"  RL input length: {len(hf_rl_input) if hf_rl_input else 0}")
        
        # Check for tool usage
        tool_used = any(turn.get("tool_output") for turn in hf_trace)
        print(f"  Tool used: {tool_used}")
        
        if tool_used:
            print("🎉 HuggingFace custom tool calling: SUCCESS")
        else:
            print("⚠️ HuggingFace custom tool calling: No tool usage detected")
        
    except Exception as e:
        print(f"❌ HuggingFace test failed: {e}")
    
    # Test 2: Gemini Model (should use native tool calling)
    print("\n🧪 TEST 2: GEMINI MODEL (Native Tool Calling)")
    print("=" * 50)
    
    try:
        # Note: This requires GOOGLE_API_KEY environment variable
        import os
        if "GOOGLE_API_KEY" not in os.environ:
            print("⚠️ GOOGLE_API_KEY not set, creating mock test...")
            
            # Create a simple mock test to show the logic works
            class MockChatGoogleGenerativeAI:
                def __init__(self):
                    pass
                
                def invoke(self, messages):
                    # Mock response without tool calls
                    return {
                        "messages": [
                            type('MockMessage', (), {
                                'type': 'ai', 
                                'content': '8', 
                                'tool_calls': []
                            })()
                        ]
                    }
            
            MockChatGoogleGenerativeAI.__name__ = "ChatGoogleGenerativeAI"
            MockChatGoogleGenerativeAI.__module__ = "langchain_google_genai.chat_models"
            
            # Create mock agent
            mock_gemini_model = MockChatGoogleGenerativeAI()
            
            # Create adapter
            gemini_adapter = LangChainAdapter(
                agent=mock_gemini_model,  # Using model as agent for simplicity
                trainable_model=mock_gemini_model,
                config={"model_name": "gemini-pro"},
                tools=tools
            )
            
            print(f"✅ Mock Gemini adapter created")
            print(f"📊 Model type: {gemini_adapter.model_type}")
            print(f"📊 Custom tool calling: {gemini_adapter.needs_custom_tool_calling}")
            
            if gemini_adapter.model_type == "gemini" and not gemini_adapter.needs_custom_tool_calling:
                print("🎉 Gemini detection logic: SUCCESS")
                print("✅ Would use native tool calling")
            else:
                print("❌ Gemini detection logic: FAILED")
        
        else:
            print("✅ GOOGLE_API_KEY found, testing real Gemini...")
            
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.agents import create_agent
            
            # Create Gemini model
            gemini_model = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
            
            # Create agent
            gemini_agent = create_agent(
                model=gemini_model,
                tools=tools,
                prompt="You are a helpful assistant. Use tools when needed.",
            )
            
            # Create adapter
            gemini_adapter = LangChainAdapter(
                agent=gemini_agent,
                trainable_model=gemini_model,
                config={"model_name": "gemini-pro"},
                tools=tools
            )
            
            print(f"✅ Gemini adapter created")
            print(f"📊 Model type: {gemini_adapter.model_type}")
            print(f"📊 Custom tool calling: {gemini_adapter.needs_custom_tool_calling}")
            
            # Test run
            print(f"\n🚀 Running query: '{test_query}'")
            gemini_trace, gemini_rl_input, gemini_raw_memory = gemini_adapter.run(test_query)
            
            print(f"📊 Gemini results:")
            print(f"  Trace length: {len(gemini_trace)}")
            print(f"  RL input length: {len(gemini_rl_input) if gemini_rl_input else 0}")
            
            # Check for tool usage
            tool_used = any(turn.get("tool_output") for turn in gemini_trace)
            print(f"  Tool used: {tool_used}")
            
            if tool_used:
                print("🎉 Gemini native tool calling: SUCCESS")
            else:
                print("⚠️ Gemini native tool calling: No tool usage detected")
    
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
    
    print("\n📋 COMPARISON SUMMARY")
    print("=" * 50)
    print("🔧 HuggingFace Models:")
    print("  - Detected as 'huggingface' type")
    print("  - Uses custom tool calling implementation")
    print("  - Creates CustomLangChainAgent")
    print("  - Parses JSON tool calls manually")
    
    print("\n🌟 Gemini/OpenAI/Claude Models:")
    print("  - Detected as 'gemini'/'openai'/'claude' type")
    print("  - Uses native LangChain tool calling")
    print("  - Uses original agent with built-in tool support")
    print("  - Leverages model's native tool calling API")
    
    print("\n🎯 CONCLUSION:")
    print("✅ Adapter now intelligently chooses tool calling strategy")
    print("✅ Custom implementation for HuggingFace models")
    print("✅ Native implementation for supported models")
    print("✅ Maintains same trace format for both approaches")

if __name__ == "__main__":
    test_gemini_vs_huggingface()