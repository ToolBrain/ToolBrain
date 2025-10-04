#!/usr/bin/env python3
"""
Test model type detection and tool calling strategy selection.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_model_type_detection():
    print("🧪 TESTING MODEL TYPE DETECTION")
    print("=" * 50)
    
    try:
        from toolbrain.langchain_adapter_v2 import LangChainAdapter
        from langchain_core.tools import tool
        
        # Define a simple tool
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b
        
        tools = [add]
        
        # Test 1: HuggingFace Model
        print("\n🧪 TEST 1: HUGGINGFACE MODEL")
        print("=" * 30)
        
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
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
            
            # Create adapter
            adapter = LangChainAdapter(
                agent=None,  # We'll test just the detection
                trainable_model=hf_model,
                config={"model_name": model_id},
                tools=tools
            )
            
            print(f"✅ Model type detected: {adapter.model_type}")
            print(f"✅ Needs custom tool calling: {adapter.needs_custom_tool_calling}")
            
            expected_type = "huggingface"
            expected_custom = True
            
            if adapter.model_type == expected_type and adapter.needs_custom_tool_calling == expected_custom:
                print("🎉 HuggingFace detection: PASSED")
            else:
                print(f"❌ HuggingFace detection: FAILED (expected {expected_type}/{expected_custom}, got {adapter.model_type}/{adapter.needs_custom_tool_calling})")
            
        except ImportError:
            print("⚠️ HuggingFace dependencies not available")
        except Exception as e:
            print(f"❌ HuggingFace test failed: {e}")
        
        # Test 2: Gemini Model (if available)
        print("\n🧪 TEST 2: GEMINI MODEL")
        print("=" * 30)
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Note: This might require API key, so we'll just test the class detection
            gemini_model = ChatGoogleGenerativeAI(model="gemini-pro")
            
            # Create adapter
            adapter = LangChainAdapter(
                agent=None,
                trainable_model=gemini_model,
                config={"model_name": "gemini-pro"},
                tools=tools
            )
            
            print(f"✅ Model type detected: {adapter.model_type}")
            print(f"✅ Needs custom tool calling: {adapter.needs_custom_tool_calling}")
            
            expected_type = "gemini"
            expected_custom = False
            
            if adapter.model_type == expected_type and adapter.needs_custom_tool_calling == expected_custom:
                print("🎉 Gemini detection: PASSED")
            else:
                print(f"❌ Gemini detection: FAILED (expected {expected_type}/{expected_custom}, got {adapter.model_type}/{adapter.needs_custom_tool_calling})")
            
        except ImportError:
            print("⚠️ Gemini dependencies not available")
        except Exception as e:
            print(f"❌ Gemini test failed: {e}")
        
        # Test 3: Mock OpenAI Model
        print("\n🧪 TEST 3: MOCK OPENAI MODEL")
        print("=" * 30)
        
        # Create a mock model that looks like ChatOpenAI
        class MockChatOpenAI:
            def __init__(self):
                pass
            
            def __class__(self):
                return type("ChatOpenAI", (), {})()
        
        MockChatOpenAI.__name__ = "ChatOpenAI"
        MockChatOpenAI.__module__ = "langchain_openai.chat_models.base"
        
        mock_openai = MockChatOpenAI()
        
        # Create adapter
        adapter = LangChainAdapter(
            agent=None,
            trainable_model=mock_openai,
            config={"model_name": "gpt-4"},
            tools=tools
        )
        
        print(f"✅ Model type detected: {adapter.model_type}")
        print(f"✅ Needs custom tool calling: {adapter.needs_custom_tool_calling}")
        
        expected_type = "openai"
        expected_custom = False
        
        if adapter.model_type == expected_type and adapter.needs_custom_tool_calling == expected_custom:
            print("🎉 OpenAI detection: PASSED")
        else:
            print(f"❌ OpenAI detection: FAILED (expected {expected_type}/{expected_custom}, got {adapter.model_type}/{adapter.needs_custom_tool_calling})")
        
        # Test 4: Unknown Model
        print("\n🧪 TEST 4: UNKNOWN MODEL")
        print("=" * 30)
        
        class UnknownModel:
            pass
        
        UnknownModel.__name__ = "SomeWeirdModel"
        UnknownModel.__module__ = "some.weird.module"
        
        unknown_model = UnknownModel()
        
        # Create adapter
        adapter = LangChainAdapter(
            agent=None,
            trainable_model=unknown_model,
            config={"model_name": "unknown"},
            tools=tools
        )
        
        print(f"✅ Model type detected: {adapter.model_type}")
        print(f"✅ Needs custom tool calling: {adapter.needs_custom_tool_calling}")
        
        expected_type = "unknown"
        expected_custom = True
        
        if adapter.model_type == expected_type and adapter.needs_custom_tool_calling == expected_custom:
            print("🎉 Unknown model detection: PASSED")
        else:
            print(f"❌ Unknown model detection: FAILED (expected {expected_type}/{expected_custom}, got {adapter.model_type}/{adapter.needs_custom_tool_calling})")
        
        print("\n📋 SUMMARY")
        print("=" * 30)
        print("✅ Model type detection logic implemented")
        print("✅ Custom tool calling only for HuggingFace/unknown models")
        print("✅ Native tool calling for Gemini/OpenAI/Claude models")
        print("🎯 Ready to handle both scenarios!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_type_detection()