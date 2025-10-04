#!/usr/bin/env python3
"""
Simplified trace format inspection for LangChain adapter.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from toolbrain.brain import Brain
from toolbrain.rewards import reward_exact_match

def main():
    print("🔍 SIMPLE TRACE FORMAT INSPECTION")
    print("=" * 50)
    
    try:
        # Import LangChain dependencies
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        # Define tools
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b
        
        tools = [add]
        
        # Setup model
        print("📥 Setting up model...")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="cpu"
        )
        
        # Create pipeline
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            return_full_text=False
        )
        
        # Wrap pipeline
        hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
        trainable_llm = ChatHuggingFace(llm=hf_pipeline)
        
        # Create LangChain agent
        print("🔧 Creating agent...")
        langchain_agent = create_agent(
            model=trainable_llm,
            tools=tools,
            prompt="You are a helpful assistant. You must call a tool to answer the user.",
        )
        
        # Create brain
        brain = Brain(
            agent=langchain_agent,
            trainable_model=trainable_llm,
            algorithm="GRPO",
            reward_func=reward_exact_match
        )
        
        # Pass tools explicitly to adapter
        if hasattr(brain.agent_adapter, '__init__'):
            from toolbrain.langchain_adapter_v2 import LangChainAdapter
            brain.agent_adapter = LangChainAdapter(
                agent=langchain_agent,
                trainable_model=trainable_llm,
                config=brain.config.to_dict() if hasattr(brain.config, 'to_dict') else vars(brain.config),
                tools=tools
            )
        
        print("✅ Brain created")
        
        # Test trace generation
        test_query = "Calculate 2 + 3"
        print(f"\n🚀 Testing query: '{test_query}'")
        
        traces, rewards, rl_inputs = brain.get_trace(test_query, reward_kwargs={})
        
        print(f"\n📊 BRAIN GET_TRACE ANALYSIS")
        print(f"{'='*40}")
        print(f"Traces type: {type(traces)}")
        print(f"Traces length: {len(traces)}")
        print(f"Rewards type: {type(rewards)}")
        print(f"Rewards length: {len(rewards) if rewards else 0}")
        print(f"RL inputs type: {type(rl_inputs)}")
        print(f"RL inputs length: {len(rl_inputs) if rl_inputs else 0}")
        
        # Analyze first trace (assuming we have at least one)
        if traces and len(traces) > 0:
            trace = traces[0]  # Get first trace
            print(f"\n📊 FIRST TRACE ANALYSIS")
            print(f"{'='*30}")
            print(f"Trace type: {type(trace)}")
            print(f"Trace length: {len(trace)}")
        else:
            print("❌ No traces generated")
            return
        
        # Analyze each element
        for i, element in enumerate(trace):
            print(f"\nElement {i+1}:")
            print(f"  Type: {type(element)}")
            
            if isinstance(element, list):
                print(f"  List length: {len(element)}")
                for j, sub_element in enumerate(element):
                    print(f"    Sub-element {j+1}:")
                    print(f"      Type: {type(sub_element)}")
                    if isinstance(sub_element, dict):
                        print(f"      Keys: {list(sub_element.keys())}")
                        # Check for Turn structure compliance
                        required_fields = ["prompt_for_model", "model_completion", "parsed_completion", "tool_output"]
                        missing_fields = [field for field in required_fields if field not in sub_element]
                        if missing_fields:
                            print(f"      ❌ Missing Turn fields: {missing_fields}")
                        else:
                            print(f"      ✅ Has all core Turn fields")
                            
                        # Check parsed_completion structure
                        if "parsed_completion" in sub_element:
                            parsed = sub_element["parsed_completion"]
                            if isinstance(parsed, dict):
                                parsed_fields = ["thought", "tool_code", "final_answer"]
                                parsed_missing = [field for field in parsed_fields if field not in parsed]
                                if parsed_missing:
                                    print(f"      ❌ Missing ParsedCompletion fields: {parsed_missing}")
                                else:
                                    print(f"      ✅ Has all ParsedCompletion fields")
            elif isinstance(element, dict):
                print(f"  Dict keys: {list(element.keys())}")
                # Check for Turn structure compliance
                required_fields = ["prompt_for_model", "model_completion", "parsed_completion"]
                missing_fields = [field for field in required_fields if field not in element]
                if missing_fields:
                    print(f"  ❌ Missing Turn fields: {missing_fields}")
                else:
                    print(f"  ✅ Has all core Turn fields")
                    
                # Check parsed_completion structure
                if "parsed_completion" in element:
                    parsed = element["parsed_completion"]
                    if isinstance(parsed, dict):
                        parsed_fields = ["thought", "tool_code", "final_answer"]
                        parsed_missing = [field for field in parsed_fields if field not in parsed]
                        if parsed_missing:
                            print(f"  ❌ Missing ParsedCompletion fields: {parsed_missing}")
                        else:
                            print(f"  ✅ Has all ParsedCompletion fields")
        
        # Analyze RL input
        if rl_inputs and len(rl_inputs) > 0:
            rl_input = rl_inputs[0]  # Get first RL input
            print(f"\n📊 FIRST RL INPUT ANALYSIS")
            print(f"{'='*30}")
            print(f"RL input type: {type(rl_input)}")
            print(f"RL input length: {len(rl_input) if rl_input else 0}")
            
            if rl_input:
                for i, segment in enumerate(rl_input):
                    print(f"\nSegment {i+1}:")
                    print(f"  Type: {type(segment)}")
                    if isinstance(segment, dict):
                        print(f"  Keys: {list(segment.keys())}")
                        if "role" in segment and "text" in segment:
                            print(f"  ✅ Valid ChatSegment structure")
                            print(f"  Role: {segment['role']}")
                            print(f"  Text preview: {segment['text'][:50]}...")
                        else:
                            print(f"  ❌ Missing ChatSegment fields")
                    else:
                        print(f"  ❌ Not a dict: {segment}")
        else:
            print("❌ No RL inputs generated")
        
        # DIAGNOSIS AND RECOMMENDATIONS
        print(f"\n🏥 DIAGNOSIS & RECOMMENDATIONS")
        print(f"{'='*40}")
        
        # Check if trace follows Turn format
        trace_follows_format = True
        if not isinstance(trace, list):
            trace_follows_format = False
            print("❌ Trace is not a list")
        else:
            for i, element in enumerate(trace):
                if isinstance(element, list):
                    print(f"❌ Turn {i+1} is a list instead of a Turn dict")
                    trace_follows_format = False
                    # Check if the list contains Turn-like dicts
                    if len(element) > 0 and isinstance(element[0], dict):
                        print(f"  💡 Suggestion: Extract dict from list for Turn {i+1}")
                elif isinstance(element, dict):
                    required_fields = ["prompt_for_model", "model_completion", "parsed_completion"]
                    missing = [f for f in required_fields if f not in element]
                    if missing:
                        print(f"❌ Turn {i+1} missing fields: {missing}")
                        trace_follows_format = False
                    else:
                        print(f"✅ Turn {i+1} follows Turn format")
        
        if trace_follows_format:
            print("🎉 TRACE FORMAT: COMPLIANT")
        else:
            print("⚠️ TRACE FORMAT: NON-COMPLIANT")
            print("\n🔧 FIXING RECOMMENDATIONS:")
            print("1. Ensure langchain_adapter_v2.py returns List[Turn] not List[List[Turn]]")
            print("2. Each Turn should be a dict with required fields")
            print("3. Check the run() method in LangChainAdapter")
        
        # Check ChatSegment format
        if rl_inputs and len(rl_inputs) > 0 and rl_inputs[0]:
            rl_input = rl_inputs[0]
            segment_valid = all(
                isinstance(s, dict) and "role" in s and "text" in s 
                for s in rl_input
            )
            if segment_valid:
                print("🎉 CHATSEGMENT FORMAT: COMPLIANT")
            else:
                print("⚠️ CHATSEGMENT FORMAT: NON-COMPLIANT")
        else:
            print("⚠️ NO RL INPUT GENERATED")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()