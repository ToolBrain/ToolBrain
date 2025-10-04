#!/usr/bin/env python3
"""
Test DIRECTLY the LangChain adapter without Brain wrapper to see raw format.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    print("🔍 DIRECT LANGCHAIN ADAPTER TEST")
    print("=" * 50)
    
    try:
        # Import dependencies
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from toolbrain.langchain_adapter_v2 import LangChainAdapter
        
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
        print("🔧 Creating LangChain agent...")
        langchain_agent = create_agent(
            model=trainable_llm,
            tools=tools,
            prompt="You are a helpful assistant. You must call a tool to answer the user.",
        )
        
        # Create adapter DIRECTLY (no Brain wrapper)
        print("🔧 Creating LangChain adapter directly...")
        adapter = LangChainAdapter(
            agent=langchain_agent,
            trainable_model=trainable_llm,
            config={"model_name": model_id},  # Simple config
            tools=tools
        )
        
        print("✅ Adapter created")
        
        # Test DIRECTLY calling adapter.run()
        test_query = "Calculate 5 + 7"
        print(f"\n🚀 Testing adapter.run() directly: '{test_query}'")
        
        # Call adapter's run method directly
        trace, rl_input, raw_memory = adapter.run(test_query)
        
        print(f"\n📊 RAW ADAPTER OUTPUT ANALYSIS")
        print(f"{'='*40}")
        print(f"Trace type: {type(trace)}")
        print(f"Trace length: {len(trace)}")
        print(f"RL input type: {type(rl_input)}")
        print(f"RL input length: {len(rl_input) if rl_input else 0}")
        print(f"Raw memory type: {type(raw_memory)}")
        print(f"Raw memory length: {len(raw_memory) if raw_memory else 0}")
        
        # Detailed trace analysis
        print(f"\n🔍 DETAILED TRACE FROM ADAPTER")
        print(f"{'='*40}")
        
        for i, turn in enumerate(trace):
            print(f"\n--- Turn {i + 1} ---")
            print(f"Type: {type(turn)}")
            
            if isinstance(turn, dict):
                print(f"Keys: {list(turn.keys())}")
                
                # Check Turn structure compliance
                required_turn_fields = [
                    "prompt_for_model", 
                    "model_completion", 
                    "parsed_completion"
                ]
                
                for field in required_turn_fields:
                    if field in turn:
                        print(f"  ✅ {field}: {type(turn[field])}")
                        if field == "parsed_completion" and isinstance(turn[field], dict):
                            parsed = turn[field]
                            parsed_fields = ["thought", "tool_code", "final_answer"]
                            for pf in parsed_fields:
                                if pf in parsed:
                                    value_preview = str(parsed[pf])[:50] + "..." if len(str(parsed[pf])) > 50 else str(parsed[pf])
                                    print(f"    ✅ {pf}: {value_preview}")
                                else:
                                    print(f"    ❌ Missing {pf}")
                    else:
                        print(f"  ❌ Missing {field}")
                
                # Check optional fields
                optional_fields = ["tool_output", "action_output", "formatted_conversation"]
                for field in optional_fields:
                    if field in turn:
                        value_preview = str(turn[field])[:50] + "..." if len(str(turn[field])) > 50 else str(turn[field])
                        print(f"  📝 {field}: {value_preview}")
                    else:
                        print(f"  ⚪ {field}: Not present")
            else:
                print(f"❌ Turn is not a dict: {turn}")
        
        # RL Input analysis
        print(f"\n🔍 DETAILED RL INPUT FROM ADAPTER")
        print(f"{'='*40}")
        
        if rl_input:
            for i, segment in enumerate(rl_input):
                print(f"\n--- Segment {i + 1} ---")
                print(f"Type: {type(segment)}")
                
                if isinstance(segment, dict):
                    print(f"Keys: {list(segment.keys())}")
                    
                    if "role" in segment and "text" in segment:
                        print(f"  ✅ role: '{segment['role']}'")
                        text_preview = segment['text'][:50] + "..." if len(segment['text']) > 50 else segment['text']
                        print(f"  ✅ text: '{text_preview}'")
                    else:
                        print(f"  ❌ Missing ChatSegment fields")
                else:
                    print(f"❌ Segment is not a dict: {segment}")
        else:
            print("❌ No RL input generated")
        
        # Final assessment
        print(f"\n🏥 DIRECT ADAPTER ASSESSMENT")
        print(f"{'='*40}")
        
        # Check if adapter returns proper Turn format
        turns_valid = all(
            isinstance(turn, dict) and
            "prompt_for_model" in turn and
            "model_completion" in turn and
            "parsed_completion" in turn
            for turn in trace
        )
        
        if turns_valid:
            print("✅ ADAPTER RETURNS VALID TURN FORMAT")
        else:
            print("❌ ADAPTER RETURNS INVALID TURN FORMAT")
        
        # Check if adapter returns proper ChatSegment format
        if rl_input:
            segments_valid = all(
                isinstance(segment, dict) and
                "role" in segment and
                "text" in segment
                for segment in rl_input
            )
            
            if segments_valid:
                print("✅ ADAPTER RETURNS VALID CHATSEGMENT FORMAT")
            else:
                print("❌ ADAPTER RETURNS INVALID CHATSEGMENT FORMAT")
        else:
            print("⚠️ ADAPTER RETURNS NO RL INPUT")
        
        print(f"\n🎯 CONCLUSION:")
        print(f"This test directly calls adapter.run() without Brain wrapper")
        print(f"Results show the RAW format from LangChain adapter")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()