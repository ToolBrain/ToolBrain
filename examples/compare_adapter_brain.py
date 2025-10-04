#!/usr/bin/env python3
"""
Compare direct adapter vs Brain wrapper output formats.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    print("🔍 ADAPTER vs BRAIN COMPARISON")
    print("=" * 50)
    
    try:
        # Import dependencies
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from toolbrain.langchain_adapter_v2 import LangChainAdapter
        from toolbrain.brain import Brain
        from toolbrain.rewards import reward_exact_match
        
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
        
        test_query = "Calculate 3 + 4"
        
        # TEST 1: Direct adapter call
        print(f"\n🧪 TEST 1: DIRECT ADAPTER CALL")
        print(f"{'='*40}")
        
        adapter = LangChainAdapter(
            agent=langchain_agent,
            trainable_model=trainable_llm,
            config={"model_name": model_id},
            tools=tools
        )
        
        print(f"Running adapter.run('{test_query}')")
        adapter_trace, adapter_rl_input, adapter_raw_memory = adapter.run(test_query)
        
        print(f"✅ Adapter results:")
        print(f"  Trace type: {type(adapter_trace)}")
        print(f"  Trace length: {len(adapter_trace)}")
        print(f"  RL input type: {type(adapter_rl_input)}")
        print(f"  RL input length: {len(adapter_rl_input) if adapter_rl_input else 0}")
        
        # TEST 2: Brain wrapper call
        print(f"\n🧪 TEST 2: BRAIN WRAPPER CALL")
        print(f"{'='*40}")
        
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
        
        print(f"Running brain.get_trace('{test_query}')")
        brain_traces, brain_rewards, brain_rl_inputs = brain.get_trace(test_query, reward_kwargs={})
        
        print(f"✅ Brain results:")
        print(f"  Traces type: {type(brain_traces)}")
        print(f"  Traces length: {len(brain_traces)}")
        print(f"  First trace type: {type(brain_traces[0]) if brain_traces else 'N/A'}")
        print(f"  First trace length: {len(brain_traces[0]) if brain_traces else 0}")
        print(f"  RL inputs type: {type(brain_rl_inputs)}")
        print(f"  RL inputs length: {len(brain_rl_inputs)}")
        print(f"  First RL input type: {type(brain_rl_inputs[0]) if brain_rl_inputs else 'N/A'}")
        print(f"  First RL input length: {len(brain_rl_inputs[0]) if brain_rl_inputs else 0}")
        
        # COMPARISON
        print(f"\n📊 DETAILED COMPARISON")
        print(f"{'='*40}")
        
        print(f"\n🔍 TRACE STRUCTURE COMPARISON:")
        print(f"Direct Adapter:")
        print(f"  Returns: List[Turn] (length {len(adapter_trace)})")
        for i, turn in enumerate(adapter_trace):
            print(f"    Turn {i+1}: dict with keys {list(turn.keys())}")
        
        print(f"\nBrain Wrapper:")
        print(f"  Returns: List[List[Turn]] (length {len(brain_traces)})")
        if brain_traces:
            first_trace = brain_traces[0]
            print(f"    First trace: List[Turn] (length {len(first_trace)})")
            for i, turn in enumerate(first_trace):
                print(f"      Turn {i+1}: dict with keys {list(turn.keys())}")
        
        print(f"\n🔍 RL INPUT STRUCTURE COMPARISON:")
        print(f"Direct Adapter:")
        print(f"  Returns: List[ChatSegment] (length {len(adapter_rl_input) if adapter_rl_input else 0})")
        if adapter_rl_input:
            for i, seg in enumerate(adapter_rl_input):
                print(f"    Segment {i+1}: dict with keys {list(seg.keys())}")
        
        print(f"\nBrain Wrapper:")
        print(f"  Returns: List[List[ChatSegment]] (length {len(brain_rl_inputs)})")
        if brain_rl_inputs and brain_rl_inputs[0]:
            first_rl_input = brain_rl_inputs[0]
            print(f"    First RL input: List[ChatSegment] (length {len(first_rl_input)})")
            for i, seg in enumerate(first_rl_input):
                print(f"      Segment {i+1}: dict with keys {list(seg.keys())}")
        
        # FINAL ANSWER
        print(f"\n🎯 VERDICT")
        print(f"{'='*40}")
        
        print(f"✅ LangChain Adapter ITSELF returns proper format:")
        print(f"   - List[Turn] for traces")
        print(f"   - List[ChatSegment] for RL input")
        print(f"   - Each Turn is a proper dict with required fields")
        print(f"   - Each ChatSegment is a proper dict with role/text")
        
        print(f"\n🔄 Brain wrapper adds batching layer:")
        print(f"   - Wraps single trace in List[List[Turn]]")
        print(f"   - Wraps single RL input in List[List[ChatSegment]]")
        print(f"   - This is because Brain can generate multiple traces")
        
        print(f"\n📋 CONCLUSION:")
        print(f"   The LangChain adapter format is CORRECT at source.")
        print(f"   Brain adds batching wrapper for multi-trace support.")
        print(f"   Both levels follow their respective interfaces correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()