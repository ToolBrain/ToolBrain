#!/usr/bin/env python3
"""
Test script to verify LangChain adapter trace format compliance.

This script tests whether the LangChain adapter generates traces that:
1. Follow the Turn structure defined in core_types.py
2. Generate proper ChatSegment format for RL training
3. Include all required fields for rewards and logging
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Dict, Any
import json
from toolbrain.core_types import Turn, ParsedCompletion, ChatSegment, Trace
from toolbrain.brain import Brain
from toolbrain.rewards import reward_exact_match


def validate_turn_structure(turn: Dict[str, Any], turn_index: int) -> List[str]:
    """Validate a single turn against the Turn TypedDict structure."""
    errors = []
    
    print(f"🔍 Validating Turn {turn_index + 1}...")
    
    # Required fields in Turn
    required_fields = [
        "prompt_for_model",
        "model_completion", 
        "parsed_completion",
        "tool_output",
        "action_output",
        "formatted_conversation"
    ]
    
    # Check required fields exist
    for field in required_fields:
        if field not in turn:
            errors.append(f"Turn {turn_index + 1}: Missing required field '{field}'")
        else:
            print(f"  ✅ Field '{field}': {type(turn[field])}")
    
    # Validate parsed_completion structure
    if "parsed_completion" in turn:
        parsed = turn["parsed_completion"]
        if not isinstance(parsed, dict):
            errors.append(f"Turn {turn_index + 1}: 'parsed_completion' must be a dict")
        else:
            required_parsed_fields = ["thought", "tool_code", "final_answer"]
            for field in required_parsed_fields:
                if field not in parsed:
                    errors.append(f"Turn {turn_index + 1}: Missing '{field}' in parsed_completion")
                else:
                    print(f"    ✅ ParsedCompletion.{field}: {type(parsed[field])}")
    
    # Validate field types
    type_checks = [
        ("prompt_for_model", str),
        ("model_completion", str),
        ("tool_output", (str, type(None))),
        ("action_output", (str, type(None))),
        ("formatted_conversation", (str, type(None)))
    ]
    
    for field, expected_type in type_checks:
        if field in turn:
            if not isinstance(turn[field], expected_type):
                errors.append(f"Turn {turn_index + 1}: '{field}' should be {expected_type}, got {type(turn[field])}")
    
    return errors


def validate_trace_structure(trace: List[Dict[str, Any]]) -> List[str]:
    """Validate the entire trace structure."""
    errors = []
    
    print(f"\n📊 TRACE STRUCTURE VALIDATION")
    print(f"{'='*50}")
    print(f"🔢 Total turns: {len(trace)}")
    
    if not isinstance(trace, list):
        errors.append("Trace must be a list")
        return errors
    
    if len(trace) == 0:
        errors.append("Trace cannot be empty")
        return errors
    
    # Validate each turn
    for i, turn in enumerate(trace):
        if not isinstance(turn, dict):
            errors.append(f"Turn {i + 1} must be a dict, got {type(turn)}")
            continue
        
        turn_errors = validate_turn_structure(turn, i)
        errors.extend(turn_errors)
    
    return errors


def validate_chat_segments(segments: List[Dict[str, Any]]) -> List[str]:
    """Validate ChatSegment format for RL training."""
    errors = []
    
    print(f"\n🗨️ CHAT SEGMENT VALIDATION")
    print(f"{'='*50}")
    print(f"🔢 Total segments: {len(segments)}")
    
    if not isinstance(segments, list):
        errors.append("ChatSegments must be a list")
        return errors
    
    for i, segment in enumerate(segments):
        print(f"🔍 Validating Segment {i + 1}...")
        
        if not isinstance(segment, dict):
            errors.append(f"Segment {i + 1} must be a dict, got {type(segment)}")
            continue
        
        # Check required fields
        if "role" not in segment:
            errors.append(f"Segment {i + 1}: Missing 'role' field")
        elif not isinstance(segment["role"], str):
            errors.append(f"Segment {i + 1}: 'role' must be string, got {type(segment['role'])}")
        else:
            print(f"  ✅ Role: '{segment['role']}'")
        
        if "text" not in segment:
            errors.append(f"Segment {i + 1}: Missing 'text' field")
        elif not isinstance(segment["text"], str):
            errors.append(f"Segment {i + 1}: 'text' must be string, got {type(segment['text'])}")
        else:
            print(f"  ✅ Text length: {len(segment['text'])} chars")
    
    return errors


def test_reward_function_compatibility(trace: List[Dict[str, Any]]) -> List[str]:
    """Test if trace is compatible with reward functions."""
    errors = []
    
    print(f"\n🎯 REWARD FUNCTION COMPATIBILITY")
    print(f"{'='*50}")
    
    # Test that reward functions can access required fields
    for i, turn in enumerate(trace):
        print(f"🔍 Testing Turn {i + 1} for reward compatibility...")
        
        # Check if turn has tool_output or action_output for reward calculation
        has_tool_output = turn.get("tool_output") is not None
        has_action_output = turn.get("action_output") is not None
        
        print(f"  📝 tool_output present: {has_tool_output}")
        print(f"  📝 action_output present: {has_action_output}")
        
        # Check if final_answer is accessible
        parsed_completion = turn.get("parsed_completion", {})
        has_final_answer = parsed_completion.get("final_answer") is not None
        
        print(f"  📝 final_answer present: {has_final_answer}")
        
        # For tool turns, we should have either tool_output or action_output
        has_tool_code = parsed_completion.get("tool_code") is not None
        if has_tool_code and not (has_tool_output or has_action_output):
            errors.append(f"Turn {i + 1}: Has tool_code but no tool_output or action_output")
    
    return errors


def main():
    """Run comprehensive trace format compliance test."""
    print("🧪 TESTING LANGCHAIN ADAPTER TRACE FORMAT COMPLIANCE")
    print("=" * 60)
    
    # Test configuration
    test_query = "Calculate 2 + 3 using the add tool"
    
    # Setup brain with LangChain agent
    print("📥 Setting up ToolBrain with LangChain agent...")
    
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
        print("📥 Initializing HuggingFace model...")
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
        
        # Create brain - sử dụng cách tương tự như ví dụ 11
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
        print("✅ Brain created successfully")
        
        # Generate trace
        print(f"\n🚀 Running test query: '{test_query}'")
        trace, rl_input, raw_memory = brain.get_trace(test_query, reward_kwargs={})
        
        print(f"📊 Generated trace: {len(trace)} turns")
        print(f"📊 Generated RL input: {len(rl_input)} segments")
        
        # Detailed trace inspection
        print(f"\n🔍 DETAILED TRACE INSPECTION")
        print(f"{'='*50}")
        
        print(f"📊 Trace type: {type(trace)}")
        print(f"📊 Trace length: {len(trace)}")
        
        for i, turn in enumerate(trace):
            print(f"\n--- Turn {i + 1} ---")
            print(f"� Turn type: {type(turn)}")
            
            if isinstance(turn, dict):
                print(f"�🔑 Keys: {list(turn.keys())}")
                for key, value in turn.items():
                    if isinstance(value, str):
                        display_value = value[:100] + "..." if len(str(value)) > 100 else value
                    elif isinstance(value, dict):
                        display_value = f"dict with keys: {list(value.keys())}"
                    else:
                        display_value = str(value)
                    print(f"  {key}: {display_value}")
            else:
                print(f"⚠️ Turn is not a dict: {turn}")
        
        # Run validation tests
        all_errors = []
        
        # Test 1: Trace structure compliance
        trace_errors = validate_trace_structure(trace)
        all_errors.extend(trace_errors)
        
        # Test 2: ChatSegment format
        if rl_input:
            segment_errors = validate_chat_segments(rl_input)
            all_errors.extend(segment_errors)
        else:
            all_errors.append("No RL input (ChatSegments) generated")
        
        # Test 3: Reward function compatibility
        reward_errors = test_reward_function_compatibility(trace)
        all_errors.extend(reward_errors)
        
        # Report results
        print(f"\n📋 FINAL COMPLIANCE REPORT")
        print(f"{'='*50}")
        
        if all_errors:
            print(f"❌ Found {len(all_errors)} compliance issues:")
            for i, error in enumerate(all_errors, 1):
                print(f"  {i}. {error}")
            
            print(f"\n🔧 RECOMMENDATIONS:")
            print(f"  - Check Turn structure in langchain_adapter_v2.py")
            print(f"  - Ensure all required fields are populated")
            print(f"  - Verify ChatSegment generation logic")
            print(f"  - Test reward function access to tool outputs")
        else:
            print(f"✅ ALL TESTS PASSED!")
            print(f"🎉 Trace format is fully compliant with ToolBrain requirements!")
            
            print(f"\n📊 COMPLIANCE SUMMARY:")
            print(f"  ✅ Turn structure: COMPLIANT")
            print(f"  ✅ Required fields: PRESENT")
            print(f"  ✅ ChatSegment format: VALID")
            print(f"  ✅ Reward compatibility: READY")
        
        # Additional metrics
        print(f"\n📈 TRACE METRICS:")
        print(f"  🔢 Total turns: {len(trace)}")
        print(f"  🔢 Total chat segments: {len(rl_input) if rl_input else 0}")
        
        tool_turns = sum(1 for turn in trace if turn.get("parsed_completion", {}).get("tool_code"))
        print(f"  🛠️ Tool-using turns: {tool_turns}")
        
        non_null_outputs = sum(1 for turn in trace if turn.get("tool_output") or turn.get("action_output"))
        print(f"  📤 Turns with outputs: {non_null_outputs}")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()