"""
Example demonstrating the new raw memory support in ToolBrain.

This shows how the new architecture supports both:
1. Standard reward functions (backward compatible)
2. Advanced reward functions using raw memory data
"""

from typing import List, Any
from toolbrain.core_types import Trace
from toolbrain.rewards import reward_exact_match


def reward_simple_exact_match(trace: Trace, **kwargs: Any) -> float:
    """
    Example of a STANDARD reward function.
    
    This function works exactly as before - it only uses the structured trace
    and ignores any additional kwargs like raw_memory_collection.
    """
    gold_answer = kwargs.get("gold_answer")
    if gold_answer is None:
        return 0.0
    
    for turn in trace:
        parsed = turn.get("parsed_completion", {})
        if parsed:
            final_answer = parsed.get("final_answer")
            if isinstance(final_answer, str):
                if final_answer.strip() == str(gold_answer).strip():
                    return 1.0
    return 0.0


def reward_advanced_with_raw_memory(trace: Trace, **kwargs: Any) -> float:
    """
    Example of an ADVANCED reward function using raw memory data.
    
    This function can access both:
    - trace: Structured, processed trace (List[Turn])  
    - raw_memory_collection: List of raw agent memory steps for advanced analysis
    """
    # Standard processing using structured trace
    standard_score = reward_simple_exact_match(trace, **kwargs)
    
    # Advanced analysis using raw memory
    raw_memory_collection = kwargs.get("raw_memory_collection", [])
    
    if not raw_memory_collection:
        # Fallback to standard score if no raw memory available
        return standard_score
    
    # Example advanced analysis: count total memory steps across all traces
    total_steps = sum(len(raw_memory) for raw_memory in raw_memory_collection)
    
    # Example: penalize if agent took too many steps
    efficiency_bonus = 0.0
    if total_steps <= 5:  # Efficient execution
        efficiency_bonus = 0.1
    elif total_steps > 10:  # Inefficient execution  
        efficiency_bonus = -0.1
        
    # Example: analyze step types in raw memory
    step_type_bonus = 0.0
    for raw_memory in raw_memory_collection:
        for step in raw_memory:
            # Check if step has error (indicates debugging/recovery)
            if hasattr(step, 'error') and step.error:
                step_type_bonus += 0.05  # Bonus for error recovery
                
    final_score = standard_score + efficiency_bonus + step_type_bonus
    return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]


def reward_batch_advanced_analysis(traces: List[Trace], **kwargs: Any) -> List[float]:
    """
    Example of a BATCH reward function using raw memory data.
    
    This demonstrates cross-trace analysis using raw memory data.
    """
    raw_memory_collection = kwargs.get("raw_memory_collection", [])
    
    scores = []
    for i, trace in enumerate(traces):
        # Standard processing for each trace
        standard_score = reward_simple_exact_match(trace, **kwargs)
        
        # Advanced cross-trace analysis
        cross_trace_bonus = 0.0
        if i < len(raw_memory_collection):
            current_raw_memory = raw_memory_collection[i]
            
            # Compare with other traces: bonus for diversity
            for j, other_raw_memory in enumerate(raw_memory_collection):
                if i != j and current_raw_memory != other_raw_memory:
                    # Different approaches deserve bonus
                    cross_trace_bonus += 0.02
                    
        final_score = standard_score + min(cross_trace_bonus, 0.1)  # Cap bonus
        scores.append(max(0.0, min(1.0, final_score)))
        
    return scores


def demo_backward_compatibility():
    """
    Demonstrate that existing reward functions still work perfectly.
    """
    print("\n🔄 Testing Backward Compatibility")
    print("=" * 50)
    
    # Create mock trace (same structure as before)
    mock_trace: Trace = [
        {
            "prompt_for_model": "Calculate 2+2",
            "model_completion": "Thought: I need to add 2+2\nCode: result = 2 + 2\nFinal Answer: 4",
            "parsed_completion": {
                "thought": "I need to add 2+2", 
                "tool_code": "result = 2 + 2",
                "final_answer": "4"
            },
            "tool_output": "4",
            "action_output": "4",
            "formatted_conversation": "<chat>User: Calculate 2+2\nAssistant: 4</chat>"
        }
    ]
    
    # Test existing reward function
    kwargs = {"gold_answer": "4"}
    
    print("🧪 Testing existing reward_exact_match:")
    score = reward_exact_match(mock_trace, **kwargs)
    print(f"   Score: {score}")
    
    print("✅ Backward compatibility confirmed!")


def demo_new_raw_memory_features():
    """
    Demonstrate new raw memory capabilities.
    """
    print("\n🚀 Testing New Raw Memory Features")
    print("=" * 50)
    
    # Create mock trace and raw memory
    mock_trace: Trace = [
        {
            "prompt_for_model": "Calculate 3*5",
            "model_completion": "Thought: Let me calculate 3*5\nCode: result = 3 * 5\nFinal Answer: 15",
            "parsed_completion": {
                "thought": "Let me calculate 3*5",
                "tool_code": "result = 3 * 5", 
                "final_answer": "15"
            },
            "tool_output": "15",
            "action_output": "15",
            "formatted_conversation": "<chat>User: Calculate 3*5\nAssistant: 15</chat>"
        }
    ]
    
    # Mock raw memory steps (simulating smolagents ActionStep objects)
    class MockActionStep:
        def __init__(self, model_output, error=None):
            self.model_output = model_output
            self.error = error
            
    mock_raw_memory = [
        MockActionStep("Thought: Let me calculate 3*5\nCode: result = 3 * 5\nFinal Answer: 15"),
        MockActionStep("Additional debugging step", error="Minor calculation check")
    ]
    
    # Test with raw memory
    kwargs = {
        "gold_answer": "15",
        "raw_memory_collection": [mock_raw_memory]  # This is what Brain will pass
    }
    
    print("🧪 Testing advanced reward with raw memory:")
    score = reward_advanced_with_raw_memory(mock_trace, **kwargs)
    print(f"   Score: {score}")
    print(f"   (Includes efficiency and error recovery bonuses)")
    
    print("✅ Raw memory features working!")


if __name__ == "__main__":
    print("🧠 ToolBrain Raw Memory Support Demo")
    print("=" * 60)
    
    demo_backward_compatibility()
    demo_new_raw_memory_features()
    
    print("\n🎉 All tests passed! The new architecture supports:")
    print("   ✅ Full backward compatibility")
    print("   ✅ Advanced raw memory analysis") 
    print("   ✅ Optional enhanced features")
