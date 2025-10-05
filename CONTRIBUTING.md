# Contributing to ToolBrain

First off, thank you for considering contributing to ToolBrain! Your help is essential for building a truly universal agent training ecosystem.

The most impactful way to contribute is by adding support for new agent frameworks. This guide will show you how.

## The Adapter Philosophy

ToolBrain is designed around a "Coach-Athlete-Interpreter" model. The core Brain (Coach) is universal. The user's Agent (Athlete) is framework-specific. Your job is to build the Adapter (the Interpreter).

An Adapter has one core responsibility: observe a third-party agent's execution and translate it into ToolBrain's standard Execution Trace format.

## Quick Start: Building Your First Adapter

### 1. Choose Your Framework
Currently supported: **SmolAgents**, **LangChain**

Requested: AutoGen, CrewAI, Agno, LlamaIndex, Haystack and others...

### 2. Create the Adapter Structure
```bash
mkdir toolbrain/adapters/yourframework
touch toolbrain/adapters/yourframework/__init__.py
touch toolbrain/adapters/yourframework/yourframework_adapter.py
```

### 3. Implement the Base Interface
```python
from ..base_adapter import BaseAgentAdapter
from ...core_types import Trace, Turn, ParsedCompletion, ChatSegment
from typing import List, Any, Tuple

class YourFrameworkAdapter(BaseAgentAdapter):
    def __init__(self, agent, trainable_model, config):
        """Initialize adapter with agent, model, and config."""
        self.agent = agent
        self._trainable_model = trainable_model
        self.config = config
        
        # Extract tools from your framework's agent
        self.tools = self._extract_tools_from_agent(agent)
        
        # Set up LoRA fine-tuning if configured
        self._set_lora_finetuning()
    
    def get_trainable_model(self) -> Any:
        """Returns the agent's underlying trainable model."""
        return self._trainable_model
    
    def run(self, query: str) -> Tuple[Trace, Any, Any]:
        """Run agent and convert execution to ToolBrain trace format."""
        # 1. Execute your framework's agent
        # 2. Capture execution steps  
        # 3. Convert to Turn objects
        # 4. Return (trace, rl_input, raw_memory)
        pass
```

### 4. The Core Challenge: Trace Conversion

Your main task is converting your framework's execution into this structure:

```python
Turn = {
    "prompt_for_model": str,     # What the LLM saw
    "model_completion": str,     # What the LLM generated
    "parsed_completion": {       # Structured breakdown
        "thought": Optional[str],
        "tool_code": Optional[str], 
        "final_answer": Optional[str]
    },
    "tool_output": Optional[str] # Tool execution result
}
```

### 5. Study Existing Adapters

**SmolAgent Pattern** (code generation):
- Agent generates Python code → executes code → captures output
- Tools embedded in execution environment
- Multi-step reasoning with `final_answer()` calls

**LangChain Pattern** (direct tool calling):
- Agent generates JSON tool calls → framework executes → captures results
- Tools managed by framework
- Stream-based execution with tool nodes

## Adapter Implementation Guide

### Required Methods Overview

Every adapter must implement these core methods:

```python
class YourFrameworkAdapter(BaseAgentAdapter):
    def __init__(self, agent, trainable_model, config):
        """Initialize adapter"""
        pass
    
    def get_trainable_model(self) -> Any:
        """Return the underlying trainable model"""
        pass
    
    def run(self, query: str) -> Tuple[Trace, Any, Any]:
        """Execute agent and return (trace, rl_input, raw_memory)"""
        pass
    
    def _extract_tools_from_agent(self, agent) -> List[Any]:
        """Extract tools from framework's agent"""
        pass
    
    def _build_rl_input_from_trace(self, trace: Trace, query: str) -> List[ChatSegment]:
        """Convert trace to RL training format"""
        pass
    
    def _set_lora_finetuning(self):
        """Apply LoRA if configured"""
        pass
```

### Method Implementation Details

#### 1. Tool Extraction Pattern
```python
def _extract_tools_from_agent(self, agent) -> List[Any]:
    """Extract tools from your framework's agent."""
    tools = []
    
    # Pattern 1: Direct attribute access
    if hasattr(agent, 'tools'):
        tools = agent.tools  # SmolAgent pattern
    
    # Pattern 2: Method call
    elif hasattr(agent, 'get_tools'):
        tools = agent.get_tools()
    
    # Pattern 3: Graph traversal (LangChain pattern)
    elif hasattr(agent, 'get_graph'):
        graph = agent.get_graph()
        for node_id, node_data in graph.nodes.items():
            if node_id == 'tools':
                tool_node = node_data.data
                if hasattr(tool_node, 'tools_by_name'):
                    tools = list(tool_node.tools_by_name.values())
    
    # Pattern 4: Registry lookup
    elif hasattr(agent, 'tool_registry'):
        tools = list(agent.tool_registry.values())
    
    return tools
```

#### 2. Agent Execution & Trace Building
```python
def run(self, query: str) -> Tuple[Trace, Any, Any]:
    """Execute agent and build trace."""
    try:
        # Execute your framework's agent
        result = self._execute_agent(query)
        
        # Convert to structured trace
        trace = self._build_trace_from_execution(result, query)
        
        # Build RL input
        rl_input = self._build_rl_input_from_trace(trace, query)
        
        # Capture raw memory for advanced analysis
        raw_memory = self._extract_raw_memory(result)
        
        return trace, rl_input, raw_memory
        
    except Exception as e:
        # Return error trace
        error_turn = Turn(
            prompt_for_model=query,
            model_completion=f"Error: {str(e)}",
            parsed_completion=ParsedCompletion(
                thought=None,
                tool_code=None,
                final_answer=f"Error: {str(e)}"
            ),
            tool_output=None,
            action_output=None,
            formatted_conversation=None
        )
        return [error_turn], None, []
```

#### 3. Trace Building Patterns

**Single-turn pattern (simple Q&A):**
```python
def _build_trace_from_execution(self, result, query):
    trace = []
    
    turn = Turn(
        prompt_for_model=query,
        model_completion=result.content,
        parsed_completion=ParsedCompletion(
            thought=result.content,
            tool_code=None,
            final_answer=result.content
        ),
        tool_output=None,
        action_output=None,
        formatted_conversation=None
    )
    trace.append(turn)
    return trace
```

**Multi-turn pattern (with tools):**
```python
def _build_trace_from_execution(self, result, query):
    trace = []
    
    for step in result.steps:
        if step.type == "llm_response":
            # LLM reasoning/tool call
            turn = Turn(
                prompt_for_model=step.input,
                model_completion=step.output,
                parsed_completion=self._parse_completion(step.output),
                tool_output=None,
                action_output=None,
                formatted_conversation=None
            )
            
        elif step.type == "tool_execution":
            # Add tool output to current turn
            if trace:
                trace[-1]["tool_output"] = step.result
        
        # Only append turn for LLM responses
        if step.type == "llm_response":
            trace.append(turn)
    return trace
```

#### 4. RL Input Building (Universal Pattern)
```python
def _build_rl_input_from_trace(self, trace: Trace, query: str) -> List[ChatSegment]:
    """Convert trace to RL training format."""
    segments = []
    
    # Add initial query
    segments.append(ChatSegment(role="other", text=f"user: {query}\n"))
    
    # Add each turn
    for turn in trace:
        # Add LLM response
        if turn.get("model_completion"):
            segments.append(ChatSegment(
                role="assistant", 
                text=turn["model_completion"]
            ))
        
        # Add tool output if present
        if turn.get("tool_output"):
            segments.append(ChatSegment(
                role="other", 
                text=f"\ntool_output: {turn['tool_output']}\n"
            ))
    
    return segments
```

#### 5. ParsedCompletion Extraction
```python
def _parse_completion(self, model_output: str) -> ParsedCompletion:
    """Parse model output into structured format."""
    import re  # Add this import
    
    # Pattern 1: JSON tool call detection
    tool_call = self._extract_json_tool_call(model_output)
    if tool_call:
        return ParsedCompletion(
            thought=model_output,
            tool_code=f"{tool_call['name']}({tool_call['arguments']})",
            final_answer=None
        )
    
    # Pattern 2: Code block detection  
    code_match = re.search(r'```python\n(.*?)\n```', model_output, re.DOTALL)
    if code_match:
        return ParsedCompletion(
            thought=model_output,
            tool_code=code_match.group(1),
            final_answer=None
        )
    
    # Pattern 3: Direct response
    return ParsedCompletion(
        thought=model_output,
        tool_code=None,
        final_answer=model_output
    )
```

#### 6. LoRA Setup (Standard Pattern)
```python
def _set_lora_finetuning(self):
    """Apply LoRA configuration if specified."""
    lora_config = self.config.get("lora_config", None)
    if not lora_config:
        return
    
    # Convert dict to LoraConfig if needed
    if isinstance(lora_config, dict):
        from peft import LoraConfig
        lora_config = LoraConfig(**lora_config)
    
    # Get the actual model
    trainable_model = self.get_trainable_model()
    actual_model = getattr(trainable_model, 'model', trainable_model)
    
    # Apply LoRA
    from peft import get_peft_model
    peft_model = get_peft_model(actual_model, lora_config)
    
    # Update model reference
    if hasattr(trainable_model, 'model'):
        trainable_model.model = peft_model
    
    # Log statistics
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    print(f"📊 LoRA applied: {trainable_params:,} / {total_params:,} params trainable ({percentage:.2f}%)")
```

### Integration with Brain System

#### Update Factory Method
Add your adapter to `toolbrain/brain.py` in the `_get_adapter_for_agent` method:

```python
def _get_adapter_for_agent(self, agent_instance: Any, trainable_model: Optional[Any]) -> BaseAgentAdapter:
    """Factory method to automatically select the appropriate adapter."""
    config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
    
    # Existing adapters
    if CompiledStateGraph and isinstance(agent_instance, CompiledStateGraph):
        return LangChainAdapter(agent_instance, trainable_model, config_dict)
    
    elif isinstance(agent_instance, CodeAgent):
        return SmolAgentAdapter(agent_instance, config_dict)
    
    # Add your framework
    elif isinstance(agent_instance, YourFrameworkAgent):  # Replace with actual class
        from .adapters.yourframework import YourFrameworkAdapter
        return YourFrameworkAdapter(agent_instance, trainable_model, config_dict)
    
    else:
        raise TypeError(f"Agent type '{type(agent_instance).__name__}' is not supported yet.")
```

#### Export in __init__.py
Add to `toolbrain/adapters/__init__.py`:

```python
from .yourframework import YourFrameworkAdapter

__all__ = [
    "SmolAgentAdapter",
    "LangChainAdapter", 
    "YourFrameworkAdapter",  # Add this
    "create_huggingface_chat_model"
]
```

## Testing Your Adapter

### 1. Create a Simple Test
```python
# examples/test_yourframework.py
from toolbrain import Brain
from your_framework import YourAgent

agent = YourAgent(model="test-model", tools=[simple_tool])
brain = Brain(agent=agent, algorithm="GRPO")

# Should work without errors
trace, _, _ = brain.agent_adapter.run("Use the tool to calculate 2+2")
assert len(trace) > 0
assert trace[0]["tool_output"] == "4"
```

### 2. Test with Training
```python
dataset = [{"query": "Calculate 2+2", "gold_answer": "4"}]
brain.train(dataset, num_iterations=1)
```

## Common Patterns & Tips

### Tool Extraction Strategies
- **Attribute access**: `agent.tools`, `agent._tools`
- **Method inspection**: `agent.get_tools()`, `agent.list_tools()`
- **Graph traversal**: LangChain's node inspection
- **Registry patterns**: Framework tool registries

### Execution Hooking
- **Callback systems**: Most frameworks support callbacks
- **Method overriding**: Patch execution methods
- **Stream interception**: Capture streaming outputs
- **Memory inspection**: Access agent's internal state

### Error Handling
```python
try:
    # Execute agent
    result = agent.run(query)
except FrameworkSpecificError:
    # Return empty trace or error trace
    return [], None, None
```

## Submission Guidelines

### File Structure
```
toolbrain/adapters/yourframework/
├── __init__.py                    # Export YourFrameworkAdapter
├── yourframework_adapter.py      # Main adapter implementation
└── utils.py                      # Helper functions (optional)

examples/
└── train_yourframework_agent.py  # Working example

tests/
└── test_yourframework_adapter.py # Basic tests
```

### Pull Request Checklist

#### Core Implementation
- [ ] Adapter implements `BaseAgentAdapter` interface completely
- [ ] All required methods implemented: `__init__`, `get_trainable_model`, `run`
- [ ] Tool extraction working: `_extract_tools_from_agent` returns valid tools
- [ ] Trace building working: `run()` returns proper `(Trace, rl_input, raw_memory)` format
- [ ] RL input building: `_build_rl_input_from_trace` returns `List[ChatSegment]`
- [ ] LoRA support: `_set_lora_finetuning` handles config properly

#### Integration
- [ ] Factory method updated: Added detection logic to `_get_adapter_for_agent`
- [ ] Adapter exported: Added to `toolbrain/adapters/__init__.py`
- [ ] Framework detection working: Brain auto-detects your adapter type

#### Testing & Examples
- [ ] Working example in `examples/` folder (e.g., `train_yourframework_agent.py`)
- [ ] Basic functionality test: Adapter generates valid traces
- [ ] Training integration test: Full RL training pipeline completes
- [ ] Tool calling test: Tools are extracted and executed correctly
- [ ] Error handling test: Graceful handling of execution failures

#### Code Quality
- [ ] Documentation in adapter docstrings (class and key methods)
- [ ] No breaking changes to existing code
- [ ] Follows existing adapter patterns (SmolAgent/LangChain style)
- [ ] Clean imports and dependencies specified
- [ ] Error handling with proper logging

#### Framework-Specific
- [ ] Installation requirements documented for the framework
- [ ] Framework limitations noted in docstrings
- [ ] Model compatibility verified (HuggingFace models work)
- [ ] Multi-turn vs single-turn execution patterns handled

#### Final Validation
- [ ] Example script runs without errors: `python examples/train_yourframework_agent.py`
- [ ] Training produces non-zero rewards on simple tool calling task
- [ ] Adapter handles both tool calling and direct response scenarios
- [ ] Memory usage reasonable (no major leaks during training)

### Documentation Requirements
- Docstring explaining framework integration approach
- Example showing typical usage
- Notes on framework-specific limitations
- Installation requirements for the framework

## Getting Help

- **Study existing adapters**: `toolbrain/adapters/smolagent/` and `toolbrain/adapters/langchain/`
- **Check core types**: `toolbrain/core_types.py` for Trace structure
- **Ask questions**: Open a GitHub issue with "Adapter Development" label
- **Start simple**: Basic tool calling first, advanced features later

## Framework Priority List

- AutoGen
- CrewAI  
- Agno
- LlamaIndex Agents
- Haystack Agents

**Your framework not listed?** We'd love to see it! Every agent framework makes ToolBrain more universal.

---

Remember: Start simple, study existing patterns, and focus on the core trace conversion. The ToolBrain community is here to help! 🚀
