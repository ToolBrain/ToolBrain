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
from ...core_types import Trace, Turn, ParsedCompletion

class YourFrameworkAdapter(BaseAgentAdapter):
    def __init__(self, agent, trainable_model, config=None):
        super().__init__(agent, trainable_model, config)
        # Extract tools from your framework's agent
        self.tools = self._extract_tools_from_agent(agent)
    
    def get_trace(self, query: str, **kwargs) -> Tuple[Trace, Any, Any]:
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

### Extract Tools
```python
def _extract_tools_from_agent(self, agent):
    # Each framework stores tools differently:
    # - SmolAgent: agent.tools
    # - LangChain: graph.nodes['tools'].data.tools_by_name
    # - YourFramework: ???
```

### Capture Execution
```python
def get_trace(self, query: str, **kwargs):
    # Hook into your framework's execution:
    # - Override execution methods
    # - Use callbacks/listeners
    # - Parse execution logs
    # - Intercept tool calls
```

### Handle Different Execution Patterns
- **Single-turn**: One prompt → one response
- **Multi-turn**: Agent → tools → agent → tools...
- **Streaming**: Real-time execution updates
- **Error handling**: Failed tool calls, parsing errors

## Testing Your Adapter

### 1. Create a Simple Test
```python
# examples/test_yourframework.py
from toolbrain import Brain
from your_framework import YourAgent

agent = YourAgent(model="test-model", tools=[simple_tool])
brain = Brain(agent=agent, algorithm="GRPO")

# Should work without errors
trace, _, _ = brain.adapter.get_trace("Use the tool to calculate 2+2")
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
- [ ] Adapter implements `BaseAgentAdapter` interface
- [ ] Working example in `examples/` folder
- [ ] Basic tests pass
- [ ] Documentation in adapter docstrings
- [ ] No breaking changes to existing code

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
