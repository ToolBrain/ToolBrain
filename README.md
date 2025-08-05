# ToolBrain 🧠

> A framework for training LLM-powered agents to use tools more effectively using Reinforcement Learning

ToolBrain is an open-source Python library that serves as a "training harness" or "coach" for LLM-powered agents. It wraps around existing agent implementations and uses Reinforcement Learning techniques to improve their underlying policy, making them more effective at using tools and solving complex tasks.

## ✨ Key Features

- **🤖 Agent-Agnostic**: Works with any agent that can produce execution traces
- **🎯 Flexible Rewards**: Pre-built reward functions for common evaluation criteria
- **🔧 Tool-Focused**: Specifically designed to improve tool usage capabilities
- **📊 Trace-Based**: Learns from structured execution traces with thoughts, code, and outputs
- **🚀 Easy Integration**: Simple API that wraps around existing agent implementations

## 🏗️ Architecture

ToolBrain operates on the principle of **trace-based learning**:

1. **Agent Execution**: Your agent runs and produces a structured trace of its reasoning
2. **Reward Calculation**: Each trace is evaluated using customizable reward functions  
3. **RL Training**: Advanced algorithms like DPO/GRPO improve the agent's policy
4. **Iteration**: The cycle repeats, gradually improving tool usage effectiveness

```
Query → Agent → Trace → Reward → RL Algorithm → Improved Agent
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A Hugging Face account and API token

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/toolbrain/toolbrain.git
cd toolbrain
pip install -e .
```

Or install from PyPI (when available):

```bash
pip install toolbrain
```

### 2. Get a Hugging Face API Token

1. Go to [Hugging Face Settings → Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token value

### 3. Set Environment Variable

Set your Hugging Face token as an environment variable:

**Linux/Mac:**
```bash
export HF_TOKEN=your_token_here
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="your_token_here"
```

**Windows (Command Prompt):**
```cmd
set HF_TOKEN=your_token_here
```

### 4. Run the Example

Run the complete example to see ToolBrain in action:

```bash
python examples/01_run_a_training_step.py
```

This will:
- Connect to the Hugging Face Inference API
- Initialize a `CodeAgent` with simple math tools
- Run the agent multiple times to collect execution traces
- Evaluate each trace using a reward function
- Pass the results to a mock RL algorithm for "training"

## 📖 Usage Example

Here's a minimal example of how to use ToolBrain:

```python
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match
from smolagents import CodeAgent, LiteLLMModel

# Set up your agent
model = LiteLLMModel(
    model_id="huggingface/HuggingFaceTB/SmolLM3-3B",
    api_key=os.getenv("HF_TOKEN")
)

def add(a: int, b: int) -> int:
    return a + b

agent = CodeAgent(tools=[add], model=model)

# Initialize the training brain  
brain = Brain(
    agent=agent,
    reward_func=reward_exact_match,
    learning_algorithm="MockDPO"
)

# Run a training step
brain.train_step(
    query="Calculate 5 + 7 using the add tool",
    gold_answer="12",
    num_group_members=10
)
```

## 🎯 Reward Functions

ToolBrain includes several pre-built reward functions:

- **`reward_exact_match`**: Rewards traces where the final answer exactly matches the expected result
- **`reward_tool_execution_success`**: Rewards traces with no tool execution errors
- **`reward_step_efficiency`**: Rewards traces that complete tasks in fewer steps

You can also create custom reward functions:

```python
from toolbrain.types import Trace

def custom_reward(trace: Trace, gold_answer: str) -> float:
    # Your custom evaluation logic here
    return 1.0 if some_condition else 0.0
```

## 🔧 Core Components

### Brain Class
The central coordinator that orchestrates training:
- Collects multiple execution traces from your agent
- Evaluates each trace using reward functions
- Passes results to RL algorithms for learning

### Trace Structure
Execution traces use a standardized format:
```python
TraceStep = {
    "type": "thought" | "tool_code" | "tool_output" | "final_answer",
    "content": "The actual content"
}
```

### Agent Integration
ToolBrain works with any agent that can produce traces. For agents that don't natively support traces, you can create a wrapper:

```python
class TraceableAgent(YourAgent):
    def run(self, query: str) -> Trace:
        # Wrap your agent's execution and convert to trace format
        result = super().run(query)
        return self._convert_to_trace(result)
```

## 🛣️ Roadmap

- [ ] **Real RL Algorithms**: Replace mock with actual DPO/GRPO implementations
- [ ] **More Reward Functions**: Add semantic similarity, tool usage patterns, etc.
- [ ] **Multi-Agent Support**: Enable training with multiple agent types
- [ ] **Advanced Metrics**: Comprehensive evaluation and monitoring tools
- [ ] **Cloud Integration**: Support for distributed training workflows

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/toolbrain/toolbrain.git
cd toolbrain
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Format code:
```bash
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **smolagents**: Provides the base `CodeAgent` implementation
- **Hugging Face**: API infrastructure for LLM access
- **OpenAI**: Compatible API format for easy integration

## 📞 Support

- 📖 [Documentation](https://toolbrain.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/toolbrain/toolbrain/issues)
- 💬 [Discussions](https://github.com/toolbrain/toolbrain/discussions)

---

**Made with ❤️ by the ToolBrain Team** 