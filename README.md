# ToolBrain 🧠

ToolBrain is a lightweight open-source Python library for training **agentic systems** with effective tool usage and built-in reinforcement learning.  
📚 Documentation & tutorials: [toolbrain.org](https://toolbrain.org)

## ✨ Key Features

- **🤖 Learning algorithms**: Supports GRPO, DPO, and supervised learning.  
- **🎯 Flexible rewards**: Define your own reward functions or use LLM-as-judge.  
- **🔧 Tool management**: Scalable retrieval for managing large tool collections.  
- **📊 Knowledge distillation**: Distill large teacher models into smaller student models for efficiency.  
- **🚀 Zero-learn**: Automatically generate training tasks.  
- **⚡ Efficient training**: Supports LoRA, Unsloth, and BitsAndBytes for resource-efficient training.  

## 🚀 Getting Started

### Prerequisites
- Python **3.10+**

### Installation

From PyPI (when available):  
```bash
pip install toolbrain
```


### 4. Run the Example

Run the complete example to see ToolBrain in action (please see under examples folder for more advanced usage examples):

```bash
python examples/01_run_a_training_step.py
```

This will:
- Initialize a `CodeAgent` with simple math tools
- Run the agent multiple times to collect execution traces
- Evaluate each trace using a reward function
- Pass the results to a mock RL GRPO algorithm for "training"

## 📖 Usage Example

Here's a minimal example of how to use ToolBrain. This script demonstrates simplified ToolBrain API:
1. Create a smolagent CodeAgent
2. Create brain with Brain() constructor (all parameters as keywords)
3. Train with explicit, self-documenting parameters


```python
from smolagents import tool, TransformersModel, CodeAgent
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match

# --- 1. Define Tools and Reward Function (User-defined) ---
@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): First addend.
        b (int): Second addend.

    Returns:
        int: Sum of a and b.
    """
    return a + b


# --- 2. Prepare Training Data ---
training_dataset = [
    {
        "query": "Use the add tool to calculate 5 + 7",
        "gold_answer": "12"
    }
]


print("🧠 ToolBrain Training Example with Reinforcement Learning")
print("=" * 60)

# 1. Create agent
model = TransformersModel(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",  # use a bigger model if you want more accuracy and faster learning
    max_new_tokens=128
)

agent = CodeAgent(
    model=model,
    tools=[add],
    max_steps=1
)

print("✅ Agent created.")

# 2. Create Brain

brain = Brain(
    agent,                          # Agent instance
    algorithm="GRPO",                # Algorithm choice
    # Customised reward function is defined here, we use a mocking reward function with value 1.0
    # for an exact gold_answer match and 0 otherwise, llm as judge can be used for automatic reward
    reward_func=reward_exact_match
)

# 3. Train with the GRPO brain for 10 training GRPO steps
brain.train(training_dataset, num_iterations=10)
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


**Made with ❤️ by the ToolBrain Team** 