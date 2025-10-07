# ToolBrain 🧠
![PyPI Version](https://img.shields.io/pypi/v/toolbrain)
[![Monthly Downloads](https://img.shields.io/badge/dynamic/json?url=https://pypistats.org/api/packages/toolbrain/recent&query=data.last_month&label=downloads/month)](https://pypistats.org/packages/toolbrain)



ToolBrain is a lightweight open-source Python library for training **agentic systems** with effective tool usage and built-in reinforcement learning.  
📚  Our website: [toolbrain.org](https://toolbrain.org) and [Documentation & tutorials](docs/source/tutorials/tutorials.md)

📚 Watch Introduction [Video](https://www.youtube.com/watch?v=LhYiIHTRw7E) 

Support us by giving ToolBrain a ⭐ on GitHub.
## ✨ Key Features

- **🤖 Learning algorithms**: Supports [GRPO](examples/02_lightgbm_hpo_training_with_grpo/run_hpo_training.py), [DPO](examples/04_lightgbm_hpo_training_with_dpo/run_hpo_training.py), and [supervised learning](examples/05_supervised_training.py).  
- **🎯 Flexible rewards**: Define your own [reward functions](examples/09_flexible_rewards.py) or use [LLM-as-judge](examples/10_llm_as_judge.py).  
- **🔧 Tool management**: Scalable [retrieval](examples/06_tool_retrieval.py) for managing large tool collections.  
- **📊 Knowledge distillation**: [Distill](examples/08_distillation.py) large teacher models into smaller student models for efficiency.  
- **🚀 Zero-learn**: Automatically [generate training tasks](examples/03_generate_training_examples.py ).  
- **⚡ Efficient training**: Supports [FP16 finetuning](examples/13_hello_world_fp16.py), LoRA, Unsloth, and [BitsAndBytes](examples/12_hello_world_bitsandbytes.py) for [resource-efficient training](examples/07_email_search_agent/).
- 🧠 **Multiple agent frameworks**: Supports SmolAgent and [LangChain](examples/11_train_langchain_agent.py), with more coming soon.

## 🚀 Getting Started

### Prerequisites
- Python **3.10+**

### Installation

Create conda env (optional)
```bash
conda create --name toolbrain python=3.12
conda activate toolbrain
```

from PyPi:

```bash
pip install toolbrain
```
Or from the source code:
```bash
git clone https://github.com/ToolBrain/ToolBrain.git
```

Enter the cloned folder and type:
```bash
pip install .
```


### Run the Example

Run the complete example to see ToolBrain in action (please see under examples folder for more advanced usage examples):

```bash
python examples/01_run_hello_world.py
```

This will:
- Initialize a `CodeAgent` with simple math tools
- Define a customised reward function
- Run the GRPO algorithm

## 📖 Usage Example

Here's a minimal example of how to use ToolBrain. This script demonstrates simplified ToolBrain API:
1. Create a smolagent CodeAgent
2. Create a brain with our main class Brain() 
3. Train the agent with the GRPO algorithm


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


# 3. Create agent
model = TransformersModel(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",  # use a bigger model for better results
    max_new_tokens=128
)

agent = CodeAgent(
    model=model,
    tools=[add],
    max_steps=1
)

# 4. Create Brain

brain = Brain(
    agent,                          # Agent instance
    algorithm="GRPO",                # Algorithm choice
    reward_func=reward_exact_match  # A reward function, you can customise any python function as reward
)

# 5. Train the agent with GRPO steps
brain.train(training_dataset, num_iterations=10)
```
 ### Results
The following plot illustrates how ToolBrain enhances the tool usage accuracy of the small Qwen/Qwen2.5-0.5B-Instruct model after just 20 training steps using GRPO.

![GRPO learning curve](data/grpo.png)
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) for details.

## 🌍 Community contributions

Our vision is for **ToolBrain** to become the universal Reinforcement Learning layer for any agentic framework. Whether you build your agents with **LangChain**, **SmolAgents**, **LlamaIndex**, **AutoGen**, or a custom solution, you should be able to make them smarter with ToolBrain.

The key to this vision is our **modular Adapter architecture**. Adding support for a new framework is as simple as implementing a new adapter that translates the agent's internal state into ToolBrain's standard *Execution Trace*.

We welcome community contributions!  
If you are using an agent framework not yet supported, we encourage you to build an adapter for it.  
Check out our [`CONTRIBUTING.md`](./CONTRIBUTING.md) guide and the existing implementations in the [`toolbrain/adapters/`](./toolbrain/adapters/) directory to get started.

## Contributors
[Quy Minh Le](https://www.linkedin.com/in/quy-minh-le-b70218333/), Minh Sao Khue Luu, [Khanh-Tung Tran](https://www.linkedin.com/in/khanh-tung-tran-83b3541ab), Duc-Hai Nguyen, Hoang-Quoc-Viet Pham,  Quan Le, [Hoang Thanh Lam](https://research.ibm.com/people/thanh-hoang) and [Harry Nguyen](https://www.ucc.ie/en/compsci/people/harrynguyen/)

---

### 🚀 Spread the Word

If you believe in ToolBrain's vision of making agent training accessible to everyone, please consider sharing it with your network!

[![Share on Twitter](https://img.shields.io/badge/-Share%20on%20Twitter-%231DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/intent/tweet?text=Just%20found%20ToolBrain%2C%20a%20lightweight%20open-source%20framework%20to%20train%20AI%20agents%20%28like%20LangChain%20or%20SmolAgent%29%20to%20use%20tools%20reliably%20with%20Reinforcement%20Learning.%20A%20gym%20for%20your%20agents%21&url=https%3A%2F%2Fgithub.com%2FToolBrain%2FToolBrain&hashtags=ToolBrain,AIAgents,ReinforcementLearning,LLM,OpenSource)
[![Share on LinkedIn](https://img.shields.io/badge/-Share%20on%20LinkedIn-%230A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2FToolBrain%2FToolBrain&title=ToolBrain%3A%20A%20Lightweight%20RL%20Framework%20for%20Training%20AI%20Agents&summary=Just%20found%20ToolBrain%2C%20a%20lightweight%20open-source%20framework%20to%20train%20AI%20agents%20%28like%20LangChain%20or%20SmolAgent%29%20to%20use%20tools%20reliably%20with%20Reinforcement%20Learning.%20A%20gym%20for%20your%20agents%21)
[![Share on Facebook](https://img.shields.io/badge/-Share%20on%20Facebook-%231877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2FToolBrain%2FToolBrain)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-%23FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2FToolBrain%2FToolBrain&title=ToolBrain%3A%20A%20Lightweight%20RL%20Framework%20for%20Training%20AI%20Agents)

---

## References
Please cite [our paper](https://arxiv.org/abs/2510.00023) with the following bibtex:
```
@misc{le2025toolbrainflexiblereinforcementlearning,
      title={ToolBrain: A Flexible Reinforcement Learning Framework for Agentic Tools}, 
      author={Quy Minh Le and Minh Sao Khue Luu and Khanh-Tung Tran and Duc-Hai Nguyen and Hoang-Quoc-Viet Pham and Quan Le and Hoang Thanh Lam and Hoang D. Nguyen},
      year={2025},
      eprint={2510.00023},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.00023}, 
}
```

**Made with ❤️ by the ToolBrain Team** 
