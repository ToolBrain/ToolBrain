"""
Configuration File for the ART-E Email Agent Use Case.

This file centralizes all settings for the training and evaluation experiments,
making it easy to manage and modify parameters without changing the core logic.

It includes:
- General settings for models, datasets, and training runs.
- A shared LoRA configuration for efficient fine-tuning.
- Separate, fine-tuned configurations for GRPO and DPO algorithms.
"""

import os
from peft import LoraConfig

# --- 1. General Settings ---
# These settings are shared across all experiments in this use case.

# Define the base model to be fine-tuned.
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Define the model to be used as the LLM-as-a-Judge.
JUDGE_MODEL_ID = "gemini/gemini-2.5-flash-lite"

# Define paths for data and outputs.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "enron_emails.db")

# Hugging Face dataset details for the questions.
QUESTIONS_DATASET_ID = "corbt/enron_emails_sample_questions"
QUESTIONS_DATASET_SPLIT = "train"  # We'll use the 'train' split for training.

# General training parameters
NUM_TRAIN_EPOCHS = 1
# NEW: Use a ratio for splitting. 0.8 means 80% for training, 20% for testing.
TRAIN_TEST_SPLIT_RATIO = 0.8
# NEW: A seed for reproducibility. Using the same seed ensures the split is always the same.
DATASET_SPLIT_SEED = 42

# Keep this for quick development runs. If set, it will override the ratio split.
MAX_TRAIN_SAMPLES = 200  # Set to None for the full training run
MAX_TEST_SAMPLES = 20


# --- 2. LoRA Configuration ---


LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# --- 3. Algorithm-Specific Configurations ---
# Each algorithm has its own set of optimal hyperparameters.

GRPO_CONFIG = {
    "epsilon": 0.2,
    "beta": 0.04,
    "opt_steps": 4,
    "lr": 1e-5,
    "max_grad_norm": 1.0,
    "chunk_len": 8,
    "num_group_members": 2,
    "batch_size": 2,
    "lora_config": LORA_CONFIG,
    "use_bitsandbytes": True,
}


DPO_CONFIG = {
    "beta": 0.1,
    "lr": 2e-5,
    "max_grad_norm": 1.0,
    "chunk_len": 256,
    "num_group_members": 4,
    "batch_size": 2,
    "lora_config": LORA_CONFIG,
}

NEW_SYSTEM_PROMPT = '''You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "Check my email if Germany has been approved for trading credit derivatives."

Thought: I will use the `search_emails` tool to find any relevant emails, and then return the final answer using the `final_answer` and `read_email` tools.
Code:
```py
search_result = search_emails(keywords=["Germany", "credit derivatives"])
if search_result:
    email_content = read_email(search_result[0]['message_id'])
    print(email_content)
    if "Germany" in email_content and "approved" in email_content:
        final_answer("Germany is approved for trading credit derivatives.")
    else:
        final_answer("Germany is not approved for trading credit derivatives.")
else:
    final_answer("No relevant emails found regarding Germany and credit derivatives.")
```<end_code>

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools, behaving like regular python functions:
```python
{%- for tool in tools.values() %}
def {{ tool.name }}({% for arg_name, arg_info in tool.inputs.items() %}{{ arg_name }}: {{ arg_info.type }}{% if not loop.last %}, {% endif %}{% endfor %}) -> {{tool.output_type}}:
    """{{ tool.description }}

    Args:
    {%- for arg_name, arg_info in tool.inputs.items() %}
        {{ arg_name }}: {{ arg_info.description }}
    {%- endfor %}
    """
{% endfor %}
```

{%- if managed_agents and managed_agents.values() | list %}
You can also give tasks to team members.
Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task'.
Given that this team member is a real human, you should be very verbose in your task, it should be a long string providing informations as detailed as necessary.
Here is a list of the team members that you can call:
```python
{%- for agent in managed_agents.values() %}
def {{ agent.name }}("Your query goes here.") -> str:
    """{{ agent.description }}"""
{% endfor %}
```
{%- endif %}

You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
'''

temp = '''
Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin!'''
# --- 4. Tool Output Configuration ---
# To prevent Out Of Memory (OOM) errors caused by very long email bodies,
# we can set a character limit for the tool output.
# The `read_email` tool will truncate the email body if it exceeds this limit.
# Set to None for no limit. A value around 4000-8000 is recommended for A100 GPUs.
MAX_TOOL_OUTPUT_CHARS = 4096
