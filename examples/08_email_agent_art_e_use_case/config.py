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
# BASE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Define the model to be used as the LLM-as-a-Judge.
# Supported models:
# - OpenAI: "gpt-3.5-turbo", "gpt-4", "gpt-4o" (with structured output)
# - Gemini: "gemini/gemini-1.5-flash", "gemini/gemini-2.5-flash-lite" (JSON parsing fallback)
# - Claude: "anthropic/claude-3-sonnet-20240229" (JSON parsing fallback)
# - Other LiteLLM supported models will use JSON parsing fallback
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
MAX_TRAIN_SAMPLES = 4  # Set to None for the full training run
MAX_TEST_SAMPLES = 1


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
    "learning_rate": 1e-5,
    "max_grad_norm": 1.0,
    "chunk_len": 8,
    "num_group_members": 2,
    "batch_size": 2,
    "lora_config": LORA_CONFIG,
    "use_bitsandbytes": True,
}


DPO_CONFIG = {
    "beta": 0.1,
    "learning_rate": 2e-5,
    "max_grad_norm": 1.0,
    "chunk_len": 256,
    "num_group_members": 4,
    "batch_size": 2,
    "lora_config": LORA_CONFIG,
}

# --- 4. Tool Output Configuration ---
# To prevent Out Of Memory (OOM) errors caused by very long email bodies,
# we can set a character limit for the tool output.
# The `read_email` tool will truncate the email body if it exceeds this limit.
# Set to None for no limit. A value around 4000-8000 is recommended for A100 GPUs.
MAX_TOOL_OUTPUT_CHARS = 16384

MAX_NEW_TOKENS = 512

# Run evaluation on the validation set every N training steps.
VALIDATION_INTERVAL = 2  # The same as ART-E

# File to log the detailed inputs and outputs of the LLM Judge for debugging.
JUDGE_LOG_FILENAME = "evaluation_judge_log.txt"

JUDGE_TOOL_BRAIN_LOG_FILENAME = "evaluation_judge_toolbrain_log.txt"


# --- 5. Prompt Template ---
# This is a direct replication of the system prompt from ART-E's rollout.py
SYSTEM_PROMPT_TEMPLATE = """You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {max_turns} turns to find the answer, so if your first seach doesn't find the answer, you can try with different keywords.
User's email address is {inbox_address}
Today's date is {query_date}"""

MAX_AGENT_TURNS = 10