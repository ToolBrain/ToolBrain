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
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Define the model to be used as the LLM-as-a-Judge.
JUDGE_MODEL_ID = "gemini/gemini-1.5-flash"

# Define paths for data and outputs.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "enron_emails.db")

# Hugging Face dataset details for the questions.
QUESTIONS_DATASET_ID = "corbt/enron_emails_sample_questions"
QUESTIONS_DATASET_SPLIT = "train" # We'll use the 'train' split for training.

# General training parameters
NUM_TRAIN_EPOCHS = 2
# NEW: Use a ratio for splitting. 0.8 means 80% for training, 20% for testing.
TRAIN_TEST_SPLIT_RATIO = 0.8 
# NEW: A seed for reproducibility. Using the same seed ensures the split is always the same.
DATASET_SPLIT_SEED = 42 

# Keep this for quick development runs. If set, it will override the ratio split.
MAX_TRAIN_SAMPLES = 200 # Set to None for the full training run


# --- 2. LoRA Configuration ---


LORA_CONFIG = LoraConfig(
    r=16,                   
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
    "lr": 2e-5,            
    "max_grad_norm": 1.0,  
    "chunk_len": 256,      

    "num_group_members": 4, 
    
    "lora_config": LORA_CONFIG
}


DPO_CONFIG = {
    "beta": 0.1,
    "lr": 2e-5,
    "max_grad_norm": 1.0,
    "chunk_len": 256,

    "num_group_members": 4, 
    "batch_size": 2, 

    "lora_config": LORA_CONFIG
}