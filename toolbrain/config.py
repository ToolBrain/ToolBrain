from peft import LoraConfig

def get_default_config():
    return {
        "epsilon": 0.2, # clipping parameter
        "beta": 0.04, # KL divergence penalty coefficient
        "opt_steps": 3, # Number of GRPO optimization steps per batch
        "lr": 1e-5, # Learning rate for optimizer
        "max_grad_norm": 1.0,
        "chunk_len": 128, # If not None, get_per_token_logps will process in chunks
        "num_group_members": 2,  # The number of group members used for GRPO training steps
        "lora_config": LoraConfig( # If set, the RL will perform LoRA finetuning instead of full fine-tuning
            r=8,  # LoRA rank
            lora_alpha=16,  # scaling
            target_modules=["q_proj", "v_proj"],  # the modules to apply LoRA
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",  # or "SEQ_2_SEQ_LM" for encoder-decoder
        )
    }