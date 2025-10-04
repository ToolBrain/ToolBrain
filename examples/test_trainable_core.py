# Test Giai đoạn 3: Truy cập và Huấn luyện Mô hình (The "Trainable Core")

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.agents import create_agent
from langchain_core.tools import tool
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

print("🔬 TESTING GIAI ĐOẠN 3: TRAINABLE CORE ACCESS")
print("=" * 60)

# Setup model
print("📥 Setting up HuggingFace model...")
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Explicit for training
    device_map="cpu"  # Keep on CPU for testing
)

# Create pipeline and wrapper
from transformers import pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    return_full_text=False
)

hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
trainable_llm = ChatHuggingFace(llm=hf_pipeline)

print("\n" + "=" * 60)
print("🧪 TEST 1: Model Access Verification")
print("=" * 60)

# Test 1: Access underlying model
try:
    print("🔍 Checking model access paths...")
    
    # Path 1: Direct pipeline access
    pipeline_model = trainable_llm.llm.pipeline.model
    pipeline_tokenizer = trainable_llm.llm.pipeline.tokenizer
    
    print(f"✅ Pipeline model type: {type(pipeline_model)}")
    print(f"✅ Pipeline tokenizer type: {type(pipeline_tokenizer)}")
    print(f"✅ Model device: {next(pipeline_model.parameters()).device}")
    print(f"✅ Model dtype: {next(pipeline_model.parameters()).dtype}")
    
    # Check if same as original
    print(f"✅ Same as base model: {pipeline_model is base_model}")
    
    # Check model parameters
    total_params = sum(p.numel() for p in pipeline_model.parameters())
    trainable_params = sum(p.numel() for p in pipeline_model.parameters() if p.requires_grad)
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"📊 Trainable ratio: {trainable_params/total_params:.2%}")
    
except Exception as e:
    print(f"❌ Model access failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🧪 TEST 2: Brain Model Extraction")
print("=" * 60)

# Test 2: Brain's model extraction
try:
    print("🔍 Testing Brain model extraction...")
    
    # Create agent 
    langchain_agent = create_agent(
        model=trainable_llm,
        tools=[add],
        prompt="You are a helpful assistant."
    )
    
    # Create Brain
    brain = Brain(
        agent=langchain_agent,
        trainable_model=trainable_llm,
        algorithm="GRPO",
        reward_func=reward_exact_match,
        learning_rate=1e-4
    )
    
    # Test trainable model extraction
    brain_model = brain.agent_adapter.get_trainable_model()
    print(f"✅ Brain extracted model type: {type(brain_model)}")
    
    if hasattr(brain_model, 'model') and hasattr(brain_model, 'tokenizer'):
        print(f"✅ Brain model inner type: {type(brain_model.model)}")
        print(f"✅ Brain tokenizer type: {type(brain_model.tokenizer)}")
        print(f"✅ Same model reference: {brain_model.model is pipeline_model}")
    else:
        print(f"⚠️ Brain model structure: {dir(brain_model)}")
        
except Exception as e:
    print(f"❌ Brain model extraction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🧪 TEST 3: LoRA Integration")
print("=" * 60)

# Test 3: LoRA setup
try:
    print("🔍 Testing LoRA integration...")
    
    from peft import LoraConfig, get_peft_model
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print(f"✅ LoRA config created: {lora_config}")
    
    # Apply LoRA to model
    model_before_params = sum(p.numel() for p in pipeline_model.parameters())
    
    # Create PEFT model
    peft_model = get_peft_model(pipeline_model, lora_config)
    
    model_after_params = sum(p.numel() for p in peft_model.parameters())
    trainable_after = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    
    print(f"✅ PEFT model created: {type(peft_model)}")
    print(f"📊 Parameters before LoRA: {model_before_params:,}")
    print(f"📊 Parameters after LoRA: {model_after_params:,}")
    print(f"📊 Trainable after LoRA: {trainable_after:,}")
    print(f"📊 LoRA efficiency: {trainable_after/model_after_params:.2%}")
    
    # Update pipeline to use PEFT model
    text_gen_pipeline.model = peft_model
    print(f"✅ Pipeline updated with PEFT model")
    
except Exception as e:
    print(f"❌ LoRA integration failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🧪 TEST 4: Gradient Flow Test")
print("=" * 60)

# Test 4: Check gradient flow
try:
    print("🔍 Testing gradient computation...")
    
    # Create simple forward pass
    test_input = tokenizer("Hello world", return_tensors="pt")
    
    # Forward pass
    model_for_test = text_gen_pipeline.model
    with torch.no_grad():
        baseline_output = model_for_test(**test_input)
    
    print(f"✅ Forward pass successful")
    print(f"✅ Output shape: {baseline_output.logits.shape}")
    
    # Test gradient computation
    model_for_test.train()  # Set to training mode
    
    # Forward with gradients
    output = model_for_test(**test_input)
    loss = output.logits.mean()  # Dummy loss
    
    print(f"✅ Loss computed: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_count = 0
    total_grad_norm = 0.0
    
    for name, param in model_for_test.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"✅ Parameters with gradients: {grad_count}")
    print(f"✅ Total gradient norm: {total_grad_norm:.4f}")
    
    if grad_count > 0 and total_grad_norm > 0:
        print("🎉 GRADIENT FLOW WORKING!")
    else:
        print("⚠️ No gradients detected")
        
except Exception as e:
    print(f"❌ Gradient flow test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🧪 TEST 5: End-to-End Training")
print("=" * 60)

# Test 5: Actual training step
try:
    print("🔍 Testing actual training step...")
    
    # Create new Brain with updated model
    brain_updated = Brain(
        agent=langchain_agent,
        trainable_model=trainable_llm,
        algorithm="GRPO",
        reward_func=reward_exact_match,
        learning_rate=1e-4,
        batch_size=1
    )
    
    # Override with tools for custom adapter  
    from toolbrain.langchain_adapter_v2 import LangChainAdapter
    brain_updated.agent_adapter = LangChainAdapter(
        agent=langchain_agent,
        trainable_model=trainable_llm,
        config=brain_updated.config.to_dict() if hasattr(brain_updated.config, 'to_dict') else vars(brain_updated.config),
        tools=[add]
    )
    
    # Test dataset
    test_dataset = [{"query": "Calculate 2 + 3", "gold_answer": "5"}]
    
    print("🚀 Running one training step...")
    
    # Store initial weights for comparison
    initial_weights = {}
    for name, param in model_for_test.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()
    
    # Run training
    brain_updated.train(test_dataset, num_iterations=1)
    
    # Check weight updates
    weight_changes = 0
    total_change = 0.0
    
    for name, param in model_for_test.named_parameters():
        if param.requires_grad and name in initial_weights:
            change = (param.data - initial_weights[name]).abs().sum().item()
            if change > 1e-8:  # Threshold for meaningful change
                weight_changes += 1
                total_change += change
    
    print(f"✅ Training completed!")
    print(f"📊 Parameters changed: {weight_changes}")
    print(f"📊 Total weight change: {total_change:.6f}")
    
    if weight_changes > 0:
        print("🎉 MODEL WEIGHTS UPDATED - TRAINING WORKING!")
    else:
        print("⚠️ No weight changes detected")
        
except Exception as e:
    print(f"❌ Training test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("📋 FINAL ASSESSMENT")
print("=" * 60)

assessment = {
    "Model Access": "✅" if 'pipeline_model' in locals() else "❌",
    "Brain Integration": "✅" if 'brain_model' in locals() else "❌", 
    "LoRA Setup": "✅" if 'peft_model' in locals() else "❌",
    "Gradient Flow": "✅" if 'grad_count' in locals() and grad_count > 0 else "❌",
    "Training Step": "✅" if 'weight_changes' in locals() and weight_changes > 0 else "❌"
}

print("🎯 GIAI ĐOẠN 3 RESULTS:")
for test, status in assessment.items():
    print(f"  {status} {test}")

all_passed = all(status == "✅" for status in assessment.values())
if all_passed:
    print("\n🏆 GIAI ĐOẠN 3: HOÀN THÀNH THÀNH CÔNG!")
    print("All 3 phases are now 100% complete! 🚀")
else:
    failed_tests = [test for test, status in assessment.items() if status == "❌"]
    print(f"\n⚠️ GIAI ĐOẠN 3: CẦN KHẮC PHỤC: {', '.join(failed_tests)}")