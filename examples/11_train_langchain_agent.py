# examples/09_train_langchain_agent.py (PHIÊN BẢN MỚI)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match

try:
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    print("Vui lòng cài đặt các gói cần thiết: pip install 'langchain>=1.0.0a10' 'langgraph' 'langchain-huggingface' 'transformers'")
    exit()

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

training_dataset = [
    {"query": "Use the add tool to calculate 5 + 7", "gold_answer": "12"},
    {"query": "What is 8 multiplied by 6?", "gold_answer": "48"},
]

print("🧠 ToolBrain Training Example with a LangChain Agent")
print("=" * 60)

print("📥 Initializing HuggingFace model...")
# Khởi tạo model và tokenizer từ transformers trước
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

# Tạo pipeline cho text generation
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    return_full_text=False
)

# Wrap pipeline trong HuggingFacePipeline
hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Sau đó wrap trong ChatHuggingFace
trainable_llm = ChatHuggingFace(llm=hf_pipeline)
print("✅ Model initialized.")

print("🔧 Creating LangChain agent...")
langchain_agent = create_agent(
    model=trainable_llm,  # Sử dụng tham số model thay vì llm
    tools=[add, multiply],
    prompt="You are a helpful assistant. You must call a tool to answer the user.",
)
print("✅ LangChain agent created.")

print("\n🧠 Initializing Brain for the LangChain agent...")
# Store tools để pass vào adapter
tools_list = [add, multiply]

# QUAN TRỌNG: Truyền cả agent và trainable_model vào Brain
brain = Brain(
    agent=langchain_agent,
    trainable_model=trainable_llm, # <<<<<<<<<<<<<<<<<<<<<<<< THAY ĐỔI QUAN TRỌNG
    algorithm="GRPO",
    reward_func=reward_exact_match
)

# Pass tools explicitly to adapter
if hasattr(brain.agent_adapter, '__init__'):
    # Re-initialize adapter with tools
    from toolbrain.langchain_adapter_v2 import LangChainAdapter
    brain.agent_adapter = LangChainAdapter(
        agent=langchain_agent,
        trainable_model=trainable_llm,
        config=brain.config.to_dict() if hasattr(brain.config, 'to_dict') else vars(brain.config),
        tools=tools_list
    )

print("✅ Brain initialized.")

print("\n🚀 Starting training...")
brain.train(training_dataset, num_iterations=5)
print("\n🎉 Training finished!")

print("\n🤖 Testing the trained agent:")
trained_langchain_agent = brain.get_agent()
response = trained_langchain_agent.invoke({"messages": [("user", "What is 100 + 200?")]})
print(f"Query: What is 100 + 200?")
print(f"Response: {response['messages'][-1].content}")