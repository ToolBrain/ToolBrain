import os
import json
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel


load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


model_id = "meta-llama/Llama-3.3-70B-Instruct" 
model = InferenceClientModel(model_id=model_id, token=HUGGINGFACEHUB_API_TOKEN) 
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

G = 3
question = "Sum all numbers from 1 to 10."
all_traces = {}

for i in range(G):
    agent.run(question)
    steps = agent.memory.get_succinct_steps()
    trace_id = f"trace_{i+1}"
    all_traces[trace_id] = steps

with open("sample_traces.json", "w") as f:
    json.dump(all_traces, f, indent=4)
