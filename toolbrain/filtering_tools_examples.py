from smolagents import ToolCollection, CodeAgent, load_tool
from smolagents import PythonExecutor, DuckDuckGoSearchTool, PythonInterpreterTool
from smolagents.local_python_executor import LocalPythonExecutor

from retriever import ToolRetriever

from huggingface_hub import login
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

def print_tools(tools):
    print("Available Tools:")
    for i, resource in enumerate(tools):
        name = getattr(resource, "name", str(resource))
        desc = getattr(resource, "description", "")
        inputs = getattr(resource, "inputs", None)
        output_type = getattr(resource, "output_type", None)
        print(f"{i}. Tool object: {resource} \nName: {name} \nDescription: {desc}"+ f"\nInputs: {inputs}. \nOutput_type: {output_type}" if inputs or output_type else "")

login(token=os.getenv("HF_TOKEN"))

# Tải công cụ tạo hình ảnh từ văn bản
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
python_executor = PythonExecutor()
python_interpreter = PythonInterpreterTool(executor=python_executor)

 # Unpack the tools from the collection
tools_list = [python_interpreter, image_generation_tool, DuckDuckGoSearchTool()]
print_tools(tools_list)
print("===================================================")
resources = {"tools": tools_list}
# agent = CodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)
retriever = ToolRetriever()
api_key = os.getenv("OPEN_AI_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)  # Replace with your actual API key
guideline = """
1. Focus on relevance to the query.
2. Select tools that can directly address the user's needs.
3. Avoid including irrelevant items.
"""
list_pairs = [("Write a programme to analyze my csv files to count how many items was sale on Sep 30th 2025.", "Data analytic"), 
("Generate an image showing the superman standing on top of Mount Fuji", "Photo Editting"), 
("Given my uploaded images. Blur region of a cat in my uploaded image.", "Photo Editting"),
("Find the capital of France", "General Knowledge"),
("Find images of Effel Tower and then generate an image showing the superman flying to top of Effel Tower", "Photo Editting")]
for query, topic in list_pairs:
    print(f"\nQuery: {query}\nTopic: {topic}")
    selected = retriever.prompt_based_retrieval(query, resources, topic=topic, llm=llm, guideline=guideline)
    print("Selected Resources:", selected)