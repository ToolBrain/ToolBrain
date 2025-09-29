import os
from toolbrain import Brain, create_agent
from toolbrain.rewards import reward_exact_match
from smolagents import tool
from toolbrain.retriever import ToolRetriever
from openai import OpenAI

@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.
    
    Args:
        a: First integer to add
        b: Second integer to add
        
    Returns:
        The sum of a and b
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.
    
    Args:
        a: First integer to multiply
        b: Second integer to multiply
        
    Returns:
        The product of a and b
    """
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """
    Divide two integers.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        The quotient of a divided by b
    """
    return a / b

@tool
def subtract(a: int, b: int) -> int:
    """
    Subtract two integers.
    
    Args:
        a: First integer
        b: Second integer to subtract from a
        
    Returns:
        The difference of a minus b
    """
    return a - b

# @tool
# def calculate(a: int, b: int, operation: str) -> int:
#     """
#     Calculate the result of a mathematical operation.
    
#     Args:
#         a: First integer
#         b: Second integer
#         operation: The operation to perform
#     Returns:
#         The result of the mathematical operation
#     """
#     return a + b if operation == "add" else a * b if operation == "multiply" else a / b if operation == "divide" else a - b if operation == "subtract" else None

if __name__ == "__main__":

    # Create agent with local model
    agent = create_agent(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",  # Use a smaller local model
        tools=[add, multiply, divide, subtract]
    )
    
    # Initialize the tool retriever
    retrieval_llm_instance = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create
    retrieval_llm_model = "gpt-4o-mini" 
    tool_retriever = ToolRetriever(llm_model=retrieval_llm_model, llm_instance=retrieval_llm_instance)

    # Initialize the training brain  
    brain = Brain(
        agent=agent,
        reward_func=reward_exact_match,
        algorithm="GRPO",
        enable_tool_retrieval=True,
        retrieval_topic="mathematics",
        retrieval_guidelines="Select only tools/functions needed for mathematical calculations",
        tool_retriever=tool_retriever
    )

    # Prepare training dataset
    training_dataset = [
        {
            "query": "Calculate 5 + 7",
            "gold_answer": "12"
        },
        {
            "query": "What is 8 multiplied by 6?",
            "gold_answer": "48"
        },
        {
            "query": "Calculate 8 divided by 2",
            "gold_answer": "4"
        },
        {
            "query": "Calculate 8 subtracted from 2",
            "gold_answer": "6"
        }
    ]

    # Run training
    brain.train(training_dataset, num_iterations=1)

    trained_agent = brain.get_agent()

    print(trained_agent.run("Calculate 6 + 9"))
    print(trained_agent.run("What is 4 multiplied by 2?"))


