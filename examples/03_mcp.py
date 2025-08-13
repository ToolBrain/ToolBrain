"""
Simple ToolBrain example - Ideal user experience.

This script demonstrates how easy it is to use ToolBrain:
1. Create a normal CodeAgent
2. Pass it to Brain
3. Run training

"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import os
from typing import List

from mcp import StdioServerParameters
from smolagents import CodeAgent, InferenceClientModel, MCPClient, tool

from toolbrain import Brain
from toolbrain.rewards import reward_exact_match

TASK_DESCRIPTION = "Generate training examples for a medical question answering task."
NUM_EXAMPLES = 1

# Initialize server parameters
server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)


def generate_training_examples(llm, tools: List, num_examples: int) -> list:
    tools_description = "\n".join([tool.description for tool in tools])
    prompt = f"""You are an expert at creating realistic scenarios for testing AI agents that interact with MCP (Model Context Protocol) servers.

Given the following available tools and resources from an MCP server, generate {num_examples} diverse, realistic scenarios that a user might want to accomplish using these tools.

AVAILABLE TOOLS:
{tools_description}

Requirements for scenarios:
1. Each scenario should be a task that can be accomplished using the available tools
2. Scenarios should vary in complexity - some simple (1-2 tool calls), some complex (multiple tool calls)
3. Scenarios should cover different use cases and tool combinations (though the task should not specify which tools to use)
4. Each scenario should be realistic - something a real user might actually want to do
5. Assign a difficulty rating from 1 (easy, single tool call) to 5 (hard, complex multi-step analysis)
6. The task should always include generating a summary of the work done and a thorough analysis and report of the results

You must respond with a JSON object containing a "scenarios" array of exactly {num_examples} objects. Each object must have:
- "task": string describing the scenario
- "difficulty": integer from 1-5 representing complexity"""

    response_schema = {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["task", "difficulty"],
                    "additionalProperties": False,
                },
                "minItems": num_examples,
                "maxItems": num_examples,
            }
        },
        "required": ["scenarios"],
        "additionalProperties": False,
    }
    response = llm(
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scenario_list", "schema": response_schema},
        },
    )
    try:
        result = json.loads(response)
    except Exception as e:
        print("Failed to parse JSON from model response.")
        raise

    # Extract scenarios
    if "scenarios" in result:
        scenarios = result["scenarios"]
    else:
        scenarios = result if isinstance(result, list) else list(result.values())[0]

    return [scenario["task"] for scenario in scenarios]


def main() -> None:
    """Demonstrate the ideal ToolBrain user experience."""

    print("ToolBrain Training Example")
    print("=" * 50)

    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "❌ HF_TOKEN environment variable is required!\n"
            "Create a .env file with: HF_TOKEN=your_token_here\n"
            "Get a token at: https://huggingface.co/settings/tokens"
        )

    print("✅ HF_TOKEN found")

    # Initialize model
    print("🔗 Connecting to Hugging Face Inference API...")
    model = InferenceClientModel(model_id="HuggingFaceTB/SmolLM3-3B", token=hf_token)
    print("✅ Model initialized")

    mcp_client = MCPClient(server_parameters)
    tools = mcp_client.get_tools()

    print(f"🛠️  Created {len(tools)} tools: {[tool.name for tool in tools]}")

    # --- USER EXPERIENCE ---
    # User creates a completely normal CodeAgent
    print("🤖 Creating CodeAgent...")
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    print("✅ Agent created")

    # User simply passes their agent to Brain
    # Brain automatically handles all trace instrumentation
    print("🧠 Initializing Brain...")
    brain = Brain(
        agent=agent, reward_func=reward_exact_match, learning_algorithm="MockDPO"
    )

    # User runs training
    print("\n🚀 Running training step...")

    queries = generate_training_examples(
        agent=agent, task_description=TASK_DESCRIPTION, num_examples=NUM_EXAMPLES
    )

    for i, query in enumerate(queries):
        try:
            brain.train_step(query=query, num_group_members=3)

            # Show training statistics
            stats = brain.get_training_stats()
            print(f"\n📊 Training Statistics:")
            print(f"  Algorithm: {stats['algorithm']}")
            print(f"  Training Steps: {stats['training_steps']}")

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return

        print(f"\n✅ Training step {i} completed successfully!")


if __name__ == "__main__":
    main()
