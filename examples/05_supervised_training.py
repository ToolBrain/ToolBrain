"""
ToolBrain Training Example - Simplified API with supervised learning

This script demonstrates the new, simplified ToolBrain API.
1. Define a configuration dictionary.
2. Pass the config to the Brain.
3. Call brain.train().
"""

from typing import List

from dotenv import load_dotenv

from toolbrain.core_types import ChatSegment

load_dotenv()

from smolagents import CodeAgent, TransformersModel
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match


# for supervised learning each training example is a list of ChatSegments with text and role, assistant's text is
# considered as gold answers for supervised learning, the learning can be multi-turns
training_datasets: List[List[ChatSegment]] = [
        [
            ChatSegment(
                role="other",
                text="You are a Python assistant. Compute the sum of 1..10 and explain briefly.",
            ),
            ChatSegment(
                role="assistant",
                text="The sum of numbers from 1 to 10 is 55, using the formula n(n+1)/2."
            ),
        ],
        [
            ChatSegment(
                role="other",
                text="Translate 'Good morning' into French.",
            ),
            ChatSegment(
                role="assistant",
                text="In French, it is 'Bonjour'."
            ),
        ],
    ]


def main():
    print("🧠 ToolBrain Flexible Training Example")
    print("=" * 60)

    print("🤖 User is creating their own agent...")

    print("📥 Initializing TransformersModel...")
    trainable_model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=512)
    print("✅ TransformersModel initialized.")

    print("🔧 Creating CodeAgent...")
    my_agent = CodeAgent(
        tools=[],
        model=trainable_model
    )
    print("✅ Agent created.")

    # User only needs to pass their agent to Brain
    brain = Brain(
        agent=my_agent,
        reward_func=reward_exact_match,
        algorithm="Supervised"
    )

    brain.train(training_datasets, num_iterations=5)


if __name__ == "__main__":
    main()