"""
ToolBrain Training Example - Use RL of ToolBrain for HPO.
The agent needs to call the  run_lightgbm(feature_fraction: float) function defined in the file lightgbm_model.py
with the right feature_faction value ranging from 0 to 1.
This example is similar to the HPO example with GRPO, the only difference is the change from GRPO to DPO

"""

from typing import Any

from smolagents import CodeAgent, TransformersModel, tool
from toolbrain import Brain, Trace
from lightgbm_model import run_lightgbm


# --- 2. Prepare Training Data ---
training_dataset = [
    {
        "query": "Use the run_lightgbm tool with a suggested value of feature_fraction (must be between 0.0 and 1.0)."
    }
]


# customised reward function, just return the lightgbm accuracy output, 0 otherwise
def reward_accuracy(trace: Trace, **kwargs: Any) -> float:
    for turn in trace:
        try:
            reward = float(turn["action_output"]) # output of the function run_lightgbm
            return reward
        except:
            # failed to call the function set reward to 0
            reward = 0.0
        return reward


def main():
    print("🧠 ToolBrain Flexible Training Example with DPO")
    print("=" * 60)

    print("🤖 User is creating their own agent...")

    print("📥 Initializing TransformersModel...")
    trainable_model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B-Instruct",
                                        max_new_tokens=128)

    print("✅ TransformersModel initialized.")

    print("🔧 Creating CodeAgent...")
    my_agent = CodeAgent(
        tools=[run_lightgbm],
        model=trainable_model,
        max_steps=1
    )
    print("✅ Agent created.")

    # User only needs to pass their agent to Brain
    brain = Brain(
        agent=my_agent,
        reward_func=reward_accuracy,
        algorithm="DPO"
    )

    brain.train(training_dataset, num_iterations=10)


if __name__ == "__main__":
    main()