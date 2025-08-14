"""
Brain module - The flexible, user-friendly interface for ToolBrain.

This module contains the Brain class which orchestrates the training process.
It automatically detects the agent type and uses the appropriate adapter.
"""

from typing import Any, List, Dict
from .core_types import Trace, RewardFunction
from .adapters import BaseAgentAdapter, SmolAgentAdapter
from .rl.grpo import GRPOAlgorithm, Policy 
from smolagents import CodeAgent


class Brain:
    """
    The flexible and intelligent trainer for ToolBrain agents.

    Users provide their pre-configured agent, and the Brain automatically
    handles the complexities of trace capture and RL training.
    """
    
    def __init__(
        self,
        agent: Any, # Any agent instance
        reward_func: RewardFunction,
        learning_algorithm: str = "GRPO",
        rl_config: Dict[str, Any] = None
    ):
        """
        Initializes the Brain by automatically selecting the correct adapter for the agent.
        """
        self.reward_func = reward_func
        self.learning_algorithm = learning_algorithm
        print(f"🧠 Initializing Brain for agent of type '{type(agent).__name__}'...")

        # --- "Adapter Factory" automatically ---
        self.agent_adapter = self._get_adapter_for_agent(agent)
        print(f"   ✅ Using adapter: {type(self.agent_adapter).__name__}")
        
        # Get trainable model from adapter
        trainable_model = self.agent_adapter.get_trainable_model()
        
        # --- Initialize RL module ---
        print(f"   - Initializing RL algorithm: {learning_algorithm}...")
        if learning_algorithm == "GRPO":
            policy = Policy(llm=trainable_model.model, tokenizer=trainable_model.tokenizer)
            self.rl_module = GRPOAlgorithm(
                policy=policy, 
                config=rl_config or {}
            )
        else:
            raise NotImplementedError(f"Algorithm '{learning_algorithm}' is not supported.")
        print("   ✅ RL module initialized.")
        
        print("\n✅ Brain is ready for training.")

    def _get_adapter_for_agent(self, agent_instance: Any) -> BaseAgentAdapter:
        """
        Factory method to automatically select the appropriate adapter for the given agent.
        """
        if isinstance(agent_instance, CodeAgent):
            return SmolAgentAdapter(agent=agent_instance)
        # Future example:
        # elif isinstance(agent_instance, AutoGenAgent):
        #     return AutoGenAdapter(agent=agent_instance)
        else:
            raise TypeError(f"Agent type '{type(agent_instance).__name__}' is not supported yet.")

    def train(self, dataset: List[Dict[str, Any]], num_iterations: int = 1):
        """
        Runs the full training process on a dataset.
        
        Args:
            dataset: A list of training examples, where each example is a dict
                     (e.g., {"query": "...", "gold_answer": "..."}).
            num_iterations: The number of training iterations (epochs).
        """
        print("\n🚀 Starting training...")
        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            
            for example in dataset:
                query = example.get("query")
                if not query:
                    continue
                
                self.train_step(query=query, reward_kwargs=example)
        
        print("\n🎉 Training finished!")

    def train_step(self, query: str, reward_kwargs: Dict[str, Any]):
        """Executes a single training step for a given query."""
        num_group_members = self.config.get("num_group_members", 10)
        
        traces: List[Trace] = []
        rewards: List[float] = []
        
        for _ in range(num_group_members):
            try:
                trace = self.agent_adapter.run(query)
                reward = float(self.reward_func(trace=trace, **reward_kwargs))
                traces.append(trace)
                rewards.append(reward)
            except Exception as e:
                print(f"    ❌ Error during agent iteration: {e}")
                continue
        
        if not traces:
            print(f"⚠️ No successful traces collected for query: '{query}'. Skipping training step.")
            return
        
        self.rl_module.train_step(traces, rewards)

    def get_agent(self) -> CodeAgent:
        """
        Returns the trained agent.
        
        The returned agent contains the fine-tuned model.
        """
        return self.agent_adapter.agent
