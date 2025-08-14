"""
Brain module - The all-in-one interface for ToolBrain.

This module contains the Brain class which encapsulates the entire process of
agent creation, training orchestration, reward calculation, and RL updates.
"""

from typing import Any, Callable, List, Dict
from .core_types import Trace, RewardFunction
from .adapters import SmolAgentAdapter, BaseAgentAdapter
from .rl.grpo import GRPOAlgorithm 
from smolagents import CodeAgent, TransformersModel

class Brain:
    """
    The all-in-one factory and trainer for ToolBrain agents.

    This class hides all implementation details. Users interact with this
    single class to configure, train, and retrieve their agent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the entire system from a single configuration dictionary.
        
        Args:
            config (Dict[str, Any]): A dictionary containing all necessary settings.
                Required keys:
                    - "model_id": str (e.g., "HuggingFaceTB/SmolLM-135M-Instruct")
                    - "tools": List[Callable]
                    - "reward_func": RewardFunction
                Optional keys:
                    - "learning_algorithm": str (default: "GRPO")
                    - "rl_config": dict (hyperparameters for the RL algorithm)
                    - "max_turns": int (for the agent adapter)
        """
        self.config = config
        print("🧠 Initializing ToolBrain...")

        # --- 1. Auto-initialize trainable model ---
        model_id = self.config.get("model_id")
        if not model_id:
            raise ValueError("Config must include a 'model_id'.")
        
        print(f"   - Loading trainable model: {model_id}...")
        self.model = TransformersModel(model_id=model_id)
        print("   ✅ Model loaded.")

        # --- 2. Auto-initialize base agent ---
        tools = self.config.get("tools", [])
        print(f"   - Initializing base CodeAgent with {len(tools)} tools...")
        original_agent = CodeAgent(tools=tools, model=self.model)
        print("   ✅ Base agent created.")

        # --- 3. Auto-initialize internal adapter ---
        print("   - Creating internal agent adapter...")
        
        self.agent_adapter = SmolAgentAdapter(
            agent=original_agent,
            max_turns=self.config.get("max_turns", 5)
        )
        print("   ✅ Adapter created.")

        # --- 4. Auto-initialize RL module ---
        learning_algorithm = self.config.get("learning_algorithm", "GRPO")
        print(f"   - Initializing RL algorithm: {learning_algorithm}...")
        
        self.reward_func = self.config.get("reward_func")
        if not self.reward_func:
            raise ValueError("Config must include a 'reward_func'.")

        if learning_algorithm == "GRPO":
            policy = Policy(llm=self.model.model, tokenizer=self.model.tokenizer)
            self.rl_module = GRPOAlgorithm(
                policy=policy, 
                config=self.config.get("rl_config", {})
            )
        else:
            raise NotImplementedError(f"Algorithm '{learning_algorithm}' is not supported.")
        print("   ✅ RL module initialized.")
        
        print("\n✅ Brain is ready for training.")

    def train(self, dataset: List[Dict[str, Any]], num_iterations: int = 10):
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