"""
Brain module - The flexible, user-friendly interface for ToolBrain.

This module contains the Brain class which orchestrates the training process.
It automatically detects the agent type and uses the appropriate adapter.
"""

from typing import Any, List, Dict, Union
from .core_types import Trace, RewardFunction, BatchRewardFunction
from .adapters import BaseAgentAdapter, SmolAgentAdapter
from .rl.grpo import GRPOAlgorithm, Policy 
from .rewards import RewardFunctionWrapper
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
        reward_func: Union[RewardFunction, BatchRewardFunction, RewardFunctionWrapper],
        config: Dict[str, Any],
        learning_algorithm: str = "GRPO",
    ):
        """
        Initializes the Brain by automatically selecting the correct adapter for the agent.
        """
        self.config = config
        
        # Auto-wrap reward function if needed
        if isinstance(reward_func, RewardFunctionWrapper):
            self.reward_func = reward_func
        else:
            self.reward_func = RewardFunctionWrapper(reward_func)
            
        self.learning_algorithm = learning_algorithm
        
        # Store original agent type for flexible return in get_agent()
        self.original_agent_type = type(agent)
        
        print(f"🧠 Initializing Brain for agent of type '{self.original_agent_type.__name__}'...")

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
                initial_policy=policy, 
                config=config
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
            return SmolAgentAdapter(agent=agent_instance, config=self.config)
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
        print(f"\n🔄 Training step for query: '{query[:50]}...'")
        num_group_members = self.config.get("num_group_members", 10)
        
        traces: List[Trace] = []
        rewards: List[float] = []
        rl_inputs: List[Any] = []

        print(f"  📊 Collecting {num_group_members} traces...")
        for i in range(num_group_members):
            try:
                print(f"    📝 Trace {i+1}/{num_group_members}")
                trace, rl_input = self.agent_adapter.run(query)
                traces.append(trace)
                rl_inputs.append(rl_input)
            except Exception as e:
                print(f"    ❌ Error during agent iteration: {e}")
                continue
        
        if not traces:
            print(f"⚠️ No successful traces collected for query: '{query}'. Skipping training step.")
            return
        
        # Compute rewards using batch scoring (supports both single and batch functions)
        print(f"  🎯 Computing rewards for {len(traces)} traces...")
        if self.reward_func.is_batch_function:
            print(f"      Using batch reward function")
        else:
            print(f"      Using single-trace reward function")
            
        try:
            rewards = self.reward_func.get_batch_scores(traces, **reward_kwargs)
            for i, reward in enumerate(rewards):
                print(f"      🎯 Trace {i+1} Reward: {reward:.3f}")
        except Exception as e:
            print(f"    ❌ Error computing rewards: {e}")
            rewards = [0.0] * len(traces)
        
        print(f"  🧠 Running RL training step with {len(traces)} traces...")
        self.rl_module.train_step(rl_inputs, rewards)
        print(f"  ✅ RL training step completed")

    def get_agent(self) -> Any:
        """
        Returns the trained agent with the same type as the input agent.
        
        The returned agent contains the fine-tuned model and preserves
        the original agent's interface and methods. This method is flexible
        and works with any agent type supported by ToolBrain adapters.
        
        Returns:
            The trained agent with the same type as the original input agent.
            For example:
            - If input was CodeAgent -> returns CodeAgent
            - If input was ConversableAgent -> returns ConversableAgent
            - If input was CustomAgent -> returns CustomAgent
        """
        return self.agent_adapter.agent
    
    def get_agent_type(self) -> type:
        """
        Returns the original agent type that was passed to the Brain.
        
        This is useful for type checking or understanding what type
        of agent the Brain is working with.
        
        Returns:
            The type of the original agent (e.g., CodeAgent, ConversableAgent, etc.)
        """
        return self.original_agent_type
    
    @staticmethod
    def is_agent_supported(agent: Any) -> bool:
        """
        Check if an agent type is supported by ToolBrain.
        
        This method can be used to validate agent compatibility
        before creating a Brain instance.
        
        Args:
            agent: The agent instance to check
            
        Returns:
            True if the agent type is supported, False otherwise
        """
        try:
            # Try to get adapter for the agent
            if isinstance(agent, CodeAgent):
                return True
            # Future: add more agent type checks here
            # elif isinstance(agent, ConversableAgent):
            #     return True
            # elif isinstance(agent, LLMChain):
            #     return True
            else:
                return False
        except Exception:
            return False