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

class GRPOAlgorithm:
    """Lightweight trainer that wraps GRPO optimization around a Policy.

    Usage:
        algo = GRPOAlgorithm(policy)
        algo.train_step(traces, rewards)
    """
    def __init__(
        self,
        initial_policy: Policy,
        ref_policy: Policy = None,
        config: dict = None,
    ) -> None:
        self.policy = initial_policy
        # the reference model, usually the initial Supervised Fine-Tuning model
        self.pi_ref = ref_policy if ref_policy else copy_model(initial_policy) 
        self.config = config
        self.training_steps = 0
        self.device = next(initial_policy.llm.parameters()).device
        self.optimizer = torch.optim.AdamW(self.policy.llm.parameters(), lr=self.config["lr"])

    def _update_policy(self, pi_theta, loss):
        """Apply one optimizer step using the algorithm's optimizer and gradient clipping."""
        model = getattr(pi_theta, "llm", None) or getattr(pi_theta, "model", None)
        if model is None:
            raise AttributeError("No model found in pi_theta")
        
        model.train()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()
        return pi_theta

    def train_step(
        self,
        traces: List[Trace],
        rewards: List[float],
    ) -> Policy:
        """Run one GRPO update over a batch of traces.

        Args:
            traces: batch of traces; each trace is a list Trace.
            rewards: list of scalar rewards, one per trace.
        Returns:
            Updated Policy (also stored in self.policy).
        """
        device = self.device
        pi_theta = copy_model(self.policy).to(self.device)
        assert len(traces) == len(rewards)

        batch = build_inputs(
            traces=traces,
            rewards=rewards,
            tokenizer=pi_theta.tokenizer
        )
 
        input_ids = batch.input_ids.to(device) # shape: (N, T)
        attention_mask = batch.attention_mask.to(device) # shape: (N, T)
        completion_mask = batch.completion_mask.to(device) # shape: (N, T)
        advantages = batch.advantages.to(device) # shape: (N, T)

        # Prepare old-policy (for ratio) and a fixed reference (for KL) log-probs.
        #   - pi_old_logps: starts as current pre-update policy; will be refreshed each grpo loss step.
        #   - pi_ref_logps: fixed reference for KL across the grpo iteration (use pre-update self.policy).
        with torch.no_grad():
            pi_old_logps = pi_theta.get_log_probs(input_ids, attention_mask) # shape: (N, T-1)
            pi_ref_logps = self.pi_ref.get_log_probs(input_ids, attention_mask) # shape: (N, T-1)

        for _ in range(self.config["mu"]):
            # Current policy log-probs
            pi_theta_logps = pi_theta.get_log_probs(input_ids, attention_mask) # shape: (N, T-1)

            loss = grpo_loss(
                pi_theta_log_probs=pi_theta_logps,
                pi_theta_old_log_probs=pi_old_logps,
                pi_ref_log_probs=pi_ref_logps,
                advantages=advantages[:,1:], # shape: (N, T-1)
                epsilon=self.config["epsilon"],
                beta=self.config["beta"],
                completion_mask=completion_mask[:,1:], # shape: (N, T-1)
            )

            # Apply update
            pi_theta = self._update_policy(pi_theta, loss)

            # Cache current log-probs as next step's old-policy (detach from graph)
            pi_old_logps = pi_theta_logps.detach()

        self.policy = pi_theta
        self.training_steps += 1
        return pi_theta

    def __repr__(self) -> str:
        return (
            f"GRPOAlgorithm(epsilon={self.config.get('epsilon')}, beta={self.config.get('beta')}, "
            f"mu={self.config.get('mu')}, steps={self.training_steps})"
        )


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
