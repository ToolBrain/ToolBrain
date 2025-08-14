"""
Brain module - Core training orchestration for ToolBrain.

This module contains the Brain class which orchestrates the training process
by coordinating agent execution, reward calculation, and RL algorithm updates.

The Brain now uses the Adapter pattern for clean separation of concerns.
"""

from typing import Any, Callable, List, Protocol
from .core_types import Trace, TraceStep, RewardFunction
from .adapters import BaseAgentAdapter


class RLAlgorithm(Protocol):
    """Protocol defining the interface for RL algorithms."""
    def train_step(self, traces: List[Trace], rewards: List[float]) -> None:
        """Perform a single training step given traces and rewards."""
        ...


class MockRLAlgorithm:
    """Mock RL algorithm for development and testing."""
    
    def __init__(self, algorithm_name: str = "MockRL") -> None:
        self.algorithm_name = algorithm_name
        self.training_steps = 0
    
    def train_step(self, traces: List[Trace], rewards: List[float]) -> None:
        """Simulate a training step by logging trace and reward information."""
        self.training_steps += 1
        
        print("=" * 60)
        print(f"🧠 {self.algorithm_name} - Training Step #{self.training_steps}")
        print("=" * 60)
        print(f"📊 Batch Size: {len(traces)} traces")
        avg = (sum(rewards) / len(rewards)) if rewards else 0.0
        rmin = min(rewards) if rewards else 0.0
        rmax = max(rewards) if rewards else 0.0
        print(f"📈 Average Reward: {avg:.3f}")
        print(f"📊 Reward Range: [{rmin:.3f}, {rmax:.3f}]")
        print()
        print("📋 Training Data Summary:")
        for i, (trace, reward) in enumerate(zip(traces, rewards)):
            print(f"  Trace {i+1}: {len(trace)} turns, reward = {reward:.3f}")
            # Show first few turns for each trace
            for j, turn in enumerate(trace[:3]):  # Show first 3 turns
                print(f"    Turn {j+1}:")
                print(f"      Prompt: {turn['prompt_for_model'][:50]}...")
                print(f"      Completion: {turn['model_completion'][:50]}...")
                
                # Show parsed completion details
                parsed = turn['parsed_completion']
                if parsed.get('thought'):
                    thought_preview = parsed['thought'][:50] + "..." if len(parsed['thought']) > 50 else parsed['thought']
                    print(f"      Thought: {thought_preview}")
                if parsed.get('tool_code'):
                    code_preview = parsed['tool_code'][:50] + "..." if len(parsed['tool_code']) > 50 else parsed['tool_code']
                    print(f"      Tool Code: {code_preview}")
                if parsed.get('final_answer'):
                    answer_preview = parsed['final_answer'][:50] + "..." if len(parsed['final_answer']) > 50 else parsed['final_answer']
                    print(f"      Final Answer: {answer_preview}")
                
                if turn['tool_output']:
                    output_preview = turn['tool_output'][:50] + "..." if len(turn['tool_output']) > 50 else turn['tool_output']
                    print(f"      Tool Output: {output_preview}")
            
            if len(trace) > 3:
                print(f"    ... and {len(trace) - 3} more turns")
            print()
        
        print()
        print("🔄 Mock training completed. Real RL algorithm would update model weights here.")
        print("=" * 60)


class GRPOAlgorithm:
    """Lightweight trainer that wraps GRPO optimization around a Policy.

    Usage:
        algo = GRPOAlgorithm(policy)
        algo.train_step(traces, rewards)
    """
    def __init__(
        self,
        policy: Policy,
        config: dict = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.training_steps = 0
        self.device = next(policy.llm.parameters()).device
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
        traces: List[List[Example]],
        rewards: List[float],
    ) -> Policy:
        """Run one GRPO update over a batch of traces.

        Args:
            traces: batch of traces; each trace is a list of (prompt, completion) pairs.
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        completion_mask = batch["completion_mask"].to(device)
        advantages = batch["advantages"].to(device)

        with torch.no_grad():
            pi_ref_logps = self.policy.get_log_probs(input_ids, attention_mask)

        for _ in range(self.config["mu"]):
            pi_theta_logps = pi_theta.get_log_probs(input_ids, attention_mask)
            loss = grpo_loss(
                pi_theta_log_probs=pi_theta_logps,
                pi_ref_log_probs=pi_ref_logps,
                advantages=advantages,
                epsilon=self.config["epsilon"],
                beta=self.config["beta"],
                completion_mask=completion_mask
            )
            pi_theta = self._update_policy(pi_theta, loss)
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
    Core training orchestrator for ToolBrain.
    
    The Brain now uses the Adapter pattern for clean separation of concerns.
    Users pass an agent adapter that conforms to the BaseAgentAdapter interface,
    making the system more explicit, testable, and extensible.
    """
    
    def __init__(
        self,
        agent_adapter: BaseAgentAdapter,
        reward_func: RewardFunction,
        learning_algorithm: str = "MockRL"
    ) -> None:
        """
        Initialize the Brain with an agent adapter.
        
        Args:
            agent_adapter: An adapter that conforms to BaseAgentAdapter interface
            reward_func: Flexible reward function callable (see RewardFunction protocol)
            learning_algorithm: Name of the RL algorithm to use
        """
        if not isinstance(agent_adapter, BaseAgentAdapter):
            raise TypeError(
                f"Expected BaseAgentAdapter instance, got {type(agent_adapter)}. "
                "Use an adapter like SmolAgentAdapter to wrap your agent."
            )
        
        self.agent_adapter = agent_adapter
        self.reward_func = reward_func
        self.learning_algorithm = learning_algorithm
        
        self.rl_module: RLAlgorithm = MockRLAlgorithm(learning_algorithm)
        
        print(f"🧠 Brain initialized with {learning_algorithm} algorithm")
        print(f"✅ Using agent adapter: {type(agent_adapter).__name__}")
    
    def train_step(
        self,
        query: str,
        num_group_members: int = 10,
        **reward_kwargs: Any
    ) -> None:
        """
        Execute a single training step.
        
        Args:
            query: The input query for the agent
            num_group_members: Number of agent runs to collect for training
            **reward_kwargs: Additional keyword arguments forwarded to reward_func
                              (e.g., gold_answer, judge_client, constraints, etc.)
        """
        print(f"\n🚀 Starting training step with query: '{query}'")
        print(f"👥 Collecting {num_group_members} traces...")
        
        traces: List[Trace] = []
        rewards: List[float] = []
        
        for i in range(num_group_members):
            print(f"  🔄 Running agent iteration {i+1}/{num_group_members}...")
            try:
                trace = self.agent_adapter.run(query)
                # Forward query and any provided kwargs to the reward function
                reward = float(self.reward_func(trace=trace, query=query, **reward_kwargs))
                traces.append(trace)
                rewards.append(reward)
                print(f"    ✅ Trace {i+1}: {len(trace)} steps, reward = {reward:.3f}")
            except Exception as e:
                print(f"    ❌ Error in iteration {i+1}: {e}")
                continue
        
        if not traces:
            raise RuntimeError("No successful traces collected for training")
        
        avg_reward = sum(rewards) / len(rewards)
        print(f"\n📊 Collected {len(traces)} traces with average reward {avg_reward:.3f}")
        
        self.rl_module.train_step(traces, rewards)
        print("\n✅ Training step completed!")
    
    def get_training_stats(self) -> dict:
        """Get current training statistics."""
        return {
            "algorithm": self.learning_algorithm,
            "training_steps": self.rl_module.training_steps,
            "adapter_type": type(self.agent_adapter).__name__
        } 
