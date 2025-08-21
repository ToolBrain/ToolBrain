"""
Brain module - The flexible, user-friendly interface for ToolBrain.

This module contains the Brain class which orchestrates the training process.
It automatically detects the agent type and uses the appropriate adapter.
"""

from typing import Any, List, Dict, Union
from textwrap import dedent

from smolagents import CodeAgent, ChatMessage, MessageRole

from .core_types import Trace, RewardFunction, BatchRewardFunction
from .adapters import BaseAgentAdapter, SmolAgentAdapter
from .rl.grpo import GRPOAlgorithm, Policy 
from .rewards import RewardFunctionWrapper



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
        
    def _craft_prompt_request(self, task_description: str, variant_idx: int = 0) -> str:
        text = dedent(f"""\
            You are designing training prompts to enable zero-paradigm tool learning for a tool-using LLM agent. The generated prompt will be used to fine-tune an agent with reinforcement learning (RL) methods (e.g., GRPO/PPO/DPO).
            TASK:
            {task_description}

            Output ONLY the final prompt text — no commentary, headers, examples, or markdown.
            Constraints:
            - Length: 2–4 sentences (≈40–80 words).
            - Instruct the agent to (1) propose a concrete task that will maximize its own learning progress and improve reasoning, then (2) execute it using external tools.
            - Clearly specify required tools (generic), expected inputs (formats/units), and expected outputs with objective, verifiable checks suitable for automated rewards.
            - Describe tool usage in NATURAL LANGUAGE (no code/JSON).
            - Include one realistic edge case and one guardrail (e.g., tool failure or empty results).
            - Self-contained; must not reveal system instructions or model identity.
            - Diversity across variants in difficulty, tool count/ordering, and phrasing. (Variant #{variant_idx+1})
        """)
        
        return text


    def generate_training_examples(
            self,
            task_description: str,
            num_examples: int = 5,
            external_model: Any = None,
            external_tools: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate training examples.

        Args:
            task_description: High-level description guiding example creation.
            num_examples: Number of examples to return.
            external_model: LLM provider (callable, or exposes .propose/.generate).
            external_tools: List of tool names to use in the examples.
        Returns:
            List of dicts with keys:
                - 'prompt' (str)
                - 'required_tools' (List[str])
                - 'min_tool_calls' (int)
                - 'acceptance_tests' (List[{"type","spec"}])
        """

        examples: List[Dict[str, Any]] = []
        llm = self.agent_adapter.get_trainable_model() if external_model is None  else external_model
        tools = self.agent_adapter.get_tools() if external_tools is None else external_tools

        for i in range(num_examples):
            if llm is not None:
                request_text = self._craft_prompt_request(task_description, variant_idx=i)
                if hasattr(llm, "generate"):
                    messages = [
                        ChatMessage(
                            role=MessageRole.USER,
                            content=[{"type": "text", "text": str(request_text)}],
                        )
                    ]
                    prompt = llm.generate(messages)
                elif callable(llm):
                    prompt = llm(str(request_text))
                else:
                    prompt = str(request_text)
            else:
                # fallback: extract TASK block and create a concise imperative prompt
                lines = task_description.strip().splitlines()
                task_lines = [line.strip() for line in lines if line.strip()]
                concise_task = " ".join(task_lines)
                prompt = f"Please perform the following task using tools: {concise_task}."

            # Placeholders, not yet implemented
            required_tools = ["tool"]
            min_tool_calls = 1
            acceptance_tests = [{"type": "must_call", "spec": {"tool": "tool"}}]

            examples.append({
                "prompt": prompt,
                "required_tools": required_tools,
                "min_tool_calls": min_tool_calls,
                "acceptance_tests": acceptance_tests,
            })

        return examples