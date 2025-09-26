"""
Brain module - The flexible, user-friendly interface for ToolBrain.

This module contains the Brain class which orchestrates the training process.
It automatically detects the agent type and uses the appropriate adapter.
"""

import gc
import json
import os
import re
from collections import deque
from typing import Any, List, Dict, Union, Tuple, Optional, Callable

import torch
import numpy as np
from textwrap import dedent
from smolagents import CodeAgent, ChatMessage, MessageRole, TransformersModel

from .learning.supervised.algo import SupervisedAlgorithm
from .rewards import RewardFunctionWrapper, create_reward_function, reward_exact_match
from .core_types import Trace, RewardFunction, BatchRewardFunction
from .adapters import BaseAgentAdapter, SmolAgentAdapter
from .learning.dpo import DPOAlgorithm, make_dpo_pairs
from .learning.grpo import GRPOAlgorithm
from .learning import Policy
from .config import BaseConfig, get_config
from .prompt import (
    build_prompt_to_generate_training_examples,
    validate_model,
    validate_tools,
    tools_to_card,
)


GRPOALiasNames = ["GRPO", "grpo"]
DPOALiasNames = ["DPO", "dpo"]
SupervisedALiasNames = ["Supervised", "supervised", "supervise"]


class Brain:
    """
    The simple and powerful trainer for ToolBrain agents.
    
    Just one constructor with all important parameters as keyword arguments.
    No need to understand config classes or factory methods.
    """
    
    def __init__(
        self,
        agent: Any,
        *,  # Force all parameters after agent to be keyword-only
        # === Core Training Settings ===
        algorithm: str = "GRPO", 
        learning_rate: float = 3e-5,
        batch_size: int = 1,
        
        # === GRPO Specific Parameters ===
        epsilon: float = 0.2,          # GRPO clip ratio
        num_group_members: int = 2,    # Number of traces per training step (reduced for memory efficiency)
        
        # === DPO Specific Parameters ===  
        beta: float = 0.1,             # DPO temperature parameter
        loss_type: str = "sigmoid",    # DPO loss type
        label_smoothing: float = 0.0,  # DPO label smoothing
        
        # === Optimization Settings ===
        max_grad_norm: float = 1.0,    # Gradient clipping
        use_bitsandbytes: bool = False, # Memory optimization
        
        # === Reward & Tools ===
        reward_func: Optional[Union[RewardFunction, BatchRewardFunction, RewardFunctionWrapper]] = None,
        enable_tool_retrieval: bool = False,
        retrieval_topic: str = "general",       # Topic for tool retrieval
        retrieval_guidelines: str = "",         # Custom guidelines for tool selection
        
        # === Advanced (Optional) ===
        config: Optional[BaseConfig] = None  # For power users only
    ):
        """
        Initialize Brain with simple, self-documenting parameters.
        
        Args:
            agent: Pre-configured agent instance
            
            # === Core Training Settings ===
            algorithm: Learning algorithm ("GRPO", "DPO", "Supervised") 
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            
            # === GRPO Parameters (only used if algorithm="GRPO") ===
            epsilon: Clip ratio for GRPO (typically 0.1-0.3)
            num_group_members: Number of traces collected per training step
            
            # === DPO Parameters (only used if algorithm="DPO") ===
            beta: Temperature parameter for DPO (typically 0.1-0.5)
            loss_type: DPO loss function ("sigmoid", "hinge")  
            label_smoothing: Label smoothing factor
            
            # === Optimization ===
            max_grad_norm: Gradient clipping threshold
            use_bitsandbytes: Enable memory-efficient training
            
            # === Reward & Tools ===
            reward_func: Reward function (defaults to exact match)
            enable_tool_retrieval: Enable intelligent tool filtering
            retrieval_topic: Domain/topic for tool retrieval (e.g., "bio medical", "data science")
            retrieval_guidelines: Custom guidelines for tool selection
            
        Example:
            >>> # Simple GRPO training
            >>> brain = Brain(agent, algorithm="GRPO", learning_rate=1e-4, epsilon=0.2)
            >>> 
            >>> # DPO training with custom parameters
            >>> brain = Brain(agent, algorithm="DPO", learning_rate=3e-5, beta=0.3)
            >>> 
            >>> # Supervised training
            >>> brain = Brain(agent, algorithm="Supervised", learning_rate=5e-5)
            >>>
            >>> # Training with tool retrieval
            >>> brain = Brain(agent, algorithm="GRPO", learning_rate=1e-4,
            ...              enable_tool_retrieval=True, retrieval_topic="data science")
        """

        
        # Store core settings
        self.agent = agent
        self.algorithm = algorithm
        self.batch_size = batch_size
        
        # Create algorithm-specific config automatically
        if config is None:
            config = get_config(algorithm)
            
        # Apply all parameters to config
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.max_grad_norm = max_grad_norm
        config.use_bitsandbytes = use_bitsandbytes
        
        # Apply algorithm-specific parameters
        if algorithm.upper() == "GRPO":
            config.epsilon = epsilon
            config.num_group_members = num_group_members

            
        elif algorithm.upper() == "DPO":
            config.beta = beta
            config.loss_type = loss_type
            config.label_smoothing = label_smoothing
            # DPO requires multiple traces
            config.num_group_members = max(num_group_members, 2)  # At least 2 for comparison
            
        elif algorithm.upper() == "SUPERVISED":
            pass  # No additional parameters needed
            
        else:
            pass  # Unknown algorithm, use config as-is
            
        self.config = config
        
        # Handle reward function
        if reward_func is None:
            reward_func = reward_exact_match

        
        if not isinstance(reward_func, RewardFunctionWrapper):
            reward_func = create_reward_function(reward_func)
        self.reward_func = reward_func
        
        # Handle tool retrieval
        self.enable_tool_retrieval = enable_tool_retrieval
        self.retrieval_topic = retrieval_topic
        self.retrieval_guidelines = retrieval_guidelines
        
        if enable_tool_retrieval:
            self._setup_tool_retrieval()

        else:
            self.retriever = None
        
        # Store original agent type for flexible return in get_agent()
        self.original_agent_type = type(agent)
        
        # Initialize adapter and algorithm
        self.agent_adapter = self._get_adapter_for_agent(agent)

        
        # Get trainable model from adapter and setup training
        self._setup_training()
        
        # Legacy compatibility
        self.reward_buffer = deque(maxlen=10)
        
    def _setup_tool_retrieval(self):
        """Setup tool retrieval components."""
        try:
            from .retriever import ToolRetriever
            self.retriever = ToolRetriever()

        except ImportError as e:

            self.retriever = None
            self.enable_tool_retrieval = False

    def _setup_training(self):
        """Setup training components."""
        # Get trainable model from adapter
        self.trainable_model = self.agent_adapter.get_trainable_model()
        
        # Create policy for RL
        if hasattr(self.trainable_model, 'model') and hasattr(self.trainable_model, 'tokenizer'):
            model = self.trainable_model.model
            tokenizer = self.trainable_model.tokenizer
        else:
            model = self.trainable_model
            tokenizer = getattr(self.trainable_model, 'tokenizer', None)
            if tokenizer is None:
                raise ValueError("Could not find tokenizer in the model")
        
        # Convert config to dict format for algorithms (backward compatibility)
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        
        # Create and store policy
        self.policy = Policy(llm=model, tokenizer=tokenizer)
        
        # Initialize the RL algorithm (maintain backward compatibility with learning_module)
        if self.algorithm in GRPOALiasNames:
            self.learning_module = GRPOAlgorithm(
                initial_policy=self.policy,
                config=config_dict
            )
        elif self.algorithm in DPOALiasNames:
            self.learning_module = DPOAlgorithm(
                initial_policy=self.policy,
                config=config_dict
            )
        elif self.algorithm in SupervisedALiasNames:
            self.learning_module = SupervisedAlgorithm(
                initial_policy=self.policy,
                config=config_dict
            )
        else:
            raise ValueError(f"Unknown learning algorithm: {self.algorithm}")
        
    def _retrieve_relevant_tools(self, query: str) -> List[Any]:
        """
        Use tool retriever to select relevant tools for the query.
        
        Args:
            query: User's query/task
            
        Returns:
            List of relevant Tool objects
        """
        if not self.enable_tool_retrieval or self.retriever is None:
            return self.agent.tools
        
        try:
            # Use retriever's direct method for smolagents tools
            relevant_tools = self.retriever.select_relevant_tools(
                query=query,
                tools_list=self.agent.tools,
                topic=self.retrieval_topic,
                guidelines=self.retrieval_guidelines or "Select tools that are directly relevant to accomplishing the task."
            )
            

            
            return relevant_tools if relevant_tools else self.agent.tools
            
        except Exception as e:

            return self.agent.tools

    def _get_adapter_for_agent(self, agent_instance: Any) -> BaseAgentAdapter:
        """
        Factory method to automatically select the appropriate adapter for the given agent.
        """
        if isinstance(agent_instance, CodeAgent):
            # Convert config to dict format for adapter (backward compatibility)
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
            return SmolAgentAdapter(agent=agent_instance, config=config_dict)
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
                     For supervised training the query is a list of text segments with role information
            num_iterations: The number of training iterations (epochs).
        """
        print("\n🚀 Starting training...")
        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            
            for example in dataset:
                if self.algorithm in GRPOALiasNames or self.algorithm in DPOALiasNames:
                    query = example.get("query")
                elif self.algorithm in SupervisedALiasNames:
                    query = example
                self.train_step(query=query, reward_kwargs=example)
        
        print("\n🎉 Training finished!")

    def get_trace(self, query: str, reward_kwargs: Dict[str, Any]):
        traces: List[Trace] = []
        rl_inputs: List[Any] = []
        raw_memory_collection: List[List[Any]] = []  # Collection of raw memory steps
        num_group_members = self.config.get("num_group_members", 2)

        
        # Store original tools and apply tool retrieval if enabled
        original_tools = None
        if self.enable_tool_retrieval:

            relevant_tools = self._retrieve_relevant_tools(query)
            
            # Backup original tools (list)
            original_tools = self.agent.tools.copy()
            
            # Use filtered tool objects
            if relevant_tools and len(relevant_tools) < len(original_tools):
                self.agent.tools = relevant_tools
                print(f"  ✅ Temporarily using {len(relevant_tools)} filtered tools")
            else:
                print(f"  ℹ️ Using all {len(original_tools)} tools (no filtering applied)")
        
        try:
            # Collect traces with filtered tools (if retrieval enabled)
            for i in range(num_group_members):
                try:
                    print(f"    📝 Trace {i + 1}/{num_group_members}")
                    trace, rl_input, raw_memory_steps = self.agent_adapter.run(query)
                    traces.append(trace)
                    rl_inputs.append(rl_input)
                    raw_memory_collection.append(raw_memory_steps)
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"    ❌ Error during agent iteration: {e}")
                    continue
        
        finally:
            # Always restore original tools if they were modified
            if original_tools is not None:
                self.agent.tools = original_tools
                print(f"  🔄 Restored original {len(original_tools)} tools")
                    
        # Compute rewards using batch scoring (supports both single and batch functions)
        print(f"  🎯 Computing rewards for {len(traces)} traces...")
        if self.reward_func.is_batch_function:
            print(f"      Using batch reward function")
        else:
            print(f"      Using single-trace reward function")

        try:
            # Add raw memory steps to reward_kwargs for advanced analysis (optional)
            enhanced_reward_kwargs = {
                **reward_kwargs,
                "raw_memory_collection": raw_memory_collection  # List of raw memory steps for each trace
            }
            rewards = self.reward_func.get_batch_scores(traces, **enhanced_reward_kwargs)
            for i, reward in enumerate(rewards):
                print(f"      🎯 Trace {i + 1} Reward: {reward:.3f}")
        except Exception as e:
            print(f"    ❌ Error computing rewards: {e}")
            rewards = [0.0] * len(traces)
        return traces, rewards, rl_inputs

    def train_step(self, query: Any, reward_kwargs: Dict[str, Any]):
        """Executes a single training step for a given query."""
        print(f"\n🔄 Training step for query: '{query[:50]}...'")
        num_group_members = self.config.get("num_group_members", 2)
        if num_group_members == 1 and self.algorithm in DPOALiasNames:
            raise NotImplementedError(f"Algorithm '{self.algorithm}' requires num_group_members > 1!")

        if self.algorithm in GRPOALiasNames or self.algorithm in DPOALiasNames:
            traces, rewards, rl_inputs = self.get_trace(query, reward_kwargs)
            # ✅ Update reward buffer
            self.reward_buffer.extend(rewards)
            avg_reward = np.mean(self.reward_buffer)
            print(
                f"📈 Sliding window avg reward (last {len(self.reward_buffer)}): {avg_reward:.4f}")
            if not traces:
                print(f"⚠️ No successful traces collected for query: '{query}'. Skipping training step.")
                return
        elif self.algorithm in SupervisedALiasNames:
            rl_inputs = query

        if self.algorithm in GRPOALiasNames:
            print(f"  🧠 Running RL training step with {len(traces)} traces...")
            self.learning_module.train_step(rl_inputs, rewards)
            print(f"  ✅ RL training step completed")
        elif self.algorithm in DPOALiasNames:
            print(f"  🧠 Sample chosen and rejected pairs from traces...")
            chosen_segments, rejected_segments = make_dpo_pairs(rl_inputs, rewards)
            total_pairs = len(chosen_segments)
            print(f"  🧠 Running DPO with total sampled pairs: {total_pairs}")

            # minibatch training
            batch_size = self.config.get("batch_size", 1)
            for start in range(0, total_pairs, batch_size):
                end = start + batch_size
                chosen_batch = chosen_segments[start:end]
                rejected_batch = rejected_segments[start:end]
                print(f"    🔹 Training on minibatch {start // batch_size + 1} "
                      f"with {len(chosen_batch)} pairs...")
                self.learning_module.train_step(chosen_batch, rejected_batch)
                torch.cuda.empty_cache()
                gc.collect()
            print(f"  ✅ RL training step completed")
        elif self.algorithm in SupervisedALiasNames:
            self.learning_module.train_step([rl_inputs])
            print(f"  ✅ Supervised training step completed")

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

            else:
                return False
        except Exception:
            return False
    
    def generate_training_examples(
            self,
            task_description: str | None = None,
            num_examples: int = 5,
            min_tool_calls: int = 1,
            max_words: int = 80,
            guidance_example: str = None,
            external_model: Any = None,
            external_tools: List[str] = None,
            self_rank: bool = False) -> List[str]:
        """
        Generate training examples using LLM.

        Args:
            task_description: High-level description guiding example creation.
            num_examples: Number of examples to return.
            min_tool_calls: Minimum tool calls per example.
            max_words: Maximum word count for the generated user query.
            external_model: LLM provider (callable, or exposes .propose/.generate).
            external_tools: List of tool names to use in the examples.
            guidance_example: An example to guide the generation style.
        Returns:
            List of strings, each a training example.
        """
        try:
            generated_examples: List[Dict[str, Any]] = []

            # By default, use agent's LLM. If external_model is  provided, use external_model to genetate examples.
            model = self.agent_adapter.get_trainable_model() if not external_model else external_model
            validate_model(model)

            # By default, use agent's tools. If external_tools is provided, use external_tools instead.
            tools = self.agent_adapter.get_callable_tools() if not external_tools else external_tools
            validate_tools(tools)
            tools_description = tools_to_card(tools)

            for _ in range(num_examples):           
                prompt = build_prompt_to_generate_training_examples(
                    tools_description=tools_description,
                    task_description=task_description,
                    min_tool_calls=min_tool_calls,
                    max_words=max_words,
                    guidance_example=guidance_example
                )

                # Turn prompt into a ChatMessage for smolagents compatibility
                messages = [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": str(prompt)}],
                    )
                ]

                llm_output = model.generate(messages)
                generated_example = llm_output[0].content if isinstance(llm_output, list) else llm_output.content
                generated_examples.append(generated_example)

            # Use the current LLM to rank the generated examples, best first
            # Return the list of ranked examples
            if self_rank:
                return self.rank_generated_examples(
                    examples=generated_examples,
                    task_description=task_description,
                    tools_description=tools_description,
                )
            else:
                return generated_examples
        finally:
            # Only delete if we used a temporary external model; keep the agent's own model alive
            if external_model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()

    def rank_generated_examples(
            self,
            examples: List[str],
            task_description: str,
            tools_description: str,
            external_model: Any = None,) -> List[str]:
        
        try:
            # Use agent's model if external_model is not provided
            model = self.agent_adapter.get_trainable_model() if not external_model else external_model
            validate_model(model)

            # Build numbered block of examples
            numbered = "\n".join([f"{i}. {str(ex).strip()}" for i, ex in enumerate(examples)])

            # Compose prompt
            prompt = dedent(f"""
            You are an expert data annotator. Your job is to judge and rank the following generated examples for a tool-using task.
            
            **Task description:**
            {task_description}

            **Tool API card:**
            {tools_description}

            **Generated examples:**
            {numbered}

            For each example, judge:
            (a) Does it directly align with the task (not off-topic)?
            (b) Does it *require* the given tools to solve? (Does it include tool arguments, realistic values, or edge cases?)
            (c) Is it concrete (not vague or theoretical-only)?

            Rank the examples from best to worst based on these criteria.
            Return ONLY a JSON array of the indices in best-to-worst order, e.g. [2,0,1].
            """)

            # Send to model
            messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": str(prompt)}],
                )
            ]
            llm_output = model.generate(messages)
            output_text = llm_output[0].content if isinstance(llm_output, list) else llm_output.content

            # Try to parse as JSON array of indices
            idx_order = None
            try:
                idx_order = json.loads(output_text)
                if not isinstance(idx_order, list):
                    idx_order = None
            except Exception:
                # Try regex to extract first bracketed array
                m = re.search(r"\[[^\[\]]+\]", output_text)
                if m:
                    try:
                        idx_order = json.loads(m.group(0))
                    except Exception:
                        pass
            # Fallback: original order
            if not isinstance(idx_order, list):
                idx_order = list(range(len(examples)))

            # Deduplicate and clamp indices, append missing
            seen = set()
            ordered = []
            for idx in idx_order:
                try:
                    idx_int = int(idx)
                except Exception:
                    continue
                if idx_int < 0 or idx_int >= len(examples):
                    continue
                if idx_int not in seen:
                    ordered.append(idx_int)
                    seen.add(idx_int)
            # Append any missing indices in original order
            for i in range(len(examples)):
                if i not in seen:
                    ordered.append(i)
            # Reorder examples
            return [examples[i] for i in ordered]
        finally:
            if external_model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()
     
    def _get_distillation_config(self) -> Dict[str, Any]:
        """Get configuration for distillation."""
        return {
            "num_traces": 100,
            "accuracy_threshold": 0.9,
            "batch_size": self.config.get("batch_size", 4),
            "learning_rate": self.config.get("learning_rate", 5e-5),
            "use_bitsandbytes": self.config.get("use_bitsandbytes", False)
        }
    
    def _get_cache_file_path(self, teacher_model_id: str, num_traces: int) -> str:
        """Generate cache file path for teacher traces."""
        model_name_safe = teacher_model_id.replace("/", "_")
        return f"teacher_{num_traces}_traces_{model_name_safe}.json"
    
    def _load_cached_traces(self, traces_file_path: str) -> Tuple[List[Trace], List[Any], List[float]]:
        """Load teacher traces from cache file."""
        print(f" Loading existing teacher traces from {traces_file_path}")
        with open(traces_file_path, 'r') as f:
            teacher_data = json.load(f)
        traces = teacher_data['traces']
        rl_inputs = teacher_data['rl_inputs']  
        rewards = teacher_data['rewards']
        print(f"✅ Loaded {len(traces)} traces from file")
        return traces, rl_inputs, rewards
    
    def _create_teacher_agent(self, teacher_model_id: str):
        """Create teacher agent with same tools as student."""
        print(f" Creating teacher model ({teacher_model_id})...")
        teacher_model = TransformersModel(model_id=teacher_model_id)

        # Set chat template if needed
        if teacher_model.tokenizer.chat_template is None:
            teacher_model.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}{% elif message['role'] == 'system' %}{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n'  + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}"

        # Get student's original tool functions (not wrapped tool objects)
        # This preserves the original function's import context like old_distill approach
        student_tools = []
        for _, tool_obj in self.agent_adapter.agent.tools.items():
            # Try to get the original function from the tool object
            if hasattr(tool_obj, 'func'):
                student_tools.append(tool_obj.func)  # Get original function
            else:
                student_tools.append(tool_obj)  # Fallback to tool object

        print(f" Teacher will use the same {len(student_tools)} tools as student: {[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in student_tools]}")

        # Create teacher agent with original tool functions (like old_distill approach)
        teacher_agent = CodeAgent(tools=student_tools, model=teacher_model, max_steps=1)
        teacher_adapter = SmolAgentAdapter(agent=teacher_agent, config={})
        print("✅ Teacher agent created")

        return teacher_adapter
    
    def _collect_teacher_traces(self, teacher_adapter, num_traces: int, dataset: List[Dict[str, Any]]) -> Tuple[List[Trace], List[Any], List[float]]:
        """Collect traces from teacher model."""
        print(f" Collecting {num_traces} traces from teacher model...")
        traces, rl_inputs, rewards = [], [], []
        
        # Use provided dataset queries
        if dataset is None:
            raise ValueError("No dataset provided.")
        
        queries = [item["query"] for item in dataset]
        
        for i in range(num_traces):
            query = queries[i % len(queries)]  # Cycle through available queries
            print(f"    Trace {i+1}/{num_traces}")
            try:
                trace, rl_input, _ = teacher_adapter.run(query)

                # Calculate reward using same function as student (for consistency)
                if dataset:
                    gold_answer = dataset[i % len(dataset)].get("gold_answer")
                    if gold_answer is not None:
                        accuracy = self.reward_func(trace, gold_answer=gold_answer)
                    else:
                        accuracy = self.reward_func(trace)
                else:
                    accuracy = self.reward_func(trace)

                traces.append(trace)
                rl_inputs.append(rl_input)
                rewards.append(accuracy)
                print(f"       Reward: {accuracy:.3f}")
            except Exception as e:
                print(f"    ❌ Error collecting trace {i+1}: {e}")
                continue
        
        return traces, rl_inputs, rewards
    
    def _save_traces_to_cache(self, traces_file_path: str, traces: List[Trace], rl_inputs: List[Any], rewards: List[float]) -> None:
        """Save teacher traces to cache file."""
        print(f"💾 Saving traces to {traces_file_path}")
        with open(traces_file_path, 'w') as f:
            json.dump({
                "traces": traces,
                "rewards": rewards,
                "rl_inputs": rl_inputs
            }, f, indent=2, default=str)
        print(f"✅ Saved {len(traces)} teacher traces to file")
    
    def _filter_high_quality_traces(self, traces: List[Trace], rl_inputs: List[Any], rewards: List[float], accuracy_threshold: float, dataset: List[Dict[str, Any]] = None) -> List[Any]:
        """Filter traces based on quality threshold using reward function."""
        print(f"\n Filtering high-quality traces (quality > {accuracy_threshold})...")
        filtered_rl_inputs = []

        for i, (trace, rl_input, _) in enumerate(zip(traces, rl_inputs, rewards)):
            # Use reward function consistently (works with any tool type)
            if dataset:
                # Use gold answer from dataset if available
                gold_answer = dataset[i % len(dataset)].get("gold_answer")
                if gold_answer is not None:
                    quality_score = self.reward_func(trace, gold_answer=gold_answer)
                else:
                    quality_score = self.reward_func(trace)
            else:
                # No dataset or gold answer, use reward function without gold answer
                quality_score = self.reward_func(trace)

            if quality_score > accuracy_threshold:
                filtered_rl_inputs.append(rl_input)

        print(f"✅ Filtered {len(filtered_rl_inputs)}/{len(traces)} high-quality traces")
        return filtered_rl_inputs
    
    def _train_student_with_traces(self, filtered_rl_inputs: List[Any], batch_size: int, learning_rate: float, use_bitsandbytes: bool) -> None:
        """Train student model with filtered teacher traces."""
        print(f"\n🎓 Starting distillation training on student model...")
        print(f"   Using {len(filtered_rl_inputs)} high-quality teacher traces")
        
        # Create supervised learning module
        distill_config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size, 
            "epochs": 1,
            "max_grad_norm": 1.0,
            "use_bitsandbytes": use_bitsandbytes,
            "lora_config": self.config.get("lora_config"),
        }
        
        distill_module = SupervisedAlgorithm(
            initial_policy=self.learning_module.policy,
            config=distill_config
        )
        
        # Train in batches
        for i in range(0, len(filtered_rl_inputs), batch_size):
            batch_end = min(i + batch_size, len(filtered_rl_inputs))
            batch_rl_inputs = filtered_rl_inputs[i:batch_end]
            
            batch_num = i // batch_size + 1
            total_batches = (len(filtered_rl_inputs) + batch_size - 1) // batch_size
            print(f"      Processing batch {batch_num}/{total_batches} (traces {i+1}-{batch_end})")
            
            try:
                distill_module.train_step(batch_rl_inputs)
            except Exception as e:
                print(f"      ⚠️ Error processing batch {batch_num}: {e}")
                # Try individual traces if batch fails
                for j, single_rl_input in enumerate(batch_rl_inputs):
                    try:
                        distill_module.train_step([single_rl_input])
                    except Exception as e2:
                        print(f"         ⚠️ Error processing individual trace {i+j+1}: {e2}")
                        continue

    # Distill Knowledge from Teacher to Student
    def distill(self, dataset: List[Dict[str, Any]], teacher_model_id: str) -> None:
        """
        Distill knowledge from a teacher model to this Brain's student model.
        
        This method handles the complete distillation pipeline:
        1. Creates a teacher agent with the same tools as the student
        2. Collects execution traces from the teacher using the provided dataset
        3. Filters high-quality traces (accuracy > 90%)
        4. Trains the student model using supervised learning
        
        Args:
            dataset: Training dataset with query/gold_answer pairs
            teacher_model_id: HuggingFace model ID for the teacher model
        """
        print("\n🎓 Distillation mode activated")
        
        # Get configuration
        config = self._get_distillation_config()
        traces_file_path = self._get_cache_file_path(teacher_model_id, config["num_traces"])
        
        # === Step 1: Load or collect teacher traces ===
        if os.path.exists(traces_file_path):
            traces, rl_inputs, rewards = self._load_cached_traces(traces_file_path)
        else:
            # Collect new traces from teacher
            teacher_adapter = self._create_teacher_agent(teacher_model_id)
            traces, rl_inputs, rewards = self._collect_teacher_traces(teacher_adapter, config["num_traces"], dataset)
            self._save_traces_to_cache(traces_file_path, traces, rl_inputs, rewards)
            
            # Clear teacher model from GPU memory after traces are collected
            del teacher_adapter
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 Teacher model cleared from GPU memory")
        
        # === Step 2: Filter high-quality traces ===
        filtered_rl_inputs = self._filter_high_quality_traces(traces, rl_inputs, rewards, config["accuracy_threshold"], dataset)
        
        # === Step 3: Train student model ===
        if len(filtered_rl_inputs) == 0:
            print("⚠️ No high-quality traces found for distillation")
            return
        
        self._train_student_with_traces(
            filtered_rl_inputs, 
            config["batch_size"], 
            config["learning_rate"], 
            config["use_bitsandbytes"]
        )
        
        print("✅ Distillation complete! Student model pre-trained with teacher knowledge")
        print("\n Starting regular training with RL...")

    # Save/Load functionality
    def save(self, output_dir: str) -> None:
        """
        Save the fine-tuned model adapters to a directory.
        
        This method:
        - Saves LoRA adapters using save_pretrained()
        - Saves tokenizer using save_pretrained() 
        
        Args:
            output_dir: Directory to save the model adapters
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving fine-tuned model to '{output_dir}'...")
        
        try:
            # Get trained model (same pattern as examples)
            trained_agent = self.get_agent()
            
            # Save model adapters and tokenizer (align with examples)
            trained_agent.model.model.save_pretrained(output_dir)
            trained_agent.model.tokenizer.save_pretrained(output_dir)
            
            print(f"✅ Model adapters successfully saved to '{output_dir}'")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise
    
    @staticmethod
    def load_agent(
        model_dir: str, 
        base_model_id: str,
        tools: List[Callable],
        **model_kwargs
    ) -> CodeAgent:
        """
        Load a fine-tuned agent from a directory.
        
        This method:
        - Creates base model (UnslothModel or TransformersModel)
        - Loads LoRA adapters using load_adapter()
        - Returns ready-to-use CodeAgent
        
        Args:
            model_dir: Directory containing the saved model adapters
            base_model_id: Base model ID (e.g., "microsoft/DialoGPT-medium")
            tools: List of tools for the agent (required)
            **model_kwargs: Additional arguments for model creation
        
        Returns:
            Loaded CodeAgent ready for use
            
        Example:
            >>> tools = [search_tool, email_tool]
            >>> agent = Brain.load_agent("./my_model", "microsoft/DialoGPT-medium", tools)
            >>> # Agent ready to use with trained adapters!
        """
        from .factory import create_agent  
        from pathlib import Path
        
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        print(f"\n📂 Loading fine-tuned agent from '{model_dir}'...")
        print(f"   Base model: {base_model_id}")
        
        try:
            # Create base agent
            agent = create_agent(
                model_id=base_model_id, 
                tools=tools,
                **model_kwargs
            )
            
            # Load LoRA adapters 
            print(f"   🔧 Loading LoRA adapters...")
            agent.model.model.load_adapter(model_dir)
            
            print(f"   ✅ Fine-tuned agent loaded successfully")
            return agent
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    @classmethod
    def load_and_continue_training(
        cls,
        model_dir: str,
        base_model_id: str,
        tools: List[Callable],
        reward_func: Optional[Union[RewardFunction, BatchRewardFunction]] = None,
        **kwargs
    ) -> 'Brain':
        """
        Load a previously trained agent and create a new Brain for continued training.
        
        Args:
            model_dir: Directory containing the saved model adapters
            base_model_id: Base model ID (e.g., "microsoft/DialoGPT-medium") 
            tools: List of tools (required)
            reward_func: Reward function for continued training
            **kwargs: Additional arguments for Brain creation
        
        Returns:
            Brain instance ready for continued training
            
        Example:
            >>> tools = [search_tool, email_tool]
            >>> brain = Brain.load_and_continue_training(
            ...     model_dir="./my_model",
            ...     base_model_id="microsoft/DialoGPT-medium", 
            ...     tools=tools,
            ...     reward_func=my_reward_func
            ... )
            >>> # Brain ready for continued training
        """
        print(f"\n🔄 Loading agent for continued training...")
        
        # Load the trained agent (using simplified method)
        agent = cls.load_agent(
            model_dir=model_dir,
            base_model_id=base_model_id, 
            tools=tools
        )
        
        # Create Brain with loaded agent (using default algorithm)
        brain = cls(
            agent=agent,
            reward_func=reward_func,
            **kwargs
        )
        
        print("✅ Brain ready for continued training")
        return brain