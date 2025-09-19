"""
Brain module - The flexible, user-friendly interface for ToolBrain.

This module contains the Brain class which orchestrates the training process.
It automatically detects the agent type and uses the appropriate adapter.
"""

import gc
import json
import os
from collections import deque
from typing import Any, List, Dict, Union, Tuple

import numpy as np

from .learning.supervised.algo import SupervisedAlgorithm
from .rewards import RewardFunctionWrapper
from .core_types import Trace, RewardFunction, BatchRewardFunction
from .adapters import BaseAgentAdapter, SmolAgentAdapter
from .learning.dpo.algo import DPOAlgorithm
from .learning.dpo.utils import make_dpo_pairs
from .learning.grpo import GRPOAlgorithm
from .learning import Policy
from smolagents import CodeAgent, ChatMessage, MessageRole, TransformersModel
import torch
from textwrap import dedent
GRPOALiasNames = ["GRPO", "grpo"]
DPOALiasNames = ["DPO", "dpo"]
SupervisedALiasNames = ["Supervised", "supervised", "supervise"]


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
        print(f"   - Initializing learning algorithm: {learning_algorithm}...")
        if learning_algorithm in  GRPOALiasNames:
            policy = Policy(llm=trainable_model.model, tokenizer=trainable_model.tokenizer)
            self.learning_module = GRPOAlgorithm(
                initial_policy=policy, 
                config=config
            )
        elif learning_algorithm in DPOALiasNames:
            policy = Policy(llm=trainable_model.model, tokenizer=trainable_model.tokenizer)
            self.learning_module = DPOAlgorithm(
                initial_policy=policy,
                config=config
            )
        elif learning_algorithm in SupervisedALiasNames:
            policy = Policy(llm=trainable_model.model, tokenizer=trainable_model.tokenizer)
            self.learning_module = SupervisedAlgorithm(
                initial_policy=policy,
                config=config
            )
        else:
            raise NotImplementedError(f"Algorithm '{learning_algorithm}' is not supported.")
        print("   ✅ Learning module initialized.")
        self.reward_buffer = deque(maxlen=10)
        
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
                     For supervised training the query is a list of text segments with role information
            num_iterations: The number of training iterations (epochs).
        """
        print("\n🚀 Starting training...")
        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            
            for example in dataset:
                if self.learning_algorithm in GRPOALiasNames or self.learning_algorithm in DPOALiasNames:
                    query = example.get("query")
                elif self.learning_algorithm in SupervisedALiasNames:
                    query = example
                self.train_step(query=query, reward_kwargs=example)
        
        print("\n🎉 Training finished!")

    def get_trace(self, query: str, reward_kwargs: Dict[str, Any]):
        traces: List[Trace] = []
        rl_inputs: List[Any] = []
        raw_memory_collection: List[List[Any]] = []  # Collection of raw memory steps
        num_group_members = self.config.get("num_group_members", 10)
        print(f"  📊 Collecting {num_group_members} traces...")
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
        num_group_members = self.config.get("num_group_members", 10)
        if num_group_members == 1 and self.learning_algorithm in DPOALiasNames:
            raise NotImplementedError(f"Algorithm '{self.learning_algorithm}' requires num_group_members > 1!")

        if self.learning_algorithm in GRPOALiasNames or  self.learning_algorithm in DPOALiasNames:
            traces, rewards, rl_inputs = self.get_trace(query, reward_kwargs)
            # ✅ Update reward buffer
            self.reward_buffer.extend(rewards)
            avg_reward = np.mean(self.reward_buffer)
            print(
                f"📈 Sliding window avg reward (last {len(self.reward_buffer)}): {avg_reward:.4f}")
            if not traces:
                print(f"⚠️ No successful traces collected for query: '{query}'. Skipping training step.")
                return
        elif self.learning_algorithm in SupervisedALiasNames:
            rl_inputs = query

        if self.learning_algorithm in GRPOALiasNames:
            print(f"  🧠 Running RL training step with {len(traces)} traces...")
            self.learning_module.train_step(rl_inputs, rewards)
            print(f"  ✅ RL training step completed")
        elif self.learning_algorithm in DPOALiasNames:
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
        elif self.learning_algorithm in SupervisedALiasNames:
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

    def generate_train_examples(
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
    


    def _get_distillation_config(self) -> Dict[str, Any]:
        """Get configuration for distillation."""
        return {
            "num_traces": 100,
            "accuracy_threshold": 0.9,
            "batch_size": self.config.get("batch_size", 4),
            "learning_rate": self.config.get("lr", 5e-5),
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
            "lr": learning_rate,
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