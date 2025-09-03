"""
Agent adapters for ToolBrain.

This module provides the Adapter pattern implementation to make different
agent libraries compatible with ToolBrain's trace-based training system.
"""

try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from abc import ABC, abstractmethod
from smolagents.models import get_clean_message_list, tool_role_conversions
from typing import Optional, List, Any

from peft import get_peft_model
from peft import prepare_model_for_kbit_training

from smolagents import CodeAgent, TransformersModel, ChatMessage, MessageRole
import io
import contextlib
import re
import torch

from .core_types import Trace, Turn, ParsedCompletion, ChatSegment

import logging


class BaseAgentAdapter(ABC):
    """Abstract base class for agent adapters."""

    @abstractmethod
    def run(self, query: str) -> Trace:
        """Execute a query and return a structured execution trace."""
        pass

    @abstractmethod
    def get_trainable_model(self) -> TransformersModel:
        """Return the underlying trainable model from the agent."""
        pass


class SmolAgentAdapter(BaseAgentAdapter):
    """Adapter for smolagents CodeAgent using a local TransformersModel."""

    def __init__(self, agent: CodeAgent, config):
        """
        Initialize the SmolAgentAdapter.

        Args:
            agent: A smolagents CodeAgent instance configured with a TransformersModel.
            config: general config
        """
        if not isinstance(agent, CodeAgent):
            raise TypeError(f"Expected CodeAgent instance, got {type(agent)}")
        if not isinstance(agent.model, TransformersModel):
            raise TypeError(
                "Training is only supported for agents using a local smolagents.TransformersModel."
            )

        self.agent = agent
        self.config = config
        self._set_lora_finetuning()
        print("✅ SmolAgentAdapter: Initialized for local model training.")

    def get_trainable_model(self) -> TransformersModel:
        """Returns the agent's underlying TransformersModel."""
        return self.agent.model

    def get_tools(self) -> List[str]:
        """Returns the list of tool names available in the agent."""
        return [tool for tool in self.agent.tools]

    def run(self, query: str):
        """
        Executes the agent and then extracts a structured, high-fidelity trace
        from the agent's memory.

        Returns:
            tuple: (structured_trace, rl_input, raw_memory_steps)
                - structured_trace: Trace (List[Turn]) - processed trace for standard use
                - rl_input: Any - input prepared for RL training
                - raw_memory_steps: List[Any] - raw agent memory steps for advanced analysis
        """
        print(f"🚀 Adapter is calling agent.run() for query: '{query[:50]}...'")

        try:
            with torch.inference_mode():
                self.agent.run(query, reset=True)

            # Extract structured trace and RL input as before
            structured_trace = self._extract_trace_from_memory()
            rl_input = self._build_input_for_rl_from_memory()

            # Capture raw memory steps for advanced analysis
            raw_memory_steps = []
            if hasattr(self.agent, "memory") and hasattr(self.agent.memory, "steps"):
                # Create a copy to avoid reference issues
                raw_memory_steps = list(self.agent.memory.steps)

            logging.info(
                f"✅ Agent run completed. Extracted a trace with {len(structured_trace)} turns, "
                f"and {len(raw_memory_steps)} raw memory steps."
            )
            return structured_trace, rl_input, raw_memory_steps

        except Exception as e:
            logging.error(
                f"❌ An exception occurred during an agent run: {e}", exc_info=True
            )

            error_turn: Turn = {
                "prompt_for_model": query,
                "model_completion": f"Adapter/Agent Runtime Error: {str(e)}",
                "parsed_completion": {
                    "thought": None,
                    "tool_code": None,
                    "final_answer": f"Adapter/Agent Runtime Error: {str(e)}",
                },
                "tool_output": None,
                "action_output": None,
                "formatted_conversation": None,
            }

            logging.warning(
                "Due to the error above, this agent run is considered FAILED. "
                "Returning rl_input=None. This run will be excluded from the training batch."
            )

            return [error_turn], None, []

    def _extract_trace_from_memory(self) -> Trace:
        """
        Parses the agent's internal memory into our standardized Trace format.
        This version leverages pre-parsed fields from ActionStep where possible
        and parses the rest from the raw model_output.

        Also generates formatted conversation text using smolagents utilities
        for consistent formatting with training data.
        """
        if not hasattr(self.agent, "memory") or not hasattr(self.agent.memory, "steps"):
            return []

        # Generate formatted conversation text using smolagents utilities
        # This is done FIRST while agent memory is still intact
        formatted_conversation = None
        try:
            messages = self.agent.write_memory_to_messages()
            messages = get_clean_message_list(
                messages,
                role_conversions=tool_role_conversions,
                flatten_messages_as_text=True,
            )
            formatted_conversation = self.agent.model.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            print(f"⚠️ Warning: Failed to generate formatted conversation: {e}")
            formatted_conversation = None

        full_trace: Trace = []

        for step in self.agent.memory.steps:
            if step.__class__.__name__ == "ActionStep":

                model_completion_str = step.model_output or ""
                tool_output_str = step.observations or ""
                action_output_str = step.action_output or ""
                if step.error:
                    tool_output_str += f"\nError: {str(step.error)}"

                parsed_completion: ParsedCompletion = {
                    "thought": None,
                    "tool_code": step.code_action,
                    "final_answer": None,
                }

                missing_parts = self._parse_missing_parts(model_completion_str)

                parsed_completion["thought"] = missing_parts.get("thought")
                if not parsed_completion["final_answer"]:
                    parsed_completion["final_answer"] = missing_parts.get(
                        "final_answer"
                    )
                if step.model_input_messages is not None:
                    prompt_for_model_str = (
                        self.agent.model.tokenizer.apply_chat_template(
                            [
                                {"content": m.content, "role": str(m.role)}
                                for m in step.model_input_messages
                            ],
                            return_tensors="pt",
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    )
                else:
                    prompt_for_model_str = ""
                current_turn: Turn = {
                    "prompt_for_model": prompt_for_model_str.strip(),
                    "model_completion": model_completion_str.strip(),
                    "parsed_completion": parsed_completion,
                    "tool_output": tool_output_str.strip() if tool_output_str else None,
                    "action_output": action_output_str,
                    "formatted_conversation": formatted_conversation,  # Add formatted text
                }
                full_trace.append(current_turn)

        return full_trace

    def _build_input_for_rl_from_memory(self) -> Any:
        """
        Parses the agent's internal memory, build input for RL learning
        """
        messages = self.agent.write_memory_to_messages()
        messages = get_clean_message_list(
            messages,
            role_conversions=tool_role_conversions,
            flatten_messages_as_text=True,
        )
        out_text = self.agent.model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        messages = self.agent.write_memory_to_messages()
        messages = get_clean_message_list(messages, flatten_messages_as_text=True)
        segments = self._segment_text_with_assistant(out_text, messages)
        return segments

    def _segment_text_with_assistant(
        self, full_text: str, messages: list
    ) -> list[dict]:
        """
        Split full_text into contiguous segments, labeling only assistant messages.
        Ensures the sum of all segments equals the original text.

        Args:
            full_text: str, rendered chat text
            messages: list of {'role': str, 'content': str}

        Returns:
            List of segments [{'role': 'assistant' or 'other', 'start': int, 'end': int, 'text': str}]
        """
        segments = []
        pos = 0  # cursor in full_text

        for msg in messages:
            if msg["role"] != "assistant":
                continue  # skip non-assistant messages

            snippet = msg["content"]
            start = full_text.find(snippet, pos)
            if start == -1:
                continue  # skip if not found
            end = start + len(snippet)

            # text before this assistant message is "other"
            if start > pos:
                segment: ChatSegment = {"role": "other", "text": full_text[pos:start]}
                segments.append(segment)

            # assistant message
            segment: ChatSegment = {"role": "assistant", "text": full_text[start:end]}
            segments.append(segment)

            pos = end  # move cursor forward

        # any remaining text after last assistant message
        if pos < len(full_text):
            segment: ChatSegment = {"role": "other", "text": full_text[pos:]}
            segments.append(segment)

        # --- Assertion: the sum of all segment texts equals the original text ---
        combined_text = "".join(seg["text"] for seg in segments)
        assert combined_text == full_text, "Segments do not cover the full text!"

        return segments

    def _parse_missing_parts(self, model_output: str) -> dict:
        """
        A helper to parse thought and a cleaned final_answer from the raw model output.
        """
        if not isinstance(model_output, str):
            return {}

        parts = {}

        thought_match = re.search(
            r"Thought:(.*?)(?:Code:|Final Answer:|$)",
            model_output,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match and thought_match.group(1):
            parts["thought"] = thought_match.group(1).strip()

        answer_match = re.search(
            r"Final Answer:(.*)", model_output, re.DOTALL | re.IGNORECASE
        )
        if answer_match and answer_match.group(1):
            raw_answer_text = answer_match.group(1).strip()

            number_match = re.search(r"[-+]?\d*\.\d+|\d+", raw_answer_text)

            if number_match:
                parts["final_answer"] = number_match.group(0)
            else:
                parts["final_answer"] = raw_answer_text

        return parts

    def _set_lora_finetuning(self):
        lora_config = self.config.get("lora_config", None)
        if lora_config:
            is_unsloth_model = (
                UNSLOTH_AVAILABLE
                and hasattr(self.agent.model, "model")
                and isinstance(self.agent.model.model, FastLanguageModel)
            )

            if is_unsloth_model:
                print(
                    "✅ Applying LoRA configuration using Unsloth's optimized method..."
                )
                self.agent.model.model = FastLanguageModel.get_peft_model(
                    self.agent.model.model, lora_config, use_gradient_checkpointing=True
                )
            else:
                print("Applying LoRA configuration using standard PEFT...")
                hf_model = get_peft_model(self.agent.model.model, lora_config)
                self.agent.model.model = hf_model

            print(f"✅ LoRA configuration is successful!")
            total_params = sum(p.numel() for p in self.agent.model.model.parameters())
            trainable_params = sum(
                p.numel()
                for p in self.agent.model.model.parameters()
                if p.requires_grad
            )
            percentage = (
                100 * trainable_params / total_params if total_params > 0 else 0
            )

            print(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({percentage:.2f}%)"
            )
