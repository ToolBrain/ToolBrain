"""
Agent adapters for ToolBrain.

This module provides the Adapter pattern implementation to make different
agent libraries compatible with ToolBrain's trace-based training system.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from smolagents import CodeAgent, TransformersModel, ChatMessage, MessageRole
import io
import contextlib
import re

from .core_types import Trace, Turn, ParsedCompletion

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Guards import safer_getattr, full_write_guard

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
    
    def __init__(self, agent: CodeAgent, max_turns: int = 5):
        """
        Initialize the SmolAgentAdapter.
        
        Args:
            agent: A smolagents CodeAgent instance configured with a TransformersModel.
            max_turns: Maximum number of interaction turns.
        """
        if not isinstance(agent, CodeAgent):
            raise TypeError(f"Expected CodeAgent instance, got {type(agent)}")
        if not isinstance(agent.model, TransformersModel):
            raise TypeError("Training is only supported for agents using a local smolagents.TransformersModel.")
        
        self.agent = agent
        self.max_turns = max_turns
        print("✅ SmolAgentAdapter: Initialized for local model training.")

    def get_trainable_model(self) -> TransformersModel:
        """Returns the agent's underlying TransformersModel."""
        return self.agent.model

    def run(self, query: str) -> Trace:
        print(f"🚀 Starting agent execution for query: '{query[:50]}...'")
        full_trace: Trace = []
        history: Trace = []

        for turn_number in range(self.max_turns):
            print(f"  📝 Turn {turn_number+1}/{self.max_turns}")
            prompt_for_this_turn = self._build_prompt_for_turn(query, history)
            print(f"    📤 Sending prompt to model ({len(prompt_for_this_turn)} chars)...")
            
            model_completion_string = self._get_llm_completion(prompt_for_this_turn)
            
            print(f"    📥 Model response received ({len(model_completion_string)} chars)")
            print(f"    DEBUG [Turn {turn_number+1}] LLM Raw Output:\n---\n{model_completion_string}\n---")

            parsed_completion = self._parse_model_completion(model_completion_string)
            
            tool_output_string = None
            if parsed_completion.get("tool_code"):
                tool_output_string = self._execute_tool_code(parsed_completion["tool_code"])
            
            current_turn: Turn = {
                "prompt_for_model": prompt_for_this_turn,
                "model_completion": model_completion_string,
                "parsed_completion": parsed_completion,
                "tool_output": tool_output_string
            }
            full_trace.append(current_turn)
            history.append(current_turn)
            
            if parsed_completion.get("final_answer"):
                print(f"    🎯 Final answer found: {parsed_completion['final_answer'][:100]}...")
                break
            
            if not parsed_completion.get("tool_code"):
                print(f"⚠️ Agent did not produce a tool call in turn {turn_number + 1}. Ending trace.")
                break
        
        print(f"✅ Agent execution completed in {len(full_trace)} turns")
        return full_trace

    def _build_prompt_for_turn(self, query: str, history: Trace) -> str:
        tool_definitions_str = self._get_tool_definitions()
        
        if not history:
            initial_prompt = f"""SYSTEM:
You are a precise and logical AI assistant. Your task is to solve the user's query by thinking step-by-step and using the provided tools. You MUST strictly follow the specified format. Do not add any extra explanations or introductory phrases.

AVAILABLE TOOLS:
{tool_definitions_str}

RESPONSE FORMAT:
You must respond with a sequence of Thought and Action blocks. An action can be `Code` or `Final Answer`.

1.  **Thought**: Reason about the user's query and decide the next action.
2.  **Code**: Write the Python code to call ONE of the available tools to make progress.
3.  **Final Answer**: Provide the final answer ONLY when the task is fully completed.

---
EXAMPLE:
USER QUERY: What is 8 multiplied by 6?

ASSISTANT RESPONSE:
Thought: The user wants to multiply 8 and 6. I have the `multiply` tool available for this. I will call this tool with the numbers 8 and 6 and print the result to the output.
Code:
print(multiply(8, 6))

---
TASK:
USER QUERY: {query}
"""
            return initial_prompt + "\nASSISTANT RESPONSE:\nThought:"

        else:
            initial_prompt = f"""SYSTEM:
You are a helpful AI assistant... (instructions)

USER QUERY: {query}
"""
            history_str = self._format_history_for_prompt(history)
            
            return initial_prompt + history_str + "\nASSISTANT RESPONSE:\nThought:"

    def _get_tool_definitions(self) -> str:
        tool_definitions = []
        if hasattr(self.agent, 'tools') and isinstance(self.agent.tools, dict):
            for tool_name, tool_func in self.agent.tools.items():
                docstring = getattr(tool_func, '__doc__', None) or "No description available."
                tool_definitions.append(f"- {tool_name}: {docstring.strip()}")
        return "\n".join(tool_definitions)

    def _format_history_for_prompt(self, history: Trace) -> str:
        """
        Formats the history of Turns into a single string for the prompt.
        """
        parts = []
        for turn in history:
            parts.append(f"\nASSISTANT RESPONSE:\n{turn['model_completion']}")
            
            if turn["tool_output"]:
                parts.append(f"\nTOOL OUTPUT:\n{turn['tool_output']}")
        
        return "".join(parts)

    def _get_llm_completion(self, prompt: str) -> str:
        try:
            print(f"      🤖 Calling model.generate()...")
            import time
            start_time = time.time()
            
            content_to_send = [{"type": "text", "text": prompt}]
            messages_to_send = [
                ChatMessage(role=MessageRole.USER, content=content_to_send)
            ]
            
            response_object = self.agent.model.generate(
                messages=messages_to_send,
                stop_sequences=["Tool Output:", "Observation:"]
            )
            
            generation_time = time.time() - start_time
            print(f"      ⚡ Model generation completed in {generation_time:.2f}s")
            
            if hasattr(response_object, 'content') and isinstance(response_object.content, str):
                return response_object.content
            else:
                print(f"      ⚠️ WARNING: Model response is not a ChatMessage with .content. Casting to string.")
                return str(response_object)

        except Exception as e:
            print(f"❌ ERROR: Local model generation failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: LLM call failed. Details: {str(e)}"

    def _parse_model_completion(self, model_output: str) -> ParsedCompletion:
        if not isinstance(model_output, str):
            model_output = str(model_output or "")
        parsed: ParsedCompletion = {"thought": None, "tool_code": None, "final_answer": None}
        thought_match = re.search(r"Thought:(.*?)(?:Code:|Final Answer:|$)", model_output, re.DOTALL)
        if thought_match and thought_match.group(1):
            parsed["thought"] = thought_match.group(1).strip()
        code_match = re.search(r"Code:(.*?)(?:Thought:|Final Answer:|$)", model_output, re.DOTALL)
        if code_match and code_match.group(1):
            code_content = code_match.group(1).strip().replace("`", "")
            if code_content.startswith("python"):
                code_content = code_content[6:].strip()
            parsed["tool_code"] = code_content
        answer_match = re.search(r"Final Answer:(.*)", model_output, re.DOTALL)
        if answer_match and answer_match.group(1):
            parsed["final_answer"] = answer_match.group(1).strip()
        if not any(parsed.values()):
            parsed["thought"] = model_output.strip()
        return parsed

    def _execute_tool_code(self, tool_code: str) -> Optional[str]:
        """
        Executes tool code in a highly secure and robust sandbox using RestrictedPython,
        incorporating advanced guards and result handling.
        """
        print(f"  🔧 Executing tool code: '{tool_code}'")

        execution_scope = safe_globals.copy()
        execution_scope.update(self.agent.tools)

        execution_scope['_getattr_'] = safer_getattr
        execution_scope['_write_'] = full_write_guard 
        
        collector = PrintCollector()
        execution_scope['_print_'] = collector

        try:
            byte_code = compile_restricted(
                tool_code,
                filename='<agent_code>',
                mode='exec'
            )
            
            exec(byte_code, execution_scope)

            result = collector.getvalue().strip() 
            return result if result else "Code executed without output."

        except Exception as e:
            import traceback
            print("--- EXECUTION ERROR ---")
            traceback.print_exc()
            print("-----------------------")
            return f"Error executing tool code: {str(e)}"
