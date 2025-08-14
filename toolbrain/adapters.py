"""
Agent adapters for ToolBrain.

This module provides the Adapter pattern implementation to make different
agent libraries compatible with ToolBrain's trace-based training system.

"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from smolagents import CodeAgent, LiteLLMModel 
import io
import contextlib
import re
from openai import OpenAI

from .core_types import Trace, Turn, ParsedCompletion

class BaseAgentAdapter(ABC):
    @abstractmethod
    def run(self, query: str) -> Trace:
        pass

class SmolAgentAdapter(BaseAgentAdapter):
    def __init__(
        self,
        agent: CodeAgent,
        model_id: str,
        api_base: str,
        api_key: str,
        max_turns: int = 5 
    ):
        if not isinstance(agent, CodeAgent):
            raise TypeError(f"Expected CodeAgent instance, got {type(agent)}")
        
        self.agent = agent
        self.max_turns = max_turns
        self.model_id = model_id
        
        if OpenAI is None:
            raise ImportError("Please install openai library: pip install openai")

        try:
            self.client = OpenAI(base_url=api_base, api_key=api_key)
            print("✅ SmolAgentAdapter: Initialized direct OpenAI client.")
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

    def run(self, query: str) -> Trace:
        full_trace: Trace = []
        history: Trace = []

        for turn_number in range(self.max_turns):
            prompt_for_this_turn = self._build_prompt_for_turn(query, history)
            
            model_completion_string = self._get_llm_completion(prompt_for_this_turn)
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
                break
            
            if not parsed_completion.get("tool_code"):
                print(f"⚠️ Agent did not produce a tool call in turn {turn_number + 1}. Ending trace.")
                break
        
        return full_trace

    def _build_prompt_for_turn(self, query: str, history: Trace) -> str:
        """Builds the prompt for the current turn based on history."""
        
        tool_definitions_str = self._get_tool_definitions()
        
        initial_prompt = f"""You are a helpful AI assistant that thinks step-by-step to solve problems using tools.

Here are the tools available:
{tool_definitions_str}

You MUST respond in the following format:
Thought: Your reasoning for the next step.
Code: The Python code to execute for the next step.
OR
Final Answer: The final answer to the user's query. 

---
User Query: {query}
"""
        history_str = self._format_history_for_prompt(history)
        
        return initial_prompt + history_str + "\nThought:"

    def _get_tool_definitions(self) -> str:
        tool_definitions = []
        if hasattr(self.agent, 'tools') and isinstance(self.agent.tools, dict):
            for tool_name, tool_func in self.agent.tools.items():
                docstring = getattr(tool_func, '__doc__', None) or "No description available."
                tool_definitions.append(f"- {tool_name}: {docstring.strip()}")
        return "\n".join(tool_definitions)

    def _format_history_for_prompt(self, history: Trace) -> str:
        parts = []
        for turn in history:
            parts.append(turn["model_completion"])
            if turn["tool_output"]:
                parts.append(f"\nTool Output:\n{turn['tool_output']}")
        return "\n".join(parts)

    def _get_llm_completion(self, prompt: str) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client is not initialized.")
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.5,
                stop=["Tool Output:", "Observation:"] 
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"❌ ERROR: Direct API call failed: {e}")
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
        print(f"  🔧 Executing tool code: '{tool_code}'")
        try:
            execution_scope = {
                "__builtins__": __builtins__
            }
            execution_scope.update(self.agent.tools)

            output_capture = io.StringIO()
            with contextlib.redirect_stdout(output_capture):
                exec(tool_code, execution_scope)
            
            result = output_capture.getvalue().strip()
            return result if result else "Code executed without output."
        except Exception as e:
            return f"Error executing tool code: {str(e)}"

