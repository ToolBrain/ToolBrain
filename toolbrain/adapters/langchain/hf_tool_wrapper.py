# Custom tool calling wrapper for HuggingFace models

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, ToolMessage


class HuggingFaceToolWrapper:
    """Wrapper to add tool calling capability to HuggingFace models"""
    
    def __init__(self, model, tools: List[BaseTool]):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = self._format_tool_descriptions()
        
    def _format_tool_descriptions(self) -> str:
        """Format tools for prompt"""
        descriptions = []
        for tool_name, tool in self.tools.items():
            # Get tool parameters from schema
            schema = tool.args_schema
            if schema:
                params = []
                for field_name, field_info in schema.model_fields.items():
                    field_type = field_info.annotation.__name__ if hasattr(field_info.annotation, '__name__') else str(field_info.annotation)
                    params.append(f'"{field_name}": {field_type}')
                param_str = ", ".join(params)
            else:
                param_str = ""
                
            descriptions.append(f"- {tool_name}({{{param_str}}}): {tool.description}")
        
        return "\n".join(descriptions)
    
    def _create_tool_prompt(self, query: str) -> str:
        """Create prompt that encourages tool calling"""
        return f"""You are a helpful assistant with access to tools. When you need to use a tool, respond with EXACTLY this JSON format:

{{"tool_call": {{"name": "tool_name", "arguments": {{"param_name": value}}}}}}

Available tools:
{self.tool_descriptions}

IMPORTANT: Use the exact parameter names shown above. For example:
- For add tool: {{"tool_call": {{"name": "add", "arguments": {{"a": 5, "b": 7}}}}}}

User: {query}"""
    
    def _parse_tool_call(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from response"""
        try:
            # Look for JSON in response
            if "{" in response_content and "}" in response_content:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                json_part = response_content[start:end]
                parsed = json.loads(json_part)
                
                if "tool_call" in parsed:
                    return parsed["tool_call"]
        except json.JSONDecodeError:
            pass
        return None
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute the tool call"""
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            try:
                result = tool.invoke(tool_args)
                return str(result)
            except Exception as e:
                return f"Error executing tool: {e}"
        else:
            return f"Tool {tool_name} not found"
    
    def invoke_with_tools(self, query: str, max_iterations: int = 3) -> Tuple[List[Any], bool]:
        """
        Invoke model with tool calling capability
        Returns: (messages, has_tool_calls)
        """
        messages = []
        has_tool_calls = False
        
        for iteration in range(max_iterations):
            # Create appropriate prompt
            if iteration == 0:
                prompt = self._create_tool_prompt(query)
            else:
                # Continue conversation
                prompt = query
                
            # Get model response
            response = self.model.invoke([("user", prompt)])
            
            # Create AI message
            ai_message = AIMessage(content=response.content)
            messages.extend([
                ("user", prompt),
                ai_message
            ])
            
            # Check for tool call
            tool_call = self._parse_tool_call(response.content)
            
            if tool_call:
                has_tool_calls = True
                print(f"🔧 Tool call detected: {tool_call}")
                
                # Execute tool
                tool_result = self._execute_tool(tool_call)
                print(f"🔧 Tool result: {tool_result}")
                
                # Create tool message
                tool_message = ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=f"call_{iteration}"
                )
                messages.append(tool_message)
                
                # Create follow-up prompt
                query = f"The tool returned: {tool_result}. Please provide the final answer to the user."
                
            else:
                # No tool call, conversation complete
                break
                
        return messages, has_tool_calls


class CustomLangChainAgent:
    """Custom agent that works with HuggingFace models"""
    
    def __init__(self, model, tools: List[BaseTool]):
        self.tool_wrapper = HuggingFaceToolWrapper(model, tools)
        
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke agent with input"""
        messages = input_dict.get("messages", [])
        if messages:
            query = messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1].content
        else:
            query = str(input_dict)
            
        result_messages, has_tools = self.tool_wrapper.invoke_with_tools(query)
        
        return {"messages": result_messages}
    
    def stream(self, input_dict: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Stream agent execution"""
        # For simplicity, we'll simulate streaming by returning chunks
        messages = input_dict.get("messages", [])
        if messages:
            query = messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1].content
        else:
            query = str(input_dict)
            
        result_messages, has_tools = self.tool_wrapper.invoke_with_tools(query)
        
        # Return chunks to simulate LangChain streaming
        chunks = []
        for i, msg in enumerate(result_messages):
            if isinstance(msg, tuple):
                continue  # Skip user messages
            elif isinstance(msg, AIMessage):
                chunks.append({"agent": {"messages": [msg]}})
            elif isinstance(msg, ToolMessage):
                chunks.append({"tools": {"messages": [msg]}})
                
        return chunks