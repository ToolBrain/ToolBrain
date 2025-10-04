# Updated LangChain adapter with custom tool calling for HuggingFace models

import logging
from typing import Any, List, Dict, Tuple

from toolbrain.adapters import BaseAgentAdapter
from toolbrain.core_types import Trace, Turn, ParsedCompletion, ChatSegment
from toolbrain.hf_tool_wrapper import CustomLangChainAgent

# LangChain specific imports
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import LLMResult
    from langgraph.graph.state import CompiledStateGraph
    from langchain_huggingface import ChatHuggingFace
    from langchain_core.tools import BaseTool
except ImportError:
    raise ImportError(
        "LangChain dependencies not found. Please run 'pip install langchain langgraph langchain-huggingface'"
    )


class ToolBrainCallbackHandler(BaseCallbackHandler):
    """A custom callback handler to capture raw LLM inputs and outputs."""
    def __init__(self):
        self.prompts: List[str] = []
        self.completions: List[str] = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.prompts.append(prompts[0])

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        completion = response.generations[0][0].text
        self.completions.append(completion)


class LangChainAdapter(BaseAgentAdapter):
    """Adapter for LangChain agents with intelligent model type detection."""

    def __init__(self, agent: Any, trainable_model: Any, config: Dict[str, Any], tools: List[BaseTool] = None):
        """
        Initializes the adapter.

        Args:
            agent: The agent instance (can be CompiledStateGraph or tools list).
            trainable_model: The model instance (ChatHuggingFace, ChatGoogleGenerativeAI, etc.).
            config: The ToolBrain configuration dictionary.
            tools: Optional list of tools for custom tool calling.
        """
        self.original_agent = agent
        self._trainable_model = trainable_model
        self.config = config
        
        # Detect model type to determine tool calling strategy
        self.model_type = self._detect_model_type(trainable_model)
        self.needs_custom_tool_calling = self._needs_custom_tool_calling()
        
        print(f"🔍 Detected model type: {self.model_type}")
        print(f"🔧 Needs custom tool calling: {self.needs_custom_tool_calling}")
        
        # Use provided tools or extract from agent
        if tools:
            self.tools = tools
            print(f"🔧 Using provided tools: {[t.name for t in tools]}")
        else:
            self.tools = self._extract_tools_from_agent(agent)
        
        # Create custom agent only for models that need it (e.g., HuggingFace)
        if self.needs_custom_tool_calling and self.tools:
            print(f"🔧 Creating custom tool-calling agent with {len(self.tools)} tools")
            self.agent = CustomLangChainAgent(trainable_model, self.tools)
        else:
            if not self.needs_custom_tool_calling:
                print(f"✅ Using native tool calling for {self.model_type} model")
            else:
                print("⚠️ No tools detected, using original agent")
            self.agent = agent

    def _extract_tools_from_agent(self, agent) -> List[BaseTool]:
        """Extract tools from various agent types"""
        tools = []
        
        # For CompiledStateGraph, tools are in the graph nodes
        if hasattr(agent, 'get_graph'):
            graph = agent.get_graph()
            print(f"🔍 Inspecting graph nodes: {list(graph.nodes.keys())}")
            
            # Look for tools node
            for node_id, node_data in graph.nodes.items():
                if node_id == 'tools' or 'tool' in str(node_id).lower():
                    print(f"🔧 Found tools node: {node_id}")
                    
                    # The tools node should contain the bound tools
                    # We need to access the original tools passed to create_agent
                    # For now, we'll check if tools were passed via the builder
                    if hasattr(agent, 'builder') and hasattr(agent.builder, 'tools'):
                        tools = agent.builder.tools
                        print(f"✅ Found tools in builder: {[t.name for t in tools]}")
                        break
        
        # Try direct access to tools attribute
        if not tools and hasattr(agent, 'tools'):
            if isinstance(agent.tools, dict):
                tools = list(agent.tools.values())
            elif isinstance(agent.tools, list):
                tools = agent.tools
        
        # If we still don't have tools, this might be a limitation
        # For now, return empty list and log a warning
        if not tools:
            print("⚠️ Could not extract tools from agent. Custom tool calling will be disabled.")
            return []
        
        # Filter to ensure we have BaseTool instances
        filtered_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                filtered_tools.append(tool)
                print(f"✅ Added tool: {tool.name}")
            elif hasattr(tool, 'name') and hasattr(tool, 'description'):
                # Convert to BaseTool if it's tool-like
                filtered_tools.append(tool)
                print(f"✅ Added tool-like object: {tool.name}")
        
        return filtered_tools

    def _detect_model_type(self, model: Any) -> str:
        """Detect the type of language model being used."""
        model_class_name = model.__class__.__name__
        model_module = model.__class__.__module__
        
        # Check for specific model types by class name first
        if "ChatHuggingFace" in model_class_name:
            return "huggingface"
        elif "ChatGoogleGenerativeAI" in model_class_name or "ChatGoogleGenAI" in model_class_name:
            return "gemini"
        elif "ChatOpenAI" in model_class_name:
            return "openai"
        elif "ChatAnthropic" in model_class_name:
            return "claude"
        elif "ChatOllama" in model_class_name:
            return "ollama"
        
        # Check by module path if class name doesn't match
        elif "huggingface" in model_module.lower():
            return "huggingface"
        elif "google" in model_module.lower() or "gemini" in model_module.lower():
            return "gemini"
        elif "openai" in model_module.lower():
            return "openai"
        elif "anthropic" in model_module.lower():
            return "claude"
        elif "ollama" in model_module.lower():
            return "ollama"
        else:
            return "unknown"
    
    def _needs_custom_tool_calling(self) -> bool:
        """Determine if this model needs custom tool calling implementation."""
        # Models that have native tool calling support in LangChain
        native_tool_calling_models = ["gemini", "openai", "claude"]
        
        # HuggingFace and unknown models need custom implementation
        return self.model_type not in native_tool_calling_models

    def get_trainable_model(self) -> Any:
        """Returns the agent's underlying trainable model."""
        # Truy cập model và tokenizer từ ChatHuggingFace qua pipeline
        if hasattr(self._trainable_model, 'llm') and hasattr(self._trainable_model.llm, 'pipeline'):
            pipeline = self._trainable_model.llm.pipeline
            if hasattr(pipeline, 'model') and hasattr(pipeline, 'tokenizer'):
                # Tạo object với model và tokenizer để tương thích với Brain
                class TrainableModelWrapper:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer
                
                return TrainableModelWrapper(pipeline.model, pipeline.tokenizer)
        
        # Fallback về ChatHuggingFace instance
        return self._trainable_model

    def run(self, query: str) -> Tuple[Trace, Any, List[Any]]:
        """
        Executes the agent and reconstructs a high-fidelity trace.
        """
        trace: Trace = []
        
        try:
            if self.needs_custom_tool_calling:
                print(f"🔍 Running query with custom tool calling: {query[:50]}...")
                return self._run_with_custom_tool_calling(query)
            else:
                print(f"🔍 Running query with native tool calling: {query[:50]}...")
                return self._run_with_native_tool_calling(query)

        except Exception as e:
            logging.error(f"An exception occurred during agent run: {e}", exc_info=True)
            return [], None, []

    def _run_with_custom_tool_calling(self, query: str) -> Tuple[Trace, Any, List[Any]]:
        """Run with custom tool calling implementation for HuggingFace models."""
        trace: Trace = []
        
        try:
            # Use custom agent
            if isinstance(self.agent, CustomLangChainAgent):
                # Get stream chunks to build trace
                chunks = self.agent.stream({"messages": [("user", query)]})
                
                print(f"📝 Received {len(chunks)} stream chunks")
                
                current_turn: Dict[str, Any] = {}
                
                for i, chunk in enumerate(chunks):
                    print(f"  Chunk {i+1}: {list(chunk.keys())}")
                    
                    if "agent" in chunk:
                        ai_message = chunk["agent"]["messages"][0]
                        
                        # Check if this is a tool calling message
                        tool_call = self._extract_tool_call_from_content(ai_message.content)
                        
                        if tool_call:
                            current_turn = {
                                "parsed_completion": ParsedCompletion(
                                    thought=ai_message.content,
                                    tool_code=f"{tool_call['name']}({str(tool_call['arguments'])})",
                                    final_answer=None
                                ),
                                "prompt_for_model": query,
                                "model_completion": ai_message.content
                            }
                        else:
                            # Final response without tool call
                            if current_turn:
                                # This is the final answer after tool execution
                                current_turn["parsed_completion"].final_answer = ai_message.content
                            else:
                                # Direct response without tools
                                current_turn = {
                                    "parsed_completion": ParsedCompletion(
                                        thought=ai_message.content,
                                        tool_code=None,
                                        final_answer=ai_message.content
                                    ),
                                    "prompt_for_model": query,
                                    "model_completion": ai_message.content
                                }
                    
                    elif "tools" in chunk:
                        tool_message = chunk["tools"]["messages"][0]
                        
                        if current_turn:
                            current_turn["tool_output"] = tool_message.content
                            # Add completed turn
                            trace.append(Turn(**current_turn))
                            current_turn = {}
                
                # Add final turn if exists
                if current_turn:
                    trace.append(Turn(**current_turn))
                    
            else:
                # Fallback to original agent
                print("⚠️ Using fallback to original agent")
                result = self.original_agent.invoke({"messages": [("user", query)]})
                
                # Create minimal trace from result
                if "messages" in result and len(result["messages"]) > 1:
                    ai_response = result["messages"][-1]
                    trace.append(Turn(
                        parsed_completion=ParsedCompletion(
                            thought=ai_response.content,
                            tool_code=None,
                            final_answer=ai_response.content
                        ),
                        prompt_for_model=query,
                        model_completion=ai_response.content,
                        tool_output=None
                    ))

            rl_input = self._build_rl_input_from_trace(trace, query)
            raw_memory_steps = trace 

            print(f"✅ Generated trace with {len(trace)} turns")
            return trace, rl_input, raw_memory_steps

        except Exception as e:
            logging.error(f"An exception occurred during custom tool calling: {e}", exc_info=True)
            return [], None, []

    def _run_with_native_tool_calling(self, query: str) -> Tuple[Trace, Any, List[Any]]:
        """Run with native tool calling for models like Gemini, OpenAI, Claude."""
        trace: Trace = []
        
        try:
            # Use original agent with native tool calling
            result = self.agent.invoke({"messages": [("user", query)]})
            
            print(f"📝 Processing native tool calling result...")
            
            # Extract messages from result
            if "messages" in result:
                messages = result["messages"]
                print(f"📝 Found {len(messages)} messages")
                
                # Build trace from messages
                current_turn: Dict[str, Any] = {}
                
                for i, message in enumerate(messages):
                    print(f"  Message {i+1}: {type(message).__name__}")
                    
                    if hasattr(message, 'type'):
                        if message.type == "ai":
                            # AI message - check for tool calls
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                # This is a tool calling message
                                tool_call = message.tool_calls[0]  # Get first tool call
                                current_turn = {
                                    "parsed_completion": ParsedCompletion(
                                        thought=message.content or f"Calling {tool_call['name']}",
                                        tool_code=f"{tool_call['name']}({str(tool_call['args'])})",
                                        final_answer=None
                                    ),
                                    "prompt_for_model": query,
                                    "model_completion": message.content or f"Tool call: {tool_call['name']}"
                                }
                            else:
                                # Regular AI response
                                if current_turn:
                                    # Final answer after tool execution
                                    current_turn["parsed_completion"].final_answer = message.content
                                else:
                                    # Direct response without tools
                                    current_turn = {
                                        "parsed_completion": ParsedCompletion(
                                            thought=message.content,
                                            tool_code=None,
                                            final_answer=message.content
                                        ),
                                        "prompt_for_model": query,
                                        "model_completion": message.content
                                    }
                        
                        elif message.type == "tool":
                            # Tool execution result
                            if current_turn:
                                current_turn["tool_output"] = message.content
                                # Add completed turn
                                trace.append(Turn(**current_turn))
                                current_turn = {}
                
                # Add final turn if exists
                if current_turn:
                    trace.append(Turn(**current_turn))
            
            else:
                # Simple response without tool calling
                content = str(result.get("output", result))
                trace.append(Turn(
                    parsed_completion=ParsedCompletion(
                        thought=content,
                        tool_code=None,
                        final_answer=content
                    ),
                    prompt_for_model=query,
                    model_completion=content,
                    tool_output=None
                ))

            rl_input = self._build_rl_input_from_trace(trace, query)
            raw_memory_steps = trace 

            print(f"✅ Generated trace with {len(trace)} turns")
            return trace, rl_input, raw_memory_steps

        except Exception as e:
            logging.error(f"An exception occurred during native tool calling: {e}", exc_info=True)
            return [], None, []

    def _extract_tool_call_from_content(self, content: str) -> Dict[str, Any]:
        """Extract tool call information from message content"""
        try:
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_part = content[start:end]
                
                import json
                parsed = json.loads(json_part)
                
                if "tool_call" in parsed:
                    return parsed["tool_call"]
        except:
            pass
        return None

    def _build_rl_input_from_trace(self, trace: Trace, initial_query: str) -> List[ChatSegment]:
        segments: List[ChatSegment] = []
        segments.append(ChatSegment(role="other", text=f"user: {initial_query}\n"))

        for turn in trace:
            if turn.get("model_completion"):
                segments.append(ChatSegment(role="assistant", text=turn["model_completion"]))
            if turn.get("tool_output"):
                segments.append(ChatSegment(role="other", text=f"\ntool_output: {turn['tool_output']}\n"))
        
        return segments