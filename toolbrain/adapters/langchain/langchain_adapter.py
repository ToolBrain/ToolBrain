# LangChain adapter for HuggingFace models with custom tool calling

import logging
from typing import Any, List, Dict, Tuple

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError):
    # ImportError: unsloth not installed
    # NotImplementedError: unsupported GPU (e.g., no NVIDIA/Intel GPU)
    FastLanguageModel = None
    UNSLOTH_AVAILABLE = False

from peft import get_peft_model
from ..base_adapter import BaseAgentAdapter
from ...core_types import Trace, Turn, ParsedCompletion, ChatSegment
from ...models import UnslothModel
from .hf_tool_wrapper import CustomLangChainAgent


def create_huggingface_chat_model(
    model_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = True,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    return_full_text: bool = False,
    **kwargs
):
    """
    Helper function to create a ChatHuggingFace model for LangChain agents.
    
    This function simplifies the process of creating a HuggingFace model wrapped
    for use with LangChain agents, handling all the pipeline setup automatically.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature for generation
        do_sample: Whether to use sampling or greedy decoding
        torch_dtype: PyTorch data type for the model
        device_map: Device mapping strategy for the model
        return_full_text: Whether to return full text or just new tokens
        **kwargs: Additional arguments passed to model initialization
        
    Returns:
        ChatHuggingFace: Ready-to-use chat model for LangChain agents
        
    Example:
        >>> model = create_huggingface_chat_model("Qwen/Qwen2.5-0.5B-Instruct")
        >>> agent = create_agent(model=model, tools=[...])
    """
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        raise ImportError(
            "Required packages not found. Please install: "
            "pip install 'langchain-huggingface' 'transformers'"
        )
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs
    )
    
    # Create text generation pipeline
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        return_full_text=return_full_text
    )
    
    # Wrap in LangChain components
    hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)
    chat_model = ChatHuggingFace(llm=hf_pipeline)
    
    return chat_model

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
    """Adapter for LangChain agents using HuggingFace models with custom tool calling."""

    def __init__(self, agent: Any, trainable_model: Any, config: Dict[str, Any]):
        """
        Initializes the adapter for HuggingFace models.

        Args:
            agent: The LangChain agent instance.
            trainable_model: The ChatHuggingFace model instance.
            config: The ToolBrain configuration dictionary.
        """
        self.original_agent = agent
        self._trainable_model = trainable_model
        self.config = config
        
        if self._trainable_model is None:
            raise ValueError("trainable_model is required for LangChain adapter. Make sure you're passing the ChatHuggingFace model.")
        
        # Extract tools from agent
        self.tools = self._extract_tools_from_agent(agent)
        
        # Create custom agent for HuggingFace models
        if self.tools:
            self.agent = CustomLangChainAgent(self._trainable_model, self.tools)
        else:
            print("⚠️ No tools detected, using original agent")
            self.agent = agent
        
        # Set up LoRA fine-tuning if configured
        self._set_lora_finetuning()

    def _extract_tools_from_agent(self, agent) -> List[BaseTool]:
        """Extract tools from LangGraph CompiledStateGraph agents"""
        tools = []
        
        if hasattr(agent, 'get_graph'):
            graph = agent.get_graph()
            
            if hasattr(agent, 'get_state') and hasattr(agent, 'invoke'):
                try:
                    if hasattr(agent, 'config') and hasattr(agent.config, 'tools'):
                        tools = agent.config.tools
                        print(f"✅ Found tools in agent config: {[t.name for t in tools]}")
                    
                    elif hasattr(agent, '_compiled') and hasattr(agent._compiled, 'tools'):
                        tools = agent._compiled.tools
                        print(f"✅ Found tools in compiled config: {[t.name for t in tools]}")
                    
                    else:
                        for node_id, node_data in graph.nodes.items():
                            if node_id == 'tools':
                                tool_node = node_data.data
                                if hasattr(tool_node, 'tools_by_name'):
                                    tools_dict = tool_node.tools_by_name
                                    tools = list(tools_dict.values())
                                    break
                                elif hasattr(tool_node, '_tools_by_name'):
                                    tools_dict = tool_node._tools_by_name
                                    tools = list(tools_dict.values())
                                    break
                                
                                elif hasattr(node_data, 'func'):
                                    func = node_data.func
                                    if hasattr(func, 'tools'):
                                        tools = func.tools
                                        break
                                    
                                    elif hasattr(func, '__closure__') and func.__closure__:
                                        for cell in func.__closure__:
                                            cell_content = cell.cell_contents
                                            if hasattr(cell_content, '__iter__') and not isinstance(cell_content, str):
                                                try:
                                                    for item in cell_content:
                                                        if hasattr(item, 'name') and hasattr(item, 'description'):
                                                            tools.append(item)
                                                except (TypeError, AttributeError):
                                                    continue
                                            elif hasattr(cell_content, 'tools'):
                                                tools = cell_content.tools
                                                break
                except Exception as e:
                    print(f"⚠️ Error during tool extraction: {e}")
        
        if not tools:
            for attr_name in ['tools', '_tools', 'tool_executor', 'toolkit']:
                if hasattr(agent, attr_name):
                    attr_value = getattr(agent, attr_name)
                    if isinstance(attr_value, list) and attr_value:
                        tools = attr_value
                        break
                    elif isinstance(attr_value, dict) and attr_value:
                        tools = list(attr_value.values())
                        break
        
        if not tools:
            return []
        
        # Filter to ensure we have valid tool instances
        filtered_tools = []
        for tool in tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description') and callable(getattr(tool, 'invoke', None)):
                filtered_tools.append(tool)
                print(f"✅ Added valid tool: {tool.name}")
            else:
                print(f"⚠️ Skipping invalid tool: {tool}")
        
        return filtered_tools

    def get_trainable_model(self) -> Any:
        """Returns the agent's underlying trainable model."""
        # Access model and tokenizer from ChatHuggingFace via pipeline
        if hasattr(self._trainable_model, 'llm') and hasattr(self._trainable_model.llm, 'pipeline'):
            pipeline = self._trainable_model.llm.pipeline
            if hasattr(pipeline, 'model') and hasattr(pipeline, 'tokenizer'):
                # Create object with model and tokenizer for Brain compatibility
                class TrainableModelWrapper:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer
                
                return TrainableModelWrapper(pipeline.model, pipeline.tokenizer)
        
        # Fallback to ChatHuggingFace instance
        return self._trainable_model

    def run(self, query: str) -> Tuple[Trace, Any, List[Any]]:
        """
        Executes the agent and reconstructs a high-fidelity trace.
        """
        print(f"🔍 Running query with custom tool calling: {query[:50]}...")
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

            return trace, rl_input, raw_memory_steps

        except Exception as e:
            logging.error(f"An exception occurred during agent execution: {e}", exc_info=True)
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

    def _set_lora_finetuning(self):
        """Set up LoRA fine-tuning if configured."""
        lora_config = self.config.get("lora_config", None)
        if not lora_config:
            return
        
        # Convert dict to LoraConfig object if needed
        if isinstance(lora_config, dict):
            from peft import LoraConfig
            lora_config = LoraConfig(**lora_config)
        
        # Only apply LoRA to HuggingFace models (all models in this adapter are HuggingFace)
        trainable_model = self.get_trainable_model()
        if trainable_model is None:
            logging.warning("Could not find trainable model for LoRA setup")
            return
        
        # Check if it's a wrapper with model attribute
        if hasattr(trainable_model, 'model'):
            actual_model = trainable_model.model
        else:
            actual_model = trainable_model
        
        is_unsloth_model = (
            UNSLOTH_AVAILABLE
            and isinstance(actual_model, UnslothModel)
        )
        
        if is_unsloth_model:
            # Apply Unsloth LoRA
            new_model = FastLanguageModel.get_peft_model(
                actual_model,
                use_gradient_checkpointing="unsloth",
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                max_seq_length=512,
            )
        else:
            # Apply standard LoRA
            new_model = get_peft_model(actual_model, lora_config)
        
        # Update the model reference
        if hasattr(trainable_model, 'model'):
            trainable_model.model = new_model
        elif hasattr(self._trainable_model, 'llm') and hasattr(self._trainable_model.llm, 'pipeline'):
            self._trainable_model.llm.pipeline.model = new_model
        
        # Log LoRA statistics
        total_params = sum(p.numel() for p in new_model.parameters())
        trainable_params = sum(
            p.numel() for p in new_model.parameters() if p.requires_grad
        )
        percentage = (100 * trainable_params / total_params if total_params > 0 else 0)
        logging.info(
            f"📊 LoRA applied: {trainable_params:,} / {total_params:,} params trainable ({percentage:.2f}%)"
        )