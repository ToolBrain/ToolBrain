# toolbrain/adapters/langchain_adapter.py (PHIÊN BẢN MỚI)

import logging
from typing import Any, List, Dict, Tuple

from toolbrain.adapters import BaseAgentAdapter
from toolbrain.core_types import Trace, Turn, ParsedCompletion, ChatSegment

# LangChain specific imports
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import LLMResult
    from langgraph.graph.state import CompiledStateGraph
    from langchain_huggingface import ChatHuggingFace
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
    """Adapter for LangChain agents created with `create_agent`."""

    def __init__(self, agent: CompiledStateGraph, trainable_model: ChatHuggingFace, config: Dict[str, Any]):
        """
        Initializes the adapter.

        Args:
            agent: The compiled LangGraph agent instance.
            trainable_model: The ChatHuggingFace model instance used by the agent.
                             This direct reference is required for training.
            config: The ToolBrain configuration dictionary.
        """
        if not isinstance(agent, CompiledStateGraph):
            raise TypeError(f"Expected LangChain CompiledStateGraph, got {type(agent)}")
        if not isinstance(trainable_model, ChatHuggingFace):
            raise TypeError(f"Expected ChatHuggingFace instance for training, got {type(trainable_model)}")
            
        self.agent = agent
        self._trainable_model = trainable_model
        self.config = config

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

    # --- Phương thức run() và _build_rl_input_from_trace() giữ nguyên như cũ ---
    def run(self, query: str) -> Tuple[Trace, Any, List[Any]]:
        """
        Executes the agent using .stream() and reconstructs a high-fidelity trace.
        """
        handler = ToolBrainCallbackHandler()
        config = {"callbacks": [handler]}
        trace: Trace = []
        
        current_turn: Dict[str, Any] = {}
        llm_call_index = 0

        try:
            stream = self.agent.stream(
                {"messages": [("user", query)]},
                config=config,
                stream_mode="updates",
            )

            for chunk in stream:
                if "agent" in chunk:
                    last_message: AIMessage = chunk["agent"]["messages"][-1]
                    
                    if last_message.tool_calls:
                        current_turn = {} 
                        
                        tool_call = last_message.tool_calls[0]
                        current_turn["parsed_completion"] = ParsedCompletion(
                            thought=last_message.content,
                            tool_code=f"{tool_call['name']}({str(tool_call['args'])})", # Đảm bảo args là string
                            final_answer=None
                        )
                        
                        if llm_call_index < len(handler.prompts):
                            current_turn["prompt_for_model"] = handler.prompts[llm_call_index]
                            current_turn["model_completion"] = handler.completions[llm_call_index]
                            llm_call_index += 1

                elif "tools" in chunk:
                    last_message: ToolMessage = chunk["tools"]["messages"][-1]
                    
                    if "parsed_completion" in current_turn:
                        current_turn["tool_output"] = str(last_message.content)
                        trace.append(Turn(**current_turn))
                        current_turn = {}

            rl_input = self._build_rl_input_from_trace(trace, query)
            raw_memory_steps = trace 

            return trace, rl_input, raw_memory_steps

        except Exception as e:
            logging.error(f"An exception occurred during LangChain agent run: {e}", exc_info=True)
            return [], None, []

    def _build_rl_input_from_trace(self, trace: Trace, initial_query: str) -> List[ChatSegment]:
        segments: List[ChatSegment] = []
        segments.append(ChatSegment(role="other", text=f"user: {initial_query}\n"))

        for turn in trace:
            if turn.get("model_completion"):
                segments.append(ChatSegment(role="assistant", text=turn["model_completion"]))
            if turn.get("tool_output"):
                segments.append(ChatSegment(role="other", text=f"\ntool_output: {turn['tool_output']}\n"))
        
        return segments