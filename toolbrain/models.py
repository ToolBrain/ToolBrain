from unsloth import FastLanguageModel

import logging
from typing import Optional, Dict, Any

from smolagents import Model, TransformersModel
from transformers import TextIteratorStreamer


class UnslothModel(TransformersModel):
    """
    An extension of the smolagents.TransformersModel that uses the Unsloth library
    for significantly faster training and lower memory usage.
    """
    def __init__(
        self,
        model_id: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        max_seq_length: int = 4096,
        max_new_tokens: int = 4096,
        **kwargs: Any
    ):
        """
        Initializes the model using Unsloth's FastLanguageModel.

        Args:
            model_id: The ID of the model to load from Hugging Face.
            model_kwargs: Additional keyword arguments for Unsloth's model loading.
            max_seq_length: The maximum sequence length for the model.
        """
        logging.info("Initializing grandparent 'Model' class...")
        Model.__init__(self, 
            flatten_messages_as_text=True, 
            model_id=model_id, 
            max_new_tokens=max_new_tokens, 
            **kwargs
        )
        
        logging.info(f"Initializing '{model_id}' with Unsloth for optimized performance...")
        
        model_kwargs = model_kwargs or {}
        
        # Load the model and tokenizer using Unsloth's optimized method
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,     
            load_in_4bit=True, 
            **model_kwargs,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Set tokenizer's pad_token to its eos_token.")

        self._is_vlm = False 
        self.model_kwargs = model_kwargs
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        logging.info("✅ Unsloth model initialized successfully and is ready to use.")

    def _prepare_completion_args(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        This method is a copy of the parent TransformersModel._prepare_completion_args,
        with a single line added to fix the dtype mismatch error when using Unsloth.
        """
        # This part is copied directly from the smolagents source code
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            **kwargs,
        )

        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)
        tools = completion_kwargs.pop("tools", None)

        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.kwargs.get("max_new_tokens")
            or self.kwargs.get("max_tokens")
            or 1024
        )
        prompt_tensor = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        prompt_tensor = prompt_tensor.to(self.model.device)
        if hasattr(prompt_tensor, "input_ids"):
            prompt_tensor = prompt_tensor["input_ids"]

        # This is the single most important change. We ensure the input tensor's
        # dtype matches the model's expected dtype before it's used.
        if hasattr(self.model, "dtype") and prompt_tensor.dtype != self.model.dtype:
            prompt_tensor = prompt_tensor.to(self.model.dtype)

        model_tokenizer = self.tokenizer
        stopping_criteria = (
            self.make_stopping_criteria(stop_sequences, tokenizer=model_tokenizer) if stop_sequences else None
        )
        completion_kwargs["max_new_tokens"] = max_new_tokens
        
        # The final dictionary is returned, now with a dtype-corrected prompt_tensor
        return dict(
            inputs=prompt_tensor,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            **completion_kwargs,
        )