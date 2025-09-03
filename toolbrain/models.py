from unsloth import FastLanguageModel

import logging
from typing import Optional, Dict, Any

from smolagents import Model, TransformersModel
from transformers import TextIteratorStreamer
from smolagents import ChatMessage, Tool
import torch


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
        **kwargs: Any,
    ):
        """
        Initializes the model using Unsloth's FastLanguageModel.

        Args:
            model_id: The ID of the model to load from Hugging Face.
            model_kwargs: Additional keyword arguments for Unsloth's model loading.
            max_seq_length: The maximum sequence length for the model.
        """
        logging.info("Initializing grandparent 'Model' class...")
        Model.__init__(
            self,
            flatten_messages_as_text=True,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        logging.info(
            f"Initializing '{model_id}' with Unsloth for optimized performance..."
        )

        model_kwargs = model_kwargs or {}

        # Load the model and tokenizer using Unsloth's optimized method
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
            **model_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Set tokenizer's pad_token to its eos_token.")

        self._is_vlm = False
        self.model_kwargs = model_kwargs
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        logging.info("✅ Unsloth model initialized successfully and is ready to use.")
