"""HuggingFace adapter implementation for sparse attention."""

import random
import string
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from ..sparse_attention.base import SparseAttention, SparseAttentionConfig
from ..sparse_attention.research_attention.base import ResearchAttention
from .base import ModelAdapter, Request, RequestResponse


class ModelAdapterHF(ModelAdapter):
    """ModelAdapter for HuggingFace integration. Provides concrete implementations for huggingface's
    transformer library.
    """

    def __init__(
        self,
        sparse_attention_config: Optional[SparseAttentionConfig],
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize HuggingFace adapter.

        Args:
            sparse_attention_config: Configuration for sparse attention. If None, adapter runs in dense-only mode.
            model_name: Name of the HuggingFace model to use
            device: Device to load the model on
            torch_dtype: Torch data type for the model
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.tokenizer: Optional[Any] = None
        self._registered_attention_name: Optional[str] = None
        self._custom_attention_fn: Optional[Callable] = None

        # Handle dense-only mode when sparse_attention_config is None
        self._sparse_attention_available: bool = sparse_attention_config is not None

        super().__init__(sparse_attention_config, model_name)

    def __del__(self) -> None:
        """Clean up registered attention functions when the adapter is destroyed."""
        self._cleanup_attention_registration()

    def create_model(self, model_name: str) -> Any:
        """Creates a model using HuggingFace transformers library.

        Args:
            model_name: Name of the model to create

        Returns:
            model: The created HuggingFace model instance
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        model_kwargs = {}
        if self.device is not None:
            model_kwargs["device_map"] = self.device
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        return model

    def process_request(self, request: Request) -> RequestResponse:
        """Processes request with optimized tokenization but independent question processing.
        Context is tokenized once but each question is processed independently to avoid KV cache contamination.

        Args:
            request: The request to process

        Returns:
            response: The response to the request
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )

        context_tokens = self.tokenizer.encode(request.context, return_tensors="pt")
        if self.device is not None:
            context_tokens = context_tokens.to(self.device)

        responses: List[str] = []

        for question in questions:
            sparse_meta_data: Dict[str, Any] = {}

            question_tokens = self.tokenizer.encode(question, return_tensors="pt")
            if self.device is not None:
                question_tokens = question_tokens.to(self.device)

            with self.enable_dense_mode():
                context_outputs = self.model(
                    context_tokens,
                    past_key_values=None,
                    use_cache=True,
                    sparse_meta_data=sparse_meta_data,
                )

            if self._sparse_attention_available:
                with self.enable_sparse_mode():
                    response_text = self._generate_response(
                        question_tokens,
                        context_outputs,
                        sparse_meta_data,
                        max_new_tokens=50,
                    )
                    responses.append(response_text)
            else:
                # Dense-only mode: process questions with dense attention
                with self.enable_dense_mode():
                    response_text = self._generate_response(
                        question_tokens,
                        context_outputs,
                        sparse_meta_data,
                        max_new_tokens=50,
                    )
                    responses.append(response_text)

        if isinstance(request.questions, str):
            return RequestResponse(responses=responses[0])
        else:
            return RequestResponse(responses=responses)

    def get_custom_attention_function(
        self, sparse_attention: SparseAttention
    ) -> Callable:
        """Returns custom_attention_fn callable with the correct signature required for HuggingFace.

        Args:
            sparse_attention: The sparse attention instance

        Returns:
            custom_attention_fn: Callable with correct signature for HuggingFace
        """

        def custom_attention_callable(
            module: torch.nn.Module,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs: Dict[str, Any],
        ):
            """Custom attention callable for HuggingFace integration."""
            if hasattr(module, "layer_idx"):
                layer_idx = getattr(module, "layer_idx", None)
                if layer_idx is not None:
                    kwargs["layer_idx"] = layer_idx

            if "sparse_meta_data" in kwargs:
                sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
                kwargs.pop("sparse_meta_data", None)
            else:
                raise ValueError(
                    "sparse_meta_data must be provided while calling model.forward()"
                )

            return sparse_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs,
            )

        return custom_attention_callable

    def _generate_unique_attention_name(self) -> str:
        """Generate a unique name not present in ALL_ATTENTION_FUNCTIONS."""
        base_name: str = "sparse_attention"
        existing_keys: List[str] = (
            ALL_ATTENTION_FUNCTIONS.valid_keys()
            + ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )

        while True:
            suffix: str = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=8)
            )
            name: str = f"{base_name}_{suffix}"

            if name not in existing_keys:
                return name

    def _ensure_attention_registered(self) -> str:
        """Ensure custom attention function is registered and return the name.
        Caches the registration to avoid repeated registration overhead.

        Returns:
            The name of the registered attention function
        """
        if self._registered_attention_name is None:
            if not self._sparse_attention_available or self.sparse_attention is None:
                raise RuntimeError(
                    "Cannot register attention function: sparse attention is not available"
                )
            self._custom_attention_fn = self.get_custom_attention_function(
                self.sparse_attention
            )
            self._registered_attention_name = self._generate_unique_attention_name()

            from transformers.masking_utils import eager_mask

            ALL_ATTENTION_FUNCTIONS[
                self._registered_attention_name
            ] = self._custom_attention_fn
            if isinstance(self.sparse_attention, ResearchAttention):
                ALL_MASK_ATTENTION_FUNCTIONS[
                    self._registered_attention_name
                ] = eager_mask
            else:
                raise NotImplementedError(
                    "Sparse attention is not supported for this model yet"
                )

        return self._registered_attention_name

    def _cleanup_attention_registration(self) -> None:
        """Clean up registered attention functions."""
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        ):
            del ALL_ATTENTION_FUNCTIONS[self._registered_attention_name]
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        ):
            del ALL_MASK_ATTENTION_FUNCTIONS[self._registered_attention_name]
        self._registered_attention_name = None
        self._custom_attention_fn = None

    @contextmanager
    def enable_sparse_mode(self) -> Generator[None, None, None]:
        """Context manager to temporarily enable sparse attention mode.

        Yields:
            None
        """
        # If sparse attention is not available, raise an error
        if not self._sparse_attention_available:
            raise RuntimeError(
                "Cannot enable sparse mode: sparse attention is not available"
            )

        # Store original implementations to restore later
        original_implementations: Dict[str, str] = {}

        # First, store the original implementations before registering custom attention
        for name, module in self.model.named_modules():
            if hasattr(module, "config") and hasattr(
                module.config, "_attn_implementation"
            ):
                original_implementations[name] = module.config._attn_implementation

        # Ensure custom attention function is registered (reuse if already registered)
        custom_attention_name: str = self._ensure_attention_registered()

        try:
            # Switch to sparse attention
            for name, module in self.model.named_modules():
                if hasattr(module, "config") and hasattr(
                    module.config, "_attn_implementation"
                ):
                    module.config._attn_implementation = custom_attention_name

            yield

        finally:
            # Restore original implementations
            for name, module in self.model.named_modules():
                if name in original_implementations:
                    module.config._attn_implementation = original_implementations[name]

    @contextmanager
    def enable_dense_mode(self) -> Generator[None, None, None]:
        """Context manager to explicitly enable dense attention mode.

        Note: This is the default mode, so this context manager is mainly
        for clarity and consistency with enable_sparse_mode.

        Yields:
            None
        """
        # Dense mode is always available - just yield
        yield

    def _generate_response(
        self,
        question_tokens: torch.Tensor,
        context_outputs: Any,
        sparse_meta_data: Dict[str, Any],
        max_new_tokens: int = 50,
    ) -> str:
        """Generate text response using greedy decoding based on kvpress pipeline approach.

        Args:
            question_tokens: The tokenized question
            context_outputs: The model outputs from processing the context
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text response
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        context_length: int = context_outputs.past_key_values.get_seq_length()

        position_ids = torch.arange(
            context_length,
            context_length + question_tokens.shape[1],
            device=self.model.device,
        ).unsqueeze(0)

        with torch.no_grad():
            question_outputs = self.model(
                input_ids=question_tokens,
                past_key_values=context_outputs.past_key_values,
                position_ids=position_ids,
                num_logits_to_keep=1,
                sparse_meta_data=sparse_meta_data,
            )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [question_outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                    past_key_values=context_outputs.past_key_values,
                    position_ids=position_ids + i,
                    sparse_meta_data=sparse_meta_data,
                )
                new_id = outputs.logits[0, -1].argmax()
                generated_ids.append(new_id)

                if new_id.item() in should_stop_token_ids:
                    break

        answer: str = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )
        return answer
