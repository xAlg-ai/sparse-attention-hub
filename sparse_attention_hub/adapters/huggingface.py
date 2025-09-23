"""HuggingFace adapter implementation for sparse attention."""

import random
import string
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..sparse_attention.base import SparseAttention, SparseAttentionConfig
from ..sparse_attention.research_attention.base import ResearchAttention
from .base import ModelAdapter, Request, RequestResponse
from tqdm import tqdm

INT_MAX = 2**31 - 1


def _apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.
    
    Args:
        logits: Input logits tensor
        temperature: Temperature parameter (1.0 = no scaling, <1.0 = sharper, >1.0 = smoother)
        
    Returns:
        Temperature-scaled logits
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature


def _apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits.
    
    Args:
        logits: Input logits tensor of shape [..., vocab_size]
        top_p: Cumulative probability threshold (0.0 to 1.0)
        
    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    if not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    
    if top_p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Convert to probabilities and compute cumulative sum
    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens to keep (cumulative probability <= top_p)
    # Keep at least one token (the highest probability one)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # Keep the top token
    
    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    
    # Set filtered tokens to -inf
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')
    
    return filtered_logits


def _sample_token(
    logits: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample next token from logits with optional temperature and top_p filtering.
    
    Args:
        logits: Logits tensor of shape [..., vocab_size]
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        temperature: Temperature for scaling logits (only used if do_sample=True)
        top_p: Top-p threshold for nucleus sampling (only used if do_sample=True)
        
    Returns:
        Sampled token indices
    """
    if not do_sample:
        # Greedy decoding
        return torch.argmax(logits, dim=-1)
    
    # Apply temperature scaling
    if temperature != 1.0:
        logits = _apply_temperature_scaling(logits, temperature)
    
    # Apply top-p filtering
    if top_p < 1.0:
        logits = _apply_top_p_filtering(logits, top_p)
    
    # Convert to probabilities and sample
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class ModelAdapterHF(ModelAdapter):
    """ModelAdapter for HuggingFace integration. Provides concrete implementations for huggingface's
    transformer library.
    """

    def __init__(
        self,
        model_name: str,
        sparse_attention_config: Optional[SparseAttentionConfig],
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        recovery_enabled: bool = False,
        recovery_interval: int = 400,
        recovery_dense_attention: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize HuggingFace adapter.

        Args:
            model_name: Name of the HuggingFace model to use
            sparse_attention_config: Configuration for sparse attention. If None, adapter runs in dense-only mode.
            model_kwargs: Additional keyword arguments for model creation
            device: Device to run the model on TODO: support dynamic and multipledevice placement
            tokenizer_kwargs: Additional keyword arguments for tokenizer creation
            recovery_enabled: Whether to enable recovery mechanism during generation
            recovery_interval: Number of tokens after which to trigger recovery
                (regenerate embeddings with full attention)
            recovery_dense_attention: Override attention implementation for recovery
                (if None, uses original implementation)
        """
        super().__init__(model_name, sparse_attention_config, **kwargs)
        self._registered_attention_name: Optional[str] = None
        self._custom_attention_fn: Optional[Callable] = None
        self._original_implementations: Dict[str, str] = {}
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        # Recovery mechanism parameters
        self.recovery_enabled = recovery_enabled
        self.recovery_interval = recovery_interval
        self.recovery_dense_attention = recovery_dense_attention

        # more useful parameters to store
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_dtype = self.model_kwargs.get("torch_dtype", torch.float32)

        # Handle dense-only mode when sparse_attention_config is None
        self._sparse_attention_available: bool = sparse_attention_config is not None

        # create model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **self.tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # TODO: support dynamic and multipledevice placement
        self.model.to(self.device)
        self.random_separator = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=100)
        )

    def __del__(self) -> None:
        """Clean up registered attention functions when the adapter is destroyed."""
        self._cleanup_attention_registration()

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> RequestResponse:
        """Processes request with optimized tokenization but independent question processing.
        Context is tokenized once but each question is processed independently to avoid KV cache contamination.

        Args:
            request: The request to process

        Returns:
            response: The response to the request
        """
        max_context_length: int = request_kwargs.get("max_context_length", INT_MAX)

        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )
        context: str = request.context

        context, questions = self._preprocess_context_and_questions(context, questions)

        context_tokens = self.tokenizer.encode(context, return_tensors="pt")
        context_tokens = context_tokens[
            :, :max_context_length
        ]  # truncate context to max_context_length
        if self.device is not None:
            context_tokens = context_tokens.to(self.device)
        print(f"Context tokens: {context_tokens.shape}")
        responses: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for question in questions:
                sparse_meta_data: Dict[str, Any] = {}

                question_tokens = self.tokenizer.encode(question, return_tensors="pt")
                if self.device is not None:
                    question_tokens = question_tokens.to(self.device)

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
                            generation_kwargs,
                            **kwargs,
                        )
                        responses.append(response_text)
                else:
                    # Dense-only mode: process questions with dense attention
                    response_text = self._generate_response(
                        question_tokens,
                        context_outputs,
                        sparse_meta_data,
                        generation_kwargs,
                        **kwargs,
                    )
                    responses.append(response_text)

        if isinstance(request.questions, str):
            return RequestResponse(responses=responses[0])
        else:
            return RequestResponse(responses=responses)

    def _preprocess_context_and_questions(
        self, context: str, questions: List[str]
    ) -> Tuple[str, List[str]]:
        """Preprocess the context and questions -- apply chat template if needed

        Args:
            context: The context to preprocess
            questions: The questions to preprocess
        """
        context = context + self.random_separator
        if self.tokenizer.chat_template is not None:
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
        new_context = context.split(self.random_separator)[0]
        new_questions = [
            question + context.split(self.random_separator)[1] for question in questions
        ]
        return new_context, new_questions

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

            ALL_ATTENTION_FUNCTIONS.register(
                self._registered_attention_name, self._custom_attention_fn
            )
            if isinstance(self.sparse_attention, ResearchAttention):
                ALL_MASK_ATTENTION_FUNCTIONS.register(
                    self._registered_attention_name, eager_mask
                )
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
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(self._registered_attention_name)
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(
                self._registered_attention_name
            )
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

        # Store original implementations as instance variable for use during generation
        self._original_implementations = original_implementations

        # Ensure custom attention function is registered (reuse if already registered)
        custom_attention_name: str = self._ensure_attention_registered()

        try:
            # Switch to sparse attention
            for name, module in self.model.named_modules():
                if hasattr(module, "config") and hasattr(
                    module.config, "_attn_implementation"
                ):
                    #print(f"Switching to sparse attention for {name}", module.config._attn_implementation, "->", custom_attention_name, flush=True)
                    module.config._attn_implementation = custom_attention_name

            yield

        finally:
            # Restore original implementations
            for name, module in self.model.named_modules():
                if name in original_implementations:
                    #print(f"Restoring original implementation for {name}", module.config._attn_implementation, "->", original_implementations[name], flush=True)
                    module.config._attn_implementation = original_implementations[name]

            # Clean up instance variable
            self._original_implementations = {}

    def _generate_response(
        self,
        question_tokens: torch.Tensor,
        context_outputs: Any,
        sparse_meta_data: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text response using greedy decoding based on kvpress pipeline approach.

        Args:
            question_tokens: The tokenized question
            context_outputs: The model outputs from processing the context
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text response

        TODO:
            move to huggingface genera`te() to leverage all possible generations
            pass generation_kwargs appropriately
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        max_new_tokens: int = generation_kwargs.get("max_new_tokens", 50)  # type: ignore
        context_length: int = context_outputs.past_key_values.get_seq_length()

        # Extract sampling parameters from generation_kwargs
        do_sample: bool = generation_kwargs.get("do_sample", False)
        temperature: float = generation_kwargs.get("temperature", 1.0)
        top_p: float = generation_kwargs.get("top_p", 1.0)

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
        
        # Use proper sampling instead of greedy argmax
        first_token = _sample_token(
            question_outputs.logits[0, -1:],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p
        )
        generated_ids = [first_token.squeeze()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        # Track newly generated tokens for recovery mechanism
        new_tokens_generated = 0
        generation_start_cache_length = context_outputs.past_key_values.get_seq_length()

        if self.recovery_enabled:
            print(f"Recovery enabled: regenerate embeddings every "
                  f"{self.recovery_interval} new tokens")
        else:
            print("Recovery disabled: using sparse attention for answer generation")
        
        # Print sampling configuration
        if do_sample:
            print(f"Sampling enabled: temperature={temperature}, top_p={top_p}")
        else:
            print("Greedy decoding enabled")

        for i in tqdm(range(max_new_tokens - 1)):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                    past_key_values=context_outputs.past_key_values,
                    position_ids=position_ids + i,
                    sparse_meta_data=sparse_meta_data,
                )
                
                # Use proper sampling instead of greedy argmax
                new_id = _sample_token(
                    outputs.logits[0, -1:],
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p
                ).squeeze()
                generated_ids.append(new_id)
                new_tokens_generated += 1

                if new_id.item() in should_stop_token_ids:
                    break

                # Check if we need to regenerate embeddings (only if recovery is enabled)
                if (self.recovery_enabled and
                        new_tokens_generated >= self.recovery_interval and
                        i < max_new_tokens - 2):  # Don't regenerate on the last token

                    print(f"Regenerating embeddings for {new_tokens_generated} "
                          f"newly generated tokens")
                    self._regenerate_embeddings_for_new_tokens(
                        context_outputs.past_key_values,
                        generated_ids[-new_tokens_generated:],  # Last N generated tokens
                        generation_start_cache_length,
                        new_tokens_generated,
                        sparse_meta_data,
                        self._original_implementations
                    )
                    new_tokens_generated = 0  # Reset counter

        answer: str = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )

        return answer

    def _regenerate_embeddings_for_new_tokens(
        self,
        cache: Any,
        new_token_ids: List[torch.Tensor],
        start_cache_length: int,
        num_new_tokens: int,
        sparse_meta_data: Dict[str, Any],
        original_implementations: Dict[str, str]
    ) -> None:
        """Regenerate embeddings for newly generated tokens using full attention.

        This removes the KV cache entries for the newly generated tokens and regenerates
        them using full attention (dense mode), then continues with sparse attention.

        Args:
            cache: The KV cache to modify
            new_token_ids: List of newly generated token IDs
            start_cache_length: Cache length when generation started
            num_new_tokens: Number of new tokens to regenerate embeddings for
            sparse_meta_data: Sparse metadata dictionary
            original_implementations: Dict mapping module names to their original
                attention implementations
        """
        current_cache_length = cache.get_seq_length()

        # Remove embeddings for the newly generated tokens (keep everything before them)
        keep_length = current_cache_length - num_new_tokens

        print(f"Removing embeddings for {num_new_tokens} tokens "
              f"(keeping first {keep_length} tokens)")

        # Truncate cache to remove new token embeddings
        for layer_idx in range(len(cache.key_cache)):
            if cache.key_cache[layer_idx] is not None:
                cache.key_cache[layer_idx] = (
                    cache.key_cache[layer_idx][:, :, :keep_length]
                )
            if cache.value_cache[layer_idx] is not None:
                cache.value_cache[layer_idx] = (
                    cache.value_cache[layer_idx][:, :, :keep_length]
                )

        # Handle quantized caches if present
        if hasattr(cache, "_quantized_key_cache"):
            for layer_idx in range(len(cache._quantized_key_cache)):
                if cache._quantized_key_cache[layer_idx] is not None:
                    cache._quantized_key_cache[layer_idx] = (
                        cache._quantized_key_cache[layer_idx][:, :, :keep_length]
                    )
                if cache._quantized_value_cache[layer_idx] is not None:
                    cache._quantized_value_cache[layer_idx] = (
                        cache._quantized_value_cache[layer_idx][:, :, :keep_length]
                    )
        # Regenerate embeddings using full attention (one forward pass)
        print(f"Regenerating embeddings using full attention for {num_new_tokens} tokens")

        # Create input tensor for the new tokens
        new_tokens_tensor = torch.stack(new_token_ids).unsqueeze(0).to(self.model.device)

        # Create position IDs for the new tokens
        position_ids = torch.arange(
            keep_length, keep_length + num_new_tokens, device=self.model.device
        ).unsqueeze(0)

        # Temporarily disable sparse mode to force dense attention
        print("Forcing dense attention for regeneration")

        # Store current sparse implementations and switch to dense implementations
        for name, module in self.model.named_modules():
            has_config = hasattr(module, "config")
            has_attn_impl = has_config and hasattr(module.config, "_attn_implementation")
            if name in original_implementations and has_attn_impl:
                # Use override if provided, otherwise use original implementation
                dense_implementation = (
                    self.recovery_dense_attention or original_implementations[name]
                )
                module.config._attn_implementation = dense_implementation
        try:
            # Regenerate embeddings with dense attention
            with torch.no_grad():
                self.model(
                    input_ids=new_tokens_tensor,
                    past_key_values=cache,
                    position_ids=position_ids,
                )

            print(f"Successfully regenerated embeddings. Cache length: {cache.get_seq_length()}")

        finally:
            # Restore sparse attention implementations
            for name, module in self.model.named_modules():
                if hasattr(module, "config") and hasattr(
                    module.config, "_attn_implementation"
                ):
                    module.config._attn_implementation = self._registered_attention_name
