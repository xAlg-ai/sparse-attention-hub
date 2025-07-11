"""HuggingFace integration for sparse attention."""

import random
import string
from typing import Any, Callable, Optional
from contextlib import contextmanager

import torch

from ..base import SparseAttention, SparseAttentionConfig
from ..research_attention.base import ResearchAttention
from ..generator import SparseAttentionGen
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS


class SparseAttentionHF(SparseAttentionGen):
    """HuggingFace-compatible sparse attention generator."""

    def __init__(self, sparse_attention: SparseAttention) -> None:
        """Initialize HF generator.
        
        Args:
            sparse_attention: Instance of SparseAttention to use for custom attention.
        """
        self.sparse_attention = sparse_attention

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "SparseAttentionHF":
        """Create SparseAttentionHF instance from configuration.
        
        Args:
            config: Configuration for the sparse attention mechanism.
            
        Returns:
            Instance of SparseAttentionHF with the appropriate sparse attention mechanism.
        """
        sparse_attention = SparseAttention.create_from_config(config)
        return cls(sparse_attention)

    def get_custom_attention_function(self) -> Callable:
        """Get the custom attention function for HuggingFace models.

        Returns:
            Callable that can be used as attention function in HF models.
        """
        def custom_attention_callable(
            module: Any,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs: Any,
        ):
            """Custom attention callable for HuggingFace integration."""
            if hasattr(module, 'layer_idx'):
                kwargs['layer_idx'] = module.layer_idx

            if 'sparse_meta_data' in kwargs:
                sparse_meta_data = kwargs['sparse_meta_data']
                kwargs.pop('sparse_meta_data', None)

            else:
                raise ValueError(
                    "sparse_meta_data must be provided while calling model.forward()"
                )
            return self.sparse_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs
            )
        
        return custom_attention_callable

    def _generate_unique_attention_name(self) -> str:
        """Generate a unique name not present in ALL_ATTENTION_FUNCTIONS."""
        base_name = "sparse_attention"
        existing_keys = ALL_ATTENTION_FUNCTIONS.valid_keys() + ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        
        while True:
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            name = f"{base_name}_{suffix}"
            
            if name not in existing_keys:
                return name

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Any:
        """Execute the sparse attention mechanism.

        This method registers a custom attention function with ALL_ATTENTION_FUNCTIONS,
        replaces the attention implementation in all attention layers of the model
        with the registered function name, and cleans up on exit.

        Parameters
        ----------
        model : PreTrainedModel
            The transformer model to apply sparse attention to.

        Examples
        --------
        >>> from sparse_attention_hub.sparse_attention.integrations.hugging_face import SparseAttentionHF
        >>> sparse_attention_hf = SparseAttentionHF.create_from_config(sparse_attention_config)
        >>> with sparse_attention_hf(model):
        ...     # sparse_meta_cache = {} # start with empty cache
        ...     outputs = model(input_ids, past_key_values=cache, sparse_meta_data=sparse_meta_cache)
        """
        original_implementations = {}
        custom_attention_name = None
        
        try:
            custom_attention_fn = self.get_custom_attention_function()
            custom_attention_name = self._generate_unique_attention_name()
            from transformers.masking_utils import eager_mask
            
            ALL_ATTENTION_FUNCTIONS.register(custom_attention_name, custom_attention_fn)
            if isinstance(self.sparse_attention, ResearchAttention):
                ALL_MASK_ATTENTION_FUNCTIONS.register(custom_attention_name, eager_mask) # this is true for research attention; will need to fix gracefully for efficient attention
            else:
                raise NotImplementedError("Sparse attention is not supported for this model yet")
            
            for name, module in model.named_modules():
                if hasattr(module, 'config') and hasattr(module.config, '_attn_implementation'):
                    original_implementations[name] = module.config._attn_implementation
                    module.config._attn_implementation = custom_attention_name
            
            yield model
            
        finally:
            for name, module in model.named_modules():
                if name in original_implementations:
                    module.config._attn_implementation = original_implementations[name]
            
            try:
                del ALL_ATTENTION_FUNCTIONS[custom_attention_name]
            except KeyError:
                pass
