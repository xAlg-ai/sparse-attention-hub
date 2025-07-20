"""Adapter classes to bridge sparse attention implementations with HuggingFace attention interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from ..sparse_attention.base import SparseAttention


class BaseAttentionFunction(ABC):
    """
    Base class for attention functions that integrate with transformers' AttentionInterface.
    This mirrors the interface from attention-interface-hub.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the attention function with configuration.

        Args:
            config: Configuration dictionary for this attention implementation
        """
        self.config = config

    @abstractmethod
    def attention_forward(
        self,
        module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Attention forward function that matches transformers' attention interface.

        Args:
            module: The attention module instance (self in the original context)
            query_states: Query tensor [batch, heads, seq_len, head_dim]
            key_states: Key tensor [batch, heads, seq_len, head_dim]
            value_states: Value tensor [batch, heads, seq_len, head_dim]
            attention_mask: Optional attention mask
            dropout: Dropout probability
            scaling: Attention scaling factor
            sliding_window: Sliding window size (for compatible models)
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (attention_output, attention_weights) where attention_weights
            can be None if not computed
        """
        pass

    @classmethod
    @abstractmethod
    def get_attention_name(cls) -> str:
        """Return the name to register this attention function under"""
        pass

    @classmethod
    def validate_config(cls, model_config) -> bool:
        """Validate that this attention function is compatible with the model config"""
        return True

    def modify_model(self, model: torch.nn.Module) -> bool:
        """
        Modify model after loading but before inference.

        Args:
            model: The loaded transformers model instance whose weights can be modified

        Returns:
            bool: True if modifications were applied, False otherwise
        """
        return False


class SparseAttentionAdapter(BaseAttentionFunction):
    """
    Adapter class that bridges SparseAttention implementations to the BaseAttentionFunction interface.
    """

    def __init__(self, sparse_attention: SparseAttention, config: Dict[str, Any]):
        """
        Initialize the adapter with a sparse attention implementation.

        Args:
            sparse_attention: The sparse attention implementation to wrap
            config: Configuration dictionary
        """
        super().__init__(config)
        self.sparse_attention = sparse_attention

    def attention_forward(
        self,
        module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using the wrapped sparse attention implementation.

        Args:
            module: The attention module instance
            query_states: Query tensor [batch, heads, seq_len, head_dim]
            key_states: Key tensor [batch, heads, seq_len, head_dim]
            value_states: Value tensor [batch, heads, seq_len, head_dim]
            attention_mask: Optional attention mask
            dropout: Dropout probability
            scaling: Attention scaling factor
            sliding_window: Sliding window size (for compatible models)
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Use the sparse attention's custom_attention method
        return self.sparse_attention.custom_attention(
            module=module,
            queries=query_states,
            keys=key_states,
            values=value_states,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )

    @classmethod
    def get_attention_name(cls) -> str:
        """Return the name to register this attention function under"""
        return "sparse_attention_adapter"

    def modify_model(self, model: torch.nn.Module) -> bool:
        """
        No model modifications needed for the base adapter.

        Args:
            model: The loaded transformers model instance

        Returns:
            bool: False, indicating no modifications were applied
        """
        return False


class ResearchAttentionAdapter(SparseAttentionAdapter):
    """
    Adapter specifically for ResearchAttention implementations.
    """

    @classmethod
    def get_attention_name(cls) -> str:
        """Return the name to register this attention function under"""
        return "research_attention"


class EfficientAttentionAdapter(SparseAttentionAdapter):
    """
    Adapter specifically for EfficientAttention implementations.
    """

    @classmethod
    def get_attention_name(cls) -> str:
        """Return the name to register this attention function under"""
        return "efficient_attention"
