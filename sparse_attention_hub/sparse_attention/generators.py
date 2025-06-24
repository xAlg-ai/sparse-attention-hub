"""Sparse attention generators and interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Callable

from .base import SparseAttention


class SparseAttentionGen(ABC):
    """Abstract base class for sparse attention generators."""

    @abstractmethod
    def get_custom_attention_function(self) -> Callable:
        """Get the custom attention function."""
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the generator callable."""
        pass


class SparseAttentionHF(SparseAttentionGen):
    """HuggingFace-compatible sparse attention generator."""

    def __init__(self, sparse_attention: SparseAttention):
        self.sparse_attention = sparse_attention

    def get_custom_attention_function(self) -> Callable:
        """Get the custom attention function for HuggingFace models.

        Returns:
            Callable that can be used as attention function in HF models.
        """
        return self.sparse_attention.custom_attention

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the sparse attention mechanism.

        Args:
            *args: Variable arguments passed to attention function
            **kwargs: Keyword arguments passed to attention function

        Returns:
            Output from the sparse attention mechanism
        """
        attention_fn = self.get_custom_attention_function()
        return attention_fn(*args, **kwargs)
