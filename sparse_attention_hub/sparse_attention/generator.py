"""Sparse attention generators and interfaces (bare metal)."""

from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager


class SparseAttentionGen(ABC):
    """Abstract base class for sparse attention generators."""

    @abstractmethod
    def get_custom_attention_function(self) -> Callable:
        """Get the custom attention function."""
        pass

    @abstractmethod
    def __call__(self, model: Any) -> ContextManager[Any]:
        """
        Context manager to apply a sparse attention method to a model.

        This method replaces the attention_interface in all attention layers of the model
        with the custom attention function returned by get_custom_attention_function.
        The custom attention function is reverted when the context manager is exited.
        """
        pass
