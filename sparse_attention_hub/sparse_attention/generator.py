"""Sparse attention generators and interfaces (bare metal)."""

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