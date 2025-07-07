"""Base classes for sparse attention mechanisms (bare metal)."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class SparseAttention(ABC):
    """Abstract base class for sparse attention mechanisms."""

    def __init__(self) -> None:
        """Initialize sparse attention mechanism."""
        # Base class initialization - no specific setup required

    @abstractmethod
    def custom_attention(self, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[Any]]:
        """Compute custom attention mechanism.

        Args:
            *args: Variable arguments for attention computation
            **kwargs: Keyword arguments for attention computation

        Returns:
            Tuple of attention output and optional attention weights.
        """
        # Abstract method - implementation required in subclass

    def pre_attention_hook_generator(self) -> None:
        """Generate pre-attention hooks for model integration."""
        # Default implementation - no hooks generated


class EfficientAttention(SparseAttention):
    """Abstract base class for efficient attention mechanisms."""

    @abstractmethod
    def custom_attention(self, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[Any]]:
        """Compute efficient attention mechanism.

        Args:
            *args: Variable arguments for attention computation
            **kwargs: Keyword arguments for attention computation

        Returns:
            Tuple of attention output and optional attention weights.
        """
        # Abstract method - implementation required in subclass


class ResearchAttention(SparseAttention):
    """Abstract base class for research attention mechanisms with maskers."""

    def __init__(self) -> None:
        """Initialize research attention mechanism."""
        super().__init__()
        self.masks: list = []

    @abstractmethod
    def custom_attention(self, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[Any]]:
        """Compute research attention mechanism with masking.

        Args:
            *args: Variable arguments for attention computation
            **kwargs: Keyword arguments for attention computation

        Returns:
            Tuple of attention output and optional attention weights.
        """
        # Abstract method - implementation required in subclass
