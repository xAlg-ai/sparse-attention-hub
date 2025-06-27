"""Base classes for sparse attention mechanisms (bare metal)."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class SparseAttention(ABC):
    """Abstract base class for sparse attention mechanisms."""

    def __init__(self) -> None:
        """Initialize sparse attention mechanism."""
        pass

    @abstractmethod
    def custom_attention(self) -> Tuple[Any, Optional[Any]]:
        """Compute custom attention mechanism.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass

    def pre_attention_hook_generator(self) -> None:
        """Generate pre-attention hooks for model integration."""
        pass


class EfficientAttention(SparseAttention):
    """Abstract base class for efficient attention mechanisms."""

    @abstractmethod
    def custom_attention(self) -> Tuple[Any, Optional[Any]]:
        """Compute efficient attention mechanism.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass


class ResearchAttention(SparseAttention):
    """Abstract base class for research attention mechanisms with maskers."""

    def __init__(self) -> None:
        """Initialize research attention mechanism."""
        super().__init__()
        self.masks: list = []

    @abstractmethod
    def custom_attention(self) -> Tuple[Any, Optional[Any]]:
        """Compute research attention mechanism with masking.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass
