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
