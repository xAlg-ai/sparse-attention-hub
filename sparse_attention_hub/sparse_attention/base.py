"""Base classes for sparse attention mechanisms."""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

from torch import Tensor

from .metadata import SparseAttentionMetadata


class SparseAttention(ABC):
    """Abstract base class for sparse attention mechanisms."""

    def __init__(self) -> None:
        self.metadata = SparseAttentionMetadata()

    @abstractmethod
    def custom_attention(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute custom attention mechanism.

        Returns:
            Tuple of attention output tensor and optional attention weights.
        """
        pass

    def pre_attention_hook_generator(self) -> None:
        """Generate pre-attention hooks for model integration."""
        pass


class EfficientAttention(SparseAttention):
    """Abstract base class for efficient attention mechanisms."""

    @abstractmethod
    def custom_attention(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute efficient attention mechanism.

        Returns:
            Tuple of attention output tensor and optional attention weights.
        """
        pass


class ResearchAttention(SparseAttention):
    """Abstract base class for research attention mechanisms with maskers."""

    def __init__(self) -> None:
        super().__init__()
        self.masks: Sequence = []

    @abstractmethod
    def custom_attention(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute research attention mechanism with masking.

        Returns:
            Tuple of attention output tensor and optional attention weights.
        """
        pass
