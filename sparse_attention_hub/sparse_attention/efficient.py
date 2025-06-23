"""Efficient attention implementations."""

from typing import Optional, Tuple

from torch import Tensor

from .base import EfficientAttention


class DoubleSparsity(EfficientAttention):
    """Double sparsity attention mechanism."""

    def custom_attention(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute double sparsity attention.

        Returns:
            Tuple of attention output tensor and optional attention weights.
        """
        # TODO: Implement double sparsity attention algorithm
        raise NotImplementedError("DoubleSparsity attention not yet implemented")


class HashAttention(EfficientAttention):
    """Hash-based attention mechanism."""

    def custom_attention(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute hash-based attention.

        Returns:
            Tuple of attention output tensor and optional attention weights.
        """
        # TODO: Implement hash attention algorithm
        raise NotImplementedError("HashAttention not yet implemented")
