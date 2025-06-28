"""Base classes for efficient attention mechanisms."""

from abc import abstractmethod
from typing import Any, Optional, Tuple

from ..base import SparseAttention


class EfficientAttention(SparseAttention):
    """Abstract base class for efficient attention mechanisms."""

    @abstractmethod
    def custom_attention(self) -> Tuple[Any, Optional[Any]]:
        """Compute efficient attention mechanism.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass 