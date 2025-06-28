"""Base classes for research attention mechanisms."""

from abc import abstractmethod
from typing import Any, Optional, Tuple

from ..base import SparseAttention


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