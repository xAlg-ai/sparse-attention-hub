"""Base classes for efficient attention mechanisms."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..base import SparseAttention, SparseAttentionConfig


@dataclass
class EfficientAttentionConfig(SparseAttentionConfig):
    """Configuration class for efficient attention mechanisms."""

    pass


class EfficientAttention(SparseAttention):
    """Abstract base class for efficient attention mechanisms."""

    def __init__(self, sparse_attention_config: SparseAttentionConfig) -> None:
        """Initialize efficient attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
        """
        super().__init__(sparse_attention_config)

    @abstractmethod
    def custom_attention(self) -> Tuple[Any, Optional[Any]]:
        """Compute efficient attention mechanism.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass

    @classmethod
    def create_from_config(
        cls, config: EfficientAttentionConfig
    ) -> "EfficientAttention":
        """Create efficient attention instance from configuration.

        Args:
            config: Configuration for the efficient attention mechanism.

        Returns:
            Instance of the efficient attention mechanism.
        """
        # Import here to avoid circular imports
        from .implementations import (
            DoubleSparsity,
            DoubleSparsityConfig,
            HashAttention,
            HashAttentionConfig,
        )

        # Registry mapping config types to concrete efficient attention classes
        _EFFICIENT_ATTENTION_REGISTRY = {
            DoubleSparsityConfig: DoubleSparsity,
            HashAttentionConfig: HashAttention,
        }

        # Look up the concrete class based on the config type
        concrete_class = _EFFICIENT_ATTENTION_REGISTRY.get(type(config))
        if concrete_class is None:
            raise ValueError(
                f"No efficient attention class found for config type: {type(config)}"
            )

        # Call the concrete class's create_from_config method
        return concrete_class.create_from_config(config)
