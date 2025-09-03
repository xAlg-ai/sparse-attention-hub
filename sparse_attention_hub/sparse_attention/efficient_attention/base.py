"""Base classes for efficient attention mechanisms."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, cast

import torch
from torch import nn

from ..base import SparseAttention, SparseAttentionConfig


@dataclass
class EfficientAttentionConfig(SparseAttentionConfig):
    """Configuration class for efficient attention mechanisms.

    This is a base configuration class for production-ready sparse attention
    implementations that prioritize efficiency and performance.
    """

    pass


class EfficientAttention(SparseAttention):
    """Abstract base class for efficient attention mechanisms.

    This class serves as the base for production-ready sparse attention
    implementations that are optimized for performance and memory efficiency,
    such as HashAttention and DoubleSparsity.
    """

    def __init__(self, sparse_attention_config: SparseAttentionConfig) -> None:
        """Initialize efficient attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
        """
        super().__init__(sparse_attention_config)

    @abstractmethod
    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute efficient attention mechanism.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "EfficientAttention":
        """Create efficient attention instance from configuration.

        Args:
            config: Configuration for the efficient attention mechanism.
                Must be an instance of EfficientAttentionConfig.

        Returns:
            Instance of the efficient attention mechanism.

        Raises:
            TypeError: If config is not an EfficientAttentionConfig.
            ValueError: If no implementation is found for the config type.
        """
        if not isinstance(config, EfficientAttentionConfig):
            raise TypeError(f"Expected EfficientAttentionConfig, got {type(config)}")

        registry = cls._get_implementation_registry()
        concrete_class = cls._get_concrete_class(config, registry)
        return concrete_class.create_from_config(config)

    @classmethod
    def _get_implementation_registry(
        cls,
    ) -> Dict[Type[EfficientAttentionConfig], Type["EfficientAttention"]]:
        """Get the registry mapping config types to implementation classes.

        Returns:
            Dictionary mapping config types to their corresponding implementation classes.
        """
        # Import here to avoid circular imports
        from .implementations import (
            DoubleSparsity,
            DoubleSparsityConfig,
            HashAttention,
            HashAttentionConfig,
        )

        return {
            DoubleSparsityConfig: DoubleSparsity,
            HashAttentionConfig: HashAttention,
        }

    @classmethod
    def _get_concrete_class(
        cls,
        config: EfficientAttentionConfig,
        registry: Dict[Type[EfficientAttentionConfig], Type["EfficientAttention"]],
    ) -> Type["EfficientAttention"]:
        """Get the concrete implementation class for the given configuration.

        Args:
            config: Configuration instance.
            registry: Registry mapping config types to implementation classes.

        Returns:
            Concrete implementation class.

        Raises:
            ValueError: If no implementation is found for the config type.
        """
        concrete_class = registry.get(type(config))
        if concrete_class is None:
            raise ValueError(
                f"No efficient attention class found for config type: {type(config)}"
            )
        return cast(Type["EfficientAttention"], concrete_class)
