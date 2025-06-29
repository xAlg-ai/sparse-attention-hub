"""Base classes for sparse attention mechanisms (bare metal)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class SparseAttentionConfig:
    """Configuration class for sparse attention mechanisms."""
    pass


class SparseAttention(ABC):
    """Abstract base class for sparse attention mechanisms."""

    def __init__(self, sparse_attention_config: SparseAttentionConfig) -> None:
        """Initialize sparse attention mechanism.
        
        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
        """
        self.sparse_attention_config = sparse_attention_config

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

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "SparseAttention":
        """Create sparse attention instance from configuration.
        
        Args:
            config: Configuration for the sparse attention mechanism.
            
        Returns:
            Instance of the sparse attention mechanism.
        """
        return cls(config)
