"""Hash attention top-K masker implementation."""

from dataclasses import dataclass
from typing import Any

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
)

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class HashAttentionTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for HashAttentionTopKMasker."""

    hat_bits: int
    hat_mlp_layers: int
    hat_mlp_hidden_size: int


class HashAttentionTopKMasker(TopKMasker):
    """Hash attention top-K masker."""

    def __init__(self, config: HashAttentionTopKMaskerConfig):
        """Initialize hash attention top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.hat_bits = config.hat_bits
        self.hat_mlp_layers = config.hat_mlp_layers
        self.hat_mlp_hidden_size = config.hat_mlp_hidden_size

    def add_mask(
        self,
        keys: Any,
        queries: Any,
        values: Any,
        attention_mask: Any,
        sparse_meta_data: Any,
        previous_mask: Any,
        **kwargs: Any,
    ) -> Any:
        """Add hash attention top-K mask."""
        # just return the same mask for now
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "HashAttentionTopKMasker":
        """Create HashAttentionTopKMasker instance from configuration."""
        if not isinstance(config, HashAttentionTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
