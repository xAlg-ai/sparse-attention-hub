"""Double sparsity top-K masker implementation."""

from dataclasses import dataclass
from typing import Any

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
)

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class DoubleSparsityTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for DoubleSparsityTopKMasker."""

    group_factor: int
    label_bits: int
    channel_config: Any  # config with stats required for double sparsity


class DoubleSparsityTopKMasker(TopKMasker):
    """Double sparsity top-K masker."""

    def __init__(self, config: DoubleSparsityTopKMaskerConfig):
        """Initialize double sparsity top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.group_factor = config.group_factor
        self.label_bits = config.label_bits
        self.channel_config = config.channel_config

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
        """Add double sparsity top-K mask."""
        # just return the same mask for now
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "DoubleSparsityTopKMasker":
        """Create DoubleSparsityTopKMasker instance from configuration."""
        if not isinstance(config, DoubleSparsityTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
