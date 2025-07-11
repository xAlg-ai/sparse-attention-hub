"""Double sparsity top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class ChannelConfig:
    """Configuration for channel statistics in double sparsity."""

    pass


@dataclass
class DoubleSparsityTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for DoubleSparsityTopKMasker."""

    group_factor: int
    label_bits: int
    channel_config: ChannelConfig  # config with stats required for double sparsity


@MaskerRegistry.register(DoubleSparsityTopKMaskerConfig)
class DoubleSparsityTopKMasker(TopKMasker):
    """Double sparsity top-K masker."""

    heavy_size: Union[float, int]
    group_factor: int
    label_bits: int
    channel_config: ChannelConfig

    def __init__(self, config: DoubleSparsityTopKMaskerConfig) -> None:
        """Initialize double sparsity top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.group_factor = config.group_factor
        self.label_bits = config.label_bits
        self.channel_config = config.channel_config

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add double sparsity top-K mask."""
        # just return the same mask for now
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "DoubleSparsityTopKMasker":
        """Create DoubleSparsityTopKMasker instance from configuration."""
        if not isinstance(config, DoubleSparsityTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
