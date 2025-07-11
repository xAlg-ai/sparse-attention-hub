"""PQ cache top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class PQCacheConfig(TopKMaskerConfig):
    """Configuration for PQCache masker."""

    pq_sub_dim: int
    pq_bits: int


@MaskerRegistry.register(PQCacheConfig)
class PQCache(TopKMasker):
    """PQ cache-based top-K masker."""

    def __init__(self, config: PQCacheConfig) -> None:
        """Initialize PQ cache masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.pq_sub_dim = config.pq_sub_dim
        self.pq_bits = config.pq_bits

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
        """Add PQ cache mask."""
        # just return the same mask for now
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
