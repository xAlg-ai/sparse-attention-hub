"""Oracle top-K masker implementation."""

from dataclasses import dataclass
from typing import Any

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class OracleTopKConfig(TopKMaskerConfig):
    """Configuration for OracleTopK masker."""

    pass


class OracleTopK(TopKMasker):
    """Oracle top-K masker."""

    def __init__(self, config: OracleTopKConfig):
        """Initialize oracle top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size

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
        """Add oracle top-K mask."""
        if previous_mask.is_full_mask():
            return previous_mask
        
        batch_size, num_heads, seq_len_queries, _ = queries.shape
        seq_len_keys = keys.shape[2]
        
        # Compute heavy_size: if float, multiply by number of keys
        heavy_size = int(self.heavy_size * seq_len_keys) if isinstance(self.heavy_size, float) else int(self.heavy_size)
        
        if seq_len_keys <= heavy_size:
            mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
            return Mask.create_full_mask(mask_shape)
        
        # Compute raw attention scores
        raw_attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Mask out positions already active in previous_mask
        previous_dense_mask = previous_mask.get_dense_mask()
        masked_scores = raw_attention_scores.clone()
        masked_scores[previous_dense_mask != 0] = float('-inf')
        
        # Get top-k indices from inactive positions
        _, top_k_indices = torch.topk(masked_scores, k=heavy_size, dim=-1, largest=True)
        data = torch.ones_like(top_k_indices, dtype=torch.float32)
        
        # Create and merge masks
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        this_mask = Mask.create_from_row_wise_idx(mask_shape, top_k_indices, data, type="index")
        
        return previous_mask.merge_mask(this_mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "OracleTopK":
        """Create OracleTopK instance from configuration."""
        if not isinstance(config, OracleTopKConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
