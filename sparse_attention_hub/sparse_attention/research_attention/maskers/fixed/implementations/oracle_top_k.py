"""Oracle top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import _get_num_key_value_groups, repeat_kv
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class OracleTopKConfig(TopKMaskerConfig):
    """Configuration for OracleTopK masker."""

    pass


@MaskerRegistry.register(OracleTopKConfig)
class OracleTopK(TopKMasker):
    """Oracle top-K masker."""

    heavy_size: Union[float, int]

    def __init__(self, config: OracleTopKConfig) -> None:
        """Initialize oracle top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size

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
        """Add oracle top-K mask to enable oracle-based attention selection."""
        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_heavy_size: int = self._calculate_effective_heavy_size(
            tensor_dims.seq_len_keys
        )

        # If sequence is small enough, use full attention
        if self._should_use_full_attention(tensor_dims, effective_heavy_size):
            return self._create_full_mask(tensor_dims, previous_mask.dtype)

        # Create oracle top-K mask
        oracle_mask: Mask = self._create_oracle_topk_mask(
            tensor_dims, effective_heavy_size, keys, queries, previous_mask
        )
        return previous_mask.merge_mask(oracle_mask, inplace=False)

    def _calculate_effective_heavy_size(self, seq_len_keys: int) -> int:
        """Calculate the effective heavy size based on configuration."""
        return self._calculate_effective_size(self.heavy_size, seq_len_keys)

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_size: int
    ) -> bool:
        """Determine if full attention should be used instead of oracle top-K attention."""
        return dims.seq_len_keys <= heavy_size

    def _create_oracle_topk_mask(
        self,
        dims: AttentionTensorDimensions,
        heavy_size: int,
        keys: torch.Tensor,
        queries: torch.Tensor,
        previous_mask: Mask,
    ) -> Mask:
        """Create oracle top-K mask using raw attention scores."""
        raw_attention_scores: torch.Tensor = self._compute_raw_attention_scores(
            keys, queries
        )
        top_k_indices: torch.Tensor = self._get_topk_indices_from_inactive_positions(
            raw_attention_scores, previous_mask, heavy_size
        )
        return self._create_mask_from_rowise_indices(
            dims, top_k_indices, keys.device, previous_mask.dtype
        )

    def _compute_raw_attention_scores(
        self, keys: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """Compute raw attention scores using query-key dot product."""
        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)
        return torch.matmul(queries, keys.transpose(-2, -1))

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "OracleTopK":
        """Create OracleTopK instance from configuration."""
        if not isinstance(config, OracleTopKConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
