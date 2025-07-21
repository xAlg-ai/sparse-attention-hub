"""Oracle top-P masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopPMasker, TopPMaskerConfig


@dataclass
class OracleTopPMaskerConfig(TopPMaskerConfig):
    """Configuration for OracleTopPMasker."""

    pass  # Inherits top_p from parent with validation


@MaskerRegistry.register(OracleTopPMaskerConfig)
class OracleTopPMasker(TopPMasker):
    """Oracle top-P attention masker."""

    def __init__(self, config: OracleTopPMaskerConfig) -> None:
        """Initialize oracle top-P masker with configuration."""
        super().__init__(config)
        assert (
            isinstance(self.top_p, float) and self.top_p is not None
        ), "top_p must be set as a float in config"
        # self.top_p is now set in the base class

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
        """Add oracle top-P mask to enable top-P attention pattern."""
        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )

        # If sequence is small enough, use full attention
        if self._should_use_full_attention(tensor_dims):
            return self._create_full_mask(tensor_dims, previous_mask.dtype)

        # Create oracle top-P attention mask
        oracle_mask: Mask = self._create_oracle_top_p_mask(
            tensor_dims, keys, queries, previous_mask
        )
        return previous_mask.merge_mask(oracle_mask, inplace=False)

    def _should_use_full_attention(self, dims: AttentionTensorDimensions) -> bool:
        """Determine if full attention should be used instead of top-P attention."""
        effective_size: int = int(self.top_p * dims.seq_len_keys)
        return dims.seq_len_keys <= effective_size

    def _compute_attention_scores(
        self, keys: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """Compute exp(attention scores) between queries and keys."""
        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)
        raw_attention_scores = queries @ keys.transpose(-2, -1)
        _max_attention_score = raw_attention_scores.max(dim=-1, keepdim=True)[0]
        adjusted = torch.exp(raw_attention_scores - _max_attention_score)
        return adjusted

    def _compute_top_p_thresholds(
        self, scores: torch.Tensor, top_p: float
    ) -> torch.Tensor:
        """Compute top-p thresholds using vectorized operations (shape-agnostic)."""
        # Sort scores in descending order along last dimension
        sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)

        # Compute cumulative sum
        cumsum = torch.cumsum(sorted_scores, dim=-1)

        # Normalize by total sum (last element)
        total_sum = cumsum[..., -1:]
        normalized_cumsum = cumsum / total_sum

        # Create top_p tensor with same shape as normalized_cumsum except last dimension
        top_p_tensor = torch.full_like(normalized_cumsum[..., :1], top_p)

        # Find positions where normalized_cumsum >= top_p
        threshold_positions = torch.searchsorted(
            normalized_cumsum, top_p_tensor, side="left"
        )

        # Prepare indices for advanced indexing (shape-agnostic)
        leading_shape = scores.shape[:-1]
        idx_grids = torch.meshgrid(
            *[torch.arange(s, device=scores.device) for s in leading_shape],
            indexing="ij",
        )
        thresholds = sorted_scores[idx_grids + (threshold_positions.squeeze(-1),)]

        # Add trailing singleton dimension for broadcasting
        return thresholds.unsqueeze(-1)

    def _create_oracle_top_p_mask(
        self,
        dims: AttentionTensorDimensions,
        keys: torch.Tensor,
        queries: torch.Tensor,
        previous_mask: Mask,
    ) -> Mask:
        """Create oracle top-P attention mask using vectorized computation."""
        # Get attention scores
        scores: torch.Tensor = self._compute_attention_scores(keys, queries)
        # Get previous dense mask and mask out already active positions
        previous_dense_mask: torch.Tensor = previous_mask.get_dense_mask()
        masked_scores: torch.Tensor = scores.clone()
        masked_scores[previous_dense_mask != 0] = float("-inf")

        # Compute thresholds using vectorized operations
        thresholds: torch.Tensor = self._compute_top_p_thresholds(
            masked_scores, self.top_p
        )

        # Create dense mask: scores >= thresholds
        dense_mask: torch.Tensor = masked_scores >= thresholds

        # Create mask object
        mask_shape: tuple = (
            dims.batch_size,
            dims.num_heads,
            dims.seq_len_queries,
            dims.seq_len_keys,
        )
        return Mask.create_mask_from_dense_mask(
            mask_shape, dense_mask, dtype=previous_mask.dtype
        )

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "OracleTopPMasker":
        """Create OracleTopPMasker instance from configuration."""
        if not isinstance(config, OracleTopPMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
