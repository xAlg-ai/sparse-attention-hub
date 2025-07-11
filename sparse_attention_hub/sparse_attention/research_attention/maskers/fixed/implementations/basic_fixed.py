"""Basic fixed pattern masker implementations."""

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import FixedMasker, FixedMaskerConfig


@dataclass
class LocalMaskerConfig(FixedMaskerConfig):
    """Configuration for LocalMasker."""

    window_size: Union[float, int]


@MaskerRegistry.register(LocalMaskerConfig)
class LocalMasker(FixedMasker):
    """Local attention masker."""

    def __init__(self, config: LocalMaskerConfig):
        """Initialize local masker with configuration."""
        super().__init__(config)
        self.window_size = config.window_size

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Any,
    ) -> Mask:
        """Add local mask."""
        if previous_mask.is_full_mask():
            return previous_mask

        batch_size = queries.shape[0]
        num_heads = queries.shape[1]
        seq_len_queries = queries.shape[2]
        seq_len_keys = keys.shape[2]

        if isinstance(self.window_size, float):
            window_size = int(self.window_size * seq_len_keys)
        else:
            window_size = int(self.window_size)

        if seq_len_keys <= window_size + seq_len_queries:
            mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
            return Mask.create_full_mask(mask_shape, dtype=previous_mask.dtype)

        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        device = keys.device

        # Vectorized local window computation
        # For query k: window_start = seq_len_keys - (seq_len_queries - k) - window_size + 1
        query_positions = torch.arange(seq_len_queries, device=device, dtype=torch.long)
        window_starts = (
            seq_len_keys - seq_len_queries - window_size + query_positions + 1
        )
        window_offsets = torch.arange(window_size, device=device, dtype=torch.long)

        all_window_indices = window_starts.unsqueeze(1) + window_offsets.unsqueeze(0)
        all_window_indices = torch.clamp(all_window_indices, 0, seq_len_keys - 1)

        row_wise_idx = (
            all_window_indices.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, seq_len_queries, window_size)
        )

        data = torch.ones(
            (batch_size, num_heads, seq_len_queries, window_size),
            dtype=previous_mask.dtype,
            device=device,
        )

        local_mask = Mask.create_from_row_wise_idx(
            shape=mask_shape, row_wise_idx=row_wise_idx, data=data, type="index", dtype=previous_mask.dtype
        )

        return previous_mask.merge_mask(local_mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "LocalMasker":
        """Create LocalMasker instance from configuration."""
        if not isinstance(config, LocalMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


@MaskerRegistry.register(FixedMaskerConfig)
class CausalMasker(FixedMasker):
    """Causal attention masker."""

    def __init__(self, config: FixedMaskerConfig):
        """Initialize causal masker with configuration."""
        super().__init__(config)

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Any,
    ) -> Mask:
        """Add causal mask."""
        # just return the same mask for now
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "CausalMasker":
        """Create CausalMasker instance from configuration."""
        if not isinstance(config, FixedMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


@dataclass
class SinkMaskerConfig(FixedMaskerConfig):
    """Configuration for SinkMasker."""

    sink_size: Union[float, int]


@MaskerRegistry.register(SinkMaskerConfig)
class SinkMasker(FixedMasker):
    """Sink attention masker."""

    def __init__(self, config: SinkMaskerConfig):
        """Initialize sink masker with configuration."""
        super().__init__(config)
        self.sink_size = config.sink_size

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Any,
    ) -> Mask:
        """Add sink mask."""
        # 1. Check if previous_mask is full mask, if so return full mask
        if previous_mask.is_full_mask():
            return previous_mask

        # Get tensor shapes
        batch_size = queries.shape[0]
        num_heads = queries.shape[1]
        seq_len_queries = queries.shape[2]
        seq_len_keys = keys.shape[2]

        # Convert sink_size to int for tensor operations
        sink_size = int(self.sink_size)

        # 2. Check if # keys is smaller than sink_size, if so, then return a full mask
        if seq_len_keys <= sink_size:
            mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
            return Mask.create_full_mask(mask_shape, dtype=previous_mask.dtype)

        # 3. Compute row_wise_idx: b,h,sq,sink_size with row_wise_idx[i,j,k,:] = arrange(0,sink_size)
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)

        # Create row_wise_idx with shape (b, h, sq, sink_size)
        # Each row contains indices [0, 1, ..., sink_size-1]
        row_wise_idx = (
            torch.arange(sink_size, device=keys.device, dtype=torch.long)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_heads, seq_len_queries, sink_size)
        )

        # Create data tensor with all ones (sink positions get weight 1.0)
        data = torch.ones(
            (batch_size, num_heads, seq_len_queries, sink_size),
            device=keys.device,
            dtype=previous_mask.dtype,
        )

        # 4. Call Mask.create_from_row_wise_idx() to get the mask
        sink_mask = Mask.create_from_row_wise_idx(
            shape=mask_shape, row_wise_idx=row_wise_idx, data=data, type="index", dtype=previous_mask.dtype
        )

        # 5. Merge this_mask with previous mask using previous_mask.merge and return the new mask
        return previous_mask.merge_mask(sink_mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SinkMasker":
        """Create SinkMasker instance from configuration."""
        if not isinstance(config, SinkMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
