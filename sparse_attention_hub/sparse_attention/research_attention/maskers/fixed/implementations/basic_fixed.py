"""Basic fixed pattern masker implementations."""

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import FixedMasker, FixedMaskerConfig


@dataclass
class LocalMaskerConfig(FixedMaskerConfig):
    """Configuration for LocalMasker."""

    window_size: Union[float, int]

    @classmethod
    def get_search_space(cls, task_name: str) -> Dict[str, Any]:
        """Get Ray Tune search space for Local masker.
        
        Args:
            task_name: Name of the benchmark task to optimize for
            
        Returns:
            Dictionary mapping parameter names to Ray Tune distributions
        """
        from ray import tune

        return {
            "window_size": tune.choice([32, 64, 128, 256])
        }


@MaskerRegistry.register(LocalMaskerConfig)
class LocalMasker(FixedMasker):
    """Local attention masker."""

    window_size: Union[float, int]

    def __init__(self, config: LocalMaskerConfig) -> None:
        """Initialize local masker with configuration."""
        super().__init__(config)
        self.window_size = config.window_size

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add local mask to enable local attention pattern."""
        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_window_size: int = self._calculate_effective_window_size(
            tensor_dims.seq_len_keys
        )

        # If sequence is small enough, use full attention
        if self._should_use_full_attention(tensor_dims, effective_window_size):
            return self._create_full_mask(tensor_dims, previous_mask.dtype)

        # Create local attention mask
        local_mask: Mask = self._create_local_mask(
            tensor_dims, effective_window_size, keys.device, previous_mask.dtype
        )
        return previous_mask.merge_mask(local_mask, inplace=False)

    def _calculate_effective_window_size(self, seq_len_keys: int) -> int:
        """Calculate the effective window size based on configuration."""
        return self._calculate_effective_size(self.window_size, seq_len_keys)

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, window_size: int
    ) -> bool:
        """Determine if full attention should be used instead of local attention."""
        return dims.seq_len_keys <= window_size + dims.seq_len_queries

    def _create_local_mask(
        self,
        dims: AttentionTensorDimensions,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mask:
        """Create local attention mask using vectorized computation."""
        window_indices: torch.Tensor = self._compute_window_indices(
            dims, window_size, device
        )
        return self._create_mask_from_rowise_indices(
            dims, window_indices, device, dtype
        )

    def _compute_window_indices(
        self, dims: AttentionTensorDimensions, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """Compute window indices for local attention using vectorized operations."""
        # For each query position, compute the start of its local window
        # Formula: window_start = seq_len_keys - (seq_len_queries - query_pos) - window_size + 1
        query_positions: torch.Tensor = torch.arange(
            dims.seq_len_queries, device=device, dtype=torch.long
        )
        window_starts: torch.Tensor = (
            dims.seq_len_keys - dims.seq_len_queries - window_size + query_positions + 1
        )

        # Create offset indices for the window
        window_offsets: torch.Tensor = torch.arange(
            window_size, device=device, dtype=torch.long
        )

        # Compute all window indices: window_start + offset for each query
        window_indices: torch.Tensor = window_starts.unsqueeze(
            1
        ) + window_offsets.unsqueeze(0)
        window_indices = torch.clamp(window_indices, 0, dims.seq_len_keys - 1)

        # Expand to match batch and head dimensions
        return (
            window_indices.unsqueeze(0)
            .unsqueeze(0)
            .expand(dims.batch_size, dims.num_heads, dims.seq_len_queries, window_size)
        )

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "LocalMasker":
        """Create LocalMasker instance from configuration."""
        if not isinstance(config, LocalMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


@MaskerRegistry.register(FixedMaskerConfig)
class CausalMasker(FixedMasker):
    """Causal attention masker."""

    def __init__(self, config: FixedMaskerConfig) -> None:
        """Initialize causal masker with configuration."""
        super().__init__(config)

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
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

    @classmethod
    def get_search_space(cls, task_name: str) -> Dict[str, Any]:
        """Get Ray Tune search space for Sink masker.
        
        Args:
            task_name: Name of the benchmark task to optimize for
            
        Returns:
            Dictionary mapping parameter names to Ray Tune distributions
        """
        from ray import tune

        return {
            "sink_size": tune.choice([4, 8, 16, 32, 64, 128])
        }


@MaskerRegistry.register(SinkMaskerConfig)
class SinkMasker(FixedMasker):
    """Sink attention masker."""

    sink_size: Union[float, int]

    def __init__(self, config: SinkMaskerConfig) -> None:
        """Initialize sink masker with configuration."""
        super().__init__(config)
        self.sink_size = config.sink_size

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add sink mask to enable sink attention pattern."""
        if previous_mask.is_full_mask():
            return previous_mask

        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )
        effective_sink_size: int = self._calculate_effective_sink_size(
            tensor_dims.seq_len_keys
        )

        # If sequence is small enough, use full attention
        if self._should_use_full_attention(tensor_dims, effective_sink_size):
            return self._create_full_mask(tensor_dims, previous_mask.dtype)

        # Create sink attention mask
        sink_mask: Mask = self._create_sink_mask(
            tensor_dims, effective_sink_size, keys.device, previous_mask.dtype
        )
        return previous_mask.merge_mask(sink_mask, inplace=False)

    def _calculate_effective_sink_size(self, seq_len_keys: int) -> int:
        """Calculate the effective sink size based on configuration."""
        return self._calculate_effective_size(self.sink_size, seq_len_keys)

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, sink_size: int
    ) -> bool:
        """Determine if full attention should be used instead of sink attention."""
        return dims.seq_len_keys <= sink_size

    def _create_sink_mask(
        self,
        dims: AttentionTensorDimensions,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mask:
        """Create sink attention mask using vectorized computation."""
        sink_indices: torch.Tensor = self._compute_sink_indices(dims, sink_size, device)
        return self._create_mask_from_rowise_indices(dims, sink_indices, device, dtype)

    def _compute_sink_indices(
        self, dims: AttentionTensorDimensions, sink_size: int, device: torch.device
    ) -> torch.Tensor:
        """Compute sink indices for sink attention pattern."""
        # Create row_wise_idx with shape (b, h, sq, sink_size)
        # Each row contains indices [0, 1, ..., sink_size-1] (first sink_size positions)
        sink_indices: torch.Tensor = (
            torch.arange(sink_size, device=device, dtype=torch.long)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(dims.batch_size, dims.num_heads, dims.seq_len_queries, sink_size)
        )

        return sink_indices

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SinkMasker":
        """Create SinkMasker instance from configuration."""
        if not isinstance(config, SinkMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
