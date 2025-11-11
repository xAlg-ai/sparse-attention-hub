"""Random sampling masker implementation.

This module provides a random sampling masker that randomly selects a fraction
of attention positions for each query. The sampling is performed independently
for each query position, allowing for efficient sparse attention computation.

The RandomSamplingMasker is useful for:
- Reducing computational complexity in attention mechanisms
- Exploring random attention patterns for research purposes
- Creating baseline comparisons for more sophisticated sampling methods
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import SamplingMasker, SamplingMaskerConfig


@dataclass
class RandomSamplingMaskerConfig(SamplingMaskerConfig):
    """Configuration for RandomSamplingMasker.

    This configuration class inherits from SamplingMaskerConfig and adds
    validation to ensure the sampling_rate is within the valid range [0, 1].

    Attributes:
        sampling_rate: Float in range [0, 1] representing the fraction of
            indices to sample for each query position. A value of 0.0 means
            no indices are sampled, while 1.0 means all indices are sampled.
    """

    sampling_rate: float  # Float in range [0,1] representing fraction of indices to sample
    search_space: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate sampling_rate after initialization."""
        if not (0.0 < self.sampling_rate <= 1.0):
            raise ValueError(
                f"sampling_rate must be in range (0, 1], got {self.sampling_rate}"
            )


@MaskerRegistry.register(RandomSamplingMaskerConfig)
class RandomSamplingMasker(SamplingMasker):
    """Random sampling masker for sparse attention computation.

    This masker implements random sampling of attention positions by randomly
    selecting a fraction of key positions for each query position. The sampling
    is performed independently for each query.


    Attributes:
        sampling_rate: The fraction of indices to sample for each query position.
            This value is set from the configuration and represents the probability
            of each index being included in the attention computation.

    Imporant Notes:
        - The sampling is performed with replacement so the final sample size might be
        smaller than the expected. Without replacement will be expensive.
        - The sampling ignores the previous mask (i.e. samples from entire range of keys)
        This is different from top-k masker behavior. This is done so as to avoid
        involved idx manipulation.
        - Merge operation adds the data in masks. It clamps the data to 1.0. keeping
        the probability of being present in the mask sane.


    Example:
        >>> config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        >>> masker = RandomSamplingMasker(config)
        >>> # Use masker.add_mask() to apply random sampling to attention masks
    """

    def __init__(self, config: RandomSamplingMaskerConfig) -> None:
        """Initialize random sampling masker with configuration.

        Args:
            config: Configuration object containing the sampling rate and other
                parameters for the random sampling masker.

        Raises:
            ValueError: If the sampling_rate in config is not in range [0, 1].
                This validation is performed in the config's __post_init__ method.
        """
        super().__init__(config)
        self.sampling_rate = config.sampling_rate

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
        """Add random sampling mask to attention computation.

        This method implements the core random sampling logic. For each query
        position, it randomly selects a fraction of key positions based on the
        sampling_rate. The selected positions are assigned a weight equal to
        the sampling_rate, representing the probability of that position being
        included in the attention computation.

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
                The keys used in the attention computation.
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).
                The queries used in the attention computation.
            values: Value tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
                The values used in the attention computation.
            attention_mask: Attention mask tensor indicating which positions are valid.
                This is typically used for padding or causal masking.
            sparse_meta_data: Dictionary containing sparse attention metadata.
                This can include additional information needed for the masker.
            previous_mask: Previous attention mask to merge with the new random sampling mask.
                If this is a full mask, it will be returned unchanged.
            **kwargs: Additional keyword arguments that may be passed to the masker.
                These are not used in the current implementation but allow for
                future extensibility.

        Returns:
            A new Mask object representing the attention pattern after applying
            random sampling. The mask is created by merging the previous mask
            with the new random sampling mask.

        Note:
            - If previous_mask is a full mask, it is returned unchanged
            - The number of indices sampled per query is int(sampling_rate * seq_len_keys)
            - Each sampled position is assigned a weight equal to sampling_rate
            - Random sampling is performed independently for each query position
            - The resulting mask is merged with the previous mask using the merge_mask method
        """
        # Step 1: Check if previous_mask is full mask, if so return full mask
        if previous_mask.is_full_mask():
            return previous_mask

        # Extract tensor dimensions
        batch_size: int = queries.shape[0]
        num_heads: int = queries.shape[1]
        seq_len_queries: int = queries.shape[2]
        seq_len_keys: int = keys.shape[2]

        # Calculate number of indices to sample per row
        num_indices_to_sample: int = int(self.sampling_rate * seq_len_keys)

        # Step 2: Compute row_wise_idx: (b, h, s, sampling_rate * #keys) tensor of random indices
        # Generate random indices for each row
        row_wise_idx: torch.Tensor = torch.randint(
            low=0,
            high=seq_len_keys,
            size=(batch_size, num_heads, seq_len_queries, num_indices_to_sample),
            device=keys.device,
            dtype=torch.long,
        )

        # Step 3: Use row-wise idx to compute this_mask using Mask.create_row_wise_idx()
        # Create data tensor with values equal to sampling_rate (probability of being sampled)
        data: torch.Tensor = torch.full_like(
            row_wise_idx,
            self.sampling_rate,
            dtype=previous_mask.dtype,
            device=keys.device,
        )

        # Create mask shape
        mask_shape: tuple[int, int, int, int] = (
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
        )

        this_mask: Mask = Mask.create_from_row_wise_idx(
            shape=mask_shape,
            row_wise_idx=row_wise_idx,
            data=data,
            mask_type="index",
            dtype=previous_mask.dtype,
        )

        # Step 4: Merge this_mask with previous mask using previous_mask.merge and return the new mask
        return previous_mask.merge_mask(this_mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "RandomSamplingMasker":
        """Create RandomSamplingMasker instance from configuration.

        This factory method creates a RandomSamplingMasker instance from a
        configuration object. It validates that the config is of the correct
        type before creating the masker.

        Args:
            config: Configuration object for the masker. Must be an instance
                of RandomSamplingMaskerConfig.

        Returns:
            A new RandomSamplingMasker instance configured with the provided config.

        Raises:
            ValueError: If the config is not an instance of RandomSamplingMaskerConfig.
                This ensures type safety and prevents configuration mismatches.

        Example:
            >>> config = RandomSamplingMaskerConfig(sampling_rate=0.3)
            >>> masker = RandomSamplingMasker.create_from_config(config)
            >>> assert isinstance(masker, RandomSamplingMasker)
        """
        if not isinstance(config, RandomSamplingMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
