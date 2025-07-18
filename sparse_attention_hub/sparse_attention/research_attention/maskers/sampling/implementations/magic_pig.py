"""Magic pig sampling masker implementation.

This module provides a MagicPig sampling masker that uses Locality Sensitive Hashing (LSH)
to efficiently approximate maximum inner product search for attention computation. It combines
LSH-based similarity matching with probability-based sampling to create sparse attention patterns.

"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import SamplingMasker, SamplingMaskerConfig


@dataclass
class MagicPigConfig(SamplingMaskerConfig):
    """Configuration for MagicPig masker.

    This configuration class inherits from SamplingMaskerConfig and adds LSH-specific
    parameters.

    Attributes:
        lsh_l: Number of LSH tables. Higher values increase the chance of finding
            similar items but also increase computational cost.
        lsh_k: Number of bits per LSH table. Higher values make each table more
            precise but reduce the probability of collisions.
    """

    lsh_l: int  # number of LSH tables
    lsh_k: int  # number of bits per LSH table

    def __post_init__(self) -> None:
        """Validate LSH parameters after initialization."""
        if self.lsh_l <= 0:
            raise ValueError(f"lsh_l must be positive, got {self.lsh_l}")
        if self.lsh_k <= 0:
            raise ValueError(f"lsh_k must be positive, got {self.lsh_k}")


@MaskerRegistry.register(MagicPigConfig)
class MagicPig(SamplingMasker):
    """Magic Pig masker using Locality Sensitive Hashing for attention computation.

    This masker implements LSH-based sampling of attention positions by using
    locality sensitive hashing to identify similar key-query pairs. The sampling
    is based on both LSH collision probability and actual LSH matches.

    Attributes:
        lsh_l: Number of LSH tables used for hashing.
        lsh_k: Number of bits per LSH table.

    Important Notes:
        - Uses the Neyshabur & Srebro technique to transform inner product search to cosine similarity
        - Computes theoretical LSH collision probabilities based on cosine similarity
        - Uses random signed projections for efficient LSH implementation
        - Ignores the sampling_rate parameter from the parent class
        - Merge operation adds the data in masks and clamps to 1.0

    Example:
        >>> config = MagicPigConfig(lsh_l=5, lsh_k=4)
        >>> masker = MagicPig(config)
        >>> # Use masker.add_mask() to apply LSH-based sampling to attention masks
    """

    def __init__(self, config: MagicPigConfig) -> None:
        """Initialize Magic Pig masker with configuration.

        Args:
            config: Configuration object containing LSH parameters.

        Raises:
            ValueError: If LSH parameters are invalid.
        """
        super().__init__(config)
        self.lsh_l = config.lsh_l
        self.lsh_k = config.lsh_k

    def _transform_for_mips(
        self, keys: torch.Tensor, queries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform keys and queries for maximum inner product search.

        Uses the technique from Neyshabur, Behnam and Srebro, Nathan.
        "On symmetric and asymmetric LSHs for inner product search."

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).

        Returns:
            Tuple of (transformed_keys, transformed_queries) with augmented dimensions.
        """
        # Normalize queries
        query_norms: torch.Tensor = torch.norm(queries, dim=-1, keepdim=True)
        queries_normalized: torch.Tensor = queries / (query_norms + 1e-8)

        # For keys, use the augmentation technique
        key_norms: torch.Tensor = torch.norm(keys, dim=-1, keepdim=True)
        max_key_norm: torch.Tensor = torch.max(key_norms)

        # Scale keys by max norm
        keys_scaled: torch.Tensor = keys / max_key_norm

        # Compute augmentation terms
        key_augmentation: torch.Tensor = (
            torch.sqrt(max_key_norm**2 - key_norms**2) / max_key_norm
        )
        query_augmentation: torch.Tensor = torch.zeros_like(queries_normalized[..., :1])

        # Concatenate augmentation terms
        keys_transformed: torch.Tensor = torch.cat(
            [keys_scaled, key_augmentation], dim=-1
        )
        queries_transformed: torch.Tensor = torch.cat(
            [queries_normalized, query_augmentation], dim=-1
        )

        return keys_transformed, queries_transformed

    def _compute_probabilities(
        self, keys: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """Compute LSH collision probabilities using cosine similarity.

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).

        Returns:
            Collision probabilities tensor with shape (batch_size, num_heads, seq_len_queries, seq_len_keys).
        """
        # Transform for MIPS
        keys_transformed: torch.Tensor
        queries_transformed: torch.Tensor
        keys_transformed, queries_transformed = self._transform_for_mips(keys, queries)

        # Compute cosine similarities
        # Normalize the transformed vectors
        keys_norm: torch.Tensor = torch.norm(keys_transformed, dim=-1, keepdim=True)
        queries_norm: torch.Tensor = torch.norm(
            queries_transformed, dim=-1, keepdim=True
        )

        keys_normalized: torch.Tensor = keys_transformed / (keys_norm + 1e-8)
        queries_normalized: torch.Tensor = queries_transformed / (queries_norm + 1e-8)

        # Compute cosine similarities
        cosine_similarities: torch.Tensor = torch.matmul(
            queries_normalized, keys_normalized.transpose(-2, -1)
        )
        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)

        # Convert to angles
        angles: torch.Tensor = torch.acos(cosine_similarities)

        # Compute LSH collision probability
        # P(collision) = (1 - theta/pi)^k for single table
        # P(collision across l tables) = 1 - (1 - p)^l
        single_table_prob: torch.Tensor = (1 - angles / torch.pi) ** self.lsh_k
        collision_prob: torch.Tensor = 1 - (1 - single_table_prob) ** self.lsh_l

        return collision_prob

    def _compute_lsh_matches(
        self, keys: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        """Compute LSH matches using random signed projections.

        Args:
            keys: Key tensor with shape (batch_size, num_heads, seq_len_keys, head_dim).
            queries: Query tensor with shape (batch_size, num_heads, seq_len_queries, head_dim).

        Returns:
            Binary matches tensor with shape (batch_size, num_heads, seq_len_queries, seq_len_keys).
        """
        # Transform for MIPS
        keys_transformed: torch.Tensor
        queries_transformed: torch.Tensor
        keys_transformed, queries_transformed = self._transform_for_mips(keys, queries)

        batch_size: int = keys.shape[0]
        num_heads: int = keys.shape[1]
        seq_len_queries: int = queries.shape[2]
        seq_len_keys: int = keys.shape[2]
        head_dim: int = keys_transformed.shape[-1]

        # Generate random projection matrix
        total_bits: int = self.lsh_k * self.lsh_l
        projection: torch.Tensor = torch.randn(
            head_dim, total_bits, device=keys.device, dtype=keys.dtype
        )

        # Compute signatures
        # Reshape for batch processing
        keys_flat: torch.Tensor = keys_transformed.view(
            -1, head_dim
        )  # (batch*heads*seq_len_keys, head_dim)
        queries_flat: torch.Tensor = queries_transformed.view(
            -1, head_dim
        )  # (batch*heads*seq_len_queries, head_dim)

        # Compute signed projections
        keys_signatures: torch.Tensor = torch.sign(
            torch.matmul(keys_flat, projection)
        )  # (batch*heads*seq_len_keys, total_bits)
        queries_signatures: torch.Tensor = torch.sign(
            torch.matmul(queries_flat, projection)
        )  # (batch*heads*seq_len_queries, total_bits)

        # Reshape back to original dimensions
        keys_signatures = keys_signatures.view(
            batch_size, num_heads, seq_len_keys, total_bits
        )
        queries_signatures = queries_signatures.view(
            batch_size, num_heads, seq_len_queries, total_bits
        )

        # Compute matches for each query-key pair
        # Expand dimensions for broadcasting
        keys_signatures_expanded: torch.Tensor = keys_signatures.unsqueeze(
            2
        )  # (batch, heads, 1, seq_len_keys, total_bits)
        queries_signatures_expanded: torch.Tensor = queries_signatures.unsqueeze(
            3
        )  # (batch, heads, seq_len_queries, 1, total_bits)

        # Compute element-wise product
        signature_matches: torch.Tensor = (
            keys_signatures_expanded * queries_signatures_expanded
        )
        # Shape: (batch, heads, seq_len_queries, seq_len_keys, total_bits)

        # Reshape to group by LSH tables
        signature_matches_grouped: torch.Tensor = signature_matches.view(
            batch_size, num_heads, seq_len_queries, seq_len_keys, self.lsh_l, self.lsh_k
        )

        # Check if at least one group (table) has all bits matching
        # Sum within each group - if sum == lsh_k, all bits match
        group_matches: torch.Tensor = (
            signature_matches_grouped.sum(dim=-1) == self.lsh_k
        ).int()
        # Shape: (batch, heads, seq_len_queries, seq_len_keys, lsh_l)

        # Check if at least one table has a match
        matches: torch.Tensor = (group_matches.sum(dim=-1) > 0).int()
        # Shape: (batch, heads, seq_len_queries, seq_len_keys)

        return matches

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
        """Add Magic Pig mask to attention computation.

        This method implements the core LSH-based sampling logic. It computes
        LSH collision probabilities and matches, then combines them to create
        a sparse attention pattern that prioritizes similar content.

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
            previous_mask: Previous attention mask to merge with the new Magic Pig mask.
                If this is a full mask, it will be returned unchanged.
            **kwargs: Additional keyword arguments that may be passed to the masker.
                These are not used in the current implementation but allow for
                future extensibility.

        Returns:
            A new Mask object representing the attention pattern after applying
            LSH-based sampling. The mask is created by merging the previous mask
            with the new Magic Pig mask.

        Note:
            - If previous_mask is a full mask, it is returned unchanged
            - The LSH parameters (lsh_k, lsh_l) control the sparsity and quality
            - Similar key-query pairs have higher probability of being attended to
            - The resulting mask is merged with the previous mask using the merge_mask method
        """
        if previous_mask.is_full_mask():
            return previous_mask

        batch_size: int = queries.shape[0]
        num_heads: int = queries.shape[1]
        seq_len_queries: int = queries.shape[2]
        seq_len_keys: int = keys.shape[2]

        probabilities: torch.Tensor = self._compute_probabilities(keys, queries)
        matches: torch.Tensor = self._compute_lsh_matches(keys, queries)
        dense_mask: torch.Tensor = matches * probabilities

        mask_shape: Tuple[int, int, int, int] = (
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
        )
        this_mask: Mask = Mask.create_mask_from_dense_mask(
            shape=mask_shape, mask=dense_mask, dtype=previous_mask.dtype
        )

        return previous_mask.merge_mask(this_mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "MagicPig":
        """Create MagicPig instance from configuration.

        This factory method creates a MagicPig instance from a
        configuration object. It validates that the config is of the correct
        type before creating the masker.

        Args:
            config: Configuration object for the masker. Must be an instance
                of MagicPigConfig.

        Returns:
            A new MagicPig instance configured with the provided config.

        Raises:
            ValueError: If the config is not an instance of MagicPigConfig.
                This ensures type safety and prevents configuration mismatches.

        Example:
            >>> config = MagicPigConfig(lsh_l=5, lsh_k=4)
            >>> masker = MagicPig.create_from_config(config)
        """
        if not isinstance(config, MagicPigConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
