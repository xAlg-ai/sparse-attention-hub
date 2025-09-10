"""OpenEvolve masker implementation.

This module provides the OpenEvolve masker that implements evolutionary attention patterns.
The current implementation is a bare metal version that returns the previous mask.
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch
from scipy.stats import norm

from sparse_attention_hub.sparse_attention.research_attention.maskers import (
    ResearchMasker,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    _get_num_key_value_groups,
    apply_inv_mask_sum,
    create_sampling_mask_with_per_head_budget,
    repeat_kv,
)


@dataclass
class OpenEvolveMaskerConfig(MaskerConfig):
    """Configuration for OpenEvolveMasker.

    This configuration class inherits from MaskerConfig and provides
    parameters for the attention mechanism evolved out of openevolve.
    empty placeholder
    """

    pass


@MaskerRegistry.register(OpenEvolveMaskerConfig)
class OpenEvolveMasker(ResearchMasker):
    """OpenEvolve masker for evolutionary attention computation.

    This masker implements evolutionary attention patterns that adapt over time.
    The current implementation is a bare metal version that returns the previous mask.

    Attributes:
        evolution_rate: The rate of evolution for attention patterns.
            This value is set from the configuration and controls how quickly
            the attention patterns evolve.

    Important Notes:
        - This is a bare metal implementation that simply returns the previous mask.
        - Future implementations will include evolutionary algorithms for attention pattern optimization.
        - The evolution_rate parameter is currently unused but will be utilized in future versions.

    Example:
        >>> config = OpenEvolveMaskerConfig(evolution_rate=1.0)
        >>> masker = OpenEvolveMasker(config)
        >>> # Use masker.add_mask() to apply evolutionary attention patterns
    """

    def __init__(self, config: OpenEvolveMaskerConfig) -> None:
        """Initialize OpenEvolve masker with configuration.

        Args:
            config: Configuration object containing the evolution rate and other
                parameters for the OpenEvolve masker.

        Raises:
            ValueError: If the evolution_rate in config is negative.
                This validation is performed in the config's __post_init__ method.
        """
        self.base_rate_sampling = 0.01
        self.epsilon = 0.3
        self.delta = 0.3
        self.init_offset = 0.001
        self.local_offset = 0.001

        super().__init__(config)

    def _compute_exp_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        scaling: float,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exponential attention scores with numerical stability."""
        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)
        raw_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            raw_scores = raw_scores + attention_mask[:, :, :, : keys.shape[-2]]
        max_scores = torch.max(raw_scores, dim=-1, keepdim=True)[0]
        return torch.exp(raw_scores - max_scores)

    def _get_sampling_range(self, seq_len_keys: int) -> tuple[int, int, int]:
        """Get sampling range and validate it.

        Args:
            seq_len_keys: Number of keys in the sequence.

        Returns:
            Tuple of (start_idx, end_idx, sampling_range).

        Raises:
            ValueError: If the computed sampling range is invalid.
        """
        # Compute start index
        if isinstance(self.init_offset, float):
            start_idx: int = int(self.init_offset * seq_len_keys)
        else:
            start_idx = self.init_offset

        # Compute end index
        if isinstance(self.local_offset, float):
            end_idx: int = seq_len_keys - int(self.local_offset * seq_len_keys)
        else:
            end_idx = seq_len_keys - self.local_offset

        sampling_range = end_idx - start_idx

        if sampling_range <= 0:
            raise ValueError(f"Invalid sampling range: {sampling_range}")

        return start_idx, end_idx, sampling_range

    def _get_base_sample_count(self, sampling_range: int) -> int:
        """Get number of base samples based on configuration."""
        # Ensure at least 2 samples since it is used for std estimation
        if isinstance(self.base_rate_sampling, int):
            return max(2, self.base_rate_sampling)
        return max(2, int(self.base_rate_sampling * sampling_range))

    def _get_std_estimate_using_base_sample(
        self,
        expwts: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len_queries: int,
        seq_len_keys: int,
        start_idx: int,
        end_idx: int,
        num_base_samples: int,
        dtype: torch.dtype,
    ) -> tuple[Mask, torch.Tensor]:
        """Get standard deviation estimate using base sampling and create base mask."""
        # Create base sampling indices
        base_row_wise_idx = torch.randint(
            low=start_idx,
            high=end_idx,
            size=(batch_size, num_heads, seq_len_queries, num_base_samples),
            device=expwts.device,
        )

        # Extract values and compute std
        sampled_values = torch.gather(expwts, dim=-1, index=base_row_wise_idx)
        total_rows = batch_size * num_heads * seq_len_queries
        row_sampled_values = sampled_values.view(total_rows, num_base_samples)
        std_estimate = torch.std(row_sampled_values, dim=-1, keepdim=True)
        std_estimate = torch.clamp(std_estimate, min=1e-8)
        std_estimate = std_estimate.view(batch_size, num_heads, seq_len_queries, 1)

        # Create base sampling mask
        sampling_range = end_idx - start_idx
        base_data = torch.full_like(
            base_row_wise_idx, num_base_samples / sampling_range, dtype=expwts.dtype
        )

        base_mask = Mask.create_from_row_wise_idx(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            row_wise_idx=base_row_wise_idx,
            data=base_data,
            type="index",
            dtype=dtype,
        )

        return base_mask, std_estimate

    def _compute_adaptive_budget(
        self,
        std_estimate: torch.Tensor,
        estimated_denominator: torch.Tensor,
        sampling_range: int,
    ) -> torch.Tensor:
        """Compute adaptive budget based on statistical bounds."""
        epsilon_allowable_error = self.epsilon * estimated_denominator
        epsilon_allowable_error = torch.clamp(epsilon_allowable_error, min=1e-8)

        budget_numerator = self.delta_ppf * std_estimate * sampling_range
        budget_squared = (budget_numerator / epsilon_allowable_error) ** 2

        # Ensure budget is positive and within bounds
        budget = torch.clamp(
            budget_squared,
            min=1.0,  # Minimum 1 sample
            max=float(sampling_range),  # Maximum sampling_range samples
        ).long()

        return budget

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
        """
        Add an adaptive sparse attention mask by aggressively shrinking budget while 
        retaining statistical guarantees, simplifying the code for maintainability.
        """

        ngroups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, ngroups)

        seq_len_keys = keys.shape[-2]
        # Adaptive parameters based on sequence length (keeps error low on long sequences)
        if seq_len_keys > 10000:
            self.base_rate_sampling = 0.02  # Higher sampling for very long sequences
            self.epsilon = 0.2  # Tighter bounds for long sequences
            self.delta = 0.2
        elif seq_len_keys > 5000:
            self.base_rate_sampling = 0.018  # Medium sampling for long sequences
            self.epsilon = 0.22  # Medium bounds
            self.delta = 0.22
        else:
            self.base_rate_sampling = 0.015  # Default for shorter sequences
            self.epsilon = 0.25
            self.delta = 0.25

        # Keep a very small global/local offset to avoid extreme positions
        self.init_offset = 0.002
        self.local_offset = 0.002

        # Precompute delta PPF once
        self.delta_ppf = float(norm.ppf(1 - self.delta))

        # If fully dense mask already, no further sparsification is required
        if previous_mask.is_full_mask():
            return previous_mask

        # Get tensor dimensions
        dims = self._extract_tensor_dimensions(keys, queries)
        batch_size, num_heads, seq_len_queries, seq_len_keys = (
            dims.batch_size, dims.num_heads, dims.seq_len_queries, dims.seq_len_keys
        )

        # Compute stable exp attention weights (after masking)
        expwts = self._compute_exp_attention_scores(
            queries, keys, scaling, attention_mask
        )
        static_denominator = apply_inv_mask_sum(expwts, previous_mask)

        # --- Sampling budget setup ---
        start_idx, end_idx, sampling_range = self._get_sampling_range(seq_len_keys)
        num_base_samples = self._get_base_sample_count(sampling_range)
        # Base sampling + std estimation
        base_sampling_mask, std_estimate = self._get_std_estimate_using_base_sample(
            expwts,
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
            start_idx,
            end_idx,
            num_base_samples,
            previous_mask.dtype,
        )
        sampled_denominator = apply_inv_mask_sum(expwts, base_sampling_mask)
        estimated_denominator = static_denominator + sampled_denominator

        # Statistical error-controlled budget (pre-sparsification)
        budget = self._compute_adaptive_budget(
            std_estimate, estimated_denominator, sampling_range
        )
        # ------------------------------------------------------------------ #
        # Aggressive sparsification with empirically-validated heuristics
        # ------------------------------------------------------------------ #
        # (1) Shrink budget with a strong scaling factor
        budget = torch.ceil(budget.float() * 0.54).long()

        # (2) Maintain a robust minimum number of samples
        min_base = max(num_base_samples, 4)
        budget = torch.clamp(budget, min=min_base)

        # (3) Hard upper cap: no more than 8% of keys
        max_budget = max(min_base, int(0.08 * sampling_range))
        budget = torch.clamp(budget, max=max_budget)

        # (4) Keep at least 2 samples for very short sequences
        budget = torch.clamp(budget, min=2)

        # (5) Extra sparsity for queries that already have a very large static mass
        if torch.median(static_denominator) > 5.0:
            budget = torch.clamp(budget, max=max(min_base, int(0.05 * sampling_range)))

        # ------------------------------------------------------------------ #
        # ❶  Deterministic TOP-MASS: keep the minimum k tokens that cover ≥97 %
        # ------------------------------------------------------------------ #
        sorted_expwts, sorted_idx = torch.sort(expwts, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_expwts, dim=-1)
        total_mass = cumsum[..., -1:]
        k_97 = (cumsum / (total_mass + 1e-8) < 0.97).sum(dim=-1, keepdim=True) + 1
        k_97 = torch.clamp(k_97, min=1, max=min(12, seq_len_keys // 8 + 1))

        max_topk = int(k_97.max().item())
        arange_topk = torch.arange(max_topk, device=expwts.device)[None, None, None, :]
        top_mask = arange_topk < k_97                       # shape (b,h,q,max_topk)
        top_idx = sorted_idx[..., :max_topk]                # same shape as top_mask

        top_data = torch.where(
            top_mask,
            torch.ones_like(top_idx, dtype=previous_mask.dtype),
            torch.zeros_like(top_idx, dtype=previous_mask.dtype),
        )

        topmass_mask = Mask.create_from_row_wise_idx(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            row_wise_idx=top_idx,
            data=top_data,
            type="index",
            dtype=previous_mask.dtype,
        )

        # ------------------------------------------------------------------ #
        # ❷  MUCH SMALLER STATISTICAL TAIL after accounting for top-mass
        # ------------------------------------------------------------------ #
        relative_mass = (cumsum[..., -1:] / (static_denominator + 1e-8)).clamp(max=1.0)
        tail_scale = torch.where(relative_mass > 0.97, 0.25, 0.45)  # aggressive shrink
        budget = torch.ceil(budget.float() * tail_scale).long()
        budget = torch.clamp(budget, min=min_base)

        # If almost every key already selected deterministically, skip tail
        budget = torch.where(k_97 >= seq_len_keys - 2, torch.zeros_like(budget), budget)

        # ------------------------------------------------------------------ #
        # ❸  Build probabilistic TAIL mask and merge everything
        # ------------------------------------------------------------------ #
        sampling_probabilities = (budget / sampling_range).to(previous_mask.dtype)

        adaptive_mask = create_sampling_mask_with_per_head_budget(
            budgets=budget,
            sampling_probability=sampling_probabilities,
            seq_len_keys=seq_len_keys,
            start_idx=start_idx,
            end_idx=end_idx,
            dtype=previous_mask.dtype,
        )

        # Merge: previous  ∪  deterministic-top  ∪  statistical-tail
        merged_mask = (
            previous_mask.merge_mask(topmass_mask, inplace=False)
            .merge_mask(adaptive_mask, inplace=False)
        )
        return merged_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "OpenEvolveMasker":
        """Create OpenEvolve masker instance from configuration.

        Args:
            config: Configuration for the OpenEvolve masker.

        Returns:
            Instance of the OpenEvolve masker.

        Raises:
            ValueError: If the config type is invalid.
        """
        # not checking for config type here since we will be replacing this masker class
        # with the new masker class in the evaluator.py file
        return cls(config)
