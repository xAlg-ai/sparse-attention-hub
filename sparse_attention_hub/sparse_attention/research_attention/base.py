"""Base classes for research attention mechanisms."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

from ..base import SparseAttention, SparseAttentionConfig
from ..utils.mask import Mask
from ..utils.mask_attention_utils import (
    get_masked_attention_output,
    get_true_attention_output,
)
from .maskers.base import MaskerConfig, ResearchMasker
from .maskers.sampling.base import SamplingMasker

MicroMetricLogger.register_metric("research_attention_density", float)
MicroMetricLogger.register_metric("research_attention_output_error", float)


@dataclass
class ResearchAttentionConfig(SparseAttentionConfig):
    """Configuration class for research attention mechanisms.

    This configuration specifies the masker components that will be applied
    sequentially to create sparse attention patterns for research purposes.

    Attributes:
        masker_configs: List of masker configurations to apply in sequence.
    """

    masker_configs: List[MaskerConfig]


class ResearchAttention(SparseAttention):
    """Base class for research attention mechanisms with configurable maskers.

    This class implements sparse attention by applying a sequence of maskers
    to create custom attention patterns. It supports metrics logging and
    validation of masker configurations.

    Attributes:
        maskers: List of research maskers to apply sequentially.
    """

    maskers: List[ResearchMasker]

    def __init__(
        self,
        sparse_attention_config: SparseAttentionConfig,
        maskers: List[ResearchMasker],
    ) -> None:
        """Initialize research attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
            maskers: List of research maskers to apply in sequence.

        Raises:
            ValueError: If more than one sampling masker is provided.
        """
        super().__init__(sparse_attention_config)
        self._validate_masker_configuration(maskers)
        self.maskers = maskers

    def _validate_masker_configuration(self, maskers: List[ResearchMasker]) -> None:
        """Validate the masker configuration.

        Args:
            maskers: List of maskers to validate.

        Raises:
            ValueError: If more than one sampling masker is provided.
        """
        sampling_masker_count = sum(
            1 for masker in maskers if isinstance(masker, SamplingMasker)
        )
        if sampling_masker_count > 1:
            raise ValueError(
                "Only one sampling masker supported for efficiency; "
                "consider implementing all sampling logic in one masker"
            )

    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute research attention mechanism with masking.

        Args:
            module: The attention module.
            queries: Query tensor of shape (b, h, sk, d).
            keys: Key tensor of shape (b, h, sq, d).
            values: Value tensor of shape (b, h, sq, d).
            attention_mask: Optional attention mask of shape (b, h, sq, sk).
            scaling: Scaling factor for attention weights.
            dropout: Dropout probability.
            sparse_meta_data: Additional metadata for sparse attention computation.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        sparse_attention_mask = self._create_initial_mask(queries, keys)
        sparse_attention_mask = self._apply_maskers(
            sparse_attention_mask=sparse_attention_mask,
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
            **kwargs,
        )

        self._log_attention_density(sparse_attention_mask, kwargs)

        attention_output, attention_weights = self._compute_masked_attention(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            **kwargs,
        )

        self._log_attention_error(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            attention_output=attention_output,
            **kwargs,
        )

        return attention_output, attention_weights

    def _create_initial_mask(self, queries: torch.Tensor, keys: torch.Tensor) -> Mask:
        """Create an initial empty mask for the attention computation.

        Args:
            queries: Query tensor.
            keys: Key tensor.

        Returns:
            Empty mask with appropriate shape.
        """
        mask_shape = (
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            keys.shape[2],
        )
        return Mask.create_empty_mask(mask_shape, dtype=queries.dtype)

    def _apply_maskers(
        self,
        sparse_attention_mask: Mask,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Apply all maskers sequentially to create the final sparse attention mask.

        Args:
            sparse_attention_mask: Initial mask to start with.
            keys: Key tensor.
            queries: Query tensor.
            values: Value tensor.
            attention_mask: Optional attention mask.
            scaling: Scaling factor.
            dropout: Dropout probability.
            sparse_meta_data: Additional metadata.
            **kwargs: Additional keyword arguments.

        Returns:
            Final sparse attention mask after applying all maskers.
        """
        for masker in self.maskers:
            sparse_attention_mask = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                previous_mask=sparse_attention_mask,
                **kwargs,
            )
        return sparse_attention_mask

    def _log_attention_density(
        self, sparse_attention_mask: Mask, kwargs: Dict[str, Any]
    ) -> None:
        """Log attention density metric if enabled.

        Args:
            sparse_attention_mask: The sparse attention mask.
            kwargs: Keyword arguments containing layer information.
        """
        if MicroMetricLogger().is_metric_enabled("research_attention_density"):
            MicroMetricLogger().log(
                "research_attention_density",
                sparse_attention_mask.get_density(),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

    def _compute_masked_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_attention_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the masked attention output and weights.

        Args:
            module: The attention module.
            queries: Query tensor.
            keys: Key tensor.
            values: Value tensor.
            attention_mask: Optional attention mask.
            scaling: Scaling factor.
            dropout: Dropout probability.
            sparse_attention_mask: Sparse attention mask.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of attention output and attention weights.
        """
        return get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
            **kwargs,
        )

    def _log_attention_error(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        attention_output: torch.Tensor,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Log attention output error metric if enabled.

        Args:
            module: The attention module.
            queries: Query tensor.
            keys: Key tensor.
            values: Value tensor.
            attention_mask: Optional attention mask.
            scaling: Scaling factor.
            dropout: Dropout probability.
            attention_output: Computed attention output.
            **kwargs: Additional keyword arguments.
        """
        if MicroMetricLogger().is_metric_enabled("research_attention_output_error"):
            true_attention_output, _ = get_true_attention_output(
                module,
                queries,
                keys,
                values,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )
            error = torch.norm(true_attention_output - attention_output) / torch.norm(
                true_attention_output
            )
            MicroMetricLogger().log(
                "research_attention_output_error",
                float(error.item()),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "ResearchAttention":
        """Create research attention instance from configuration.

        Args:
            config: Configuration for the research attention mechanism.
                Must be an instance of ResearchAttentionConfig.

        Returns:
            Instance of the research attention mechanism with configured maskers.

        Raises:
            TypeError: If config is not a ResearchAttentionConfig.
        """
        if not isinstance(config, ResearchAttentionConfig):
            raise TypeError(f"Expected ResearchAttentionConfig, got {type(config)}")

        maskers = cls._create_maskers_from_config(config.masker_configs)
        return cls(config, maskers)

    @classmethod
    def _create_maskers_from_config(
        cls, masker_configs: List[MaskerConfig]
    ) -> List[ResearchMasker]:
        """Create research masker objects from their configurations.

        Args:
            masker_configs: List of masker configurations.

        Returns:
            List of configured research masker instances.
        """
        maskers = []
        for masker_config in masker_configs:
            masker = ResearchMasker.create_masker_from_config(masker_config)
            maskers.append(masker)
        return maskers
