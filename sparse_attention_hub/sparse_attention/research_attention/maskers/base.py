"""Base classes for research maskers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch

from ...utils.mask import Mask


@dataclass
class MaskerConfig:
    """Base configuration class for all maskers."""


class ResearchMasker(ABC):
    """Abstract base class for research maskers."""

    # Registry mapping config types to concrete masker classes
    _MASKER_REGISTRY = None  # Will be initialized on first use

    def __init__(self, config: MaskerConfig):
        """Initialize masker with configuration."""
        self.config = config

    @abstractmethod
    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[Any, Any], # want to keep it general here.
        previous_mask: Mask,
        **kwargs: Any,
    ) -> Mask:
        """Add mask to attention computation."""
        pass

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: MaskerConfig) -> "ResearchMasker":
        """Create masker instance from configuration.

        Args:
            config: Configuration for the masker.

        Returns:
            Instance of the masker.
        """
        pass

    @classmethod
    def create_masker_from_config(cls, config: MaskerConfig) -> "ResearchMasker":
        """Create masker instance from configuration using the registry.

        Args:
            config: Configuration for the masker.

        Returns:
            Instance of the concrete masker class.

        Raises:
            ValueError: If no masker class is found for the config type.
        """

        if cls._MASKER_REGISTRY is None:
            from .fixed import (
                CausalMasker,
                DoubleSparsityTopKMasker,
                DoubleSparsityTopKMaskerConfig,
                FixedMaskerConfig,
                HashAttentionTopKMasker,
                HashAttentionTopKMaskerConfig,
                LocalMasker,
                LocalMaskerConfig,
                OracleTopK,
                OracleTopKConfig,
                PQCache,
                PQCacheConfig,
                SinkMasker,
                SinkMaskerConfig,
                TopKMasker,
                TopKMaskerConfig,
                TopPMasker,
                TopPMaskerConfig,
            )
            from .sampling import (
                MagicPig,
                MagicPigConfig,
                RandomSamplingMasker,
                RandomSamplingMaskerConfig,
            )

            cls._MASKER_REGISTRY = {
                LocalMaskerConfig: LocalMasker,
                FixedMaskerConfig: CausalMasker,  # Default for FixedMaskerConfig
                SinkMaskerConfig: SinkMasker,
                OracleTopKConfig: OracleTopK,
                TopKMaskerConfig: TopKMasker,
                TopPMaskerConfig: TopPMasker,
                HashAttentionTopKMaskerConfig: HashAttentionTopKMasker,
                DoubleSparsityTopKMaskerConfig: DoubleSparsityTopKMasker,
                PQCacheConfig: PQCache,
                RandomSamplingMaskerConfig: RandomSamplingMasker,
                MagicPigConfig: MagicPig,
            }
        masker_class = cls._MASKER_REGISTRY.get(type(config))
        if masker_class is None:
            raise ValueError(f"No masker class found for config type: {type(config)}")

        # Import here to avoid circular imports
        from typing import Type, cast

        # Cast to help mypy understand the type
        masker_class = cast(Type[ResearchMasker], masker_class)
        return masker_class.create_from_config(config)
