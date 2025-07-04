"""Base classes for research maskers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

@dataclass
class MaskerConfig:
    """Base configuration class for all maskers."""
    pass


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
        keys: Any,
        queries: Any,
        values: Any,
        attention_mask: Any,
        sparse_meta_data: Any,
        previous_mask: Any,
        **kwargs
    ) -> Any:
        """Add mask to attention computation."""
        pass

    @abstractmethod
    def get_attention_numerator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention numerator with mask applied."""
        pass

    @abstractmethod
    def get_attention_denominator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention denominator with mask applied."""
        pass

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: MaskerConfig) -> "ResearchMasker":
        """Create masker instance from configuration."""
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
                LocalMasker, LocalMaskerConfig,
                CausalMasker, FixedMaskerConfig,
                SinkMasker, SinkMaskerConfig,
                OracleTopK, OracleTopKConfig,
                TopKMasker, TopKMaskerConfig,
                TopPMasker, TopPMaskerConfig,
                HashAttentionTopKMasker, HashAttentionTopKMaskerConfig,
                DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig,
                PQCache, PQCacheConfig
            )
            from .sampling import (
                RandomSamplingMasker, RandomSamplingMaskerConfig,
                MagicPig, MagicPigConfig
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
        return masker_class.create_from_config(config) 