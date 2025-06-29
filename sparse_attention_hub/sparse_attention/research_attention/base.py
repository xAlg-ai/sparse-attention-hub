"""Base classes for research attention mechanisms."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..base import SparseAttention, SparseAttentionConfig
from .maskers.base import ResearchMasker, MaskerConfig
from .maskers.fixed import (
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
from .maskers.sampling import (
    RandomSamplingMasker, RandomSamplingMaskerConfig,
    MagicPig, MagicPigConfig
)


@dataclass
class ResearchAttentionConfig(SparseAttentionConfig):
    """Configuration class for research attention mechanisms."""
    masker_configs: List[MaskerConfig]


class ResearchAttention(SparseAttention):
    """Base class for research attention mechanisms with maskers."""

    def __init__(self, sparse_attention_config: SparseAttentionConfig, maskers: List[ResearchMasker]) -> None:
        """Initialize research attention mechanism.
        
        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
            maskers: List of research maskers to apply.
        """
        super().__init__(sparse_attention_config)
        self.maskers = maskers

    def custom_attention(self) -> Tuple[Any, Optional[Any]]:
        """Compute research attention mechanism with masking.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        # Default implementation - can be overridden by subclasses
        return None, None

    @classmethod
    def create_from_config(cls, config: ResearchAttentionConfig) -> "ResearchAttention":
        """Create research attention instance from configuration.
        
        Args:
            config: Configuration for the research attention mechanism.
            
        Returns:
            Instance of the research attention mechanism.
        """
        # Create ResearchMasker objects from the configs using the factory method
        maskers = []
        for masker_config in config.masker_configs:
            masker = ResearchMasker.create_masker_from_config(masker_config)
            maskers.append(masker)
        
        return cls(config, maskers) 