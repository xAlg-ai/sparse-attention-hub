"""Sparse attention implementations and interfaces."""

from .base import SparseAttention, SparseAttentionConfig
from .efficient_attention import EfficientAttention, EfficientAttentionConfig
from .efficient_attention.implementations import (
    DoubleSparsity, 
    HashAttention,
    DoubleSparsityConfig,
    HashAttentionConfig,
    ChannelConfig
)
from .generator import SparseAttentionGen
from .integrations import SparseAttentionHF
from .metadata import SparseAttentionMetadata
from .research_attention import ResearchAttention, ResearchAttentionConfig
from .research_attention.maskers import ResearchMasker
from .research_attention.maskers.fixed import (
    CausalMasker,
    DoubleSparsityTopKMasker,
    FixedMasker,
    HashAttentionTopKMasker,
    LocalMasker,
    OracleTopK,
    PQCache,
    SinkMasker,
    TopKMasker,
    TopPMasker,
    DoubleSparsityTopKMaskerConfig,
    HashAttentionTopKMaskerConfig,
)
from .research_attention.maskers.sampling import MagicPig, RandomSamplingMasker, SamplingMasker
from .utils import Mask

__all__ = [
    "SparseAttention",
    "SparseAttentionConfig",
    "EfficientAttention",
    "EfficientAttentionConfig",
    "ResearchAttention",
    "ResearchAttentionConfig",
    "DoubleSparsity",
    "HashAttention",
    "DoubleSparsityConfig",
    "HashAttentionConfig",
    "ChannelConfig",
    "DoubleSparsityTopKMaskerConfig",
    "HashAttentionTopKMaskerConfig",
    "ResearchMasker",
    "SamplingMasker",
    "FixedMasker",
    "TopKMasker",
    "TopPMasker",
    "LocalMasker",
    "CausalMasker",
    "SinkMasker",
    "PQCache",
    "OracleTopK",
    "HashAttentionTopKMasker",
    "DoubleSparsityTopKMasker",
    "RandomSamplingMasker",
    "MagicPig",
    "SparseAttentionGen",
    "SparseAttentionHF",
    "SparseAttentionMetadata",
    "Mask",
]
