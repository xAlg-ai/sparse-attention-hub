"""Sparse attention implementations and interfaces."""

from .base import SparseAttention
from .efficient_attention import EfficientAttention
from .efficient_attention.implementations import DoubleSparsity, HashAttention
from .generator import SparseAttentionGen
from .integrations import SparseAttentionHF
from .metadata import SparseAttentionMetadata
from .research_attention import ResearchAttention
from .research_attention.maskers import ResearchMasker
from .research_attention.maskers.fixed import (
    CausalMasker,
    RDoubleSparsity,
    FixedMasker,
    RHashAttention,
    LocalMasker,
    OracleTopK,
    PQCache,
    SinkMasker,
    TopKMasker,
    TopPMasker,
)
from .research_attention.maskers.sampling import MagicPig, RandomSamplingMasker, SamplingMasker
from .utils import Mask

__all__ = [
    "SparseAttention",
    "EfficientAttention",
    "ResearchAttention",
    "DoubleSparsity",
    "HashAttention",
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
    "RHashAttention",
    "RDoubleSparsity",
    "RandomSamplingMasker",
    "MagicPig",
    "SparseAttentionGen",
    "SparseAttentionHF",
    "SparseAttentionMetadata",
    "Mask",
]
