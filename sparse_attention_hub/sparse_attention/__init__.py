"""Sparse attention implementations and interfaces."""

from .base import SparseAttention
from .efficient_attention import EfficientAttention
from .efficient_attention.implementations import DoubleSparsity, HashAttention
from .generator import SparseAttentionGen
from .integrations import SparseAttentionHF
from .metadata import SparseAttentionMetadata
from .research_attention import ResearchAttention
from .research_attention.maskers import (
    FixedMasker,
    ResearchMasker,
    SamplingMasker,
    TopKMasker,
    TopPMasker,
)
from .research_attention.maskers.fixed import CausalMasker, LocalMasker, SinkMasker
from .research_attention.maskers.fixed.implementations import (
    DoubleSparsity as RDoubleSparsity,
    HashAttention as RHashAttention,
    OracleTopK,
    PQCache,
)
from .research_attention.maskers.sampling import RandomSamplingMasker
from .research_attention.maskers.sampling.implementations import MagicPig
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
