"""Sparse attention implementations and interfaces."""
# pylint: disable=duplicate-code

from .base import EfficientAttention, ResearchAttention, SparseAttention
from .efficient import DoubleSparsity, HashAttention
from .generators import SparseAttentionGen, SparseAttentionHF
from .maskers import (
    FixedMasker,
    RCausalMasker,
    RDoubleSparsity,
    ResearchMasker,
    RHashAttention,
    RLocalMasker,
    RMagicPig,
    ROracletopK,
    RPQCache,
    RRandomSampling,
    RSinkMasker,
    SamplingMasker,
    topKMasker,
    topPMasker,
)
from .metadata import SparseAttentionMetadata

__all__ = [
    "SparseAttention",
    "EfficientAttention",
    "ResearchAttention",
    "DoubleSparsity",
    "HashAttention",
    "ResearchMasker",
    "SamplingMasker",
    "FixedMasker",
    "topKMasker",
    "topPMasker",
    "RLocalMasker",
    "RCausalMasker",
    "RSinkMasker",
    "RPQCache",
    "ROracletopK",
    "RHashAttention",
    "RDoubleSparsity",
    "RRandomSampling",
    "RMagicPig",
    "SparseAttentionGen",
    "SparseAttentionHF",
    "SparseAttentionMetadata",
]
