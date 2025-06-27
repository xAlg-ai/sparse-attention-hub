"""Research maskers for attention mechanisms (bare metal)."""

from .base import (
    ResearchMasker,
    SamplingMasker,
    FixedMasker,
    topKMasker,
    topPMasker,
)
from .fixed import RLocalMasker, RCausalMasker, RSinkMasker
from .topk import RPQCache, ROracletopK, RHashAttention, RDoubleSparsity
from .sampling import RRandomSampling, RMagicPig

__all__ = [
    # Base classes
    "ResearchMasker",
    "SamplingMasker", 
    "FixedMasker",
    "topKMasker",
    "topPMasker",
    # Fixed maskers
    "RLocalMasker",
    "RCausalMasker", 
    "RSinkMasker",
    # Top-K maskers
    "RPQCache",
    "ROracletopK",
    "RHashAttention",
    "RDoubleSparsity",
    # Sampling maskers
    "RRandomSampling",
    "RMagicPig",
] 