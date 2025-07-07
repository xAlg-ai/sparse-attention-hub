"""Research maskers for attention mechanisms (bare metal)."""

# Import all masker classes from the maskers package
from .maskers.base import (
    FixedMasker,
    ResearchMasker,
    SamplingMasker,
    topKMasker,
    topPMasker,
)
from .maskers.fixed import RCausalMasker, RLocalMasker, RSinkMasker
from .maskers.sampling import RMagicPig, RRandomSampling
from .maskers.topk import RDoubleSparsity, RHashAttention, ROracletopK, RPQCache

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
