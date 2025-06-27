"""Research maskers for attention mechanisms (bare metal)."""

# Import all masker classes from the new package structure
from .maskers import (  # Base classes; Fixed maskers; Top-K maskers; Sampling maskers
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
