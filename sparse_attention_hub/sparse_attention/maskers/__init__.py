"""Research maskers for attention mechanisms (bare metal)."""

from .base import FixedMasker, ResearchMasker, SamplingMasker, topKMasker, topPMasker
from .fixed import RCausalMasker, RLocalMasker, RSinkMasker
from .mask import Mask
from .sampling import RMagicPig, RRandomSampling
from .topk import RDoubleSparsity, RHashAttention, ROracletopK, RPQCache

__all__ = [
    # Base classes
    "ResearchMasker",
    "SamplingMasker",
    "FixedMasker",
    "topKMasker",
    "topPMasker",
    # Mask class
    "Mask",
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
