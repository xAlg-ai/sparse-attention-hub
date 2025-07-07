"""Research maskers for attention mechanisms (bare metal)."""
# pylint: disable=duplicate-code

from .base import FixedMasker, ResearchMasker, SamplingMasker, topKMasker, topPMasker
from .fixed import RCausalMasker, RLocalMasker, RSinkMasker
from .sampling import RMagicPig, RRandomSampling
from .topk import RDoubleSparsity, RHashAttention, ROracletopK, RPQCache

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
