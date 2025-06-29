"""Sampling masker implementations."""

from .magic_pig import MagicPig, MagicPigConfig
from .random_sampling import RandomSamplingMasker, RandomSamplingMaskerConfig

__all__ = [
    "MagicPig", 
    "RandomSamplingMasker",
    "MagicPigConfig",
    "RandomSamplingMaskerConfig"
] 