"""Sampling-based maskers."""

from .base import SamplingMasker, SamplingMaskerConfig
from .implementations import (
    RandomSamplingMasker, 
    MagicPig,
    RandomSamplingMaskerConfig,
    MagicPigConfig
)

__all__ = [
    "SamplingMasker", 
    "RandomSamplingMasker", 
    "MagicPig",
    "SamplingMaskerConfig",
    "RandomSamplingMaskerConfig",
    "MagicPigConfig"
] 