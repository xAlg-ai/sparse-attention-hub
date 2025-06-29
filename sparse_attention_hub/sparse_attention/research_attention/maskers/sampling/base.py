"""Base sampling masker implementations."""

from dataclasses import dataclass
from typing import Union

from ..base import ResearchMasker, MaskerConfig


@dataclass
class SamplingMaskerConfig(MaskerConfig):
    """Base configuration for sampling maskers."""
    sampling_rate: Union[float, int]


class SamplingMasker(ResearchMasker):
    """Abstract base class for sampling-based maskers."""

    def __init__(self, config: SamplingMaskerConfig):
        """Initialize sampling masker with configuration."""
        super().__init__(config)

    @classmethod
    def create_from_config(cls, config: SamplingMaskerConfig) -> "SamplingMasker":
        """Create sampling masker instance from configuration."""
        return cls(config)

    pass 