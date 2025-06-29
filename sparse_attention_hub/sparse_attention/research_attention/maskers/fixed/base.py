"""Base fixed pattern masker implementations."""

from dataclasses import dataclass
from typing import Union

from ..base import ResearchMasker, MaskerConfig


@dataclass
class FixedMaskerConfig(MaskerConfig):
    """Base configuration for fixed pattern maskers."""
    pass


class FixedMasker(ResearchMasker):
    """Abstract base class for fixed pattern maskers."""

    def __init__(self, config: FixedMaskerConfig):
        """Initialize fixed masker with configuration."""
        super().__init__(config)

    @classmethod
    def create_from_config(cls, config: FixedMaskerConfig) -> "FixedMasker":
        """Create fixed masker instance from configuration."""
        return cls(config)


@dataclass
class TopKMaskerConfig(FixedMaskerConfig):
    """Base configuration for top-K maskers."""
    heavy_size: Union[float, int]


class TopKMasker(FixedMasker):
    """Abstract base class for top-K maskers."""

    def __init__(self, config: TopKMaskerConfig):
        """Initialize top-K masker with configuration."""
        super().__init__(config)

    @classmethod
    def create_from_config(cls, config: TopKMaskerConfig) -> "TopKMasker":
        """Create top-K masker instance from configuration."""
        return cls(config)


@dataclass
class TopPMaskerConfig(FixedMaskerConfig):
    """Base configuration for top-P maskers."""
    pass


class TopPMasker(FixedMasker):
    """Abstract base class for top-P maskers."""

    def __init__(self, config: TopPMaskerConfig):
        """Initialize top-P masker with configuration."""
        super().__init__(config)

    @classmethod
    def create_from_config(cls, config: TopPMaskerConfig) -> "TopPMasker":
        """Create top-P masker instance from configuration."""
        return cls(config) 