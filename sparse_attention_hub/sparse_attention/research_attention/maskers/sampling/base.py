"""Base sampling masker implementations."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Union

from ..base import MaskerConfig, ResearchMasker


@dataclass
class SamplingMaskerConfig(MaskerConfig):
    """Base configuration for sampling maskers."""

    sampling_rate: Union[float, int]


class SamplingMasker(ResearchMasker):
    """Abstract base class for sampling-based maskers."""

    def __init__(self, config: SamplingMaskerConfig):
        """Initialize sampling masker with configuration."""
        super().__init__(config)

    @abstractmethod
    def add_mask(
        self,
        keys: Any,
        queries: Any,
        values: Any,
        attention_mask: Any,
        sparse_meta_data: Any,
        previous_mask: Any,
        **kwargs: Any,
    ) -> Any:
        """Add sampling mask to attention computation."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SamplingMasker":
        """Create sampling masker instance from configuration."""
        if not isinstance(config, SamplingMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
