"""Base sampling masker implementations."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch

from ....utils.mask import Mask
from ..base import MaskerConfig, ResearchMasker


@dataclass
class SamplingMaskerConfig(MaskerConfig):
    """Base configuration for sampling maskers."""

    sampling_rate: float  # Float in range [0,1] representing fraction of indices to sample

    def __post_init__(self) -> None:
        """Validate sampling_rate after initialization."""
        if not (0.0 <= self.sampling_rate <= 1.0):
            raise ValueError(f"sampling_rate must be in range [0,1], got {self.sampling_rate}")


class SamplingMasker(ResearchMasker):
    """Abstract base class for sampling-based maskers."""

    def __init__(self, config: SamplingMaskerConfig) -> None:
        """Initialize sampling masker with configuration."""
        super().__init__(config)

    @abstractmethod
    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add sampling mask to attention computation."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SamplingMasker":
        """Create sampling masker instance from configuration."""
        if not isinstance(config, SamplingMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
