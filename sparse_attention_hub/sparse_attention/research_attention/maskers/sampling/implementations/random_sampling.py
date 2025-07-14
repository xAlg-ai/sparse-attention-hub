"""Random sampling masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import SamplingMasker, SamplingMaskerConfig


@dataclass
class RandomSamplingMaskerConfig(SamplingMaskerConfig):
    """Configuration for RandomSamplingMasker."""

    pass


@MaskerRegistry.register(RandomSamplingMaskerConfig)
class RandomSamplingMasker(SamplingMasker):
    """Random sampling masker."""

    def __init__(self, config: RandomSamplingMaskerConfig) -> None:
        """Initialize random sampling masker with configuration."""
        super().__init__(config)
        self.sampling_rate = config.sampling_rate

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
        """Add random sampling mask."""
        # just return the same mask for now
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "RandomSamplingMasker":
        """Create RandomSamplingMasker instance from configuration."""
        if not isinstance(config, RandomSamplingMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
