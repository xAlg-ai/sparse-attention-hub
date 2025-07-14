"""Magic pig sampling masker implementation."""

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
class MagicPigConfig(SamplingMaskerConfig):
    """Configuration for MagicPig masker."""

    lsh_l: int  # number of tables
    lsh_k: int  # number of bits per table


@MaskerRegistry.register(MagicPigConfig)
class MagicPig(SamplingMasker):
    """Magic Pig masker."""

    sampling_rate: float
    lsh_l: int
    lsh_k: int

    def __init__(self, config: MagicPigConfig) -> None:
        """Initialize Magic Pig masker with configuration."""
        super().__init__(config)
        self.sampling_rate = config.sampling_rate
        self.lsh_l = config.lsh_l
        self.lsh_k = config.lsh_k

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
        """Add Magic Pig mask."""
        # just return the same mask for now
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "MagicPig":
        """Create MagicPig instance from configuration."""
        if not isinstance(config, MagicPigConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
