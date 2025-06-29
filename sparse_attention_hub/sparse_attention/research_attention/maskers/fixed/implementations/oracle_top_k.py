"""Oracle top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, List, Union

from ...base import ResearchMasker
from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class OracleTopKConfig(TopKMaskerConfig):
    """Configuration for OracleTopK masker."""
    pass


class OracleTopK(TopKMasker):
    """Oracle top-K masker."""

    def __init__(self, config: OracleTopKConfig):
        """Initialize oracle top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size

    def add_mask(
        self,
        keys: Any,
        queries: Any,
        values: Any,
        previous_attention_mask: Any,
        prev_num: Any,
        prev_den: Any,
        maskers: List[ResearchMasker],
    ) -> None:
        """Add oracle top-K mask."""
        # Bare metal implementation - no functionality
        pass

    def get_attention_numerator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention numerator."""
        # Bare metal implementation - no functionality
        pass

    def get_attention_denominator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention denominator."""
        # Bare metal implementation - no functionality
        pass

    @classmethod
    def create_from_config(cls, config: OracleTopKConfig) -> "OracleTopK":
        """Create OracleTopK instance from configuration."""
        return cls(config) 