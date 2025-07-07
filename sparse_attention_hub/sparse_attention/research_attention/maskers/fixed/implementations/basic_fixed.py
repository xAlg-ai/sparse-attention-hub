"""Basic fixed pattern masker implementations."""

from dataclasses import dataclass
from typing import Any, List, Union

from ...base import ResearchMasker
from ..base import FixedMasker, FixedMaskerConfig


@dataclass
class LocalMaskerConfig(FixedMaskerConfig):
    """Configuration for LocalMasker."""

    window_size: Union[float, int]


class LocalMasker(FixedMasker):
    """Local attention masker."""

    def __init__(self, config: LocalMaskerConfig):
        """Initialize local masker with configuration."""
        super().__init__(config)
        self.window_size = config.window_size

    def add_mask(
        self,
        keys: Any,
        queries: Any,
        values: Any,
        attention_mask: Any,
        sparse_meta_data: Any,
        previous_mask: Any,
        **kwargs,
    ) -> Any:
        """Add local mask."""
        # just return the same mask for now
        return previous_mask

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
    def create_from_config(cls, config: LocalMaskerConfig) -> "LocalMasker":
        """Create LocalMasker instance from configuration."""
        return cls(config)


class CausalMasker(FixedMasker):
    """Causal attention masker."""

    def __init__(self, config: FixedMaskerConfig):
        """Initialize causal masker with configuration."""
        super().__init__(config)

    def add_mask(
        self,
        keys: Any,
        queries: Any,
        values: Any,
        attention_mask: Any,
        sparse_meta_data: Any,
        previous_mask: Any,
        **kwargs,
    ) -> Any:
        """Add causal mask."""
        # just return the same mask for now
        return previous_mask

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
    def create_from_config(cls, config: FixedMaskerConfig) -> "CausalMasker":
        """Create CausalMasker instance from configuration."""
        return cls(config)


@dataclass
class SinkMaskerConfig(FixedMaskerConfig):
    """Configuration for SinkMasker."""

    sink_size: Union[float, int]


class SinkMasker(FixedMasker):
    """Sink attention masker."""

    def __init__(self, config: SinkMaskerConfig):
        """Initialize sink masker with configuration."""
        super().__init__(config)
        self.sink_size = config.sink_size

    def add_mask(
        self,
        keys: Any,
        queries: Any,
        values: Any,
        attention_mask: Any,
        sparse_meta_data: Any,
        previous_mask: Any,
        **kwargs,
    ) -> Any:
        """Add sink mask."""
        # just return the same mask for now
        return previous_mask

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
    def create_from_config(cls, config: SinkMaskerConfig) -> "SinkMasker":
        """Create SinkMasker instance from configuration."""
        return cls(config)
