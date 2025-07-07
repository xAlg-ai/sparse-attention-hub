"""Base fixed pattern masker implementations."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Union

from ..base import MaskerConfig, ResearchMasker


@dataclass
class FixedMaskerConfig(MaskerConfig):
    """Base configuration for fixed pattern maskers."""

    pass


class FixedMasker(ResearchMasker):
    """Abstract base class for fixed pattern maskers."""

    def __init__(self, config: FixedMaskerConfig):
        """Initialize fixed masker with configuration."""
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
        """Add fixed mask to attention computation."""
        pass

    @abstractmethod
    def get_attention_numerator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention numerator with fixed mask applied."""
        pass

    @abstractmethod
    def get_attention_denominator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention denominator with fixed mask applied."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "FixedMasker":
        """Create fixed masker instance from configuration."""
        if not isinstance(config, FixedMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
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
        """Add top-K mask to attention computation."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "TopKMasker":
        """Create top-K masker instance from configuration."""
        if not isinstance(config, TopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
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
        """Add top-P mask to attention computation."""
        pass

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "TopPMasker":
        """Create top-P masker instance from configuration."""
        if not isinstance(config, TopPMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
