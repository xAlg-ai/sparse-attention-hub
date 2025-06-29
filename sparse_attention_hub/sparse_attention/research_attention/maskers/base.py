"""Base classes for research maskers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class MaskerConfig:
    """Base configuration class for all maskers."""
    pass


class ResearchMasker(ABC):
    """Abstract base class for research maskers."""

    def __init__(self, config: MaskerConfig):
        """Initialize masker with configuration."""
        self.config = config

    @abstractmethod
    def add_mask(
        self,
        keys: Any,
        queries: Any,
        values: Any,
        previous_attention_mask: Any,
        prev_num: Any,
        prev_den: Any,
        maskers: List["ResearchMasker"],
    ) -> None:
        """Add mask to attention computation."""
        pass

    @abstractmethod
    def get_attention_numerator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention numerator with mask applied."""
        pass

    @abstractmethod
    def get_attention_denominator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention denominator with mask applied."""
        pass

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: MaskerConfig) -> "ResearchMasker":
        """Create masker instance from configuration."""
        pass 