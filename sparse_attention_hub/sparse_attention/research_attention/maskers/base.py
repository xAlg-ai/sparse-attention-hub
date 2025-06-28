"""Base classes for research maskers."""

from abc import ABC, abstractmethod
from typing import Any, List


class ResearchMasker(ABC):
    """Abstract base class for research maskers."""

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


class SamplingMasker(ResearchMasker):
    """Abstract base class for sampling-based maskers."""

    pass


class FixedMasker(ResearchMasker):
    """Abstract base class for fixed pattern maskers."""

    pass


class TopKMasker(FixedMasker):
    """Abstract base class for top-K maskers."""

    pass


class TopPMasker(FixedMasker):
    """Abstract base class for top-P maskers."""

    pass 