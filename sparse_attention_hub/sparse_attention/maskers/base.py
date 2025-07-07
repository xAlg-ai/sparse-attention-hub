"""Base classes for research maskers."""

# pylint: disable=too-many-positional-arguments

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
        # Implementation placeholder

    @abstractmethod
    def get_attention_numerator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention numerator with mask applied."""
        # Implementation placeholder

    @abstractmethod
    def get_attention_denominator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention denominator with mask applied."""
        # Implementation placeholder


class SamplingMasker(ResearchMasker):
    """Abstract base class for sampling-based maskers."""

    # Implementation placeholder


class FixedMasker(ResearchMasker):
    """Abstract base class for fixed pattern maskers."""

    # Implementation placeholder


class topKMasker(FixedMasker):  # pylint: disable=invalid-name
    """Abstract base class for top-K maskers."""

    # Implementation placeholder


class topPMasker(FixedMasker):  # pylint: disable=invalid-name
    """Abstract base class for top-P maskers."""

    # Implementation placeholder
