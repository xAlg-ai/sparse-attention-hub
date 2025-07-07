"""Sampling-based masker implementations."""

# pylint: disable=too-many-positional-arguments

from typing import Any, List

from .base import ResearchMasker, SamplingMasker


class RRandomSampling(SamplingMasker):
    """Random sampling masker."""

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
        """Add random sampling mask."""
        # Skeleton implementation - functionality to be added
        # Implementation placeholder

    def get_attention_numerator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention numerator."""
        # Skeleton implementation - functionality to be added
        # Implementation placeholder

    def get_attention_denominator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention denominator."""
        # Skeleton implementation - functionality to be added
        # Implementation placeholder


class RMagicPig(SamplingMasker):
    """Magic Pig masker."""

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
        """Add Magic Pig mask."""
        # Skeleton implementation - functionality to be added
        # Implementation placeholder

    def get_attention_numerator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention numerator."""
        # Skeleton implementation - functionality to be added
        # Implementation placeholder

    def get_attention_denominator(
        self, keys: Any, queries: Any, values: Any, mask: Any
    ) -> Any:
        """Get attention denominator."""
        # Skeleton implementation - functionality to be added
        # Implementation placeholder
