"""Fixed pattern masker implementations."""
# pylint: disable=too-many-positional-arguments

from typing import Any, List

from .base import FixedMasker, ResearchMasker


class RLocalMasker(FixedMasker):
    """Local attention masker."""

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
        """Add local mask."""
        # Skeleton implementation - functionality to be added

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


class RCausalMasker(FixedMasker):
    """Causal attention masker."""

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
        """Add causal mask."""
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


class RSinkMasker(FixedMasker):
    """Sink attention masker."""

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
        """Add sink mask."""
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
