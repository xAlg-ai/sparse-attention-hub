"""Granularity enumeration for plotting."""

from enum import Enum


class Granularity(Enum):
    """Enumeration for different granularity levels in plotting."""

    PER_TOKEN = "per_token"
    PER_HEAD = "per_head"
    PER_LAYER = "per_layer"

    def __str__(self) -> str:
        """String representation of granularity."""
        return self.value

    def __repr__(self) -> str:
        """Detailed string representation of granularity."""
        return f"Granularity.{self.name}"
