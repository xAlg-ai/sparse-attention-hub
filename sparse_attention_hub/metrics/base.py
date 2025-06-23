"""Base metric interface."""

from abc import ABC, abstractmethod
from typing import Any


class MicroMetric(ABC):
    """Abstract base class for micro metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """Compute the metric value.

        Args:
            *args: Variable arguments for metric computation
            **kwargs: Keyword arguments for metric computation

        Returns:
            Computed metric value
        """
        pass

    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the metric."""
        return self.__str__()
