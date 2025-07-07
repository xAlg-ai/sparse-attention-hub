"""Base benchmark interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    def __init__(self, name: str, subsets: Optional[List[str]] = None):
        self.name = name
        self.subsets = subsets or []

    @abstractmethod
    def create_hugging_face_dataset(self) -> Any:
        """Create a HuggingFace dataset for the benchmark.

        Returns:
            HuggingFace dataset instance
        """
        # Abstract method - implementation required in subclass

    @abstractmethod
    def run_benchmark(self, model: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run the benchmark on a model.

        Args:
            model: Model to benchmark
            **kwargs: Additional benchmark parameters

        Returns:
            Benchmark results
        """
        # Abstract method - implementation required in subclass

    @abstractmethod
    def calculate_metrics(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Calculate metrics for benchmark results.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels/values

        Returns:
            Dictionary of metric names to values
        """
        # Abstract method - implementation required in subclass

    def get_subsets(self) -> List[str]:
        """Get available benchmark subsets.

        Returns:
            List of subset names
        """
        return self.subsets

    def validate_subset(self, subset: str) -> bool:
        """Validate if subset exists.

        Args:
            subset: Subset name to validate

        Returns:
            True if subset exists, False otherwise
        """
        return subset in self.subsets
