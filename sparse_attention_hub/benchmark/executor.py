"""Benchmark execution and management."""

from typing import Any, Dict, List, Optional

from .base import Benchmark
from .storage import ResultStorage


class BenchmarkExecutor:
    """Executes benchmarks and manages results."""

    def __init__(self, result_storage: Optional[ResultStorage] = None):
        self.result_storage = result_storage or ResultStorage()
        self._registered_benchmarks: Dict[str, Benchmark] = {}

    def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark for execution.

        Args:
            benchmark: Benchmark instance to register
        """
        self._registered_benchmarks[benchmark.name] = benchmark

    def evaluate(
        self,
        benchmark: Benchmark,
        model: Any,
        subset: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate a model on a benchmark.

        Args:
            benchmark: Benchmark to run
            model: Model to evaluate
            subset: Optional subset to run (if None, runs all)
            **kwargs: Additional evaluation parameters

        Returns:
            Evaluation results
        """
        # Validate subset if provided
        if subset is not None and not benchmark.validate_subset(subset):
            raise ValueError(
                f"Invalid subset '{subset}' for benchmark '{benchmark.name}'"
            )

        # Run benchmark
        results = benchmark.run_benchmark(model, subset=subset, **kwargs)

        # Store results
        result_id = self.result_storage.store(
            [
                f"Benchmark: {benchmark.name}",
                f"Subset: {subset or 'all'}",
                f"Results: {results}",
            ]
        )

        results["result_id"] = result_id
        return results

    def evaluate_multiple(
        self, benchmark_names: List[str], model: Any, **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate a model on multiple benchmarks.

        Args:
            benchmark_names: List of benchmark names to run
            model: Model to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary mapping benchmark names to results
        """
        results = {}

        for benchmark_name in benchmark_names:
            if benchmark_name not in self._registered_benchmarks:
                raise ValueError(f"Benchmark '{benchmark_name}' not registered")

            benchmark = self._registered_benchmarks[benchmark_name]
            results[benchmark_name] = self.evaluate(benchmark, model, **kwargs)

        return results

    def get_registered_benchmarks(self) -> List[str]:
        """Get list of registered benchmark names.

        Returns:
            List of benchmark names
        """
        return list(self._registered_benchmarks.keys())

    def compare_results(self, result_ids: List[str]) -> Dict[str, Any]:
        """Compare results from multiple benchmark runs.

        Args:
            result_ids: List of result IDs to compare

        Returns:
            Comparison analysis
        """
        # TODO: Implement result comparison
        raise NotImplementedError("Result comparison not yet implemented")
