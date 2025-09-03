"""Benchmark registry and utility functions.

This module provides the automatic benchmark registry system and utility functions
for managing benchmark discovery, instantiation, and validation.
"""

import importlib
import logging
import pkgutil
from typing import Any, Dict, List, Optional

from .base import Benchmark


# Global registry for benchmarks
_BENCHMARK_REGISTRY: Dict[str, type] = {}


def register_benchmark(name: Optional[str] = None, aliases: Optional[List[str]] = None):
    """Decorator to automatically register benchmark classes.

    This decorator allows benchmark classes to register themselves automatically,
    eliminating the need for manual registry maintenance.

    Args:
        name: Custom name for the benchmark (defaults to class.benchmark_name)
        aliases: Optional list of alternative names for the benchmark

    Example:
        >>> @register_benchmark("my_benchmark", aliases=["my_bench", "mb"])
        >>> class MyBenchmark(Benchmark):
        ...     benchmark_name = "my_benchmark"
        ...     # ... rest of implementation
    """

    def decorator(benchmark_class: type) -> type:
        # Get benchmark name from parameter, class attribute, or class name
        if name is not None:
            benchmark_name = name
        elif (
            hasattr(benchmark_class, "benchmark_name")
            and benchmark_class.benchmark_name
        ):
            benchmark_name = benchmark_class.benchmark_name
        else:
            benchmark_name = benchmark_class.__name__.lower()

        # Register main name
        _BENCHMARK_REGISTRY[benchmark_name.lower()] = benchmark_class

        # Register aliases
        if aliases:
            for alias in aliases:
                _BENCHMARK_REGISTRY[alias.lower()] = benchmark_class

        return benchmark_class

    return decorator


def get_registered_benchmarks() -> Dict[str, type]:
    """Get all registered benchmark classes.

    Returns:
        Dictionary mapping benchmark names to their classes
    """
    return _BENCHMARK_REGISTRY.copy()


def get_available_benchmark_names() -> List[str]:
    """Get list of all available benchmark names.

    Returns:
        Sorted list of registered benchmark names
    """
    return sorted(_BENCHMARK_REGISTRY.keys())


def create_benchmark_instance(
    benchmark_name: str, subsets: Optional[List[str]] = None
) -> Benchmark:
    """Factory function to create benchmark instances by name.

    Uses the automatic registry to instantiate benchmark classes.
    No manual maintenance required - benchmarks register themselves.

    Args:
        benchmark_name: Name of the benchmark to create
        subsets: Optional list of subsets to run for the benchmark

    Returns:
        Instantiated benchmark object

    Raises:
        ValueError: If benchmark_name is not registered

    Example:
        >>> benchmark = create_benchmark_instance("longbench", ["narrativeqa"])
        >>> benchmark = create_benchmark_instance("mock_benchmark")
    """
    # Normalize benchmark name
    benchmark_name = benchmark_name.strip().lower()

    if benchmark_name not in _BENCHMARK_REGISTRY:
        available_benchmarks: List[str] = get_available_benchmark_names()
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. Available benchmarks: {available_benchmarks}"
        )

    benchmark_class = _BENCHMARK_REGISTRY[benchmark_name]

    try:
        # Create instance with optional subsets
        benchmark_instance: Benchmark = benchmark_class(subsets_to_run=subsets)
        return benchmark_instance

    except Exception as e:
        raise RuntimeError(f"Failed to instantiate benchmark '{benchmark_name}': {e}")


def ensure_benchmarks_loaded() -> None:
    """Ensure all benchmark modules are loaded to trigger registration.

    This function imports all benchmark modules to ensure their decorators
    have been executed and they're registered in the global registry.
    """
    import benchmark

    # Get all modules in the benchmark package
    benchmark_package_path = benchmark.__path__

    for importer, module_name, ispkg in pkgutil.iter_modules(benchmark_package_path):
        if (
            not ispkg
            and module_name != "__init__"
            and module_name != "base"
            and module_name != "benchmark_registry"
        ):
            try:
                # Import the module to trigger @register_benchmark decorators
                module_full_name = f"benchmark.{module_name}"
                importlib.import_module(module_full_name)
            except ImportError as e:
                # Some modules might not be importable (missing dependencies, etc.)
                # This is OK - we just skip them
                logging.debug(f"Could not import benchmark module {module_name}: {e}")
                continue


def validate_benchmark_config(
    benchmark_name: str, subsets: Optional[List[str]] = None
) -> None:
    """Validate a benchmark configuration against available benchmarks.

    This function checks if the benchmark exists and if specified subsets are valid.

    Args:
        benchmark_name: Name of the benchmark to validate
        subsets: Optional list of subsets to validate

    Raises:
        ValueError: If benchmark or subsets are invalid

    Example:
        >>> validate_benchmark_config("longbench", ["narrativeqa"])  # Passes validation
        >>>
        >>> validate_benchmark_config("invalid_benchmark")  # Raises ValueError
    """
    try:
        # Create temporary benchmark instance to validate
        benchmark = create_benchmark_instance(benchmark_name, subsets=None)

        # Validate subsets if specified
        if subsets is not None:
            available_datasets: List[str] = benchmark.get_available_datasets()
            invalid_subsets: set[str] = set(subsets) - set(available_datasets)
            if invalid_subsets:
                raise ValueError(
                    f"Invalid subsets for {benchmark_name}: {invalid_subsets}. "
                    f"Available datasets: {available_datasets}"
                )

    except Exception as e:
        raise ValueError(f"Invalid benchmark configuration: {e}")


def get_benchmark_subsets(benchmark_name: str) -> Optional[List[str]]:
    """Get available subsets for a benchmark using the standard interface.

    This is a utility function that can help users determine what subsets are
    available for a given benchmark using the standard get_available_datasets() method.

    Args:
        benchmark_name: Name of the benchmark to inspect

    Returns:
        List of available subset names, or None if benchmark doesn't support subsets

    Example:
        >>> subsets = get_benchmark_subsets("longbench")
        >>> print(subsets)  # ['narrativeqa', 'qasper', 'multifieldqa_en', ...]
    """
    try:
        benchmark_instance = create_benchmark_instance(benchmark_name)
        return benchmark_instance.get_available_datasets()

    except Exception as e:
        # If we can't instantiate the benchmark or get subset info, return None
        logging.warning(f"Could not get subsets for benchmark '{benchmark_name}': {e}")
        return None
