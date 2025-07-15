"""Configuration classes and utilities for BenchmarkExecutor.

This module provides configuration dataclasses and factory functions for orchestrating
parallel benchmark execution across multiple GPUs using multiprocessing.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import multiprocessing

from sparse_attention_hub.sparse_attention.base import SparseAttentionConfig

from .base import Benchmark


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark execution.
    
    During stub generation, this expands to individual BenchmarkStub per subset.
    Each BenchmarkStub will have a single subset value for resumability.
    
    Attributes:
        benchmark_name: Name of the benchmark (e.g., "longbench", "mock_benchmark")
        subsets: Optional list of dataset subsets to run (e.g., ["narrativeqa", "qasper"])
                If None, runs all available subsets for the benchmark
    
    Example:
        >>> config = BenchmarkConfig("longbench", ["narrativeqa", "qasper"])
        >>> # This will create 2 BenchmarkStub instances during matrix expansion
    """
    
    benchmark_name: str
    subsets: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        """Validate benchmark configuration after initialization.
        
        Raises:
            ValueError: If benchmark_name is empty or invalid.
        """
        if not self.benchmark_name or not self.benchmark_name.strip():
            raise ValueError("benchmark_name cannot be empty")
        
        self.benchmark_name = self.benchmark_name.strip()
        
        if self.subsets is not None:
            # Remove empty strings and strip whitespace
            self.subsets = [subset.strip() for subset in self.subsets if subset.strip()]
            if not self.subsets:
                self.subsets = None
    
    def validate_with_benchmark_instance(self, benchmark: Benchmark) -> None:
        """Validate subsets against actual benchmark instance.
        
        Args:
            benchmark: Benchmark instance to validate against
            
        Raises:
            ValueError: If any subset is not available in the benchmark.
        """
        if self.subsets is not None:
            available_datasets: List[str] = benchmark.get_available_datasets()
            invalid_subsets: set[str] = set(self.subsets) - set(available_datasets)
            if invalid_subsets:
                raise ValueError(
                    f"Invalid subsets for {self.benchmark_name}: {invalid_subsets}. "
                    f"Available datasets: {available_datasets}"
                )


@dataclass
class AdapterConfig:
    """Base configuration for model adapters (sparse attention config is part of matrix).
    
    Note: sparse_attention_config is now part of the execution matrix, not base config.
    This allows testing different sparse attention configurations with the same adapter base.
    
    Attributes:
        adapter_name: Name of the adapter type (default: "huggingface")
        model_kwargs: Additional keyword arguments for model creation
        tokenizer_kwargs: Additional keyword arguments for tokenizer creation
    
    Example:
        >>> config = AdapterConfig(
        ...     model_kwargs={"torch_dtype": torch.bfloat16},
        ...     tokenizer_kwargs={"padding_side": "left"}
        ... )
    """
    
    adapter_name: str = "huggingface"
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Initialize default values and validate configuration."""
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}
        
        if not self.adapter_name or not self.adapter_name.strip():
            raise ValueError("adapter_name cannot be empty")
        
        self.adapter_name = self.adapter_name.strip()


@dataclass
class BenchmarkStub:
    """A single benchmark execution task (model × sparse_config × benchmark × subset).
    
    This represents one atomic execution unit that will be processed by a worker process.
    Each stub corresponds to a specific combination of model, sparse attention config,
    benchmark, and subset (if applicable).
    
    Attributes:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        sparse_config_name: Human-readable name for sparse config (e.g., "dense", "streaming")
        sparse_attention_config: SparseAttentionConfig instance or None for dense
        benchmark_name: Name of the benchmark (e.g., "longbench")
        subset: Specific subset name (e.g., "narrativeqa") or None for full benchmark
        adapter_config: Base adapter configuration (without sparse attention)
        generation_kwargs: Parameters for model inference/generation
        result_dir: Full path to directory where results should be saved
    
    Example:
        >>> stub = BenchmarkStub(
        ...     model_name="microsoft/Phi-4-mini-instruct",
        ...     sparse_config_name="dense",
        ...     sparse_attention_config=None,
        ...     benchmark_name="longbench",
        ...     subset="narrativeqa",
        ...     adapter_config=adapter_config,
        ...     generation_kwargs={"max_new_tokens": 256},
        ...     result_dir="/path/to/results/phi4_dense_longbench_narrativeqa"
        ... )
    """
    
    model_name: str
    sparse_config_name: str
    sparse_attention_config: Optional[SparseAttentionConfig]
    benchmark_name: str
    subset: Optional[str]
    adapter_config: AdapterConfig
    generation_kwargs: Dict[str, Any]
    result_dir: str
    
    def __post_init__(self) -> None:
        """Validate benchmark stub configuration."""
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not self.sparse_config_name or not self.sparse_config_name.strip():
            raise ValueError("sparse_config_name cannot be empty")
        if not self.benchmark_name or not self.benchmark_name.strip():
            raise ValueError("benchmark_name cannot be empty")
        if not self.result_dir or not self.result_dir.strip():
            raise ValueError("result_dir cannot be empty")
        
        # Trim whitespace
        self.model_name = self.model_name.strip()
        self.sparse_config_name = self.sparse_config_name.strip()
        self.benchmark_name = self.benchmark_name.strip()
        self.result_dir = self.result_dir.strip()
        
        if self.subset is not None:
            self.subset = self.subset.strip() if self.subset.strip() else None


@dataclass
class BenchmarkResult:
    """Result from a single benchmark execution.
    
    Attributes:
        stub: The benchmark stub that was executed
        metrics: Evaluation metrics returned by benchmark.run_benchmark()
        execution_time: Time taken to execute the benchmark in seconds
        success: Whether the benchmark executed successfully
        error_message: Error message if execution failed
    """
    
    stub: BenchmarkStub
    metrics: Optional[Dict[str, Any]]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkFailure:
    """Information about a failed benchmark execution.
    
    Attributes:
        stub: The benchmark stub that failed
        error_message: Description of the failure
        error_type: Type of error that occurred
        execution_time: Time spent before failure in seconds
    """
    
    stub: BenchmarkStub
    error_message: str
    error_type: str
    execution_time: float


@dataclass
class ExecutionProgress:
    """Tracks execution progress across all processes.
    
    Attributes:
        total_stubs: Total number of benchmark stubs to execute
        skipped_stubs: Number of stubs skipped due to existing results (resumability)
        completed_stubs: Number of stubs completed successfully
        failed_stubs: Number of stubs that failed execution
        current_executions: Mapping of GPU ID to current task description
    """
    
    total_stubs: int
    skipped_stubs: int = 0
    completed_stubs: int = 0
    failed_stubs: int = 0
    current_executions: Dict[int, str] = field(default_factory=dict)
    
    @property
    def pending_stubs(self) -> int:
        """Calculate number of stubs pending execution."""
        return self.total_stubs - self.skipped_stubs - self.completed_stubs - self.failed_stubs
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage (0-100)."""
        if self.total_stubs == 0:
            return 100.0
        return (self.completed_stubs + self.failed_stubs) / (self.total_stubs - self.skipped_stubs) * 100.0


@dataclass
class ExecutionResults:
    """Aggregated results from benchmark execution.
    
    Attributes:
        execution_summary: Summary metadata about the execution
        individual_results: List of successful benchmark results
        failed_executions: List of failed benchmark executions
        execution_time: Total execution time in seconds
        progress: Final progress statistics
    """
    
    execution_summary: Dict[str, Any]
    individual_results: List[BenchmarkResult]
    failed_executions: List[BenchmarkFailure]
    execution_time: float
    progress: ExecutionProgress


# =============================================================================
# Automatic Benchmark Registry (Task 1.1.2 - Improved)
# =============================================================================

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
        elif hasattr(benchmark_class, 'benchmark_name') and benchmark_class.benchmark_name:
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
    benchmark_name: str, 
    subsets: Optional[List[str]] = None
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
    import importlib
    import pkgutil
    import benchmark
    
    # Get all modules in the benchmark package
    benchmark_package_path = benchmark.__path__
    
    for importer, module_name, ispkg in pkgutil.iter_modules(benchmark_package_path):
        if not ispkg and module_name != '__init__' and module_name != 'base':
            try:
                # Import the module to trigger @register_benchmark decorators
                module_full_name = f"benchmark.{module_name}"
                importlib.import_module(module_full_name)
            except ImportError as e:
                # Some modules might not be importable (missing dependencies, etc.)
                # This is OK - we just skip them
                import logging
                logging.debug(f"Could not import benchmark module {module_name}: {e}")
                continue


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    """Validate a benchmark configuration against available benchmarks.
    
    This function checks if the benchmark exists and if specified subsets are valid.
    
    Args:
        config: BenchmarkConfig to validate
        
    Raises:
        ValueError: If benchmark or subsets are invalid
        
    Example:
        >>> config = BenchmarkConfig("longbench", ["narrativeqa"])
        >>> validate_benchmark_config(config)  # Passes validation
        >>> 
        >>> config = BenchmarkConfig("invalid_benchmark")
        >>> validate_benchmark_config(config)  # Raises ValueError
    """
    try:
        # Create temporary benchmark instance to validate
        benchmark = create_benchmark_instance(config.benchmark_name, subsets=None)
        
        # Validate subsets if specified
        config.validate_with_benchmark_instance(benchmark)
        
    except Exception as e:
        raise ValueError(f"Invalid benchmark configuration: {e}")


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in directory paths.
    
    Replaces special characters that are not filesystem-safe with underscores.
    This ensures model names can be used as directory names across different operating systems.
    
    Args:
        model_name: Original model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        
    Returns:
        Sanitized model name safe for filesystem use
        
    Example:
        >>> sanitized = sanitize_model_name("meta-llama/Llama-3.1-8B-Instruct")
        >>> print(sanitized)
        "meta_llama_Llama_3.1_8B_Instruct"
    """
    # Characters that need to be replaced: / \ : * ? " < > |
    special_chars: str = r'\/\:*?"<>|'
    
    sanitized: str = model_name
    for char in special_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove multiple consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized


def construct_result_directory(
    base_result_dir: str,
    model_name: str,
    sparse_config_name: str,
    benchmark_name: str,
    subset: Optional[str] = None
) -> str:
    """Construct result directory path following the standard hierarchy.
    
    Creates a path following the pattern:
    base_dir/sanitized_model/sparse_config/benchmark_subset/
    
    Note: No timestamp in path to enable resumability across runs.
    
    Args:
        base_result_dir: Base directory for all benchmark results
        model_name: Original model name (will be sanitized)
        sparse_config_name: Name of sparse attention configuration
        benchmark_name: Name of the benchmark
        subset: Optional subset name (appended to benchmark_name with underscore)
        
    Returns:
        Complete path to result directory
        
    Example:
        >>> path = construct_result_directory(
        ...     "./results", "meta-llama/Llama-3.1-8B", 
        ...     "dense", "longbench", "narrativeqa"
        ... )
        >>> print(path)
        "./results/meta_llama_Llama_3.1_8B/dense/longbench_narrativeqa"
    """
    sanitized_model: str = sanitize_model_name(model_name)
    
    # Construct benchmark directory name
    benchmark_dir: str = benchmark_name
    if subset is not None:
        benchmark_dir = f"{benchmark_name}_{subset}"
    
    # Build path components (no timestamp for resumability)
    path_components: List[str] = [
        base_result_dir,
        sanitized_model,
        sparse_config_name,
        benchmark_dir
    ]
    
    result_path: str = os.path.join(*path_components)
    return str(Path(result_path).resolve())


def create_result_directory(
    result_dir: str,
    create_parents: bool = True
) -> bool:
    """Create result directory with proper error handling.
    
    Creates the specified directory and all necessary parent directories.
    This function is safe to call multiple times on the same path.
    
    Args:
        result_dir: Path to the directory to create
        create_parents: Whether to create parent directories if they don't exist
        
    Returns:
        True if directory was created or already exists, False if creation failed
        
    Raises:
        ValueError: If result_dir is empty or invalid
        PermissionError: If insufficient permissions to create directory
        OSError: If directory creation fails for other reasons
        
    Example:
        >>> success = create_result_directory("/path/to/results/model_experiment")
        >>> print(success)  # True if successful
    """
    if not result_dir or not result_dir.strip():
        raise ValueError("result_dir cannot be empty")
    
    result_path = Path(result_dir.strip())
    
    try:
        # Create directory and parents if needed
        result_path.mkdir(parents=create_parents, exist_ok=True)
        
        # Verify directory was created and is writable
        if not result_path.exists():
            return False
        
        if not result_path.is_dir():
            raise OSError(f"Path exists but is not a directory: {result_path}")
        
        # Test write permissions by creating a temporary file
        test_file = result_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()  # Remove test file
        except (PermissionError, OSError) as e:
            raise PermissionError(f"Directory created but not writable: {result_path}") from e
        
        return True
        
    except PermissionError:
        raise
    except OSError as e:
        raise OSError(f"Failed to create directory {result_path}: {e}") from e


def validate_result_directory_path(
    result_dir: str,
    check_parent_writable: bool = True
) -> None:
    """Validate that a result directory path is valid and can be created.
    
    Performs validation without actually creating the directory.
    
    Args:
        result_dir: Path to validate
        check_parent_writable: Whether to check if parent directory is writable
        
    Raises:
        ValueError: If path is invalid or unsafe
        PermissionError: If parent directory is not writable
        OSError: If path validation fails
        
    Example:
        >>> validate_result_directory_path("/valid/path/to/results")  # Passes
        >>> validate_result_directory_path("")  # Raises ValueError
    """
    if not result_dir or not result_dir.strip():
        raise ValueError("result_dir cannot be empty")
    
    result_path = Path(result_dir.strip())
    
    # Check for potentially dangerous paths
    try:
        # Resolve to absolute path to detect path traversal attempts
        absolute_path = result_path.resolve()
        
        # Basic security check - ensure path doesn't try to escape reasonable bounds
        if ".." in str(result_path):
            # Allow .. only if resolved path is still reasonable
            path_str = str(absolute_path)
            if any(dangerous in path_str for dangerous in ["/etc", "/usr", "/bin", "/sbin", "/sys", "/proc"]):
                raise ValueError(f"Path appears to target system directories: {result_path}")
        
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path format: {result_path}") from e
    
    # Check if parent directory exists and is writable (if requested)
    if check_parent_writable:
        parent_dir = result_path.parent
        
        if not parent_dir.exists():
            # Check if we can create the parent directories
            try:
                # Test by checking the first existing parent
                existing_parent = parent_dir
                while not existing_parent.exists() and existing_parent.parent != existing_parent:
                    existing_parent = existing_parent.parent
                
                if existing_parent.exists() and not os.access(existing_parent, os.W_OK):
                    raise PermissionError(f"Cannot write to parent directory tree: {existing_parent}")
                    
            except (OSError, AttributeError) as e:
                raise PermissionError(f"Cannot validate parent directory permissions: {parent_dir}") from e
        
        elif not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"Parent directory is not writable: {parent_dir}")


def ensure_result_directory(
    result_dir: str,
    validate_first: bool = True
) -> str:
    """Ensure result directory exists and is ready for use.
    
    This is a convenience function that combines validation and creation.
    
    Args:
        result_dir: Path to the result directory
        validate_first: Whether to validate the path before creating
        
    Returns:
        Absolute path to the created directory
        
    Raises:
        ValueError: If path validation fails
        PermissionError: If directory cannot be created due to permissions
        OSError: If directory creation fails
        
    Example:
        >>> abs_path = ensure_result_directory("./results/experiment_1")
        >>> print(abs_path)  # "/absolute/path/to/results/experiment_1"
    """
    if validate_first:
        validate_result_directory_path(result_dir, check_parent_writable=True)
    
    success = create_result_directory(result_dir, create_parents=True)
    if not success:
        raise OSError(f"Failed to create result directory: {result_dir}")
    
    # Return absolute path
    return str(Path(result_dir).resolve()) 


# =============================================================================
# Matrix Expansion and Resumability Functions (Task 1.3)
# =============================================================================

def generate_benchmark_stubs(
    model_names: List[str],
    sparse_attention_configs: List[Tuple[str, Optional['SparseAttentionConfig']]],
    benchmark_configs: List[BenchmarkConfig],
    adapter_config: AdapterConfig,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    base_result_dir: str = "./benchmark_results"
) -> List[BenchmarkStub]:
    """Generate all benchmark stubs for the model × sparse_config × benchmark × subset matrix.
    
    Creates BenchmarkStub instances for every combination of:
    - model_names × sparse_attention_configs × benchmark_configs × subsets
    
    Args:
        model_names: List of model names to benchmark
        sparse_attention_configs: List of (name, config) tuples for sparse attention
        benchmark_configs: List of benchmark configurations
        adapter_config: Configuration for model adapter
        generation_kwargs: Optional generation parameters
        base_result_dir: Base directory for storing results
        
    Returns:
        List of BenchmarkStub instances representing all combinations
        
    Example:
        >>> stubs = generate_benchmark_stubs(
        ...     model_names=["model1", "model2"], 
        ...     sparse_attention_configs=[("dense", None), ("sparse", config)],
        ...     benchmark_configs=[BenchmarkConfig("longbench", ["narrativeqa"])],
        ...     adapter_config=AdapterConfig()
        ... )
        >>> len(stubs)  # 2 models × 2 configs × 1 benchmark × 1 subset = 4
        4
    """
    
    if generation_kwargs is None:
        generation_kwargs = {}
        
    stubs: List[BenchmarkStub] = []
    
    # Expand the full matrix: model × sparse_config × benchmark × subset
    for model_name in model_names:
        for sparse_config_name, sparse_attention_config in sparse_attention_configs:
            for benchmark_config in benchmark_configs:
                # Handle benchmarks with explicit subsets
                if benchmark_config.subsets:
                    for subset in benchmark_config.subsets:
                        result_dir = construct_result_directory(
                            base_result_dir=base_result_dir,
                            model_name=model_name,
                            sparse_config_name=sparse_config_name,
                            benchmark_name=benchmark_config.benchmark_name,
                            subset=subset
                        )
                        
                        stub = BenchmarkStub(
                            model_name=model_name,
                            sparse_config_name=sparse_config_name,
                            sparse_attention_config=sparse_attention_config,
                            benchmark_name=benchmark_config.benchmark_name,
                            subset=subset,
                            adapter_config=adapter_config,
                            generation_kwargs=generation_kwargs.copy(),
                            result_dir=result_dir
                        )
                        stubs.append(stub)
                        
                else:
                    # Handle benchmarks with no subsets (run full benchmark)
                    result_dir = construct_result_directory(
                        base_result_dir=base_result_dir,
                        model_name=model_name,
                        sparse_config_name=sparse_config_name,
                        benchmark_name=benchmark_config.benchmark_name,
                        subset=None
                    )
                    
                    stub = BenchmarkStub(
                        model_name=model_name,
                        sparse_config_name=sparse_config_name,
                        sparse_attention_config=sparse_attention_config,
                        benchmark_name=benchmark_config.benchmark_name,
                        subset=None,
                        adapter_config=adapter_config,
                        generation_kwargs=generation_kwargs.copy(),
                        result_dir=result_dir
                    )
                    stubs.append(stub)
    
    return stubs


def filter_existing_results(
    stubs: List[BenchmarkStub],
    check_result_files: bool = True,
    required_files: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[List[BenchmarkStub], List[BenchmarkStub]]:
    """Filter benchmark stubs to skip already completed experiments (resumability).
    
    Checks each stub's result_dir to determine if the experiment has already been
    completed. This enables resuming interrupted benchmark runs.
    
    Args:
        stubs: List of benchmark stubs to filter
        check_result_files: Whether to check for specific result files (not just directory)
        required_files: List of filenames that must exist for completion (default: ["results.json"])
        verbose: Whether to log skipped combinations
        
    Returns:
        Tuple of (pending_stubs, completed_stubs) where:
        - pending_stubs: Stubs that need to be executed
        - completed_stubs: Stubs that have already been completed
        
    Example:
        >>> pending, completed = filter_existing_results(all_stubs)
        >>> print(f"Found {len(completed)} completed, {len(pending)} pending")
    """
    import logging
    from pathlib import Path
    
    if required_files is None:
        required_files = ["results.json"]
    
    pending_stubs: List[BenchmarkStub] = []
    completed_stubs: List[BenchmarkStub] = []
    
    for stub in stubs:
        result_path = Path(stub.result_dir)
        
        # Check if result directory exists
        if not result_path.exists():
            pending_stubs.append(stub)
            continue
            
        # If not checking result files, just check directory existence
        if not check_result_files:
            completed_stubs.append(stub)
            if verbose:
                logging.info(f"Skipping completed experiment: {stub.result_dir}")
            continue
            
        # Check for required result files
        all_files_exist = True
        for required_file in required_files:
            file_path = result_path / required_file
            if not file_path.exists() or not file_path.is_file():
                all_files_exist = False
                break
                
        if all_files_exist:
            completed_stubs.append(stub)
            if verbose:
                logging.info(f"Skipping completed experiment (has {required_files}): {stub.result_dir}")
        else:
            pending_stubs.append(stub)
            if verbose:
                logging.info(f"Pending experiment (missing results): {stub.result_dir}")
    
    if verbose:
        total = len(stubs)
        completed_count = len(completed_stubs)
        pending_count = len(pending_stubs)
        logging.info(f"Resumability check: {completed_count}/{total} completed, {pending_count}/{total} pending")
    
    return pending_stubs, completed_stubs


def get_benchmark_subsets(benchmark_name: str) -> Optional[List[str]]:
    """Get available subsets for a benchmark by introspecting the benchmark class.
    
    This is a utility function that can help users determine what subsets are
    available for a given benchmark without having to know the internals.
    
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
        
        # Try to get available subsets - different benchmarks may have different ways
        # to expose this information
        if hasattr(benchmark_instance, 'get_available_subsets'):
            return benchmark_instance.get_available_subsets()  # type: ignore
        elif hasattr(benchmark_instance, 'subsets'):
            return list(benchmark_instance.subsets)  # type: ignore
        elif hasattr(benchmark_instance, 'AVAILABLE_SUBSETS'):
            return list(benchmark_instance.AVAILABLE_SUBSETS)  # type: ignore
        else:
            # Benchmark doesn't expose subset information
            return None
            
    except Exception as e:
        # If we can't instantiate the benchmark or get subset info, return None
        import logging
        logging.warning(f"Could not get subsets for benchmark '{benchmark_name}': {e}")
        return None 


# =============================================================================
# GPU Management and Validation Functions (Task 1.4)
# =============================================================================

def validate_gpu_availability(gpu_ids: List[int]) -> None:
    """Validate that all specified GPUs are available and accessible.
    
    Args:
        gpu_ids: List of GPU device IDs to validate
        
    Raises:
        RuntimeError: If CUDA is not available
        ValueError: If any GPU ID is invalid or inaccessible
        
    Example:
        >>> validate_gpu_availability([0, 1])  # Validates GPUs 0 and 1
    """
    import torch
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found")
    
    # Validate each GPU ID
    for gpu_id in gpu_ids:
        if not isinstance(gpu_id, int):
            raise ValueError(f"GPU ID must be an integer, got {type(gpu_id)}: {gpu_id}")
        
        if gpu_id < 0:
            raise ValueError(f"GPU ID must be non-negative, got: {gpu_id}")
        
        if gpu_id >= device_count:
            raise ValueError(f"GPU ID {gpu_id} is not available. Only {device_count} GPU(s) found: {list(range(device_count))}")
        
        # Test basic GPU accessibility
        try:
            device = torch.device(f"cuda:{gpu_id}")
            # Try to create a small tensor on the device
            test_tensor = torch.tensor([1.0], device=device)
            del test_tensor
        except Exception as e:
            raise ValueError(f"GPU {gpu_id} is not accessible: {e}")


def create_gpu_pool(gpu_ids: List[int]) -> 'multiprocessing.Queue[int]':
    """Create a multiprocessing queue with available GPU IDs for worker processes.
    
    Args:
        gpu_ids: List of GPU device IDs to include in the pool
        
    Returns:
        multiprocessing.Queue containing GPU IDs for workers to acquire
        
    Example:
        >>> gpu_pool = create_gpu_pool([0, 1, 2])
        >>> gpu_id = gpu_pool.get()  # Acquire a GPU
        >>> gpu_pool.put(gpu_id)     # Release the GPU back to pool
    """
    import multiprocessing
    
    # Validate GPUs first
    validate_gpu_availability(gpu_ids)
    
    # Create queue and populate with GPU IDs
    gpu_queue: multiprocessing.Queue[int] = multiprocessing.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    
    return gpu_queue


def acquire_gpu_from_pool(gpu_pool: 'multiprocessing.Queue[int]', timeout: float = 30.0) -> int:
    """Acquire a GPU ID from the pool for exclusive use by a worker process.
    
    Args:
        gpu_pool: Multiprocessing queue containing available GPU IDs
        timeout: Maximum time to wait for a GPU (seconds)
        
    Returns:
        GPU device ID acquired from the pool
        
    Raises:
        TimeoutError: If no GPU becomes available within timeout
        
    Example:
        >>> gpu_id = acquire_gpu_from_pool(gpu_pool, timeout=60.0)
        >>> # Use gpu_id for computation
        >>> release_gpu_to_pool(gpu_pool, gpu_id)
    """
    import queue
    
    try:
        gpu_id = gpu_pool.get(timeout=timeout)
        return gpu_id
    except queue.Empty:
        raise TimeoutError(f"No GPU became available within {timeout} seconds")


def release_gpu_to_pool(gpu_pool: 'multiprocessing.Queue[int]', gpu_id: int) -> None:
    """Release a GPU ID back to the pool for other workers to use.
    
    Args:
        gpu_pool: Multiprocessing queue to return the GPU to
        gpu_id: GPU device ID to release
        
    Example:
        >>> gpu_id = acquire_gpu_from_pool(gpu_pool)
        >>> # Use gpu_id for computation
        >>> release_gpu_to_pool(gpu_pool, gpu_id)
    """
    gpu_pool.put(gpu_id)


def validate_gpu_for_model(gpu_id: int, model_name: str, check_memory: bool = True) -> Dict[str, Any]:
    """Validate that a GPU can handle a specific model (minimal validation).
    
    Performs basic checks without loading the full model to avoid memory overhead.
    
    Args:
        gpu_id: GPU device ID to validate
        model_name: Name of the model to validate against
        check_memory: Whether to check available GPU memory
        
    Returns:
        Dictionary with validation results and GPU information
        
    Example:
        >>> info = validate_gpu_for_model(0, "meta-llama/Llama-3.1-8B-Instruct")
        >>> print(f"Available memory: {info['memory_available_gb']:.2f} GB")
    """
    import torch
    
    # Validate GPU accessibility
    validate_gpu_availability([gpu_id])
    
    # Get GPU properties
    device = torch.device(f"cuda:{gpu_id}")
    props = torch.cuda.get_device_properties(gpu_id)
    
    # Get memory information
    memory_total = props.total_memory
    memory_reserved = torch.cuda.memory_reserved(gpu_id)
    memory_allocated = torch.cuda.memory_allocated(gpu_id)
    memory_available = memory_total - memory_reserved
    
    # Convert to GB for readability
    memory_total_gb = memory_total / (1024 ** 3)
    memory_available_gb = memory_available / (1024 ** 3)
    memory_allocated_gb = memory_allocated / (1024 ** 3)
    
    validation_info = {
        "gpu_id": gpu_id,
        "gpu_name": props.name,
        "memory_total_gb": memory_total_gb,
        "memory_available_gb": memory_available_gb,
        "memory_allocated_gb": memory_allocated_gb,
        "compute_capability": f"{props.major}.{props.minor}",
        "model_name": model_name,
        "validation_passed": True
    }
    
    # Basic memory check - warn if less than 2GB available
    if check_memory and memory_available_gb < 2.0:
        import logging
        logging.warning(f"Low GPU memory on device {gpu_id}: {memory_available_gb:.2f} GB available")
        validation_info["low_memory_warning"] = True
    
    return validation_info


def set_cuda_visible_devices(gpu_id: int) -> None:
    """Set CUDA_VISIBLE_DEVICES to limit a process to a specific GPU.
    
    This should be called at the beginning of worker processes to ensure
    they only see their assigned GPU.
    
    Args:
        gpu_id: GPU device ID to make visible to this process
        
    Example:
        >>> set_cuda_visible_devices(1)  # Process will only see GPU 1 as device 0
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def cleanup_gpu_memory(gpu_id: Optional[int] = None) -> None:
    """Clean up GPU memory by emptying CUDA cache.
    
    Args:
        gpu_id: Specific GPU to clean up, or None for current device
        
    Example:
        >>> cleanup_gpu_memory(0)  # Clean up GPU 0
        >>> cleanup_gpu_memory()   # Clean up current device
    """
    import torch
    
    if torch.cuda.is_available():
        if gpu_id is not None:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache() 