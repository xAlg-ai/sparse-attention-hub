"""Path and directory management utilities for benchmark execution.

This module contains utilities for:
- Sanitizing model names for filesystem use
- Constructing result directory paths
- Creating and validating directories
- Ensuring result directories exist
"""

import os
import logging
from pathlib import Path
from typing import Optional


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
        sanitized = sanitized.replace(char, "_")

    # Remove multiple consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    return sanitized


def construct_result_directory(
    base_result_dir: str,
    model_name: str,
    sparse_config_name: str,
    benchmark_name: str,
    subset: Optional[str] = None,
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
    path_components: list[str] = [
        base_result_dir,
        sanitized_model,
        sparse_config_name,
        benchmark_dir,
    ]

    result_path: str = os.path.join(*path_components)
    return str(Path(result_path).resolve())


def create_result_directory(result_dir: str, create_parents: bool = True) -> bool:
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
            raise PermissionError(
                f"Directory created but not writable: {result_path}"
            ) from e

        return True

    except PermissionError:
        raise
    except OSError as e:
        raise OSError(f"Failed to create directory {result_path}: {e}") from e


def validate_result_directory_path(
    result_dir: str, check_parent_writable: bool = True
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
        >>> validate_result_directory_path("/path/to/results")
        >>> # No exception means path is valid
    """
    if not result_dir or not result_dir.strip():
        raise ValueError("result_dir cannot be empty")

    result_path = Path(result_dir.strip())

    # Check for unsafe path components
    try:
        result_path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {result_dir}") from e

    # Check if path already exists
    if result_path.exists():
        if not result_path.is_dir():
            raise OSError(f"Path exists but is not a directory: {result_path}")
        return  # Directory exists and is valid

    # Check parent directory
    parent_path = result_path.parent
    if not parent_path.exists():
        if check_parent_writable:
            raise OSError(f"Parent directory does not exist: {parent_path}")
    elif check_parent_writable:
        # Test if parent is writable
        test_file = parent_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            raise PermissionError(
                f"Parent directory not writable: {parent_path}"
            ) from e


def ensure_result_directory(result_dir: str, validate_first: bool = True) -> str:
    """Ensure result directory exists, creating it if necessary.

    This is a convenience function that combines validation and creation.

    Args:
        result_dir: Path to the result directory
        validate_first: Whether to validate before creating

    Returns:
        Absolute path to the result directory

    Raises:
        ValueError: If path is invalid
        PermissionError: If directory cannot be created due to permissions
        OSError: If directory creation fails

    Example:
        >>> path = ensure_result_directory("/path/to/results")
        >>> print(path)  # "/absolute/path/to/results"
    """
    if validate_first:
        validate_result_directory_path(result_dir)

    success = create_result_directory(result_dir)
    if not success:
        raise OSError(f"Failed to create result directory: {result_dir}")

    return str(Path(result_dir).resolve())
