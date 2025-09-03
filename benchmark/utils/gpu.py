"""GPU management and validation utilities for benchmark execution.

This module contains utilities for:
- Validating GPU availability and capabilities
- Managing GPU pools for multiprocessing
- Setting CUDA device visibility
- Cleaning up GPU memory
"""

import logging
import multiprocessing
import os
import queue
import time
from typing import Any, Dict, List, Optional, Union

import torch


def get_cuda_visible_devices() -> List[int]:
    """Get the CUDA_VISIBLE_DEVICES environment variable.

    Returns:
        CUDA_VISIBLE_DEVICES environment variable, or None if not set
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        all_devices = [i for i in range(torch.cuda.device_count())]
        return all_devices
    else:
        return [int(id) for id in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]


def validate_gpu_availability(gpu_ids: List[int]) -> None:
    """Validate that specified GPUs are available and accessible.

    Checks that CUDA is available, GPUs exist, and they are accessible.

    Args:
        gpu_ids: List of GPU device IDs to validate

    Raises:
        RuntimeError: If CUDA is not available
        ValueError: If any GPU ID is invalid or inaccessible
        OSError: If GPU access fails

    Example:
        >>> validate_gpu_availability([0, 1])  # Validates GPUs 0 and 1
        >>> # No exception means GPUs are available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if not gpu_ids:
        raise ValueError("gpu_ids cannot be empty")

    num_gpus: int = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found")

    # Check each GPU ID
    for gpu_id in gpu_ids:
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError(f"Invalid GPU ID: {gpu_id}")

        # if gpu_id >= num_gpus:
        #     raise ValueError(f"GPU {gpu_id} does not exist. Available GPUs: 0-{num_gpus-1}")

        # Test GPU access
        try:
            with torch.cuda.device(gpu_id):
                # Try to allocate a small tensor to test access
                test_tensor = torch.zeros(1, device=f"cuda:{gpu_id}")
                del test_tensor
                torch.cuda.empty_cache()
        except Exception as e:
            raise OSError(f"Cannot access GPU {gpu_id}: {e}") from e


def create_gpu_pool(gpu_ids: List[int]) -> "multiprocessing.Queue[int]":
    """Create a multiprocessing queue to manage GPU access.

    Creates a thread-safe queue that workers can use to acquire and release GPUs.

    Args:
        gpu_ids: List of GPU device IDs to include in the pool

    Returns:
        Multiprocessing queue containing available GPU IDs

    Example:
        >>> gpu_pool = create_gpu_pool([0, 1, 2])
        >>> gpu_id = acquire_gpu_from_pool(gpu_pool)
        >>> # Use GPU...
        >>> release_gpu_to_pool(gpu_pool, gpu_id)
    """
    validate_gpu_availability(gpu_ids)

    # Create queue and populate with GPU IDs
    gpu_pool: "multiprocessing.Queue[int]" = multiprocessing.Queue()
    for gpu_id in gpu_ids:
        gpu_pool.put(gpu_id)

    return gpu_pool


def acquire_gpu_from_pool(
    gpu_pool: "multiprocessing.Queue[int]", timeout: float = 30.0
) -> int:
    """Acquire a GPU from the pool with timeout.

    Blocks until a GPU becomes available or timeout is reached.

    Args:
        gpu_pool: GPU pool queue
        timeout: Maximum time to wait for GPU in seconds

    Returns:
        GPU device ID that was acquired

    Raises:
        queue.Empty: If no GPU becomes available within timeout

    Example:
        >>> gpu_id = acquire_gpu_from_pool(gpu_pool, timeout=60.0)
        >>> print(f"Acquired GPU {gpu_id}")
    """
    try:
        gpu_id: int = gpu_pool.get(timeout=timeout)
        return gpu_id
    except queue.Empty:
        raise queue.Empty(f"No GPU available within {timeout} seconds")


def release_gpu_to_pool(gpu_pool: "multiprocessing.Queue[int]", gpu_id: int) -> None:
    """Release a GPU back to the pool.

    Returns the GPU to the pool so other workers can use it.

    Args:
        gpu_pool: GPU pool queue
        gpu_id: GPU device ID to release

    Example:
        >>> release_gpu_to_pool(gpu_pool, 0)
        >>> # GPU 0 is now available for other workers
    """
    gpu_pool.put(gpu_id)


def validate_gpu_for_model(
    gpu_id: int, model_name: str, check_memory: bool = True
) -> Dict[str, Any]:
    """Validate that a GPU is suitable for running a specific model.

    Performs various checks including memory availability and compute capability.

    Args:
        gpu_id: GPU device ID to validate
        model_name: Name of the model (for logging purposes)
        check_memory: Whether to check available GPU memory

    Returns:
        Dictionary containing validation results and GPU information

    Raises:
        ValueError: If GPU is not suitable for the model
        OSError: If GPU access fails

    Example:
        >>> info = validate_gpu_for_model(0, "llama-7b")
        >>> print(f"GPU memory: {info['memory_gb']} GB")
    """
    try:
        with torch.cuda.device(gpu_id):
            # Get GPU properties
            props = torch.cuda.get_device_properties(gpu_id)

            # Basic validation
            if props.major < 7:  # Require compute capability 7.0+
                raise ValueError(
                    f"GPU {gpu_id} has insufficient compute capability: {props.major}.{props.minor}"
                )

            # Memory validation
            memory_info: Dict[str, Any] = {}
            if check_memory:
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                cached_memory = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                free_memory = total_memory - allocated_memory

                memory_info = {
                    "total_gb": total_memory,
                    "allocated_gb": allocated_memory,
                    "cached_gb": cached_memory,
                    "free_gb": free_memory,
                }

                # Warn if memory is low (less than 2GB free)
                if free_memory < 2.0:
                    logging.warning(
                        f"GPU {gpu_id} has low memory for {model_name}: "
                        f"{free_memory:.1f}GB free out of {total_memory:.1f}GB total"
                    )

            return {
                "gpu_id": gpu_id,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "memory_gb": props.total_memory / (1024**3),
                **memory_info,
            }

    except Exception as e:
        raise OSError(f"Failed to validate GPU {gpu_id} for {model_name}: {e}") from e


def cleanup_gpu_memory(gpu_id: Optional[int] = None) -> None:
    """Clean up GPU memory by emptying CUDA cache.

    Args:
        gpu_id: Specific GPU to clean up, or None for current device

    Example:
        >>> cleanup_gpu_memory(0)  # Clean up GPU 0
        >>> cleanup_gpu_memory()   # Clean up current device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
