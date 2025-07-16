"""Utility functions for benchmark execution and management.

This package contains utility functions organized by purpose:
- paths: Path and directory management utilities
- gpu: GPU management and validation utilities  
- validation: General validation utilities
"""

# Import all utilities from submodules for backward compatibility
from .paths import (
    sanitize_model_name,
    construct_result_directory,
    create_result_directory,
    validate_result_directory_path,
    ensure_result_directory
)

from .gpu import (
    validate_gpu_availability,
    create_gpu_pool,
    acquire_gpu_from_pool,
    release_gpu_to_pool,
    validate_gpu_for_model,
    set_cuda_visible_devices,
    cleanup_gpu_memory
)

# Re-export all utilities for backward compatibility
__all__ = [
    # Path utilities
    "sanitize_model_name",
    "construct_result_directory", 
    "create_result_directory",
    "validate_result_directory_path",
    "ensure_result_directory",
    
    # GPU utilities
    "validate_gpu_availability",
    "create_gpu_pool",
    "acquire_gpu_from_pool",
    "release_gpu_to_pool",
    "validate_gpu_for_model",
    "set_cuda_visible_devices",
    "cleanup_gpu_memory"
] 