"""GPU utility functions for ModelServer operations."""

from typing import Optional

import torch


def cleanup_gpu_memory(gpu_id: Optional[int] = None) -> None:
    """Clean up GPU memory by emptying CUDA cache.
    
    Args:
        gpu_id: Specific GPU to clean up, or None for current device
        
    Example:
        >>> cleanup_gpu_memory(0)  # Clean up GPU 0
        >>> cleanup_gpu_memory()   # Clean up current device
    """
    if torch.cuda.is_available():
        try:
            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        except Exception:
            # Silently handle GPU errors - cleanup is best effort
            pass
