"""Model management utilities for ModelServer."""

from typing import Any, Dict, Optional

from .key_generation import hash_kwargs


def generate_model_key(
    model_name: str, gpu_id: Optional[int], model_kwargs: Dict[str, Any]
) -> str:
    """Generate a unique key for a model based on its parameters.

    Args:
        model_name: Name of the model
        gpu_id: GPU ID where model is placed (None for CPU)
        model_kwargs: Additional model creation arguments

    Returns:
        Unique string key for the model
    """
    gpu_str = str(gpu_id) if gpu_id is not None else "cpu"
    kwargs_hash = hash_kwargs(model_kwargs)
    return f"{model_name}|{gpu_str}|{kwargs_hash}"


def generate_tokenizer_key(
    tokenizer_name: str, tokenizer_kwargs: Dict[str, Any]
) -> str:
    """Generate a unique key for a tokenizer based on its parameters.

    Args:
        tokenizer_name: Name of the tokenizer
        tokenizer_kwargs: Additional tokenizer creation arguments

    Returns:
        Unique string key for the tokenizer
    """
    kwargs_hash = hash_kwargs(tokenizer_kwargs)
    return f"{tokenizer_name}|{kwargs_hash}"
