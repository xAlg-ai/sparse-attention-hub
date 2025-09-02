"""Key generation utilities for ModelServer."""

import hashlib
from typing import Any, Dict


def hash_kwargs(kwargs: Dict[str, Any]) -> str:
    """Generate a consistent hash for a dictionary of keyword arguments.

    Args:
        kwargs: Dictionary to hash

    Returns:
        Hexadecimal hash string

    Note:
        This function sorts the dictionary items to ensure consistent hashing
        regardless of insertion order.
    """
    if not kwargs:
        return "empty"

    # Convert to sorted tuple of items for consistent hashing
    try:
        # Handle nested dictionaries and convert to string representation
        sorted_items = sorted(kwargs.items())
        str_repr = str(sorted_items)
        return hashlib.md5(str_repr.encode("utf-8")).hexdigest()[:8]
    except Exception:
        # Fallback to string representation if sorting fails
        str_repr = str(kwargs)
        return hashlib.md5(str_repr.encode("utf-8")).hexdigest()[:8]
