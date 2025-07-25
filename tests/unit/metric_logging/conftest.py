"""Pytest configuration for metric_logging tests."""

import pytest
from typing import Dict, Any


@pytest.fixture
def sample_metric_data() -> Dict[str, Any]:
    """Sample metric data for testing."""
    return {
        "attention_score": 0.85,
        "sparsity_ratio": 0.3,
        "computation_time": 0.0012,
        "memory_usage": 1024
    }


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Sample metadata for testing."""
    return {
        "layer": 5,
        "head": 2,
        "attention_type": "topk",
        "sequence_length": 512,
        "batch_size": 4
    }


@pytest.fixture
def complex_metadata() -> Dict[str, Any]:
    """Complex metadata structure for testing."""
    return {
        "nested": {
            "key": "value",
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2}
        },
        "array": [{"x": 1}, {"y": 2}],
        "boolean": True,
        "null": None,
        "string": "test_value",
        "number": 42.5
    }


@pytest.fixture
def metric_identifiers() -> list[str]:
    """List of metric identifiers for testing."""
    return [
        "attention_score",
        "sparsity_ratio", 
        "computation_time",
        "memory_usage",
        "accuracy",
        "throughput"
    ]


@pytest.fixture
def metric_types() -> Dict[str, type]:
    """Mapping of metric identifiers to their expected types."""
    return {
        "attention_score": float,
        "sparsity_ratio": float,
        "computation_time": float,
        "memory_usage": int,
        "accuracy": float,
        "throughput": float
    } 