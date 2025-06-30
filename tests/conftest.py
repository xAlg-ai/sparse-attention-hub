"""Pytest configuration and shared fixtures for sparse attention hub tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_attention_scores() -> np.ndarray:
    """Generate sample attention scores for testing."""
    return np.random.randn(32, 8, 512, 512)


@pytest.fixture
def sample_sparse_mask() -> np.ndarray:
    """Generate sample sparse attention mask for testing."""
    mask = np.random.rand(32, 8, 512, 512) > 0.5
    return mask.astype(np.bool_)


@pytest.fixture
def small_sequence_length() -> int:
    """Small sequence length for fast tests."""
    return 128


@pytest.fixture
def medium_sequence_length() -> int:
    """Medium sequence length for integration tests."""
    return 512


@pytest.fixture
def large_sequence_length() -> int:
    """Large sequence length for performance tests."""
    return 2048


@pytest.fixture
def num_heads() -> int:
    """Number of attention heads for testing."""
    return 8


@pytest.fixture
def batch_size() -> int:
    """Batch size for testing."""
    return 4
