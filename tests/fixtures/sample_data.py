"""Sample data and test utilities for sparse attention tests."""

from typing import Any, Dict

import torch


def create_sample_attention_data(
    batch_size: int = 4, num_heads: int = 8, seq_len: int = 512, hidden_dim: int = 64
) -> Dict[str, torch.Tensor]:
    """Create sample attention data for testing."""
    return {
        "query": torch.randn(batch_size, num_heads, seq_len, hidden_dim),
        "key": torch.randn(batch_size, num_heads, seq_len, hidden_dim),
        "value": torch.randn(batch_size, num_heads, seq_len, hidden_dim),
        "attention_scores": torch.randn(batch_size, num_heads, seq_len, seq_len),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


def create_sparse_mask(
    batch_size: int = 4, num_heads: int = 8, seq_len: int = 512, sparsity: float = 0.5
) -> torch.Tensor:
    """Create a sparse attention mask for testing."""
    mask = torch.rand(batch_size, num_heads, seq_len, seq_len) > sparsity
    return mask.bool()


def create_benchmark_data(num_samples: int = 100, seq_len: int = 512) -> Dict[str, Any]:
    """Create sample benchmark data."""
    return {
        "texts": [f"Sample text {i}" for i in range(num_samples)],
        "expected_outputs": [f"Expected output {i}" for i in range(num_samples)],
        "metadata": {
            "dataset_name": "test_dataset",
            "num_samples": num_samples,
            "avg_seq_len": seq_len,
        },
    }
