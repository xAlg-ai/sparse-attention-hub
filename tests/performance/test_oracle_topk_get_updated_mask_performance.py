"""Performance tests for OracleTopK get_updated_mask implementations.

This module compares the performance of old vs new implementations
to determine the optimal default mode.
"""

import time
from typing import List, Tuple, Dict
import pytest
import torch
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    OracleTopK,
    OracleTopKConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


class TestOracleTopKGetUpdatedMaskPerformance:
    """Performance comparison tests for get_updated_mask implementations."""

    def _create_test_inputs(
        self,
        batch_size: int,
        num_heads_queries: int,
        num_heads_keys: int,
        seq_len_queries: int,
        seq_len_keys: int,
        head_dim: int,
        previous_mask_pattern: str = "empty",
        use_attention_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mask, AttentionTensorDimensions]:
        """Create test inputs for performance testing."""
        # Create inputs
        keys: torch.Tensor = torch.randn(batch_size, num_heads_keys, seq_len_keys, head_dim)
        queries: torch.Tensor = torch.randn(batch_size, num_heads_queries, seq_len_queries, head_dim)
        
        # Create attention mask if needed
        attention_mask: torch.Tensor = None
        if use_attention_mask:
            attention_mask = torch.zeros(batch_size, 1, seq_len_queries, seq_len_keys)
            for i in range(seq_len_queries):
                if i < seq_len_keys:
                    attention_mask[:, :, i, i+1:] = float('-inf')
        
        # Create previous mask
        mask_shape: tuple = (batch_size, num_heads_queries, seq_len_queries, seq_len_keys)
        if previous_mask_pattern == "empty":
            previous_mask: Mask = Mask.create_empty_mask(mask_shape, dtype=torch.float32, device=torch.device("cpu"))
        elif previous_mask_pattern == "partial":
            previous_mask_data: torch.Tensor = torch.zeros(mask_shape)
            previous_mask_data[:, :, :, :2] = 1.0
            previous_mask = Mask.create_mask_from_dense_mask(mask_shape, previous_mask_data, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown previous_mask_pattern: {previous_mask_pattern}")
        
        # Create tensor dimensions
        tensor_dims: AttentionTensorDimensions = AttentionTensorDimensions(
            batch_size=batch_size,
            num_heads=num_heads_queries,
            seq_len_queries=seq_len_queries,
            seq_len_keys=seq_len_keys,
        )
        
        return keys, queries, attention_mask, previous_mask, tensor_dims

    def _benchmark_implementation(
        self,
        mode: str,
        masker: OracleTopK,
        tensor_dims: AttentionTensorDimensions,
        effective_heavy_size: int,
        keys: torch.Tensor,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
        previous_mask: Mask,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """Benchmark a specific implementation mode.
        
        Returns:
            Dictionary with timing statistics (mean, std, min, max)
        """
        # Warmup
        for _ in range(warmup_iterations):
            _ = masker.get_updated_mask(
                tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask, mode=mode
            )
        
        # Actual benchmark
        times: List[float] = []
        for _ in range(num_iterations):
            start_time: float = time.perf_counter()
            _ = masker.get_updated_mask(
                tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask, mode=mode
            )
            end_time: float = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        times_tensor: torch.Tensor = torch.tensor(times)
        return {
            "mean": float(torch.mean(times_tensor)),
            "std": float(torch.std(times_tensor)),
            "min": float(torch.min(times_tensor)),
            "max": float(torch.max(times_tensor)),
            "median": float(torch.median(times_tensor)),
        }

    @pytest.mark.performance
    def test_performance_small_sequences(self):
        """Test performance with small sequences (typical for early layers)."""
        batch_size: int = 2
        num_heads: int = 8
        seq_len: int = 512
        head_dim: int = 64
        heavy_size: float = 0.1
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads, num_heads, seq_len, seq_len, head_dim, "empty", False
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        
        print(f"\n{'='*70}")
        print(f"Small Sequences (batch={batch_size}, heads={num_heads}, seq_len={seq_len})")
        print(f"{'='*70}")
        print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
        print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
        speedup: float = old_stats['mean'] / new_stats['mean']
        print(f"Speedup: {speedup:.2f}x")
        
        # Log results
        assert old_stats['mean'] > 0
        assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_medium_sequences(self):
        """Test performance with medium sequences."""
        batch_size: int = 4
        num_heads: int = 16
        seq_len: int = 1024
        head_dim: int = 64
        heavy_size: float = 0.05
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads, num_heads, seq_len, seq_len, head_dim, "empty", False
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        
        print(f"\n{'='*70}")
        print(f"Medium Sequences (batch={batch_size}, heads={num_heads}, seq_len={seq_len})")
        print(f"{'='*70}")
        print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
        print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
        speedup: float = old_stats['mean'] / new_stats['mean']
        print(f"Speedup: {speedup:.2f}x")
        
        assert old_stats['mean'] > 0
        assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_large_sequences(self):
        """Test performance with large sequences."""
        batch_size: int = 1
        num_heads: int = 8
        seq_len: int = 2048
        head_dim: int = 64
        heavy_size: float = 0.02
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads, num_heads, seq_len, seq_len, head_dim, "empty", False
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask,
            num_iterations=50  # Fewer iterations for large sequences
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask,
            num_iterations=50
        )
        
        print(f"\n{'='*70}")
        print(f"Large Sequences (batch={batch_size}, heads={num_heads}, seq_len={seq_len})")
        print(f"{'='*70}")
        print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
        print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
        speedup: float = old_stats['mean'] / new_stats['mean']
        print(f"Speedup: {speedup:.2f}x")
        
        assert old_stats['mean'] > 0
        assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_gqa_scenario(self):
        """Test performance with GQA (different query/key heads)."""
        batch_size: int = 2
        num_heads_queries: int = 16
        num_heads_keys: int = 4
        seq_len: int = 1024
        head_dim: int = 64
        heavy_size: float = 0.05
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads_queries, num_heads_keys, seq_len, seq_len, head_dim, "empty", False
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        
        print(f"\n{'='*70}")
        print(f"GQA Scenario (batch={batch_size}, q_heads={num_heads_queries}, kv_heads={num_heads_keys}, seq_len={seq_len})")
        print(f"{'='*70}")
        print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
        print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
        speedup: float = old_stats['mean'] / new_stats['mean']
        print(f"Speedup: {speedup:.2f}x")
        
        assert old_stats['mean'] > 0
        assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_with_partial_mask(self):
        """Test performance with partial previous mask (realistic chaining scenario)."""
        batch_size: int = 2
        num_heads: int = 12
        seq_len: int = 1024
        head_dim: int = 64
        heavy_size: float = 0.05
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads, num_heads, seq_len, seq_len, head_dim, "partial", False
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        
        print(f"\n{'='*70}")
        print(f"Partial Mask (batch={batch_size}, heads={num_heads}, seq_len={seq_len})")
        print(f"{'='*70}")
        print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
        print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
        speedup: float = old_stats['mean'] / new_stats['mean']
        print(f"Speedup: {speedup:.2f}x")
        
        assert old_stats['mean'] > 0
        assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_comprehensive_comparison(self):
        """Comprehensive performance comparison across multiple configurations."""
        test_configs = [
            # (name, batch_size, num_heads, seq_len, heavy_size, mask_pattern)
            ("Tiny", 1, 4, 256, 0.1, "empty"),
            ("Small", 2, 8, 512, 0.08, "empty"),
            ("Medium", 4, 12, 1024, 0.05, "empty"),
            ("Large", 1, 8, 2048, 0.03, "empty"),
            ("XLarge", 1, 8, 4096, 0.02, "empty"),
            ("Small+Partial", 2, 8, 512, 0.08, "partial"),
            ("Medium+Partial", 2, 12, 1024, 0.05, "partial"),
        ]
        
        print(f"\n{'='*90}")
        print(f"COMPREHENSIVE PERFORMANCE COMPARISON")
        print(f"{'='*90}")
        print(f"{'Config':<20} {'Old (ms)':<15} {'New (ms)':<15} {'Speedup':<10}")
        print(f"{'-'*90}")
        
        total_old_time: float = 0.0
        total_new_time: float = 0.0
        
        for name, batch_size, num_heads, seq_len, heavy_size, mask_pattern in test_configs:
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)
            
            keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
                batch_size, num_heads, num_heads, seq_len, seq_len, 64, mask_pattern, False
            )
            
            effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len)
            
            num_iters: int = 50 if seq_len >= 2048 else 100
            
            old_stats: Dict[str, float] = self._benchmark_implementation(
                "old", masker, tensor_dims, effective_heavy_size, keys, queries, 
                attention_mask, previous_mask, num_iterations=num_iters
            )
            new_stats: Dict[str, float] = self._benchmark_implementation(
                "new", masker, tensor_dims, effective_heavy_size, keys, queries, 
                attention_mask, previous_mask, num_iterations=num_iters
            )
            
            speedup: float = old_stats['mean'] / new_stats['mean']
            total_old_time += old_stats['mean']
            total_new_time += new_stats['mean']
            
            print(f"{name:<20} {old_stats['mean']:>10.4f}     {new_stats['mean']:>10.4f}     {speedup:>6.2f}x")
        
        print(f"{'-'*90}")
        overall_speedup: float = total_old_time / total_new_time
        print(f"{'OVERALL':<20} {total_old_time:>10.4f}     {total_new_time:>10.4f}     {overall_speedup:>6.2f}x")
        print(f"{'='*90}\n")
        
        # Print recommendation
        if overall_speedup > 1.1:
            print(f"✓ RECOMMENDATION: Use 'new' mode as default (avg {overall_speedup:.2f}x faster)")
        elif overall_speedup < 0.9:
            print(f"✓ RECOMMENDATION: Use 'old' mode as default (avg {1/overall_speedup:.2f}x faster)")
        else:
            print(f"✓ RECOMMENDATION: Performance similar, choose 'new' for cleaner implementation")

