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
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mask, AttentionTensorDimensions]:
        """Create test inputs for performance testing."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create inputs on device
        keys: torch.Tensor = torch.randn(batch_size, num_heads_keys, seq_len_keys, head_dim, device=device)
        queries: torch.Tensor = torch.randn(batch_size, num_heads_queries, seq_len_queries, head_dim, device=device)
        
        # Create attention mask if needed
        attention_mask: torch.Tensor = None
        if use_attention_mask:
            attention_mask = torch.zeros(batch_size, 1, seq_len_queries, seq_len_keys, device=device)
            for i in range(seq_len_queries):
                if i < seq_len_keys:
                    attention_mask[:, :, i, i+1:] = float('-inf')
        
        # Create previous mask
        mask_shape: tuple = (batch_size, num_heads_queries, seq_len_queries, seq_len_keys)
        if previous_mask_pattern == "empty":
            previous_mask: Mask = Mask.create_empty_mask(mask_shape, dtype=torch.float32, device=device)
        elif previous_mask_pattern == "partial":
            previous_mask_data: torch.Tensor = torch.zeros(mask_shape, device=device)
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
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Actual benchmark
        times: List[float] = []
        for _ in range(num_iterations):
            start_time: float = time.perf_counter()
            _ = masker.get_updated_mask(
                tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask, mode=mode
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
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
        """Test performance with long context: B=1, Q=1, H=32, CTX=32768."""
        batch_size: int = 1
        num_heads: int = 32
        seq_len_queries: int = 1
        seq_len_keys: int = 32768
        head_dim: int = 128
        heavy_size: float = 0.1
        
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads, num_heads, seq_len_queries, seq_len_keys, head_dim, "empty", False, device
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len_keys)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        
        print(f"\n{'='*70}")
        print(f"Long Context (batch={batch_size}, heads={num_heads}, queries={seq_len_queries}, keys={seq_len_keys})")
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
        """Test performance with long context and attention mask: B=1, Q=1, H=32, CTX=32768."""
        batch_size: int = 1
        num_heads: int = 32
        seq_len_queries: int = 1
        seq_len_keys: int = 32768
        head_dim: int = 128
        heavy_size: float = 0.05
        
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads, num_heads, seq_len_queries, seq_len_keys, head_dim, "empty", True, device
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len_keys)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        
        print(f"\n{'='*70}")
        print(f"Long Context with Attention Mask (batch={batch_size}, heads={num_heads}, queries={seq_len_queries}, keys={seq_len_keys})")
        print(f"{'='*70}")
        print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
        print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
        speedup: float = old_stats['mean'] / new_stats['mean']
        print(f"Speedup: {speedup:.2f}x")
        
        assert old_stats['mean'] > 0
        assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_varying_heavy_size(self):
        """Test performance with varying heavy_size: B=1, Q=1, H=32, CTX=32768."""
        batch_size: int = 1
        num_heads: int = 32
        seq_len_queries: int = 1
        seq_len_keys: int = 32768
        head_dim: int = 128
        
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        print(f"Long Context Varying Heavy Size: B={batch_size}, H={num_heads}, Q={seq_len_queries}, CTX={seq_len_keys}")
        
        for heavy_size in [0.02, 0.05, 0.1]:
            print(f"\n--- Testing heavy_size={heavy_size} ---")
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)
            
            keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
                batch_size, num_heads, num_heads, seq_len_queries, seq_len_keys, head_dim, "empty", False, device
            )
            
            effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len_keys)
        
            old_stats: Dict[str, float] = self._benchmark_implementation(
                "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask,
                num_iterations=50
            )
            new_stats: Dict[str, float] = self._benchmark_implementation(
                "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask,
                num_iterations=50
            )
            
            print(f"{'='*70}")
            print(f"Heavy Size {heavy_size} (batch={batch_size}, heads={num_heads}, queries={seq_len_queries}, keys={seq_len_keys})")
            print(f"{'='*70}")
            print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
            print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
            speedup: float = old_stats['mean'] / new_stats['mean']
            print(f"Speedup: {speedup:.2f}x")
            
            assert old_stats['mean'] > 0
            assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_gqa_scenario(self):
        """Test performance with GQA (different query/key heads): B=1, Q=1, H=32, CTX=32768."""
        batch_size: int = 1
        num_heads_queries: int = 32
        num_heads_keys: int = 8  # GQA: 4x fewer KV heads
        seq_len_queries: int = 1
        seq_len_keys: int = 32768
        head_dim: int = 128
        heavy_size: float = 0.05
        
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        print(f"GQA Long Context: B={batch_size}, Q_heads={num_heads_queries}, KV_heads={num_heads_keys}, Q={seq_len_queries}, CTX={seq_len_keys}")
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, "empty", False, device
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len_keys)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        
        print(f"\n{'='*70}")
        print(f"GQA Scenario (batch={batch_size}, q_heads={num_heads_queries}, kv_heads={num_heads_keys}, queries={seq_len_queries}, keys={seq_len_keys})")
        print(f"{'='*70}")
        print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
        print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
        speedup: float = old_stats['mean'] / new_stats['mean']
        print(f"Speedup: {speedup:.2f}x")
        
        assert old_stats['mean'] > 0
        assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_with_partial_mask(self):
        """Test performance with partial previous mask: B=1, Q=1, H=32, CTX=32768."""
        batch_size: int = 1
        num_heads: int = 32
        seq_len_queries: int = 1
        seq_len_keys: int = 32768
        head_dim: int = 128
        heavy_size: float = 0.05
        
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        print(f"Long Context with Partial Mask: B={batch_size}, H={num_heads}, Q={seq_len_queries}, CTX={seq_len_keys}")
        
        config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
        masker: OracleTopK = OracleTopK(config)
        
        keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
            batch_size, num_heads, num_heads, seq_len_queries, seq_len_keys, head_dim, "partial", False, device
        )
        
        effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len_keys)
        
        old_stats: Dict[str, float] = self._benchmark_implementation(
            "old", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        new_stats: Dict[str, float] = self._benchmark_implementation(
            "new", masker, tensor_dims, effective_heavy_size, keys, queries, attention_mask, previous_mask
        )
        
        print(f"\n{'='*70}")
        print(f"Partial Mask (batch={batch_size}, heads={num_heads}, queries={seq_len_queries}, keys={seq_len_keys})")
        print(f"{'='*70}")
        print(f"Old implementation: {old_stats['mean']:.4f} ± {old_stats['std']:.4f} ms")
        print(f"New implementation: {new_stats['mean']:.4f} ± {new_stats['std']:.4f} ms")
        speedup: float = old_stats['mean'] / new_stats['mean']
        print(f"Speedup: {speedup:.2f}x")
        
        assert old_stats['mean'] > 0
        assert new_stats['mean'] > 0

    @pytest.mark.performance
    def test_performance_comprehensive_comparison(self):
        """Comprehensive performance comparison for long context: B=1, Q=1, H=32, CTX=32768."""
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        # All tests use the same long context configuration
        batch_size: int = 1
        num_heads: int = 32
        seq_len_queries: int = 1
        seq_len_keys: int = 32768
        head_dim: int = 128
        
        test_configs = [
            # (name, heavy_size, mask_pattern)
            ("Heavy 2%", 0.02, "empty"),
            ("Heavy 5%", 0.05, "empty"),
            ("Heavy 10%", 0.1, "empty"),
            ("Heavy 5%+Partial", 0.05, "partial"),
            ("Heavy 10%+Partial", 0.1, "partial"),
        ]
        
        print(f"\n{'='*90}")
        print(f"COMPREHENSIVE PERFORMANCE COMPARISON")
        print(f"Long Context: B={batch_size}, H={num_heads}, Q={seq_len_queries}, CTX={seq_len_keys}")
        print(f"{'='*90}")
        print(f"{'Config':<20} {'Old (ms)':<15} {'New (ms)':<15} {'Speedup':<10}")
        print(f"{'-'*90}")
        
        total_old_time: float = 0.0
        total_new_time: float = 0.0
        
        for name, heavy_size, mask_pattern in test_configs:
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)
            
            keys, queries, attention_mask, previous_mask, tensor_dims = self._create_test_inputs(
                batch_size, num_heads, num_heads, seq_len_queries, seq_len_keys, head_dim, mask_pattern, False, device
            )
            
            effective_heavy_size: int = masker._calculate_effective_heavy_size(seq_len_keys)
            
            num_iters: int = 50
            
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

