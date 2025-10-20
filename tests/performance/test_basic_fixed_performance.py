"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-10-20
:summary: Performance tests comparing different implementations of LocalMasker and SinkMasker.
"""

import time
import pytest
import torch


@pytest.mark.performance
class TestLocalMaskerPerformance:
    """Performance tests for LocalMasker implementations."""

    def test_local_masker_performance(self):
        """Compare performance of 3 LocalMasker implementations."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        # Test configurations: (batch_size, num_heads, seq_len_queries, seq_len_keys, description)
        test_configs: list = [
            (2, 8, 128, 512, "Small"),
            (4, 16, 256, 1024, "Medium"),
            (8, 32, 512, 2048, "Large"),
        ]

        window_size: int = 64
        num_warmup: int = 5
        num_iterations: int = 20

        print("\n" + "="*80)
        print("LocalMasker Performance Comparison")
        print("="*80)

        for batch_size, num_heads, seq_len_queries, seq_len_keys, desc in test_configs:
            print(f"\n{desc} Config: batch={batch_size}, heads={num_heads}, "
                  f"queries={seq_len_queries}, keys={seq_len_keys}")
            print("-" * 80)

            head_dim: int = 64
            
            # Create mock inputs
            keys: torch.Tensor = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
            queries: torch.Tensor = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
            
            local_config: LocalMaskerConfig = LocalMaskerConfig(window_size=window_size)
            local_masker: LocalMasker = LocalMasker(local_config)
            
            mask_shape: tuple = (batch_size, num_heads, seq_len_queries, seq_len_keys)
            
            tensor_dims = local_masker._extract_tensor_dimensions(keys, queries)
            effective_window_size: int = local_masker._calculate_effective_window_size(
                tensor_dims.seq_len_keys
            )
            
            # Skip if would use full attention
            if local_masker._should_use_full_attention(tensor_dims, effective_window_size):
                print(f"Skipping - would use full attention")
                continue
            
            results: dict = {}
            
            # Test each implementation
            for mode_name, mode in [("Sparse", "sparse"), ("Dense1 (broadcast)", "dense1"), ("Dense2 (triu)", "dense2")]:
                times: list = []
                
                # Warmup
                for _ in range(num_warmup):
                    empty_mask: Mask = Mask.create_empty_mask(
                        mask_shape, dtype=torch.float32, device=torch.device("cpu")
                    )
                    _ = local_masker.get_updated_mask(
                        tensor_dims, effective_window_size, keys, empty_mask, mode=mode
                    )
                
                # Actual timing
                for _ in range(num_iterations):
                    empty_mask = Mask.create_empty_mask(
                        mask_shape, dtype=torch.float32, device=torch.device("cpu")
                    )
                    
                    start_time: float = time.perf_counter()
                    result: Mask = local_masker.get_updated_mask(
                        tensor_dims, effective_window_size, keys, empty_mask, mode=mode
                    )
                    end_time: float = time.perf_counter()
                    
                    times.append(end_time - start_time)
                
                avg_time: float = sum(times) / len(times)
                min_time: float = min(times)
                max_time: float = max(times)
                std_time: float = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
                
                results[mode_name] = {
                    'avg': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'std': std_time
                }
                
                print(f"{mode_name:20s}: avg={avg_time*1000:7.3f}ms, "
                      f"min={min_time*1000:7.3f}ms, max={max_time*1000:7.3f}ms, "
                      f"std={std_time*1000:7.3f}ms")
            
            # Find fastest
            fastest: str = min(results.items(), key=lambda x: x[1]['avg'])[0]
            print(f"\n{'FASTEST':20s}: {fastest}")
            
            # Calculate speedup relative to sparse
            if fastest != "Sparse":
                speedup: float = results["Sparse"]['avg'] / results[fastest]['avg']
                print(f"{'Speedup vs Sparse':20s}: {speedup:.2f}x")

        print("\n" + "="*80)


@pytest.mark.performance
class TestSinkMaskerPerformance:
    """Performance tests for SinkMasker implementations."""

    def test_sink_masker_performance(self):
        """Compare performance of 2 SinkMasker implementations."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker,
            SinkMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        # Test configurations: (batch_size, num_heads, seq_len_queries, seq_len_keys, description)
        test_configs: list = [
            (2, 8, 128, 512, "Small"),
            (4, 16, 256, 1024, "Medium"),
            (8, 32, 512, 2048, "Large"),
        ]

        sink_size: int = 32
        num_warmup: int = 5
        num_iterations: int = 20

        print("\n" + "="*80)
        print("SinkMasker Performance Comparison")
        print("="*80)

        for batch_size, num_heads, seq_len_queries, seq_len_keys, desc in test_configs:
            print(f"\n{desc} Config: batch={batch_size}, heads={num_heads}, "
                  f"queries={seq_len_queries}, keys={seq_len_keys}")
            print("-" * 80)

            head_dim: int = 64
            
            # Create mock inputs
            keys: torch.Tensor = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
            queries: torch.Tensor = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
            
            sink_config: SinkMaskerConfig = SinkMaskerConfig(sink_size=sink_size)
            sink_masker: SinkMasker = SinkMasker(sink_config)
            
            mask_shape: tuple = (batch_size, num_heads, seq_len_queries, seq_len_keys)
            
            tensor_dims = sink_masker._extract_tensor_dimensions(keys, queries)
            effective_sink_size: int = sink_masker._calculate_effective_sink_size(
                tensor_dims.seq_len_keys
            )
            
            # Skip if would use full attention
            if sink_masker._should_use_full_attention(tensor_dims, effective_sink_size):
                print(f"Skipping - would use full attention")
                continue
            
            results: dict = {}
            
            # Test each implementation
            for mode_name, mode in [("Sparse", "sparse"), ("Dense", "dense")]:
                times: list = []
                
                # Warmup
                for _ in range(num_warmup):
                    empty_mask: Mask = Mask.create_empty_mask(
                        mask_shape, dtype=torch.float32, device=torch.device("cpu")
                    )
                    _ = sink_masker.get_updated_mask(
                        tensor_dims, effective_sink_size, keys, empty_mask, mode=mode
                    )
                
                # Actual timing
                for _ in range(num_iterations):
                    empty_mask = Mask.create_empty_mask(
                        mask_shape, dtype=torch.float32, device=torch.device("cpu")
                    )
                    
                    start_time: float = time.perf_counter()
                    result: Mask = sink_masker.get_updated_mask(
                        tensor_dims, effective_sink_size, keys, empty_mask, mode=mode
                    )
                    end_time: float = time.perf_counter()
                    
                    times.append(end_time - start_time)
                
                avg_time: float = sum(times) / len(times)
                min_time: float = min(times)
                max_time: float = max(times)
                std_time: float = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
                
                results[mode_name] = {
                    'avg': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'std': std_time
                }
                
                print(f"{mode_name:20s}: avg={avg_time*1000:7.3f}ms, "
                      f"min={min_time*1000:7.3f}ms, max={max_time*1000:7.3f}ms, "
                      f"std={std_time*1000:7.3f}ms")
            
            # Find fastest
            fastest: str = min(results.items(), key=lambda x: x[1]['avg'])[0]
            print(f"\n{'FASTEST':20s}: {fastest}")
            
            # Calculate speedup relative to sparse
            if fastest != "Sparse":
                speedup: float = results["Sparse"]['avg'] / results[fastest]['avg']
                print(f"{'Speedup vs Sparse':20s}: {speedup:.2f}x")

        print("\n" + "="*80)


