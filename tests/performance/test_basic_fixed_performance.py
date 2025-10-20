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
        """Compare performance of 3 LocalMasker implementations: B=1, Q=1, H=32, CTX=32768."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        # Setup device
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        # Long context attention configuration
        batch_size: int = 1
        num_heads: int = 32
        seq_len_queries: int = 1
        seq_len_keys: int = 32768
        
        # Test configurations: (window_size, description)
        test_configs: list = [
            (128, "Window 128"),
            (256, "Window 256"),
            (512, "Window 512"),
        ]

        num_warmup: int = 5
        num_iterations: int = 20

        print("\n" + "="*80)
        print("LocalMasker Performance Comparison")
        print(f"Long Context: B={batch_size}, H={num_heads}, Q={seq_len_queries}, CTX={seq_len_keys}")
        print("="*80)

        for window_size, desc in test_configs:
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
                        mask_shape, dtype=torch.float32, device=device
                    )
                    _ = local_masker.get_updated_mask(
                        tensor_dims, effective_window_size, keys, empty_mask, mode=mode
                    )
                
                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Actual timing
                for _ in range(num_iterations):
                    empty_mask = Mask.create_empty_mask(
                        mask_shape, dtype=torch.float32, device=device
                    )
                    
                    start_time: float = time.perf_counter()
                    result: Mask = local_masker.get_updated_mask(
                        tensor_dims, effective_window_size, keys, empty_mask, mode=mode
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
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
        """Compare performance of 2 SinkMasker implementations: B=1, Q=1, H=32, CTX=32768."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker,
            SinkMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        # Setup device
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        # Long context attention configuration
        batch_size: int = 1
        num_heads: int = 32
        seq_len_queries: int = 1
        seq_len_keys: int = 32768
        
        # Test configurations: (sink_size, description)
        test_configs: list = [
            (32, "Sink 32"),
            (64, "Sink 64"),
            (128, "Sink 128"),
        ]

        num_warmup: int = 5
        num_iterations: int = 20

        print("\n" + "="*80)
        print("SinkMasker Performance Comparison")
        print(f"Long Context: B={batch_size}, H={num_heads}, Q={seq_len_queries}, CTX={seq_len_keys}")
        print("="*80)

        for sink_size, desc in test_configs:
            print(f"\n{desc}: batch={batch_size}, heads={num_heads}, "
                  f"queries={seq_len_queries}, keys={seq_len_keys}, sink={sink_size}")
            print("-" * 80)

            head_dim: int = 128
            
            # Create mock inputs on device
            keys: torch.Tensor = torch.randn(batch_size, num_heads, seq_len_keys, head_dim, device=device)
            queries: torch.Tensor = torch.randn(batch_size, num_heads, seq_len_queries, head_dim, device=device)
            
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
                        mask_shape, dtype=torch.float32, device=device
                    )
                    _ = sink_masker.get_updated_mask(
                        tensor_dims, effective_sink_size, keys, empty_mask, mode=mode
                    )
                
                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Actual timing
                for _ in range(num_iterations):
                    empty_mask = Mask.create_empty_mask(
                        mask_shape, dtype=torch.float32, device=device
                    )
                    
                    start_time: float = time.perf_counter()
                    result: Mask = sink_masker.get_updated_mask(
                        tensor_dims, effective_sink_size, keys, empty_mask, mode=mode
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
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


