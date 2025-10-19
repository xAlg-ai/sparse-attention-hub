# Performance Benchmarks for Long Context Sparse Attention

## Test Configuration

**Long Context Scenario:**
- Batch size (B) = 1
- Number of heads (H) = 32  
- Query length (Q) = 1
- Key length (K) = 32768

This configuration represents a typical long-context attention scenario, such as incremental decoding where we attend to a long context.

## Benchmark Methodology

To ensure fair comparison, masks are prepared in their optimal internal representations before benchmarking:
- **Sparse mode**: Uses index representation (`from_index=True`) with indices, ptr, and data arrays
- **Dense mode**: Uses dense tensor representation (`from_dense_mask=True`)

This eliminates conversion overhead and provides a true performance comparison of the operations themselves.

## Benchmark Results

### 1. `apply_mask` Operation

| Sparsity Level | Sparse Mode (ms) | Dense Mode (ms) | Speedup | Winner |
|----------------|------------------|-----------------|---------|--------|
| 10% (Very Dense) | 0.059 | 0.008 | **~7.4x** | Dense |
| 50% (Medium) | 0.056 | 0.009 | **~6.2x** | Dense |
| 90% (Very Sparse) | 0.051 | 0.008 | **~6.4x** | Dense |

**Conclusion:** Dense mode is consistently **~6-7x faster** across all sparsity levels.

### 2. `apply_inv_mask` Operation

| Sparsity Level | Sparse Mode (ms) | Dense Mode (ms) | Speedup | Winner |
|----------------|------------------|-----------------|---------|--------|
| 10% (Very Dense) | 0.095 | 0.058 | **~1.6x** | Dense |
| 50% (Medium) | 0.090 | 0.056 | **~1.6x** | Dense |
| 90% (Very Sparse) | 0.086 | 0.052 | **~1.7x** | Dense |

**Conclusion:** Dense mode is consistently **~1.6-1.7x faster** across all sparsity levels.

### 3. `merge_mask` Operation

| Sparsity Level | Sparse Mode (ms) | Dense Mode (ms) | Speedup | Winner |
|----------------|------------------|-----------------|---------|--------|
| 10% (Very Dense) | ~1.6 | 0.020 | **~80x** | Dense |
| 50% (Medium) | ~0.97 | 0.018 | **~54x** | Dense |
| 90% (Very Sparse) | ~0.76 | 0.019 | **~40x** | Dense |

**Conclusion:** Dense mode is **~40-80x faster** depending on sparsity level, with larger speedups for denser masks.

## Summary

For long context scenarios (K=32768), **dense mode significantly outperforms sparse mode** across all operations and sparsity levels:

- **apply_mask**: Dense is ~6-7x faster
- **apply_inv_mask**: Dense is ~1.6-1.7x faster  
- **merge_mask**: Dense is ~40-80x faster

### Why Dense Mode is Faster for Long Context

Even with masks pre-converted to their optimal representations, dense mode outperforms sparse mode due to:

1. **Contiguous Memory Access**: Dense operations benefit from sequential memory access patterns that are cache-friendly
2. **Better Vectorization**: GPUs can efficiently parallelize dense matrix operations across contiguous memory
3. **Scatter/Gather Overhead**: Sparse operations require index lookups and scattered memory writes, which have inherent overhead even with optimized representations
4. **Large K Dimension**: With K=32768, the long sequence dimension makes dense vectorized operations highly efficient, while sparse operations must process each non-zero element individually
5. **Memory Bandwidth**: For moderately sparse masks (10-90% sparsity), dense operations can leverage full memory bandwidth, while sparse operations are limited by irregular access patterns

**Note**: Sparse mode may still be beneficial for:
- Extremely sparse masks (>99% sparsity) with very large K
- Memory-constrained environments where storing dense masks is prohibitive
- Specific hardware architectures optimized for sparse operations

### Default Mode Selection

Based on these results, **dense mode has been set as the default** for all mask operations in the codebase:

```python
def apply_mask(self, input_tensor: torch.Tensor, mode: Literal["sparse", "dense"] = "dense")
def apply_inv_mask(self, input_tensor: torch.Tensor, mode: Literal["sparse", "dense"] = "dense")  
def merge_mask(self, other_mask: "Mask", inplace: bool, mode: Literal["sparse", "dense"] = "dense")
```

Users can still explicitly specify `mode="sparse"` if needed for specific use cases where sparsity might provide benefits (e.g., extremely sparse masks with very small matrices).

## Running the Benchmarks

To reproduce these results:

```bash
pytest tests/performance/test_long_context_timing.py -v -s
```

For a comprehensive summary:

```bash
pytest tests/performance/test_long_context_timing.py::TestLongContextTiming::test_comprehensive_timing_summary -v -s
```

## Hardware Configuration

Tests were run on:
- GPU: NVIDIA GPU with CUDA support
- Framework: PyTorch with CUDA backend
- Iterations: 100 benchmark iterations with 10 warmup iterations per test

## Implementation Details

The benchmark suite (`tests/performance/test_long_context_timing.py`) includes:
- **Fair representation handling**: Masks are pre-converted to their optimal representations (index for sparse, dense for dense) before benchmarking
- **Proper synchronization**: CUDA synchronization before and after timing to ensure accurate measurements
- **Comprehensive coverage**: Tests across 10%, 50%, and 90% sparsity levels
- **Automated recommendations**: Reports which mode is faster for each operation

This ensures that the benchmarks measure the true performance of the operations themselves, not conversion overhead.

