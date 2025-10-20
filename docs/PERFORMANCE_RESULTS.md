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

---

# LocalMasker and SinkMasker Implementation Performance

## Test Methodology

- **Warmup runs**: 5 iterations per implementation
- **Timing runs**: 20 iterations per implementation  
- **Timing method**: `time.perf_counter()` for high-precision measurements
- **Test configurations**: Small, Medium, and Large workloads
- **Hardware**: CPU benchmarks

## LocalMasker Performance Results

Compared 3 implementations:
1. **Sparse**: Original sparse implementation using `_create_local_mask`
2. **Dense1 (broadcast)**: Dense implementation using broadcasting operations
3. **Dense2 (triu)**: Dense implementation using `torch.triu_indices`

### Results Summary

| Config | Batch | Heads | Queries | Keys | Sparse (ms) | Dense1 (ms) | Dense2 (ms) | **Winner** | Speedup |
|--------|-------|-------|---------|------|-------------|-------------|-------------|------------|---------|
| Small  | 2     | 8     | 128     | 512  | 0.296       | 0.266       | **0.163**   | Dense2     | 1.81x   |
| Medium | 4     | 16    | 256     | 1024 | 1.743       | 1.771       | **1.696**   | Dense2     | 1.03x   |
| Large  | 8     | 32    | 512     | 2048 | 16.845      | 16.434      | **15.740**  | Dense2     | 1.07x   |

### LocalMasker Recommendation

**✅ Active: Dense2 (triu) - `update_core` method (DEFAULT)**

- Consistently fastest across all workload sizes
- Speedup ranges from 1.03x to 1.81x over sparse implementation
- Most efficient for small workloads (81% improvement)
- **Now set as default mode in `LocalMasker.add_mask()`**

**Implementation:**
```python
def update_core(self, dense_mask: torch.Tensor, effective_window_size: int) -> torch.Tensor:
    """Update the core of the masker using torch.triu_indices."""
    i, j = torch.triu_indices(
        dense_mask.shape[2], 
        dense_mask.shape[3], 
        dense_mask.shape[3] - dense_mask.shape[2] - effective_window_size + 1,
        device=dense_mask.device, 
        dtype=torch.long
    )
    dense_mask[..., i, j] = 1.0
    i, j = torch.triu_indices(
        dense_mask.shape[2], 
        dense_mask.shape[3], 
        dense_mask.shape[3] - dense_mask.shape[2] + 1, 
        device=dense_mask.device
    )
    dense_mask[..., i, j] = 0.0
    return dense_mask
```

## SinkMasker Performance Results

Compared 2 implementations:
1. **Sparse**: Original sparse implementation using `_create_sink_mask`
2. **Dense**: Dense implementation with direct indexing

### Results Summary

| Config | Batch | Heads | Queries | Keys | Sparse (ms) | Dense (ms) | **Winner** | Speedup |
|--------|-------|-------|---------|------|-------------|------------|------------|---------|
| Small  | 2     | 8     | 128     | 512  | 0.254       | **0.066**  | Dense      | 3.84x   |
| Medium | 4     | 16    | 256     | 1024 | 1.676       | **1.429**  | Dense      | 1.17x   |
| Large  | 8     | 32    | 512     | 2048 | 13.160      | **12.102** | Dense      | 1.09x   |

### SinkMasker Recommendation

**✅ Active: Dense implementation - `get_updated_mask_dense` method (DEFAULT)**

- Consistently fastest across all workload sizes
- Speedup ranges from 1.09x to 3.84x over sparse implementation  
- Exceptionally efficient for small workloads (284% improvement)
- **Now set as default mode in `SinkMasker.add_mask()`**

**Implementation:**
```python
def get_updated_mask_dense(
    self, 
    tensor_dims: AttentionTensorDimensions, 
    effective_sink_size: int, 
    keys: torch.Tensor, 
    previous_mask: Mask
) -> Mask:
    """Dense implementation for sink masking."""
    mask = previous_mask.get_dense_mask()
    mask[..., :effective_sink_size] = 1.0
    return Mask.create_mask_from_dense_mask(mask.shape, mask, previous_mask.dtype)
```

## Key Insights

1. **Dense implementations outperform sparse** for both LocalMasker and SinkMasker
2. **Small workloads benefit most** from dense implementations (up to 3.84x speedup)
3. **torch.triu_indices is more efficient** than broadcasting for LocalMasker
4. **Direct indexing is highly efficient** for SinkMasker's simple pattern

## Running the Benchmarks

To reproduce these results:

```bash
pytest tests/performance/test_basic_fixed_performance.py -v -s
```

For individual tests:

```bash
# LocalMasker performance
pytest tests/performance/test_basic_fixed_performance.py::TestLocalMaskerPerformance::test_local_masker_performance -v -s

# SinkMasker performance
pytest tests/performance/test_basic_fixed_performance.py::TestSinkMaskerPerformance::test_sink_masker_performance -v -s
```

## Default Mode Changes

Based on the performance benchmarks above, the following changes have been implemented:

### LocalMasker
- **Old default**: Sparse mode (`get_updated_mask_1`)
- **New default**: Dense2/triu mode (`get_updated_mask_3` using `update_core`)
- **Impact**: Up to 1.8x performance improvement

Users can still explicitly use sparse mode by calling:
```python
# Override to use sparse mode if needed
local_masker.get_updated_mask(..., mode="sparse")
```

### SinkMasker
- **Old default**: Sparse mode (`get_updated_mask_sparse`)
- **New default**: Dense mode (`get_updated_mask_dense`)
- **Impact**: Up to 3.8x performance improvement

Users can still explicitly use sparse mode by calling:
```python
# Override to use sparse mode if needed
sink_masker.get_updated_mask(..., mode="sparse")
```

### Backward Compatibility

The API remains fully backward compatible. Users who were explicitly specifying modes will see no changes. The performance improvements apply automatically to code using the default behavior.

---

*Last updated: Performance benchmarks for LocalMasker and SinkMasker implementations, with defaults updated to use fastest implementations*

