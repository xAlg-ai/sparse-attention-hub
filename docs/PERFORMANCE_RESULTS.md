# Performance Benchmarks for Long Context Sparse Attention

## 📊 Executive Summary (CUDA Results)

All tests run on CUDA with the long-context configuration: **B=1, H=32, Q=1, CTX=32768**

| Component | Winner | Speedup | Recommendation |
|-----------|--------|---------|----------------|
| **apply_mask** | Dense | 6-7x faster | Use dense mode (default) |
| **apply_inv_mask** | Dense | 1.6x faster | Use dense mode (default) |
| **merge_mask** | Dense | 25-75x faster | Use dense mode (default) |
| **LocalMasker** | Dense2 (triu) | 2.2-2.5x faster | Use Dense2 mode (default) |
| **SinkMasker** | Dense | ~5x faster | Use dense mode (default) |
| **create_from_row_wise_idx** | Dense | 3-5x faster | Use dense mode (default) |
| **OracleTopK** | New impl | 1.05-1.06x faster | Use new implementation |

**Key Finding**: Dense mode operations consistently outperform sparse operations on CUDA by 2-75x depending on the operation, making dense mode the optimal default for GPU workloads.

---

## Test Configuration

**Long Context Scenario:**
- Batch size (B) = 1
- Number of heads (H) = 32  
- Query length (Q) = 1
- Key length (K) = 32768
- Hidden dimension (D) = 128
- Device: CUDA (with CPU fallback)

This configuration represents a typical long-context attention scenario, such as incremental decoding where we attend to a long context.

**Test Date:** October 20, 2025  
**Hardware:** NVIDIA GPU with CUDA support  
**PyTorch Version:** Latest with CUDA synchronization

## Benchmark Methodology

To ensure fair comparison, masks are prepared in their optimal internal representations before benchmarking:
- **Sparse mode**: Uses index representation (`from_index=True`) with indices, ptr, and data arrays
- **Dense mode**: Uses dense tensor representation (`from_dense_mask=True`)

This eliminates conversion overhead and provides a true performance comparison of the operations themselves.

## Benchmark Results

### 1. `apply_mask` Operation

| Sparsity Level | Sparse Mode (ms) | Dense Mode (ms) | Speedup | Winner |
|----------------|------------------|-----------------|---------|--------|
| 10% (Very Dense) | 0.0410 | 0.0065 | **6.3x** | Dense |
| 50% (Medium) | 0.0430 | 0.0064 | **6.7x** | Dense |
| 90% (Very Sparse) | 0.0421 | 0.0070 | **6.0x** | Dense |

**Conclusion:** Dense mode is consistently **6-7x faster** across all sparsity levels.

### 2. `apply_inv_mask` Operation

| Sparsity Level | Sparse Mode (ms) | Dense Mode (ms) | Speedup | Winner |
|----------------|------------------|-----------------|---------|--------|
| 10% (Very Dense) | 0.0705 | 0.0448 | **1.6x** | Dense |
| 50% (Medium) | 0.0709 | 0.0449 | **1.6x** | Dense |
| 90% (Very Sparse) | 0.0710 | 0.0439 | **1.6x** | Dense |

**Conclusion:** Dense mode is consistently **1.6x faster** across all sparsity levels.

### 3. `merge_mask` Operation

| Sparsity Level | Sparse Mode (ms) | Dense Mode (ms) | Speedup | Winner |
|----------------|------------------|-----------------|---------|--------|
| 10% (Very Dense) | 1.283 | 0.0172 | **74.6x** | Dense |
| 50% (Medium) | 0.971 | 0.0169 | **57.5x** | Dense |
| 90% (Very Sparse) | 0.428 | 0.0169 | **25.3x** | Dense |

**Conclusion:** Dense mode is **25-75x faster** depending on sparsity level, with larger speedups for denser masks.

## Summary

For long context scenarios (K=32768), **dense mode significantly outperforms sparse mode** across all operations and sparsity levels:

- **apply_mask**: Dense is 6-7x faster
- **apply_inv_mask**: Dense is 1.6x faster  
- **merge_mask**: Dense is 25-75x faster

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
# All performance tests
pytest tests/performance/ -v -s

# Individual test suites
pytest tests/performance/test_long_context_timing.py -v -s
pytest tests/performance/test_create_from_row_wise_idx_performance.py -v -s
pytest tests/performance/test_basic_fixed_performance.py -v -s
pytest tests/performance/test_oracle_topk_get_updated_mask_performance.py -v -s
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

## LocalMasker Performance Results (CUDA)

Compared 3 implementations on CUDA with long context (B=1, H=32, Q=1, CTX=32768):
1. **Sparse**: Original sparse implementation using `_create_local_mask`
2. **Dense1 (broadcast)**: Dense implementation using broadcasting operations
3. **Dense2 (triu)**: Dense implementation using `torch.triu_indices`

### Results Summary

| Window Size | Sparse (ms) | Dense1 (ms) | Dense2 (ms) | **Winner** | Speedup |
|-------------|-------------|-------------|-------------|------------|---------|
| 128         | 0.328       | 0.175       | **0.131**   | Dense2     | 2.51x   |
| 256         | 0.264       | 0.158       | **0.118**   | Dense2     | 2.23x   |
| 512         | 0.281       | 0.159       | **0.111**   | Dense2     | 2.52x   |

### LocalMasker Recommendation

**✅ Active: Dense2 (triu) - `update_core` method (DEFAULT)**

- Consistently fastest across all window sizes on CUDA
- Speedup ranges from 2.23x to 2.52x over sparse implementation  
- Provides consistent ~2.5x performance improvement for long context scenarios
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

## SinkMasker Performance Results (CUDA)

Compared 2 implementations on CUDA with long context (B=1, H=32, Q=1, CTX=32768):
1. **Sparse**: Original sparse implementation using `_create_sink_mask`
2. **Dense**: Dense implementation with direct indexing

### Results Summary

| Sink Size | Sparse (ms) | Dense (ms) | **Winner** | Speedup |
|-----------|-------------|------------|------------|---------|
| 32        | 0.205       | **0.038**  | Dense      | 5.32x   |
| 64        | 0.202       | **0.040**  | Dense      | 5.02x   |
| 128       | 0.205       | **0.041**  | Dense      | 4.95x   |

### SinkMasker Recommendation

**✅ Active: Dense implementation - `get_updated_mask_dense` method (DEFAULT)**

- Consistently fastest across all sink sizes on CUDA
- Speedup is approximately **5x** over sparse implementation  
- Exceptionally efficient for long context scenarios
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

1. **Dense implementations outperform sparse** for both LocalMasker and SinkMasker on CUDA
2. **Long context scenarios benefit significantly** from dense implementations
   - LocalMasker: 2.2-2.5x speedup
   - SinkMasker: ~5x speedup
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

# create_from_row_wise_idx Performance Benchmarks

## Test Methodology

- **Warmup runs**: 5 iterations per implementation
- **Timing runs**: 20 iterations per implementation
- **Timing method**: `time.perf_counter()` with CUDA synchronization for high-precision measurements
- **Test scenario**: Long-context attention (B=1, H=32, Q=1, K=32768)
- **Hardware**: CUDA GPU benchmarks

## Overview

The `create_from_row_wise_idx` method has two implementation modes:
1. **Sparse mode**: Creates the mask in index/sparse representation first, then optionally converts to dense
2. **Dense mode**: Creates the mask in dense representation using `scatter_`, then optionally converts to index

## Performance Results Summary

### Test Configuration

**Long Context Attention Scenario:**
- Batch size (B) = 1
- Number of heads (H) = 32  
- Query length (Q) = 1
- Key length (K) = 32768
- Effective mask shape = (32, 32768)

This represents a typical long-context attention scenario, such as incremental decoding where we attend to a long context.

### Key Finding (CUDA)

**On CUDA, dense mode consistently outperforms sparse mode:**

| Output Type | Winner | Performance Characteristics |
|-------------|--------|------------------------------|
| **`mask_type="dense"`** | Dense mode | 3.34x - 5.00x faster (100% win rate) |
| **`mask_type="index"`** | Dense mode | 2.29x - 3.05x faster (100% win rate) |

**🏆 Recommendation: For CUDA/GPU workloads, dense mode is strongly preferred**
- Dense mode is **3-4x faster** across all sparsity levels for both output types
- Use `mode="dense"` as default for CUDA operations (current default)
- Sparse mode may still be useful for CPU-only workloads or extremely sparse cases

### Detailed Results: Long Context Attention (Shape: 32×32768)

#### For `mask_type="dense"` (Creating Dense Masks) - CUDA Results

| Sparsity | Tokens (k) | Sparse (ms) | Dense (ms) | Speedup | Winner |
|----------|------------|-------------|------------|---------|--------|
| 0.1% | 32 | 0.156 | 0.037 | **4.22x** | Dense |
| 0.2% | 64 | 0.159 | 0.038 | **4.22x** | Dense |
| 0.4% | 128 | 0.160 | 0.036 | **4.42x** | Dense |
| 1% | 328 | 0.159 | 0.036 | **4.38x** | Dense |
| 2% | 656 | 0.153 | 0.035 | **4.42x** | Dense |
| 5% | 1638 | 0.163 | 0.033 | **5.00x** | Dense |
| 10% | 3276 | 0.153 | 0.035 | **4.39x** | Dense |
| 20% | 6553 | 0.151 | 0.034 | **4.44x** | Dense |
| 50% | 16384 | 0.153 | 0.037 | **4.19x** | Dense |
| 80% | 26214 | 0.157 | 0.042 | **3.76x** | Dense |
| 90% | 29491 | 0.152 | 0.043 | **3.53x** | Dense |
| 99% | 32440 | 0.152 | 0.046 | **3.34x** | Dense |

**Summary**: Dense mode wins 12/12 configurations (100%), with 3.34x - 5.00x speedup

**Key Insights:**
- Consistent 3.3x - 5.0x speedup across all sparsity levels on CUDA
- Peak performance at 5% sparsity (5.0x speedup)
- Even at 99% density, dense mode is still 3.34x faster

#### For `mask_type="index"` (Creating Sparse/Index Masks) - CUDA Results

| Sparsity | Tokens (k) | Sparse (ms) | Dense (ms) | Speedup | Winner |
|----------|------------|-------------|------------|---------|--------|
| 0.1% | 32 | 0.103 | 0.037 | **2.75x** | Dense |
| 0.2% | 64 | 0.105 | 0.039 | **2.73x** | Dense |
| 0.4% | 128 | 0.104 | 0.036 | **2.92x** | Dense |
| 1% | 328 | 0.099 | 0.034 | **2.94x** | Dense |
| 2% | 656 | 0.107 | 0.037 | **2.85x** | Dense |
| 5% | 1638 | 0.104 | 0.034 | **3.05x** | Dense |
| 10% | 3276 | 0.098 | 0.033 | **2.97x** | Dense |
| 20% | 6553 | 0.098 | 0.033 | **2.99x** | Dense |
| 50% | 16384 | 0.096 | 0.035 | **2.71x** | Dense |
| 80% | 26214 | 0.100 | 0.042 | **2.39x** | Dense |
| 90% | 29491 | 0.103 | 0.042 | **2.42x** | Dense |
| 99% | 32440 | 0.099 | 0.043 | **2.29x** | Dense |

**Summary**: Dense mode wins 12/12 configurations (100%), with 2.29x - 3.05x speedup

**Key Insights:**
- Consistent 2.3x - 3.0x speedup across all sparsity levels on CUDA
- Dense mode performs well even when outputting index format
- Peak performance at 5% sparsity (3.05x speedup)

## Detailed Sparsity Impact Analysis - CUDA

Fixed configuration: Long context attention (B=1, H=32, Q=1, K=32768)  
Mask shape = (32, 32768), mask_type = "dense"

| Sparsity | k (tokens) | Sparse (ms) | Dense (ms) | Speedup | Winner |
|----------|------------|-------------|------------|---------|--------|
| 0.1% | 32 | 0.155 | 0.034 | **4.53x** | Dense |
| 0.2% | 64 | 0.153 | 0.035 | **4.33x** | Dense |
| 0.5% | 164 | 0.157 | 0.033 | **4.72x** | Dense |
| 1% | 328 | 0.153 | 0.032 | **4.82x** | Dense |
| 2% | 656 | 0.154 | 0.038 | **4.10x** | Dense |
| 5% | 1638 | 0.157 | 0.035 | **4.47x** | Dense |
| 10% | 3276 | 0.157 | 0.035 | **4.45x** | Dense |
| 20% | 6553 | 0.149 | 0.035 | **4.27x** | Dense |
| 30% | 9830 | 0.150 | 0.035 | **4.31x** | Dense |
| 50% | 16384 | 0.152 | 0.038 | **3.98x** | Dense |
| 70% | 22937 | 0.152 | 0.040 | **3.75x** | Dense |
| 80% | 26214 | 0.153 | 0.040 | **3.81x** | Dense |
| 90% | 29491 | 0.154 | 0.042 | **3.66x** | Dense |
| 95% | 31129 | 0.153 | 0.043 | **3.56x** | Dense |
| 99% | 32440 | 0.153 | 0.047 | **3.29x** | Dense |

**Key Insights**:
- Consistently 3.3x-4.8x faster on CUDA across all sparsity levels
- No anomalies observed - dense mode wins decisively across entire range
- Peak performance at 1% sparsity (4.82x speedup)
- Dense mode performance remains excellent even at 99% density (3.29x speedup)

## Key Insights (CUDA)

1. **Dense mode dominates on CUDA**: Unlike CPU benchmarks, dense mode consistently outperforms sparse mode on CUDA across all configurations and sparsity levels.

2. **For `mask_type="dense"` (dense mask output)**:
   - Dense mode wins 100% of configurations (12/12)
   - Speedup ranges from 3.34x to 5.00x across all sparsity levels
   - No anomalies or exceptions - consistent performance advantage

3. **For `mask_type="index"` (sparse/index mask output)**:
   - Dense mode still wins 100% of configurations (12/12)  
   - Speedup ranges from 2.29x to 3.05x across all sparsity levels
   - Even when outputting sparse format, dense mode is faster

4. **Consistent performance across sparsity levels**: Dense mode maintains 3-5x speedup regardless of whether the mask is 0.1% or 99% dense.

5. **CUDA optimization favors dense operations**: Dense mode leverages GPU parallelism better than sparse operations, with superior memory access patterns and vectorization.

## Why Dense Mode Wins for Dense Output

1. **Direct creation**: PyTorch's `scatter_` directly creates the dense mask without intermediate representations.

2. **Optimized scatter_ operation**: `scatter_` is highly optimized for dense tensor indexing.

3. **Contiguous memory**: Dense operations benefit from sequential memory access patterns that are cache-friendly.

4. **No conversion overhead**: When `mask_type="dense"`, the result is immediately ready without conversion.

## Why Sparse Mode Wins for Index Output

1. **Direct index creation**: Builds the sparse/index representation directly from `row_wise_idx` without intermediate steps.

2. **Avoids dense materialization**: Never creates a full dense tensor (32×32768 = ~1M elements), saving memory and computation.

3. **Constant complexity**: Processing time proportional to number of non-zero elements (k), not total matrix size (32768).

4. **Efficient for sparse attention**: Ideal for sparse attention patterns where k << K (e.g., 328 tokens out of 32768).

## Recommendation

**Choose mode based on your output needs:**

| Your Use Case | Recommended Mode | Why |
|---------------|------------------|-----|
| Need dense mask output (`mask_type="dense"`) | `mode="dense"` | 1.8x-5.8x faster (except 2-5% sparsity) |
| Need sparse/index output (`mask_type="index"`) | `mode="sparse"` | Consistently faster & stable performance |
| Unsure / Mixed usage | `mode="sparse"` (default) | Safe choice, reasonable for both cases |

**Current default** (`mode="sparse"`) is well-chosen because:
- It's the natural default for sparse attention use cases
- It performs well for index output (83% win rate)
- It's still reasonable for dense output (only 1.5-3x slower, not catastrophic)
- Sparse attention libraries typically work with sparse/index representations internally

## Running the Benchmarks

To reproduce these results:

```bash
# Full performance benchmark (24 configurations: 12 sparsity levels × 2 mask types)
pytest tests/performance/test_create_from_row_wise_idx_performance.py::TestCreateFromRowWiseIdxPerformance::test_create_from_row_wise_idx_performance -v -s

# Detailed sparsity analysis (15 sparsity levels from 0.1% to 99%)
pytest tests/performance/test_create_from_row_wise_idx_performance.py::TestCreateFromRowWiseIdxPerformance::test_create_from_row_wise_idx_detailed_analysis -v -s
```

## Stress Testing

Comprehensive stress tests verify that both modes produce **identical results** across:
- 7 different shape configurations (2D, 3D, 4D)
- 2 mask types (dense, index)
- 2 dtypes (float32, float64)
- Various edge cases (duplicates, high density, small values, large batch dimensions)

### Duplicate Index Handling

Both modes handle duplicate indices identically:
- **Last value wins** (same as PyTorch's `scatter_` behavior)
- Example: `indices=[[0,0,0]]`, `data=[[1.0,2.0,3.0]]` → position 0 gets value 3.0

Run stress tests:

```bash
# Main stress test (28 test cases)
pytest tests/unit/sparse_attention/utils/test_mask.py::TestMask::test_create_from_row_wise_idx_sparse_dense_equivalence -v -s

# Edge cases (duplicates, high density, etc.)
pytest tests/unit/sparse_attention/utils/test_mask.py::TestMask::test_create_from_row_wise_idx_sparse_dense_edge_cases -v -s
```

All stress tests pass ✅, confirming both modes are functionally equivalent.

---

# OracleTopK get_updated_mask Performance Benchmarks

## Test Methodology

- **Warmup runs**: 10 iterations per implementation  
- **Timing runs**: 100 iterations per implementation (50 for varying heavy_size test)
- **Timing method**: `time.perf_counter()` with CUDA synchronization
- **Test scenario**: Long-context attention (B=1, H=32, Q=1, K=32768)
- **Hardware**: CUDA GPU benchmarks

## Overview

The OracleTopK masker has two implementations:
1. **Old implementation**: Original implementation
2. **New implementation**: Optimized implementation with improved memory access patterns

## Performance Results (CUDA)

### Long Context Scenario (B=1, H=32, Q=1, CTX=32768)

| Test Configuration | Old (ms) | New (ms) | Speedup | Winner |
|--------------------|----------|----------|---------|--------|
| Heavy 10% | 0.4795 | 0.4547 | **1.05x** | New |
| Heavy 5% + Attention Mask | 0.4776 | 0.4555 | **1.05x** | New |
| Heavy 2% | 0.4723 | 0.4515 | **1.05x** | New |
| Heavy 5% | 0.4773 | 0.4563 | **1.05x** | New |
| Heavy 10% (repeat) | 0.4765 | 0.4545 | **1.05x** | New |
| GQA (Q_heads=32, KV_heads=8) | 0.8335 | 0.8088 | **1.03x** | New |
| Partial Mask | 0.4640 | 0.4367 | **1.06x** | New |

### Comprehensive Comparison

| Configuration | Old (ms) | New (ms) | Speedup |
|---------------|----------|----------|---------|
| Heavy 2% | 0.4736 | 0.4514 | 1.05x |
| Heavy 5% | 0.4733 | 0.4515 | 1.05x |
| Heavy 10% | 0.4761 | 0.4542 | 1.05x |
| Heavy 5%+Partial | 0.4642 | 0.4372 | 1.06x |
| Heavy 10%+Partial | 0.4680 | 0.4378 | 1.07x |
| **OVERALL** | **2.3551** | **2.2320** | **1.06x** |

## Key Insights

1. **Consistent modest improvement**: New implementation provides a steady ~5-6% speedup across all configurations
2. **Partial mask benefits most**: Configurations with partial masks see slightly better improvements (1.06-1.07x)
3. **GQA scenario**: Even with grouped query attention, new implementation is ~3% faster
4. **Stable across heavy_size**: Performance improvement is consistent regardless of heavy_size parameter (2%, 5%, 10%)

## Recommendation

✅ **New implementation is recommended as default** for OracleTopK
- Consistent 5-6% performance improvement
- No regressions observed across any configuration
- Better memory access patterns provide reliable speedup

## Running the Benchmarks

```bash
# All OracleTopK performance tests
pytest tests/performance/test_oracle_topk_get_updated_mask_performance.py -v -s

# Individual tests
pytest tests/performance/test_oracle_topk_get_updated_mask_performance.py::TestOracleTopKGetUpdatedMaskPerformance::test_performance_small_sequences -v -s
pytest tests/performance/test_oracle_topk_get_updated_mask_performance.py::TestOracleTopKGetUpdatedMaskPerformance::test_performance_comprehensive_comparison -v -s
```

---

*Last updated: October 20, 2025 - Complete CUDA performance benchmarks for all mask operations, maskers, and OracleTopK implementations*

