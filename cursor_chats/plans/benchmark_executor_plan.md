# BenchmarkExecutor Implementation Plan

## Overview

This document outlines the design and implementation plan for a new `BenchmarkExecutor` class that orchestrates parallel benchmark execution across multiple GPUs using multiprocessing. The goal is to efficiently run combinations of models and benchmarks in parallel while managing GPU resources.

## Current State Analysis

### Existing Components
- **`Benchmark` (Abstract Base Class)**: Defines interface with `run_benchmark(adapter, result_dir)`
- **`ModelAdapterHF`**: HuggingFace adapter that creates models and handles sparse attention
- **Simple `BenchmarkExecutor`**: Single model/benchmark execution with result storage #aditya( I removed the executor.py code. we want to fresh start)
- **Available Benchmarks**: LongBench, MockBenchmark, AIME2024/2025, InfiniteBench, Loogle, etc.

### Current Limitations
- Sequential execution only
- No multi-GPU support
- No parallel model/benchmark combinations
- Limited resource management
- No resumable execution (re-runs completed benchmarks)

## Proposed Architecture

### Core Design Principles
1. **Minimal Resource Overhead**: Each process should only load one model at a time
2. **Dynamic GPU Allocation**: Processes acquire GPUs dynamically as they become available
3. **Fault Tolerance**: Failed benchmark runs shouldn't crash other processes
4. **Resumable Execution**: Skip benchmark stubs that already have results (directory exists)
5. **GPU Validation**: Ensure model adapters can properly initialize on assigned GPUs
6. **Progress Tracking**: Real-time monitoring of execution progress

### Class Interface

```python
class BenchmarkExecutor:
    """Orchestrates parallel benchmark execution across multiple GPUs."""
    
    def __init__(
        self,
        gpu_ids: List[int],
        max_concurrent_runs: int,
        base_result_dir: str = "./benchmark_results"
    ):
        """Initialize the executor.
        
        Args:
            gpu_ids: List of GPU IDs to use for execution (e.g., [0, 1, 2, 3])
            max_concurrent_runs: Maximum number of parallel processes to run across GPUs
            base_result_dir: Base directory for storing benchmark results
            
        Note:
            If OOM errors occur, user should reduce max_concurrent_runs.
            Processes will be distributed across all specified GPU IDs.
        """
    
    def run_benchmark_matrix(
        self,
        model_names: List[str],
        sparse_attention_configs: List[Tuple[str, Optional[SparseAttentionConfig]]],
        benchmark_configs: List[BenchmarkConfig], 
        adapter_config: AdapterConfig,
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> ExecutionResults:
        """Run all combinations of models, sparse attention configs, and benchmarks in parallel.
        
        Args:
            model_names: List of HuggingFace model names
            sparse_attention_configs: List of (config_name, config) tuples
                                    e.g., [("dense", None), ("streaming", streaming_config)]
            benchmark_configs: List of benchmark configurations 
            adapter_config: Base configuration for model adapters (without sparse attention)
            generation_kwargs: Generation parameters for model inference
            
        Returns:
            Aggregated results from all benchmark runs
        """
```

### Configuration Classes

```python
@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark execution."""
    benchmark_name: str  # e.g., "longbench", "mock_benchmark"
    subsets: Optional[List[str]] = None  # e.g., ["narrativeqa", "qasper"]
    
    # Note: During stub generation, this expands to individual BenchmarkStub per subset
    # Each BenchmarkStub will have a single subset value for resumability
    
@dataclass 
class AdapterConfig:
    """Base configuration for model adapters (sparse attention config is part of matrix)."""
    adapter_name: str = "huggingface"  # Future: support other adapters
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    # Note: sparse_attention_config is now part of the execution matrix, not base config

@dataclass
class BenchmarkStub:
    """A single benchmark execution task (model × sparse_config × benchmark × subset)."""
    model_name: str
    sparse_config_name: str  # e.g., "dense", "streaming", "hash_attention"
    sparse_attention_config: Optional[SparseAttentionConfig]  # None for dense
    benchmark_name: str  # e.g., "longbench"
    subset: Optional[str]  # e.g., "narrativeqa" or None for full benchmark
    adapter_config: AdapterConfig  # Base config without sparse attention
    generation_kwargs: Dict[str, Any]
    result_dir: str  # Full path including sparse_config_name and subset information
```

### Execution Flow

1. **Planning Phase**: Generate all benchmark stubs (adaptes = (model_name x sparse_attention_configs) × benchmark × subset combinations)
2. **Resumability Check**: Filter out stubs that already have result directories (subset-level)
3. **GPU Validation**: Validate that specified GPU IDs are accessible
4. **GPU Pool Creation**: Create shared pool from user-specified GPU IDs
5. **Dynamic Allocation**: Workers request GPUs from pool as needed
6. **Model Validation**: Test model adapter initialization on assigned GPU before execution
7. **Parallel Execution**: Distribute remaining stubs across worker processes
8. **Progress Monitoring**: Track completion status and collect results
9. **Result Aggregation**: Combine results from all processes

## Implementation Details

### Phase 1: Core Multiprocessing Framework

#### Worker Process Function
```python
def _benchmark_worker(
    stub_queue: multiprocessing.Queue,
    gpu_pool: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    error_queue: multiprocessing.Queue
) -> None:
    """Worker function to execute benchmarks with dynamic GPU allocation."""
    while True:
        # 1. Get next benchmark stub from queue
        # 2. Acquire GPU from pool
        # 3. Validate model adapter can initialize on GPU
        # 4. Create model adapter with stub's sparse_attention_config
        # 5. Create benchmark instance with stub's subset configuration
        # 6. Execute benchmark.run_benchmark(adapter, stub.result_dir)
        # 7. Release GPU back to pool
        # 8. Report results/errors
```

#### Process Pool Management
- Use `multiprocessing.Process` with shared queues for dynamic work distribution
- GPU pool as shared `multiprocessing.Queue` for dynamic allocation
- Implement process-safe result collection
- Handle worker failures gracefully and GPU cleanup

### Phase 2: Dynamic GPU Resource Management

#### GPU Pool Management
```python
def _create_gpu_pool(self) -> multiprocessing.Queue:
    """Create a shared pool from user-specified GPU IDs."""
    gpu_pool = multiprocessing.Queue()
    self._validate_gpu_availability()
    for gpu_id in self.gpu_ids:
        gpu_pool.put(gpu_id)
    return gpu_pool

def _validate_gpu_availability(self) -> None:
    """Validate that specified GPUs are accessible."""
    # Check torch.cuda.device_count() >= max(self.gpu_ids)
    # Validate each GPU in self.gpu_ids is accessible
    # Raise error if any GPU is unavailable
```

#### GPU Validation Strategy
```python
def _validate_gpu_for_model(self, gpu_id: int, model_name: str) -> bool:
    """Test if model can be loaded on specific GPU."""
    # Temporarily set CUDA_VISIBLE_DEVICES
    # Try to create a minimal ModelAdapterHF instance
    # Return True if successful, False if failed
```

#### Device Isolation
- Workers set `CUDA_VISIBLE_DEVICES` to single GPU before model loading
- Automatic GPU release back to pool after benchmark completion
- Handle GPU memory cleanup between benchmarks

### Phase 3: Progress Tracking & Results

#### Resumability Detection (Subset-Level)
```python
def _filter_existing_results(self, stubs: List[BenchmarkStub]) -> List[BenchmarkStub]:
    """Filter out benchmark stubs that already have result directories at subset level."""
    filtered_stubs = []
    for stub in stubs:
        result_dir = self._get_result_directory(stub)  # Includes sparse_config and subset in path
        if not Path(result_dir).exists():
            filtered_stubs.append(stub)
        else:
            subset_info = f"_{stub.subset}" if stub.subset else ""
            print(f"Skipping {stub.model_name}/{stub.sparse_config_name}/{stub.benchmark_name}{subset_info} - results exist")
    return filtered_stubs

def _sanitize_model_name(self, model_name: str) -> str:
    """Sanitize model name for use in directory paths."""
    # Replace special characters: / \ : * ? " < > |
    # meta-llama/Llama-3.1-8B-Instruct -> meta_llama_Llama_3.1_8B_Instruct
    return model_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
```

#### Progress Monitoring
```python
class ExecutionProgress:
    """Tracks execution progress across all processes."""
    total_stubs: int
    skipped_stubs: int  # Already completed (resumability)
    completed_stubs: int
    failed_stubs: int
    current_executions: Dict[int, str]  # gpu_id -> current task description
```

#### Result Aggregation
```python
@dataclass
class ExecutionResults:
    """Aggregated results from benchmark execution."""
    execution_summary: Dict[str, Any]
    individual_results: List[BenchmarkResult]
    failed_executions: List[BenchmarkFailure]
    execution_time: float
```

## Directory Structure

**Note**: This will be the standard directory structure we implement. No existing directory creation code will be reused - fresh implementation.

```
benchmark_results/
├── execution_20250115_143022/           # Timestamped execution
│   ├── execution_summary.json          # Overall summary
│   ├── meta_llama_Llama_3.1_8B_Instruct/  # Per-model directories (sanitized names)
│   │   ├── dense/                       # Per-sparse-config subdirectories
│   │   │   ├── longbench_narrativeqa/   # Per-benchmark-subset subdirectories
│   │   │   │   ├── raw_results.csv      # Required by Benchmark.run_benchmark()
│   │   │   │   └── metrics.json         # Required by Benchmark.run_benchmark()
│   │   │   ├── longbench_qasper/        # Separate directory per subset
│   │   │   │   ├── raw_results.csv
│   │   │   │   └── metrics.json
│   │   │   └── mock_benchmark/          # Full benchmark (no subsets)
│   │   │       ├── raw_results.csv
│   │   │       └── metrics.json
│   │   └── streaming/                   # Different sparse attention config
│   │       ├── longbench_narrativeqa/
│   │       │   ├── raw_results.csv
│   │       │   └── metrics.json
│   │       └── longbench_qasper/
│   │           ├── raw_results.csv
│   │           └── metrics.json
│   └── microsoft_Phi_4_mini_instruct/   # Sanitized model name
│       ├── dense/
│       │   └── longbench_hotpotqa/
│       │       ├── raw_results.csv
│       │       └── metrics.json
│       └── hash_attention/
│           └── mock_benchmark/
│               ├── raw_results.csv
│               └── metrics.json
```

**Resumability**: If a `sanitized_model_name/sparse_config_name/benchmark_name_subset/` directory exists, skip that specific benchmark stub.

## Error Handling Strategy

### Process-Level Errors
- Model loading failures
- CUDA out-of-memory errors
- Dataset loading issues
- Generation timeouts

### Recovery Mechanisms
- Skip failed model/benchmark combinations
- Continue with remaining tasks
- Detailed error logging and reporting
- Optional retry logic for transient failures

## Configuration Examples

### Basic Usage
```python
# Specify which GPUs to use and max concurrent processes
executor = BenchmarkExecutor(
    gpu_ids=[0, 1],  # Use GPUs 0 and 1
    max_concurrent_runs=4,  # Run up to 4 processes across these GPUs
    base_result_dir="./my_benchmark_results"
)

model_names = ["microsoft/Phi-4-mini-instruct", "meta-llama/Llama-3.1-8B-Instruct"]
sparse_attention_configs = [
    ("dense", None),  # Dense baseline
    ("streaming", streaming_config),  # StreamingLLM config
]
benchmark_configs = [
    BenchmarkConfig("longbench", ["narrativeqa", "qasper"]),  # Creates 2 subset stubs
    BenchmarkConfig("mock_benchmark")  # Creates 1 stub (no subsets)
]
adapter_config = AdapterConfig(
    model_kwargs={"torch_dtype": torch.bfloat16}
    # Note: sparse_attention_config removed - now part of matrix
)

# This creates 12 total stubs: 2 models × 2 sparse_configs × (2 longbench subsets + 1 mock_benchmark)
results = executor.run_benchmark_matrix(
    model_names=model_names,
    sparse_attention_configs=sparse_attention_configs,
    benchmark_configs=benchmark_configs, 
    adapter_config=adapter_config,
    generation_kwargs={"max_new_tokens": 256}
)

# Results automatically saved with sparse_config/subset-level directories
# Resumable: re-running skips completed model/sparse_config/benchmark/subset combinations
# If OOM occurs, reduce max_concurrent_runs
```

### Advanced Configuration - Multiple Sparse Attention Comparisons
```python
# Use all 4 available GPUs with high concurrency
executor = BenchmarkExecutor(
    gpu_ids=[0, 1, 2, 3],  # Specify all GPUs to use
    max_concurrent_runs=8,  # High concurrency across 4 GPUs
    base_result_dir="./experiments/sparse_attention_comparison"
)

# Comprehensive sparse attention configuration comparison
model_names = ["meta-llama/Llama-3.1-8B-Instruct", "microsoft/Phi-4-mini-instruct"]
sparse_attention_configs = [
    ("dense", None),  # Dense baseline
    ("streaming", streaming_config),  # StreamingLLM
    ("hash_attention", hash_attention_config),  # Hash attention
    ("sink_local", sink_local_config),  # Sink + Local masking
]
benchmark_configs = [
    BenchmarkConfig("longbench", ["narrativeqa", "qasper", "hotpotqa"]),
    BenchmarkConfig("mock_benchmark")
]
adapter_config = AdapterConfig(
    model_kwargs={"torch_dtype": torch.bfloat16},
    tokenizer_kwargs={"padding_side": "left"}
)

# This creates 32 stubs: 2 models × 4 sparse_configs × (3 longbench subsets + 1 mock_benchmark)
# All run in single execution with automatic resumability
results = executor.run_benchmark_matrix(
    model_names=model_names,
    sparse_attention_configs=sparse_attention_configs,
    benchmark_configs=benchmark_configs,
    adapter_config=adapter_config,
    generation_kwargs={"max_new_tokens": 512}
)

# Results organized by model/sparse_config/benchmark hierarchy
# Resumable: can restart and skip completed combinations
# Perfect for comprehensive attention mechanism studies
```

## Future Extensions

### Phase 4: Advanced Features (Future)
1. **Dynamic GPU Allocation**: Add/remove GPUs during execution
2. **Checkpoint/Resume**: Save intermediate state for long-running experiments  
3. **Distributed Execution**: Support multiple machines
4. **Resource Monitoring**: GPU memory/utilization tracking
5. **Adaptive Scheduling**: Prioritize faster benchmarks
6. **Result Analysis**: Built-in comparison and visualization tools

### Extensibility Points
- **Adapter Support**: Plugin system for non-HuggingFace adapters
- **Custom Schedulers**: Alternative task distribution strategies
- **Result Exporters**: Custom result format support
- **Notification Systems**: Integration with monitoring tools

## Implementation Phases

### Phase 1: Minimal Viable Product (MVP)
- [ ] Basic `BenchmarkExecutor` class structure (no ResultStorage dependency)
- [ ] Dynamic GPU pool management with worker queues
- [ ] Core benchmark stub generation and filtering (resumability)
- [ ] GPU validation before model loading
- [ ] Basic result directory management
- [ ] Essential error handling and GPU cleanup

**Estimated Effort**: 2-3 days
**Goal**: Functional parallel execution with dynamic GPU allocation and resumability

### Phase 2: Production Ready
- [ ] Robust error handling and recovery
- [ ] Progress tracking and monitoring (including resumability stats)
- [ ] Comprehensive result aggregation and summary generation
- [ ] GPU resource validation and availability checking
- [ ] Configuration validation and sensible defaults
- [ ] Performance optimization for large benchmark matrices

**Estimated Effort**: 2-3 days  
**Goal**: Reliable execution suitable for research experiments with production-grade robustness

### Phase 3: Quality & Polish
- [ ] Comprehensive unit tests
- [ ] Integration tests with real benchmarks
- [ ] Documentation and examples
- [ ] Performance optimization
- [ ] Memory management improvements

**Estimated Effort**: 2-3 days
**Goal**: Production-quality implementation ready for team use

## Key Changes Based on Feedback

### Implemented Changes
1. **✅ Removed ResultStorage**: Simplified to directory-based result management only (storage.py deleted)
2. **✅ Sparse Attention Config Matrix**: Added sparse_attention_config as matrix dimension (model × sparse_config × benchmark × subset)
3. **✅ Subset-Level Resumability**: Skip benchmark stubs with existing result directories at sparse_config/subset granularity
4. **✅ User-Specified GPU Lists**: Take gpu_ids as input rather than auto-discovery
5. **✅ Dynamic GPU Allocation**: Workers request GPUs from shared pool (no static assignment)
6. **✅ GPU Validation**: Test model loading on GPU before benchmark execution
7. **✅ Model Name Sanitization**: Handle special characters in model names for directory paths
8. **✅ Fresh Implementation**: No reuse of existing executor.py code

### Architecture Improvements
- **Queue-based Work Distribution**: More efficient than static task assignment
- **Fault Tolerance**: GPU failures don't block other workers
- **Resource Efficiency**: GPUs released immediately after use
- **Better Scalability**: Handles variable benchmark execution times

## Discussion Points

### 1. GPU Memory Management
**Question**: How should we handle GPU memory between benchmark runs?
**Options**:
- Clear cache after each benchmark (safer, slower)
- Rely on automatic garbage collection (faster, riskier)
- Per-benchmark memory limits

**Recommendation**: Start with cache clearing, optimize later with memory monitoring.

### 2. Process vs Thread Parallelism
**Question**: Should we use multiprocessing or multithreading?
**Analysis**: 
- **Multiprocessing Pros**: True parallelism, GPU isolation, fault tolerance
- **Multiprocessing Cons**: Memory overhead, IPC complexity
- **Threading Pros**: Lower overhead, shared memory
- **Threading Cons**: GIL limitations, shared GPU state issues

**Recommendation**: Multiprocessing for GPU isolation and fault tolerance.

### 3. Result Storage Strategy
**Question**: How should we organize results in the file system?
**Options**:
- Flat structure with naming conventions
- Hierarchical by model/benchmark
- Database-backed storage

**Recommendation**: Hierarchical file structure with JSON metadata for simplicity.

### 4. Configuration Complexity
**Question**: How complex should the configuration system be?
**Balance**: 
- Too simple: Limited flexibility
- Too complex: Hard to use

**Recommendation**: Start with simple dataclasses, add complexity as needed.

### 5. Benchmark Discovery
**Question**: Should benchmarks be auto-discovered or explicitly configured?
**Options**:
- Auto-discovery from benchmark registry
- Explicit benchmark class instantiation
- String-based benchmark names with factory

**Recommendation**: String-based names with factory pattern for simplicity.

## Next Steps

1. **Review and Discuss**: Gather feedback on architecture and design decisions
2. **Prioritize Features**: Decide which features are essential for MVP  
3. **Define Interfaces**: Finalize the public API surface
4. **Implementation Planning**: Break down into specific development tasks
5. **Testing Strategy**: Plan unit/integration testing approach

## Finalized Design Decisions ✅

1. **✅ Storage**: Remove ResultStorage, use directory-based management only
2. **✅ Matrix Design**: Support model × sparse_config × benchmark × subset combinations
3. **✅ Resumability**: Sparse_config/subset-level granularity
4. **✅ GPU Management**: User specifies gpu_ids list, max_concurrent_runs controls processes
5. **✅ Model Names**: Sanitize special characters for directory paths

## Remaining Questions for Implementation

1. **GPU Validation Depth**: How thorough should the GPU validation be? (minimal test vs full model loading)
2. **Error Recovery**: Should failed benchmarks be automatically retried or require manual intervention?
3. **Progress Reporting**: What level of progress detail is needed? (console output, log files, real-time dashboards)
4. **Benchmark Scale**: What's the typical scale of experiments? (helps optimize queue sizes, memory management)

## Summary

This reworked plan addresses all feedback points and provides a much more practical implementation approach:

### Core Strengths
- **Comprehensive Matrix Support**: Full model × sparse_config × benchmark × subset combinations
- **Resumable Execution**: Never re-run completed combinations at granular level
- **Dynamic Resource Allocation**: Efficient GPU utilization regardless of benchmark timing
- **Simplified Architecture**: No unnecessary storage abstraction layers
- **Robust Validation**: GPU compatibility testing before expensive model loading
- **Research-Focused**: Perfect for attention mechanism comparison studies
- **Fresh Implementation**: Clean, purpose-built solution

### Implementation Priority
1. **Phase 1**: Basic dynamic execution with resumability (MVP)
2. **Phase 2**: Production robustness and monitoring
3. **Phase 3**: Polish and optimization

This design balances simplicity with the sophisticated resource management needed for large-scale benchmark execution across multiple GPUs. 