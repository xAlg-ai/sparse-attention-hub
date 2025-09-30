# Ray Tune Benchmark Suite

A distributed benchmark suite for sparse attention configurations using Ray for parallel execution.

## Quick Start

### 1. Optimize Configurations
Find optimal sparse attention configurations for your models:

```bash
python3 benchmark/raytune/run_optimize_configs.py \
  --objective sparsity_10 \
  --optimal-configs-dir temp \
  --num-samples 1 \
  --search-max-new-tokens 5 \
  --search-max-context-length 32678 \
  --search-max-requests 2 \
  --actors-per-gpu 1
```

### 2. Run Benchmarks
Execute benchmarks using the optimized configurations:

```bash
python3 benchmark/raytune/run_config_dir.py \
  --configs-dir temp/run_20250930_174624 \
  --max-new-tokens 100 \
  --max-context-length 32678 \
  --max-requests 2 \
  --actors-per-gpu 1 \
  --benchmark-results-dir ./test_bench.1/
```

## Workflow

1. **Phase 1**: Use `run_optimize_configs.py` to search for optimal sparse attention parameters
2. **Phase 2**: Use `run_config_dir.py` to run full benchmarks with the found configurations

## Features

- **Distributed Execution**: Ray-based parallel processing across multiple GPUs
- **Automatic Resource Management**: Efficient GPU utilization and task scheduling
- **Sparse Attention Support**: Multiple masker types and configurations
- **Comprehensive Metrics**: Detailed performance and accuracy measurements
