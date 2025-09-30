# Ray Tune Benchmark Suite

A distributed benchmark suite for sparse attention configurations using Ray for parallel execution.

## Quick Start

### 1. Optimize Configurations
Find optimal sparse attention configurations for your models:

```bash
python3 benchmark/raytune/run_optimize_configs.py \
  --objective sparsity_10 \
  --optimal-configs-dir <base_dir>> \
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
  --configs-dir <base_dir/config_dir> \
  --max-new-tokens 100 \
  --max-context-length 32678 \
  --max-requests 2 \
  --actors-per-gpu 1 \
  --benchmark-results-dir ./test_bench.1/
```

## Workflow

### Phase 1: Configuration Optimization
Use `run_optimize_configs.py` to search for optimal sparse attention parameters:

**Configuration Sources:**
- **Models**: Defined in `get_run_configuration()` function
- **Tasks**: Specified in the configuration
- **Sparse Configs**: Two types handled:
  - `to_optimize_configs`: Configurations that need hyperparameter search
  - `optimal_configs`: Pre-optimized configurations (used as-is)
- **Search Spaces**: Each config type can have its own search space defined separately. Example:

```python
# Create a ResearchAttentionConfig with custom search spaces
config = ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=128),
    LocalMaskerConfig(window_size=128),
    OracleTopKConfig(heavy_size=0.10),
    AdaptiveSamplingMaskerConfig(
        base_rate_sampling=0.1,
        epsilon=0.25,
        delta=0.25,
        init_offset=128,
        local_offset=128
    )
])

# Define search spaces for specific maskers
config.masker_configs[2].search_space = {
    "heavy_size": tune.grid_search([0.01, 0.05, 0.1, 0.2])
}
config.masker_configs[3].search_space = {
    "base_rate_sampling": tune.grid_search([0.01, 0.02, 0.05]),
    "epsilon": tune.grid_search([0.05, 0.1, 0.2]),
    "delta": tune.grid_search([0.05, 0.1, 0.2])
}
``` 

**Output**: Optimal configurations are written to `<base_dir>/run_<timestamp>/` directory with individual JSON files per model-task-config combination.

### Phase 2: Benchmark Execution
Use `run_config_dir.py` to run full benchmarks with the found configurations:

**Input**: Pass the config directory (e.g., `<base_dir>/run_<timestamp>/`) containing all the JSON configuration files generated in Phase 1.

**Output**: Benchmark results saved to the specified `--benchmark-results-dir`.

## Features

- **Distributed Execution**: Ray-based parallel processing across multiple GPUs
- **Automatic Resource Management**: Efficient GPU utilization and task scheduling
- **Sparse Attention Support**: Multiple masker types and configurations
- **Comprehensive Metrics**: Detailed performance and accuracy measurements
