# Sparse Attention Hyperparameter Optimization

A generic optimization system for sparse attention configs using Ray Tune with automatic search space generation.

## Quick Start

```bash
# Demo with both global and per-task optimization
python benchmark/optimizer/scripts/demo_optimized_benchmarks.py

# Available configs
python -c "from benchmark.optimizer.hyperparameter_optimization import list_available_optimizers; print(list_available_optimizers())"
```

## Supported Configs

- **`magic_pig`**: Composite config (SinkMasker + LocalMasker + MagicPig)
- **`hash_attention`**: Composite config (SinkMasker + LocalMasker + OracleTopK)  
- **`local_masker`**: Window-based attention
- **`sink_masker`**: Sink token attention
- **`adaptive_sampling`**: Adaptive sampling masker

## Optimization Modes

### Global Optimization
Single best config for all benchmark subsets.

### Per-Task Optimization  
Task-specific configs per benchmark subset (recommended).

## Core Usage

```python
from benchmark.optimizer.optimized_executor import create_optimized_benchmark_executor
from benchmark.optimizer.hyperparameter_optimization import OptimizationConfig

# Create optimization config
opt_config = OptimizationConfig(
    enabled=True,
    num_samples=10,
    use_per_task_config=True,  # Per-task optimization
    cache_dir="./cache"
)

# Create and run executor
executor = create_optimized_benchmark_executor(
    gpu_ids=[0],
    optimization_config=opt_config
)

executor.run_benchmark_matrix(
    model_names=["meta-llama/Llama-3.1-8B-Instruct"],
    sparse_attention_configs=[("magic_pig", None)],  # None = optimize
    benchmark_configs=[BenchmarkConfig("loogle", ["shortdep_qa"])],
    adapter_config=AdapterConfig()
)
```

## Architecture

The system uses **generic optimizers** that automatically introspect dataclass configs:

- **GenericConfigOptimizer**: Auto-generates search spaces from any dataclass
- **CompositeConfigOptimizer**: Combines multiple masker configs with parameter prefixing
- **Automatic caching**: Results saved as JSON for reuse

```
Phase 1: Optimization (Ray Tune)
├── Auto search space from dataclass fields
├── Type-aware parameter generation (int/float/bool/string)
└── JSON cache storage

Phase 2: Benchmark Execution
├── Load optimized configs from cache
├── Fallback to defaults if optimization disabled
└── Resumability support
```

## Adding New Configs

### Single Config
```python
def create_my_config_optimizer() -> GenericOptimizerAdapter:
    from my_module import MyConfig
    
    optimizer = create_optimizer_for_config(
        config_class=MyConfig,
        config_name="my_config",
        overrides={
            "param1": tune.choice([1, 2, 4]),
            "param2": tune.uniform(0.1, 0.9)
        }
    )
    return GenericOptimizerAdapter(optimizer)

# Add to registry
SPARSE_OPTIMIZERS["my_config"] = create_my_config_optimizer()
```

### Composite Config
```python
def create_my_composite_optimizer() -> GenericOptimizerAdapter:
    composite_optimizer = create_composite_optimizer(
        masker_configs=[ConfigA, ConfigB, ConfigC],
        config_name="my_composite",
        overrides={
            "configa_param1": tune.choice([1, 2, 4]),
            "configb_param2": tune.uniform(0.1, 0.9)
        }
    )
    return GenericOptimizerAdapter(composite_optimizer)
```

## Cache Management

```python
from benchmark.optimizer.hyperparameter_optimization import get_cached_optimization

# Retrieve cached config
config = get_cached_optimization(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    config_type="magic_pig",
    benchmark_name="loogle", 
    subset="shortdep_qa",
    cache_dir="./cache"
)
```

## Files

- `optimized_executor.py`: Main benchmark executor with optimization
- `hyperparameter_optimization.py`: Optimization logic and registry
- `generic_config_optimizer.py`: Auto search space generation
- `scripts/demo_optimized_benchmarks.py`: Interactive demo
