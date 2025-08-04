# Sparse Attention Hyperparameter Optimization

Automatic hyperparameter optimization for sparse attention configurations using Ray Tune with declarative search space generation.

## Core Components

- **`hyperparameter_optimization.py`**: Main optimization logic, caching, and registry
- **`generic_config_optimizer.py`**: Auto search space generation from dataclasses
- **`optimized_executor.py`**: Extended benchmark executor with optimization
- **`scripts/`**: CLI tools and demos

## Usage

```python
from benchmark.optimizer.optimized_executor import run_optimized_benchmarks
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

# Run with automatic optimization
run_optimized_benchmarks(
    model_names=["meta-llama/Llama-3.1-8B-Instruct"],
    sparse_attention_configs=[("magic_pig", None)],  # None = optimize
    benchmark_configs=[BenchmarkConfig("loogle", ["shortdep_qa"])],
    adapter_config=AdapterConfig(),
    result_dir="./results",
    enable_optimization=True,
    optimization_samples=20
)
```

## CLI

```bash
# Quick optimization demo
python benchmark/optimizer/scripts/run_optimized_benchmarks.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --benchmark loogle \
    --config magic_pig \
    --samples 10

# Full evaluation
python benchmark/optimizer/scripts/run_optimized_benchmarks.py \
    --model microsoft/Phi-4-mini-instruct \
    --benchmark infinite_bench,ruler \
    --config dense,magic_pig \
    --samples 20
```

## Supported Configs

- `magic_pig`: Composite (Sink + Local + MagicPig)
- `local_masker`: Window-based attention
- `sink_masker`: Sink token attention
- `adaptive_sampling`: Adaptive sampling
- `hash_attention`: Composite (Sink + Local + OracleTopK)

## Architecture

```
Phase 1: Optimization
├── Auto search space from dataclass fields
├── Ray Tune with ASHA scheduler + HyperOpt
└── JSON cache storage

Phase 2: Benchmark Execution  
├── Load optimized configs from cache
├── Per-task vs global optimization
└── Fallback to defaults
```

## Adding New Configs

```python
from benchmark.optimizer.generic_config_optimizer import auto_register_config

# Auto-register any dataclass config
auto_register_config(MyConfig, "my_config")

# Config class defines its own search space
@dataclass
class MyConfig:
    param1: int = 10
    param2: float = 0.5
    
    @classmethod
    def get_default_search_space(cls):
        return {
            "param1": tune.choice([8, 10, 12, 16]),
            "param2": tune.uniform(0.1, 0.9)
        }
```

## Cache Management

```python
from benchmark.optimizer.hyperparameter_optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(OptimizationConfig(cache_dir="./cache"))

# Get cached config
config = optimizer.get_cached_best_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    config_type="magic_pig", 
    benchmark_name="loogle",
    subset="shortdep_qa"
)
```

## Key Features

- **Declarative**: Config classes define own search spaces
- **Auto-composition**: Composite configs from individual configs  
- **Per-task optimization**: Separate configs per benchmark subset
- **Caching**: JSON-based result storage and reuse
- **Backward compatible**: Supports manual overrides
