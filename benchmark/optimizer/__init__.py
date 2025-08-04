"""Sparse Attention Hyperparameter Optimization Module."""

from .hyperparameter_optimization import (
    OptimizationConfig,
    OptimizedSparseConfig,
    HyperparameterOptimizer,
    get_sparse_optimizer,
    list_available_optimizers,
    optimize_sparse_configs
)

from .optimized_executor import (
    OptimizedBenchmarkExecutor,
    run_optimized_benchmarks,
    create_optimized_benchmark_executor
)

from .generic_config_optimizer import (
    create_optimizer_for_config,
    create_composite_optimizer,
    auto_create_composite_optimizer,
    auto_register_config
)

__all__ = [
    # Core classes
    "OptimizationConfig",
    "OptimizedSparseConfig", 
    "HyperparameterOptimizer",
    "OptimizedBenchmarkExecutor",
    
    # Factory functions
    "get_sparse_optimizer",
    "list_available_optimizers",
    "optimize_sparse_configs",
    "run_optimized_benchmarks",
    "create_optimized_benchmark_executor",
    
    # Generic optimizer functions
    "create_optimizer_for_config",
    "create_composite_optimizer", 
    "auto_create_composite_optimizer",
    "auto_register_config"
]
