"""
Extended BenchmarkExecutor with Hyperparameter Optimization Integration

This module extends the existing BenchmarkExecutor to include hyperparameter optimization
as a preprocessing step before benchmark execution.

Key features:
1. Two-phase execution: optimization then benchmarking
2. Caching of optimization results to avoid redundant computation
3. Backward compatibility with existing BenchmarkExecutor usage
4. Seamless integration with existing stub/worker system
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from benchmark.hyperparameter_optimization import (
    OptimizationConfig, HyperparameterOptimizer, OptimizedSparseConfig,
    optimize_sparse_configs
)
from sparse_attention_hub.sparse_attention.base import SparseAttentionConfig


class OptimizedBenchmarkExecutor(BenchmarkExecutor):
    """Extended BenchmarkExecutor with hyperparameter optimization capabilities.
    
    This executor operates in two phases:
    1. Hyperparameter Optimization: Uses Ray Tune to find optimal hyperparameters
       for each (model, sparse_config_type, benchmark, subset) combination
    2. Benchmark Execution: Runs benchmarks using optimized configurations
    
    The optimization results are cached to avoid redundant computation.
    """
    
    def __init__(
        self,
        gpu_ids: List[int],
        max_concurrent_runs: int,
        base_result_dir: str = "./benchmark_results",
        optimization_config: Optional[OptimizationConfig] = None,
        enable_optimization: bool = True,
        **kwargs
    ):
        """Initialize OptimizedBenchmarkExecutor.
        
        Args:
            gpu_ids: List of GPU device IDs to use for parallel execution
            max_concurrent_runs: Maximum number of concurrent benchmark runs
            base_result_dir: Base directory for storing benchmark results
            optimization_config: Configuration for hyperparameter optimization
            enable_optimization: Whether to enable optimization phase
            **kwargs: Arguments passed to parent BenchmarkExecutor
        """
        super().__init__(
            gpu_ids=gpu_ids,
            max_concurrent_runs=max_concurrent_runs,
            base_result_dir=base_result_dir,
            **kwargs
        )
        
        self.optimization_config = optimization_config or OptimizationConfig()
        self.enable_optimization = enable_optimization and self.optimization_config.enabled
        self.optimizer = None
        
        if self.enable_optimization:
            self.optimizer = HyperparameterOptimizer(self.optimization_config)
        
        self.logger = logging.getLogger(__name__)
    
    def run_benchmark_matrix(
        self, 
        model_names: List[str],
        sparse_attention_configs: List[Tuple[str, Optional[SparseAttentionConfig]]],
        benchmark_configs: List[BenchmarkConfig],
        adapter_config: AdapterConfig,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Run benchmarks with optional hyperparameter optimization.
        
        This method extends the parent class by adding an optimization phase
        before benchmark execution.
        
        Args:
            model_names: List of model names to benchmark
            sparse_attention_configs: List of (name, config) sparse attention configurations
            benchmark_configs: List of benchmark configurations  
            adapter_config: Adapter configuration
            generation_kwargs: Optional generation parameters
            request_kwargs: Optional request processing parameters
        """
        
        self.logger.info("Starting OptimizedBenchmarkExecutor")
        self.logger.info(f"Optimization enabled: {self.enable_optimization}")
        self.logger.info(f"Models: {model_names}")
        self.logger.info(f"Sparse configs: {[name for name, _ in sparse_attention_configs]}")
        self.logger.info(f"Benchmarks: {[bc.benchmark_name for bc in benchmark_configs]}")
        
        if self.enable_optimization:
            # Phase 1: Hyperparameter Optimization
            self.logger.info("=== Phase 1: Hyperparameter Optimization ===")
            optimized_configs = self._run_optimization_phase(
                sparse_attention_configs,
                model_names,
                benchmark_configs,
                adapter_config
            )
            
            # Convert optimized configs back to the format expected by parent class
            final_sparse_configs = [
                (name, opt_config.create_optimized_config())
                for name, opt_config in optimized_configs
            ]
            
            self.logger.info("=== Phase 2: Benchmark Execution ===")
        else:
            self.logger.info("=== Benchmark Execution (No Optimization) ===")
            final_sparse_configs = sparse_attention_configs
        
        # Phase 2: Run benchmarks using parent class implementation
        results = super().run_benchmark_matrix(
            model_names=model_names,
            sparse_attention_configs=final_sparse_configs,
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config,
            generation_kwargs=generation_kwargs,
            request_kwargs=request_kwargs
        )
        
        if self.enable_optimization:
            self._log_optimization_summary(optimized_configs, self.base_result_dir)
        
        return results
    
    def _run_optimization_phase(
        self,
        sparse_attention_configs: List[Tuple[str, Optional[SparseAttentionConfig]]],
        model_names: List[str],
        benchmark_configs: List[BenchmarkConfig],
        adapter_config: AdapterConfig
    ) -> List[Tuple[str, OptimizedSparseConfig]]:
        """Run the hyperparameter optimization phase."""
        
        self.logger.info(f"Running hyperparameter optimization with {self.optimization_config.num_samples} samples")
        
        # Filter to only sparse configs that need optimization
        configs_to_optimize = [
            (name, config) for name, config in sparse_attention_configs
            if name != "dense" and config is not None  # Skip dense and None configs
        ]
        
        if not configs_to_optimize:
            self.logger.info("No sparse attention configs to optimize")
            return [
                (name, OptimizedSparseConfig(
                    config_type=name,
                    base_config=config,
                    is_optimized=False
                ))
                for name, config in sparse_attention_configs
            ]
        
        self.logger.info(f"Optimizing {len(configs_to_optimize)} sparse attention configurations")
        
        # Run optimization
        optimized_configs = optimize_sparse_configs(
            sparse_configs=sparse_attention_configs,
            model_names=model_names,
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config,
            optimization_config=self.optimization_config
        )
        
        return optimized_configs
    
    def _log_optimization_summary(
        self,
        optimized_configs: List[Tuple[str, OptimizedSparseConfig]],
        result_dir: Path
    ) -> None:
        """Log summary of optimization results."""
        
        summary_file = result_dir / "optimization_summary.txt"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write("Hyperparameter Optimization Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for config_name, opt_config in optimized_configs:
                f.write(f"Config: {config_name}\n")
                f.write(f"Optimized: {opt_config.is_optimized}\n")
                
                if opt_config.is_optimized and opt_config.optimized_params:
                    f.write("Optimized Parameters:\n")
                    for param, value in opt_config.optimized_params.items():
                        f.write(f"  {param}: {value}\n")
                    
                    if opt_config.optimization_metadata:
                        f.write("Optimization Metadata:\n")
                        for key, value in opt_config.optimization_metadata.items():
                            f.write(f"  {key}: {value}\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
        
        self.logger.info(f"Optimization summary saved to {summary_file}")


def create_optimized_benchmark_executor(
    gpu_ids: List[int],
    max_concurrent_runs: int = 2,
    base_result_dir: str = "./benchmark_results",
    optimization_config: Optional[OptimizationConfig] = None,
    enable_optimization: bool = True
) -> OptimizedBenchmarkExecutor:
    """Factory function to create OptimizedBenchmarkExecutor with sensible defaults.
    
    Args:
        gpu_ids: List of GPU device IDs to use for parallel execution
        max_concurrent_runs: Maximum number of concurrent benchmark runs
        base_result_dir: Base directory for storing benchmark results
        optimization_config: Hyperparameter optimization configuration
        enable_optimization: Whether to enable optimization
        
    Returns:
        Configured OptimizedBenchmarkExecutor
    """
    
    if optimization_config is None:
        optimization_config = OptimizationConfig(
            enabled=enable_optimization,
            num_samples=20,
            max_concurrent=min(4, max_concurrent_runs),
            optimization_metric="combined_score",
            optimization_mode="min",
            cache_dir="./hyperparameter_cache",
            quick_eval_requests=10
        )
    
    return OptimizedBenchmarkExecutor(
        gpu_ids=gpu_ids,
        max_concurrent_runs=max_concurrent_runs,
        base_result_dir=base_result_dir,
        optimization_config=optimization_config,
        enable_optimization=enable_optimization
    )


# Convenience functions for backward compatibility
def run_optimized_benchmarks(
    model_names: List[str],
    sparse_attention_configs: List[Tuple[str, Optional[SparseAttentionConfig]]],
    benchmark_configs: List[BenchmarkConfig],
    adapter_config: AdapterConfig,
    result_dir: str,
    gpu_ids: Optional[List[int]] = None,
    max_concurrent_runs: int = 2,
    enable_optimization: bool = True,  
    optimization_samples: int = 20
) -> None:
    """Convenience function to run optimized benchmarks with minimal setup.
    
    This function provides a simple interface for running benchmarks with
    automatic hyperparameter optimization.
    
    Args:
        model_names: List of model names to benchmark
        sparse_attention_configs: List of (name, config) sparse attention configurations
        benchmark_configs: List of benchmark configurations
        adapter_config: Adapter configuration
        result_dir: Directory to store results
        gpu_ids: List of GPU IDs to use (None for [0])
        max_concurrent_runs: Maximum concurrent benchmark runs
        enable_optimization: Whether to run hyperparameter optimization
        optimization_samples: Number of optimization samples per config
    """
    
    if gpu_ids is None:
        gpu_ids = [0]  # Default to GPU 0
    
    # Create optimization config
    optimization_config = OptimizationConfig(
        enabled=enable_optimization,
        num_samples=optimization_samples,
        max_concurrent=min(4, max_concurrent_runs),
        cache_dir=f"{result_dir}/hyperparameter_cache"
    )
    
    # Create and run executor
    executor = create_optimized_benchmark_executor(
        gpu_ids=gpu_ids,
        max_concurrent_runs=max_concurrent_runs,
        base_result_dir=result_dir,
        optimization_config=optimization_config,
        enable_optimization=enable_optimization
    )
    
    try:
        executor.run_benchmark_matrix(
            model_names=model_names,
            sparse_attention_configs=sparse_attention_configs,
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config
        )
    finally:
        # BenchmarkExecutor doesn't have a shutdown method, so just pass
        pass


if __name__ == "__main__":
    # Example usage
    from benchmark.executor_config import BenchmarkConfig, AdapterConfig
    from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig
    
    # Example configuration
    model_names = ["meta-llama/Llama-3.1-8B-Instruct"]
    
    # Create sparse attention configs
    magic_pig_config = ResearchAttentionConfig(
        masker_configs=[MagicPigConfig(lsh_l=8, lsh_k=32)]
    )
    
    sparse_configs = [
        ("dense", None),
        ("magic_pig", magic_pig_config)
    ]
    
    # Create benchmark configs
    benchmark_configs = [
        BenchmarkConfig(
            benchmark_name="loogle",
            subsets=["shortdep_qa"]
        )
    ]
    
    # Create adapter config
    adapter_config = AdapterConfig(
        model_kwargs={"torch_dtype": "auto"},
        tokenizer_kwargs={}
    )
    
    # Run optimized benchmarks
    run_optimized_benchmarks(
        model_names=model_names,
        sparse_attention_configs=sparse_configs,
        benchmark_configs=benchmark_configs,
        adapter_config=adapter_config,
        result_dir="./optimized_benchmark_results",
        gpu_ids=[0],
        max_concurrent_runs=1,
        enable_optimization=True,
        optimization_samples=10
    )
