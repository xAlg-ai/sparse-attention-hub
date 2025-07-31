"""
Hyperparameter Optimization Extension for BenchmarkExecutor

This module extends the existing BenchmarkExecutor with Ray Tune hyperparameter optimization.
Before running benchmarks, it optimizes sparse attention hyperparameters for each 
(model, sparse_config_type, benchmark, subset) combination.

Architecture:
1. OptimizedSparseConfig: Extends the existing sparse config system
2. HyperparameterOptimizer: Manages Ray Tune optimization per (model, config_type, task) triplet
3. Extended BenchmarkExecutor: Integrates optimization phase before benchmark execution
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
import torch
from abc import ABC, abstractmethod

from sparse_attention_hub.sparse_attention.base import SparseAttentionConfig
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

# Import Ray Tune components
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization.
    
    Attributes:
        enabled: Whether to run hyperparameter optimization
        num_samples: Number of Ray Tune samples per optimization
        max_concurrent: Maximum concurrent trials
        optimization_metric: Metric to optimize (e.g., "combined_score")
        optimization_mode: Optimization mode ("min" or "max")
        cache_dir: Directory to cache optimization results
        optimization_timeout: Timeout per optimization in seconds
        quick_eval_requests: Number of requests for quick evaluation during optimization
    """
    enabled: bool = True
    num_samples: int = 20
    max_concurrent: int = 4
    optimization_metric: str = "combined_score"
    optimization_mode: str = "min"
    cache_dir: str = "./hyperparameter_cache"
    optimization_timeout: float = 7200.0  # 2 hours
    quick_eval_requests: int = 10  # Small number for fast optimization


@dataclass
class OptimizedSparseConfig:
    """Container for a sparse attention config with optimization metadata.
    
    This extends the existing sparse config system to track optimization status.
    
    Attributes:
        config_type: Type of sparse attention (e.g., "magic_pig", "hash_attention")
        base_config: Base sparse attention config (may be None for optimization)
        optimized_params: Optimized hyperparameters from Ray Tune
        optimization_metadata: Metadata about the optimization process
        is_optimized: Whether this config has been optimized
    """
    config_type: str
    base_config: Optional[SparseAttentionConfig] = None
    optimized_params: Optional[Dict[str, Any]] = None
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)
    is_optimized: bool = False
    
    def create_optimized_config(self) -> Optional[SparseAttentionConfig]:
        """Create the final sparse attention config using optimized parameters."""
        if not self.is_optimized or not self.optimized_params:
            if self.base_config:
                return self.base_config
            elif self.config_type == "dense":
                # For dense configs, return None to indicate dense attention
                return None
            else:
                raise ValueError(f"No optimized parameters or base config for {self.config_type}")
        
        # Delegate to the appropriate config creator based on type
        return create_sparse_config_from_params(self.config_type, self.optimized_params)


class SparseConfigOptimizer(ABC):
    """Abstract base class for sparse attention hyperparameter optimizers.
    
    Each sparse attention type (Magic Pig, Hash Attention, etc.) should implement
    this interface to define its optimization behavior.
    """
    
    @abstractmethod
    def get_search_space(self) -> Dict[str, Any]:
        """Define Ray Tune search space for this sparse attention type."""
        pass
    
    @abstractmethod
    def create_config_from_params(self, params: Dict[str, Any]) -> SparseAttentionConfig:
        """Create sparse attention config from optimized parameters."""
        pass
    
    @abstractmethod  
    def get_default_request_kwargs(self, benchmark_name: str, subset: Optional[str]) -> Dict[str, Any]:
        """Get task-specific request_kwargs for optimization."""
        pass
    
    def get_optimization_metric_weights(self) -> Dict[str, float]:
        """Return weights for combining multiple optimization metrics."""
        return {"attention_error": 1.0, "density": 0.1}


class MagicPigOptimizer(SparseConfigOptimizer):
    """Hyperparameter optimizer for Magic Pig attention."""
    
    def get_search_space(self) -> Dict[str, Any]:
        return {
            "lsh_l": tune.choice([4, 8, 16, 32]),
            "lsh_k": tune.choice([8, 16, 32, 64]),  # Remove 128 for int64 compatibility
            "center": tune.choice([True, False]),
            "packing": tune.choice(["int64", "float32"]),
            "seed": tune.choice([42, 123, 456])
        }
    
    def create_config_from_params(self, params: Dict[str, Any]) -> SparseAttentionConfig:
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig
        
        masker_config = MagicPigConfig(
            lsh_l=params["lsh_l"],
            lsh_k=params["lsh_k"],
            center=params["center"],
            packing=params["packing"],
            seed=params.get("seed", 42)
        )
        
        return ResearchAttentionConfig(masker_configs=[masker_config])
    
    def get_default_request_kwargs(self, benchmark_name: str, subset: Optional[str]) -> Dict[str, Any]:
        # Task-specific optimization parameters - using minimal requests for quick iteration
        if benchmark_name == "loogle":
            if subset and "longdep" in subset:
                return {"max_requests": 2, "max_context_length": 512}  # Ultra-fast iteration
            else:
                return {"max_requests": 2, "max_context_length": 512}  # Ultra-fast iteration
        else:
            return {"max_requests": 2, "max_context_length": 512}  # Ultra-fast iteration


# Registry of available optimizers
SPARSE_OPTIMIZERS: Dict[str, SparseConfigOptimizer] = {
    "magic_pig": MagicPigOptimizer(),
    # "hash_attention": HashAttentionOptimizer(),  # To be implemented
    # Add other optimizers here
}


def create_sparse_config_from_params(config_type: str, params: Dict[str, Any]) -> SparseAttentionConfig:
    """Factory function to create sparse config from optimized parameters."""
    if config_type not in SPARSE_OPTIMIZERS:
        raise ValueError(f"Unknown sparse config type: {config_type}")
    
    return SPARSE_OPTIMIZERS[config_type].create_config_from_params(params)


class HyperparameterOptimizer:
    """Manages hyperparameter optimization for sparse attention configurations.
    
    This class handles:
    1. Caching optimization results to avoid re-optimization
    2. Running Ray Tune optimization for new (model, config_type, task) combinations
    3. Extracting and storing optimized hyperparameters
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def get_cache_key(self, model_name: str, config_type: str, benchmark_name: str, subset: Optional[str]) -> str:
        """Generate cache key for optimization results."""
        subset_str = f"_{subset}" if subset else ""
        return f"{model_name}_{config_type}_{benchmark_name}{subset_str}".replace("/", "_").replace("-", "_")
    
    def create_optimized_config_from_cache(self, model_name: str, config_type: str, benchmark_name: str, subset: Optional[str] = None) -> Optional[Any]:
        """Create optimized sparse attention config from cached parameters.
        
        Args:
            model_name: Name of the model
            config_type: Type of sparse attention config (e.g., "magic_pig")
            benchmark_name: Name of the benchmark
            subset: Optional subset name
            
        Returns:
            Optimized sparse attention config object if cached, None otherwise
            
        Example:
            >>> optimizer = HyperparameterOptimizer(config)
            >>> optimized_config = optimizer.create_optimized_config_from_cache(
            ...     "meta-llama/Llama-3.1-8B-Instruct", 
            ...     "magic_pig", 
            ...     "loogle", 
            ...     "shortdep_qa"
            ... )
            >>> if optimized_config:
            ...     # Use the optimized config for benchmarking
            ...     pass
        """
        cached_result = self.get_cached_best_config(model_name, config_type, benchmark_name, subset)
        
        if not cached_result:
            return None
            
        # Get the optimizer for this config type
        if config_type not in SPARSE_OPTIMIZERS:
            self.logger.warning(f"No optimizer available for config type: {config_type}")
            return None
            
        optimizer = SPARSE_OPTIMIZERS[config_type]
        best_params = cached_result.get("best_params", {})
        
        try:
            # Create config from cached parameters
            optimized_config = optimizer.create_config_from_params(best_params)
            self.logger.info(f"Created optimized {config_type} config from cache")
            return optimized_config
        except Exception as e:
            self.logger.error(f"Failed to create config from cached params: {e}")
            return None

    def get_cached_best_config(self, model_name: str, config_type: str, benchmark_name: str, subset: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get best config from cache without running optimization.
        
        Args:
            model_name: Name of the model
            config_type: Type of sparse attention config (e.g., "magic_pig")
            benchmark_name: Name of the benchmark
            subset: Optional subset name
            
        Returns:
            Dictionary with best parameters if cached, None otherwise
            
        Example:
            >>> optimizer = HyperparameterOptimizer(config)
            >>> best_params = optimizer.get_cached_best_config(
            ...     "meta-llama/Llama-3.1-8B-Instruct", 
            ...     "magic_pig", 
            ...     "loogle", 
            ...     "shortdep_qa"
            ... )
            >>> if best_params:
            ...     print(f"Best parameters: {best_params['best_params']}")
        """
        cache_key = self.get_cache_key(model_name, config_type, benchmark_name, subset)
        cached_result = self.load_cached_optimization(cache_key)
        
        if cached_result:
            self.logger.info(f"Found cached optimization for {cache_key}")
            return cached_result
        else:
            self.logger.info(f"No cached optimization found for {cache_key}")
            return None

    def load_cached_optimization(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached optimization results if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_file}: {e}")
        return None
    
    def save_optimization_results(self, cache_key: str, results: Dict[str, Any]) -> None:
        """Save optimization results to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved optimization results to {cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save cache {cache_file}: {e}")
    
    def optimize_hyperparameters(
        self, 
        model_name: str, 
        config_type: str, 
        benchmark_name: str, 
        subset: Optional[str],
        adapter_config: AdapterConfig
    ) -> Dict[str, Any]:
        """Run Ray Tune optimization for a specific (model, config_type, task) combination."""
        
        # Check cache first
        cache_key = self.get_cache_key(model_name, config_type, benchmark_name, subset)
        cached_result = self.load_cached_optimization(cache_key)
        if cached_result:
            self.logger.info(f"Using cached optimization for {cache_key}")
            return cached_result
        
        # Get the optimizer for this config type
        if config_type not in SPARSE_OPTIMIZERS:
            raise ValueError(f"No optimizer available for config type: {config_type}")
        
        optimizer = SPARSE_OPTIMIZERS[config_type]
        
        # Create evaluation function for this specific combination
        def evaluate_config(params):
            return self._evaluate_single_config(
                params, model_name, config_type, benchmark_name, subset, adapter_config, optimizer
            )
        
        # Set up Ray Tune
        search_space = optimizer.get_search_space()
        search_alg = HyperOptSearch(metric=self.config.optimization_metric, mode=self.config.optimization_mode)
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=self.config.max_concurrent)
        
        scheduler = ASHAScheduler(
            metric=self.config.optimization_metric,
            mode=self.config.optimization_mode,
            max_t=1,
            grace_period=1
        )
        
        # Run optimization
        self.logger.info(f"Starting hyperparameter optimization for {cache_key}")
        
        tuner = tune.Tuner(
            tune.with_resources(evaluate_config, resources={"cpu": 2, "gpu": 0.5}),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=self.config.num_samples,
            ),
            run_config=ray.air.RunConfig(
                name=f"hyperopt_{cache_key}",
                storage_path=str((self.cache_dir / "ray_results").resolve()),
            )
        )
        
        results = tuner.fit()
        best_result = results.get_best_result(self.config.optimization_metric, self.config.optimization_mode)
        
        # Package results
        optimization_results = {
            "model_name": model_name,
            "config_type": config_type,
            "benchmark_name": benchmark_name,
            "subset": subset,
            "best_params": dict(best_result.config),
            "best_metrics": dict(best_result.metrics),
            "optimization_metadata": {
                "num_samples": self.config.num_samples,
                "optimization_metric": self.config.optimization_metric,
                "search_space_size": self._calculate_search_space_size(search_space)
            }
        }
        
        # Save to cache
        self.save_optimization_results(cache_key, optimization_results)
        
        self.logger.info(f"Completed optimization for {cache_key}: {best_result.metrics}")
        return optimization_results
    
    def _evaluate_single_config(
        self, 
        params: Dict[str, Any],
        model_name: str,
        config_type: str, 
        benchmark_name: str,
        subset: Optional[str],
        adapter_config: AdapterConfig,
        optimizer: SparseConfigOptimizer
    ) -> Dict[str, float]:
        """Evaluate a single hyperparameter configuration."""
        
        try:
            # Create sparse attention config
            sparse_config = optimizer.create_config_from_params(params)
            
            # Create adapter
            from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
            adapter = ModelAdapterHF(
                model_name=model_name,
                sparse_attention_config=sparse_config,
                model_kwargs=adapter_config.model_kwargs,
                tokenizer_kwargs=adapter_config.tokenizer_kwargs,
                device="cuda"
            )
            
            # Create benchmark
            from benchmark.benchmark_registry import create_benchmark_instance
            benchmark = create_benchmark_instance(
                benchmark_name=benchmark_name,
                subsets=[subset] if subset else None
            )
            
            # Get task-specific request kwargs
            request_kwargs = optimizer.get_default_request_kwargs(benchmark_name, subset)
            request_kwargs["max_requests"] = self.config.quick_eval_requests  # Override for quick evaluation
            
            # Set up result directory
            from ray import train
            trial_dir = train.get_context().get_trial_dir()
            result_dir = Path(trial_dir) / "benchmark_results"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure metric logging
            from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
            metric_logger = MicroMetricLogger()
            metric_logger.configure_logging(
                log_path=str(result_dir),
                enabled_metrics=["research_attention_density", "research_attention_output_error"]
            )
            
            # Run benchmark
            generation_kwargs = {"max_new_tokens": 32}
            benchmark_results = benchmark.run_benchmark(
                adapter, 
                str(result_dir), 
                generation_kwargs=generation_kwargs,
                request_kwargs=request_kwargs
            )
            
            # Extract attention metrics
            attention_metrics = self._extract_attention_metrics(result_dir)
            
            # Calculate combined score using optimizer weights
            weights = optimizer.get_optimization_metric_weights()
            combined_score = sum(
                attention_metrics.get(metric, 0.0) * weight 
                for metric, weight in weights.items()
            )
            
            # Add penalty for high density
            if attention_metrics.get("density", 0.0) > 0.5:
                combined_score += 5.0
            
            metrics = {
                "attention_error": attention_metrics.get("attention_error", 10.0),
                "density": attention_metrics.get("density", 1.0),
                "combined_score": combined_score,
                "benchmark_score": benchmark_results.get("overall_score", 0.0)
            }
            
            # Clean up
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating config {params}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return poor metrics on error
            return {
                "attention_error": 10.0,
                "density": 1.0,
                "combined_score": 11.0,
                "benchmark_score": 0.0
            }
    
    def _extract_attention_metrics(self, result_dir: Path) -> Dict[str, float]:
        """Extract attention error and density from micro metrics log."""
        micro_metrics_file = result_dir / "micro_metrics.jsonl"
        
        attention_errors = []
        densities = []
        
        if micro_metrics_file.exists():
            try:
                with open(micro_metrics_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            metric_name = data.get('metric', '')
                            value = data.get('value')
                            
                            if metric_name == 'research_attention_output_error' and value is not None:
                                attention_errors.append(float(value))
                            elif metric_name == 'research_attention_density' and value is not None:
                                densities.append(float(value))
            except Exception as e:
                self.logger.warning(f"Error reading micro metrics: {e}")
        
        if not attention_errors or not densities:
            return {"attention_error": 10.0, "density": 1.0}
        
        return {
            "attention_error": sum(attention_errors) / len(attention_errors),
            "density": sum(densities) / len(densities)
        }
    
    def _calculate_search_space_size(self, search_space: Dict[str, Any]) -> int:
        """Calculate approximate search space size."""
        size = 1
        for param_spec in search_space.values():
            if hasattr(param_spec, '_spec') and param_spec._spec['type'] == 'choice':
                size *= len(param_spec._spec['categories'])
            else:
                size *= 10  # Conservative estimate for continuous parameters
        return size


def optimize_sparse_configs(
    sparse_configs: List[Tuple[str, Optional[SparseAttentionConfig]]],
    model_names: List[str],
    benchmark_configs: List[BenchmarkConfig],
    adapter_config: AdapterConfig,
    optimization_config: OptimizationConfig
) -> List[Tuple[str, OptimizedSparseConfig]]:
    """
    Optimize hyperparameters for all sparse attention configurations.
    
    This function takes the existing sparse_configs and optimizes their hyperparameters
    for each (model, benchmark, subset) combination, returning OptimizedSparseConfig objects.
    
    Args:
        sparse_configs: List of (name, config) tuples - the config can be None for optimization
        model_names: List of model names to optimize for
        benchmark_configs: List of benchmark configs to optimize for  
        adapter_config: Base adapter configuration
        optimization_config: Hyperparameter optimization configuration
        
    Returns:
        List of (name, OptimizedSparseConfig) tuples with optimized configurations
    """
    
    if not optimization_config.enabled:
        # Return configs as-is if optimization disabled
        return [
            (name, OptimizedSparseConfig(
                config_type=name,
                base_config=config,
                is_optimized=False
            ))
            for name, config in sparse_configs
        ]
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_gpus=2, num_cpus=8)
    
    optimizer = HyperparameterOptimizer(optimization_config)
    optimized_configs = []
    
    for config_name, base_config in sparse_configs:
        # Skip dense configs (no optimization needed)
        if config_name == "dense" or base_config is None:
            optimized_configs.append((config_name, OptimizedSparseConfig(
                config_type=config_name,
                base_config=base_config,
                is_optimized=False
            )))
            continue
        
        # For sparse configs, run optimization for each (model, benchmark, subset) combination
        # We'll optimize once per unique task and use the best overall parameters
        all_optimizations = []
        
        for model_name in model_names:
            for benchmark_config in benchmark_configs:
                # Handle benchmark subsets
                subsets = benchmark_config.subsets or [None]
                for subset in subsets:
                    try:
                        result = optimizer.optimize_hyperparameters(
                            model_name=model_name,
                            config_type=config_name,
                            benchmark_name=benchmark_config.benchmark_name,
                            subset=subset,
                            adapter_config=adapter_config
                        )
                        all_optimizations.append(result)
                    except Exception as e:
                        logging.error(f"Optimization failed for {model_name}/{config_name}/{benchmark_config.benchmark_name}/{subset}: {e}")
        
        if all_optimizations:
            # Use the best optimization result (lowest combined_score)
            best_optimization = min(all_optimizations, key=lambda x: x["best_metrics"]["combined_score"])
            
            optimized_config = OptimizedSparseConfig(
                config_type=config_name,
                base_config=base_config,
                optimized_params=best_optimization["best_params"],
                optimization_metadata=best_optimization["optimization_metadata"],
                is_optimized=True
            )
        else:
            # Fall back to base config if optimization failed
            optimized_config = OptimizedSparseConfig(
                config_type=config_name,
                base_config=base_config,
                is_optimized=False
            )
        
        optimized_configs.append((config_name, optimized_config))
    
    return optimized_configs
