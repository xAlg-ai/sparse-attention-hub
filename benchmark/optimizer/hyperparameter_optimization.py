"""
Ray Tune Hyperparameter Optimization for Sparse Attention Benchmarks

This module provides a comprehensive system for optimizing sparse attention configurations
using Ray Tune with advanced schedulers and search algorithms. It now supports both
legacy optimizers and a new generic config optimizer that can automatically generate
search spaces for any dataclass config.

Key Features:
- Generic config introspection and automatic search space generation  
- Support for any config class via dataclass introspection
- Legacy support for existing optimizers (MagicPigOptimizer)
- Per-task optimization with separate cache files
- ASHA scheduler with HyperOpt search for efficient optimization
- Comprehensive metrics tracking and analysis
- Robust caching and config retrieval system

The system creates optimized benchmark executors that automatically use optimized
configurations when available, falling back to default configs when needed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Type
import json
import logging
from abc import ABC, abstractmethod

from sparse_attention_hub.sparse_attention.base import SparseAttentionConfig
from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from benchmark.optimizer.generic_config_optimizer import (
    create_optimizer_for_config, 
    create_composite_optimizer
)

# Import Ray Tune components
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search import ConcurrencyLimiter
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    # Define dummy classes for when Ray is not available
    class tune:
        @staticmethod
        def choice(choices): return choices[0]
        @staticmethod
        def uniform(low, high): return (low + high) / 2

# Import torch for CUDA cleanup
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
        use_per_task_config: Whether to use best config per task/subset vs global best
    """
    enabled: bool = True
    num_samples: int = 3
    max_concurrent: int = 4
    optimization_metric: str = "combined_score"
    optimization_mode: str = "min"
    cache_dir: str = "./hyperparameter_cache"
    optimization_timeout: float = 7200.0  # 2 hours
    quick_eval_requests: int = 2  # Small number for fast optimization
    use_per_task_config: bool = True  # Use per-task configs by default


@dataclass
class OptimizedSparseConfig:
    """Container for a sparse attention config with optimization metadata.
    
    This extends the existing sparse config system to track optimization status.
    
    Attributes:
        config_type: Type of sparse attention (e.g., "magic_pig", "hash_attention")
        base_config: Base sparse attention config (may be None for optimization)
        optimized_params: Optimized hyperparameters from Ray Tune (global best)
        optimization_metadata: Metadata about the optimization process
        is_optimized: Whether this config has been optimized
        per_task_configs: Dict mapping (benchmark, subset) to optimized params for per-task configs
    """
    config_type: str
    base_config: Optional[SparseAttentionConfig] = None
    optimized_params: Optional[Dict[str, Any]] = None
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)
    is_optimized: bool = False
    per_task_configs: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = field(default_factory=dict)
    
    def create_optimized_config(self, benchmark_name: Optional[str] = None, subset: Optional[str] = None, use_per_task: bool = False) -> Optional[SparseAttentionConfig]:
        """Create the final sparse attention config using optimized parameters.
        
        Args:
            benchmark_name: Name of benchmark for per-task config lookup
            subset: Name of subset for per-task config lookup  
            use_per_task: Whether to use per-task config if available
            
        Returns:
            SparseAttentionConfig object with optimized parameters
        """
        if not self.is_optimized:
            if self.base_config:
                return self.base_config
            elif self.config_type == "dense":
                # For dense configs, return None to indicate dense attention
                return None
            else:
                raise ValueError(f"No optimized parameters or base config for {self.config_type}")
        
        # Choose which parameters to use
        params_to_use = None
        
        if use_per_task and benchmark_name and (benchmark_name, subset) in self.per_task_configs:
            # Use per-task config if available and requested
            params_to_use = self.per_task_configs[(benchmark_name, subset)]
        elif self.optimized_params:
            # Fall back to global best config
            params_to_use = self.optimized_params
        else:
            raise ValueError(f"No optimized parameters available for {self.config_type}")
        
        # Delegate to the appropriate config creator based on type
        return create_sparse_config_from_params(self.config_type, params_to_use)
    
    def get_config_for_task(self, benchmark_name: str, subset: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the optimized parameters for a specific task.
        
        Args:
            benchmark_name: Name of the benchmark
            subset: Optional subset name
            
        Returns:
            Dict of optimized parameters for this task, or None if not available
        """
        task_key = (benchmark_name, subset)
        return self.per_task_configs.get(task_key, self.optimized_params)


class SparseConfigOptimizer(ABC):
    """Abstract base class for sparse attention hyperparameter optimizers.
    
    Each sparse attention type can implement this interface to define its 
    optimization behavior. The new GenericConfigOptimizer provides automatic
    implementation for most dataclass configs.
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


class GenericOptimizerAdapter(SparseConfigOptimizer):
    """Adapter to make GenericConfigOptimizer and CompositeConfigOptimizer compatible with the legacy interface."""
    
    def __init__(self, optimizer):
        """Initialize with either GenericConfigOptimizer or CompositeConfigOptimizer."""
        self.optimizer = optimizer
        
    def get_search_space(self) -> Dict[str, Any]:
        """Delegate to optimizer."""
        return self.optimizer.create_search_space()
    
    def create_config_from_params(self, params: Dict[str, Any]) -> SparseAttentionConfig:
        """Create sparse attention config from params."""
        # For composite optimizers, this will create ResearchAttentionConfig directly
        # For single optimizers, we need to wrap in ResearchAttentionConfig
        config = self.optimizer.create_config_from_params(params)
        
        # Check if it's already a ResearchAttentionConfig (from composite optimizer)
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        if isinstance(config, ResearchAttentionConfig):
            return config
        else:
            # Single masker - wrap it in ResearchAttentionConfig
            return ResearchAttentionConfig(masker_configs=[config])
    
    def get_default_request_kwargs(self, benchmark_name: str, subset: Optional[str]) -> Dict[str, Any]:
        """Get task-specific request_kwargs for optimization."""
        # Use minimal requests for quick optimization
        return {"max_requests": 2, "max_context_length": 512}
    
    @property
    def config_type_name(self) -> str:
        """Get the name of the config type for caching."""
        return self.optimizer.config_type_name


# Factory functions for creating optimizers for common config types
def create_magic_pig_optimizer() -> GenericOptimizerAdapter:
    """Create composite optimizer for MagicPig with sink and local maskers."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import (
        LocalMaskerConfig, SinkMaskerConfig
    )
    
    # Create composite optimizer with all three masker types
    composite_optimizer = create_composite_optimizer(
        masker_configs=[SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig],
        config_name="magic_pig",
        overrides={
            # SinkMasker overrides
            "sinkmasker_sink_size": tune.choice([16, 32, 64, 96, 128]),
            # LocalMasker overrides  
            "localmasker_window_size": tune.choice([16, 32, 64, 96, 128]),
            # MagicPig overrides
            "magicpig_lsh_l": tune.choice([4, 6, 8, 10, 12]),
            "magicpig_lsh_k": tune.choice([2, 4, 6, 8]),
            "magicpig_center": tune.choice([True]),
            "magicpig_packing": tune.choice(["int64"]),
            "magicpig_seed": tune.choice([42]),
        }
    )
    return GenericOptimizerAdapter(composite_optimizer)

def create_local_masker_optimizer() -> GenericOptimizerAdapter:
    """Create optimizer for LocalMasker configurations."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import LocalMaskerConfig
    
    generic_optimizer = create_optimizer_for_config(
        config_class=LocalMaskerConfig,
        config_name="local_masker",
        overrides={
            "window_size": tune.uniform(0.1, 0.8),
        }
    )
    return GenericOptimizerAdapter(generic_optimizer)

def create_sink_masker_optimizer() -> GenericOptimizerAdapter:
    """Create optimizer for SinkMasker configurations."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import SinkMaskerConfig
    
    generic_optimizer = create_optimizer_for_config(
        config_class=SinkMaskerConfig,
        config_name="sink_masker",
        overrides={
            "sink_size": tune.uniform(0.05, 0.3),
        }
    )
    return GenericOptimizerAdapter(generic_optimizer)

def create_adaptive_sampling_optimizer() -> GenericOptimizerAdapter:
    """Create optimizer for AdaptiveSampling configurations."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.adaptive_sampling import AdaptiveSamplingMaskerConfig
    
    generic_optimizer = create_optimizer_for_config(
        config_class=AdaptiveSamplingMaskerConfig,
        config_name="adaptive_sampling",
        overrides={
            "base_rate_sampling": tune.uniform(0.05, 0.3),
            "epsilon": tune.uniform(0.01, 0.2),
            "delta": tune.uniform(0.01, 0.1),
        }
    )
    return GenericOptimizerAdapter(generic_optimizer)

def create_hash_attention_optimizer() -> GenericOptimizerAdapter:
    """Create composite optimizer for HashAttention with sink, local, and oracle maskers."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import (
        SinkMaskerConfig, LocalMaskerConfig
    )
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.oracle_top_k import OracleTopKConfig
    
    # Create composite optimizer with all three masker types (matching experimental setup)
    composite_optimizer = create_composite_optimizer(
        masker_configs=[SinkMaskerConfig, LocalMaskerConfig, OracleTopKConfig],
        config_name="hash_attention",
        overrides={
            # SinkMasker overrides - using smaller sizes as per experimental setup
            "sinkmasker_sink_size": tune.choice([32, 64, 128]),
            # LocalMasker overrides - window_size parameter  
            "localmasker_window_size": tune.choice([64, 128, 256, 512]),
            # OracleTopK overrides - heavy_size parameter
            "oracletopk_heavy_size": tune.choice([128, 256, 512]),
        }
    )
    return GenericOptimizerAdapter(composite_optimizer)


# Registry of available optimizers - now using generic optimizers
SPARSE_OPTIMIZERS: Dict[str, SparseConfigOptimizer] = {
    "magic_pig": create_magic_pig_optimizer(),
    "local_masker": create_local_masker_optimizer(), 
    "sink_masker": create_sink_masker_optimizer(),
    "adaptive_sampling": create_adaptive_sampling_optimizer(),
    "hash_attention": create_hash_attention_optimizer(),
}


def get_sparse_optimizer(config_type: str) -> SparseConfigOptimizer:
    """Get a sparse optimizer by config type name.
    
    Args:
        config_type: The config type name (e.g., "magic_pig", "local_masker")
        
    Returns:
        The optimizer instance
        
    Raises:
        ValueError: If config_type is not found
    """
    if config_type not in SPARSE_OPTIMIZERS:
        available = list(SPARSE_OPTIMIZERS.keys())
        raise ValueError(f"Unknown config type: {config_type}. Available: {available}")
    
    return SPARSE_OPTIMIZERS[config_type]


def list_available_optimizers() -> List[str]:
    """List all available optimizer config types."""
    return list(SPARSE_OPTIMIZERS.keys())

# Legacy MagicPigOptimizer for backward compatibility
class MagicPigOptimizer(SparseConfigOptimizer):
    """Legacy hyperparameter optimizer for Magic Pig attention.
    
    This class is kept for backward compatibility. New code should use
    the generic optimizer via create_magic_pig_optimizer().
    """
    
    def get_search_space(self) -> Dict[str, Any]:
        return {
            "sink_size": tune.choice([16, 32, 64, 96, 128]),
            "window_size": tune.choice([16, 32, 64, 96, 128]),
            "lsh_l": tune.choice([16, 32, 64, 96, 128]),
            "lsh_k": tune.choice([4, 8, 16, 32, 64]),  # Remove 128 for int64 compatibility
            "center": tune.choice([True]),
            "packing": tune.choice(["int64"]),
            "seed": tune.choice([42])
        }
    
    def create_config_from_params(self, params: Dict[str, Any]) -> SparseAttentionConfig:
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig, SinkMaskerConfig
        )

        masker_configs = [
            SinkMaskerConfig(sink_size=params.get("sink_size", 64)),
            LocalMaskerConfig(window_size=params.get("window_size", 128)),
            MagicPigConfig(
                lsh_l=params.get("lsh_l", 8),
                lsh_k=params.get("lsh_k", 32),
                center=params.get("center", True),
                packing=params.get("packing", "int64"),
                seed=params.get("seed", 42)
            )
        ]
        
        return ResearchAttentionConfig(masker_configs=masker_configs)
    
    def get_default_request_kwargs(self, benchmark_name: str, subset: Optional[str]) -> Dict[str, Any]:
        # Task-specific optimization parameters - using minimal requests for quick iteration
        if benchmark_name == "loogle":
            if subset and "longdep" in subset:
                return {"max_requests": 2, "max_context_length": 512}  # Ultra-fast iteration
            else:
                return {"max_requests": 2, "max_context_length": 512}  # Ultra-fast iteration
        else:
            return {"max_requests": 2, "max_context_length": 512}  # Ultra-fast iteration


def create_sparse_config_from_params(config_type: str, params: Dict[str, Any]) -> SparseAttentionConfig:
    """Factory function to create sparse config from optimized parameters."""
    if config_type not in SPARSE_OPTIMIZERS:
        raise ValueError(f"Unknown sparse config type: {config_type}")
    
    return SPARSE_OPTIMIZERS[config_type].create_config_from_params(params)


def register_config_optimizer(config_class: Type, config_name: str, overrides: Optional[Dict[str, Any]] = None) -> None:
    """Register a new config type for optimization.
    
    This function allows easy registration of new config types without modifying
    the core optimization code.
    
    Args:
        config_class: The dataclass config type to optimize
        config_name: Name for the optimizer registry
        overrides: Optional manual overrides for specific fields
        
    Example:
        >>> from sparse_attention_hub.sparse_attention.research_attention.maskers import RandomSamplingMaskerConfig
        >>> register_config_optimizer(
        ...     RandomSamplingMaskerConfig, 
        ...     "random_sampling",
        ...     overrides={"sampling_rate": tune.uniform(0.1, 0.9)}
        ... )
        >>> # Now "random_sampling" can be used in optimization
    """
    generic_optimizer = create_optimizer_for_config(config_class, config_name, overrides)
    adapter = GenericOptimizerAdapter(generic_optimizer)
    SPARSE_OPTIMIZERS[config_name] = adapter


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
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to create cache directory {self.cache_dir}: {e}")
        self.logger = logging.getLogger(__name__)
    
    def get_cache_key(self, model_name: str, config_type: str, benchmark_name: str, subset: Optional[str]) -> str:
        """Generate cache key for optimization results."""
        subset_str = f"_{subset}" if subset else ""
        # Clean up the strings to avoid filesystem issues
        clean_model = model_name.replace("/", "_").replace("-", "_").replace(":", "_")
        clean_config = config_type.replace("/", "_").replace("-", "_").replace(":", "_")
        clean_benchmark = benchmark_name.replace("/", "_").replace("-", "_").replace(":", "_")
        clean_subset = subset_str.replace("/", "_").replace("-", "_").replace(":", "_") if subset_str else ""
        
        cache_key = f"{clean_model}_{clean_config}_{clean_benchmark}{clean_subset}"
        
        # Ensure the key is not too long for filesystem limits
        if len(cache_key) > 200:
            # Hash the key if it's too long
            import hashlib
            cache_key = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
            
        return cache_key
    
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
            self.logger.info(f"Created optimized {config_type} config from cache for {benchmark_name}/{subset}")
            return optimized_config
        except Exception as e:
            self.logger.error(f"Failed to create config from cached params: {e}")
            return None
    
    def get_all_cached_configs_for_type(self, model_name: str, config_type: str) -> Dict[Tuple[str, Optional[str]], Dict[str, Any]]:
        """Get all cached configurations for a specific model and config type.
        
        Args:
            model_name: Name of the model
            config_type: Type of sparse attention config
            
        Returns:
            Dict mapping (benchmark_name, subset) tuples to cached config results
        """
        all_configs = {}
        
        # Look for all cache files matching the pattern
        cache_pattern = f"{model_name.replace('/', '_').replace('-', '_')}_{config_type}_*"
        
        for cache_file in self.cache_dir.glob(f"{cache_pattern}.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_result = json.load(f)
                
                benchmark_name = cached_result.get("benchmark_name")
                subset = cached_result.get("subset")
                
                if benchmark_name:
                    task_key = (benchmark_name, subset)
                    all_configs[task_key] = cached_result
                    
            except (IOError, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error loading cache file {cache_file}: {e}")
        
        return all_configs
    
    def create_optimized_sparse_config_from_cache(self, model_name: str, config_type: str, use_per_task: bool = True) -> Optional[OptimizedSparseConfig]:
        """Create an OptimizedSparseConfig from all cached results for a config type.
        
        Args:
            model_name: Name of the model
            config_type: Type of sparse attention config
            use_per_task: Whether to enable per-task configuration usage
            
        Returns:
            OptimizedSparseConfig with all cached results, or None if no cache found
        """
        all_cached_configs = self.get_all_cached_configs_for_type(model_name, config_type)
        
        if not all_cached_configs:
            self.logger.info(f"No cached configurations found for {model_name}/{config_type}")
            return None
        
        # Find the globally best configuration (lowest combined_score)
        best_global_result = min(
            all_cached_configs.values(), 
            key=lambda x: x["best_metrics"]["combined_score"]
        )
        
        # Extract per-task configurations
        per_task_configs = {}
        for task_key, cached_result in all_cached_configs.items():
            per_task_configs[task_key] = cached_result["best_params"]
        
        # Create OptimizedSparseConfig
        optimized_config = OptimizedSparseConfig(
            config_type=config_type,
            base_config=None,  # We'll use optimized params
            optimized_params=best_global_result["best_params"],
            optimization_metadata=best_global_result["optimization_metadata"],
            is_optimized=True,
            per_task_configs=per_task_configs
        )
        
        self.logger.info(f"Created OptimizedSparseConfig for {config_type} with {len(per_task_configs)} per-task configs")
        return optimized_config

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
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                self.logger.warning(f"Failed to load cache {cache_file}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error loading cache {cache_file}: {e}")
        return None
    
    def save_optimization_results(self, cache_key: str, results: Dict[str, Any]) -> None:
        """Save optimization results to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            # Ensure parent directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved optimization results to {cache_file}")
        except (IOError, OSError, UnicodeEncodeError) as e:
            self.logger.error(f"Failed to save cache {cache_file}: {e}")
            # Check if it's a permission issue
            if "Permission denied" in str(e):
                self.logger.error(f"Permission denied - check write permissions for {self.cache_dir}")
            elif "No space left on device" in str(e):
                self.logger.error(f"Disk full - cannot write to {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving cache {cache_file}: {e}")
    
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
            combined_score = 0.0
            for metric, weight in weights.items():
                metric_value = attention_metrics.get(metric, 0.0)
                # Handle NaN values by replacing with a penalty
                if isinstance(metric_value, float) and (metric_value != metric_value):  # Check for NaN
                    metric_value = 10.0  # Penalty for NaN values
                combined_score += metric_value * weight
            
            # Add penalty for high density
            if attention_metrics.get("density", 0.0) > 0.5:
                combined_score += 5.0
            
            # Ensure combined_score is not NaN
            if isinstance(combined_score, float) and (combined_score != combined_score):  # Check for NaN
                combined_score = 20.0  # High penalty for NaN combined score
            
            metrics = {
                "attention_error": attention_metrics.get("attention_error", 10.0),
                "density": attention_metrics.get("density", 1.0),
                "combined_score": combined_score,
                "benchmark_score": benchmark_results.get("overall_score", 0.0)
            }
            
            # Clean up
            del adapter
            if TORCH_AVAILABLE and torch.cuda.is_available():
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
                with open(micro_metrics_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                metric_name = data.get('metric', '')
                                value = data.get('value')
                                
                                if metric_name == 'research_attention_output_error' and value is not None:
                                    # Handle NaN values properly
                                    try:
                                        error_value = float(value)
                                        if not (error_value != error_value):  # Check if not NaN
                                            attention_errors.append(error_value)
                                    except (ValueError, TypeError):
                                        pass  # Skip invalid values
                                elif metric_name == 'research_attention_density' and value is not None:
                                    try:
                                        density_value = float(value)
                                        if not (density_value != density_value):  # Check if not NaN  
                                            densities.append(density_value)
                                    except (ValueError, TypeError):
                                        pass  # Skip invalid values
                            except (json.JSONDecodeError, ValueError, TypeError) as e:
                                self.logger.debug(f"Skipping malformed line in metrics: {e}")
                                continue
            except (IOError, OSError, UnicodeDecodeError) as e:
                self.logger.warning(f"Error reading micro metrics: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error reading micro metrics: {e}")
        
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
        # Skip dense and fixed sparse configs (no optimization needed)
        if config_name == "dense" or config_name == "streaming_conservative":
            optimized_configs.append((config_name, OptimizedSparseConfig(
                config_type=config_name,
                base_config=base_config,
                is_optimized=False
            )))
            continue
        
        # For sparse configs, run optimization for each (model, benchmark, subset) combination
        all_optimizations = []
        per_task_configs = {}
        
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
                        
                        # Store per-task config for this (benchmark, subset) combination
                        task_key = (benchmark_config.benchmark_name, subset)
                        per_task_configs[task_key] = result["best_params"]
                        
                    except Exception as e:
                        logging.error(f"Optimization failed for {model_name}/{config_name}/{benchmark_config.benchmark_name}/{subset}: {e}")
        
        if all_optimizations:
            # Use the best optimization result (lowest combined_score) as global best
            best_optimization = min(all_optimizations, key=lambda x: x["best_metrics"]["combined_score"])
            
            optimized_config = OptimizedSparseConfig(
                config_type=config_name,
                base_config=base_config,
                optimized_params=best_optimization["best_params"],  # Global best
                optimization_metadata=best_optimization["optimization_metadata"],
                is_optimized=True,
                per_task_configs=per_task_configs  # Per-task configs
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
