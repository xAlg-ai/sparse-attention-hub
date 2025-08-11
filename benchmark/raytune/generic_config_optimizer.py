"""Task-specific config optimizer for sparse attention configs.

This module provides optimizers that work with masker configs that define their own
search spaces, enabling per-task optimization and caching.

Key Features:
- Each masker config defines its own get_search_space() method
- Per-task optimization and caching
- Support for composite configs (ResearchAttentionConfig with multiple maskers)
- Task-specific parameter tuning
- Benchmark integration
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, List

from ray import tune


class SparseConfigOptimizer(ABC):
    """Base class for sparse attention config optimizers."""

    @abstractmethod
    def create_search_space(self, task_name: str) -> Dict[str, Any]:
        """Create Ray Tune search space for the config type and task."""
        pass

    @abstractmethod
    def create_config_from_params(self, params: Dict[str, Any]) -> Any:
        """Create config instance from optimization parameters."""
        pass

    @abstractmethod
    def optimize_for_task(self, task_name: str, num_samples: int = 10) -> Any:
        """Run optimization for a specific task and return best config."""
        pass

    @property
    @abstractmethod
    def config_type_name(self) -> str:
        """Get the name of the config type for caching."""
        pass


class CompositeConfigOptimizer(SparseConfigOptimizer):
    """Optimizer for composite configs like ResearchAttentionConfig with multiple maskers."""
    
    def __init__(self, masker_configs: List[Type], config_name: str, overrides: Optional[Dict[str, Any]] = None):
        """Initialize composite optimizer.
        
        Args:
            masker_configs: List of masker config classes to optimize
            config_name: Name for caching purposes
            overrides: Optional manual overrides for specific fields (prefixed by masker name)
        """
        self.masker_configs = masker_configs
        self._config_name = config_name
        self.overrides = overrides or {}
        self.logger = logging.getLogger(__name__)
        
        # Validate that all masker configs have get_search_space method
        for masker_class in masker_configs:
            if not hasattr(masker_class, 'get_search_space'):
                raise ValueError(f"Masker config {masker_class.__name__} must implement get_search_space() method")
        
        # Cache for task-specific best configs
        self.task_cache = {}
    
    def create_search_space(self, task_name: str) -> Dict[str, Any]:
        """Create combined search space from all masker configs for a specific task."""
        combined_space = {}
        
        for masker_class in self.masker_configs:
            masker_name = masker_class.__name__.lower().replace('config', '')
            
            # Get search space from the masker config class
            masker_space = masker_class.get_search_space(task_name)
            
            # Apply any overrides for this masker
            prefix = f"{masker_name}_"
            for key, value in self.overrides.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    masker_space[param_name] = value
            
            # Prefix each parameter with masker name to avoid conflicts
            for param_name, param_space in masker_space.items():
                combined_space[f"{masker_name}_{param_name}"] = param_space
                
        return combined_space
    
    def create_config_from_params(self, params: Dict[str, Any]) -> Any:
        """Create ResearchAttentionConfig from optimization parameters."""
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        
        masker_instances = []
        
        for masker_class in self.masker_configs:
            masker_name = masker_class.__name__.lower().replace('config', '')
            
            # Extract parameters for this masker
            masker_params = {}
            prefix = f"{masker_name}_"
            for param_name, param_value in params.items():
                if param_name.startswith(prefix):
                    masker_params[param_name[len(prefix):]] = param_value
            
            # Create masker instance
            masker_instance = masker_class(**masker_params)
            masker_instances.append(masker_instance)
        
        return ResearchAttentionConfig(masker_configs=masker_instances)
    
    def optimize_for_task(self, task_name: str, num_samples: int = 10) -> Any:
        """Run optimization for a specific task and return best config."""
        # Check cache first
        cache_key = f"{task_name}_{num_samples}"
        if cache_key in self.task_cache:
            self.logger.info(f"Using cached best config for task {task_name}")
            return self.task_cache[cache_key]
        
        self.logger.info(f"Starting optimization for task {task_name} with {num_samples} samples")
        
        # Create search space for this task
        search_space = self.create_search_space(task_name)
        
        # Run Ray Tune optimization
        analysis = tune.run(
            self._objective_function,
            config=search_space,
            num_samples=num_samples,
            resources_per_trial={"cpu": 1, "gpu": 0.25},
            name=f"optimize_{self._config_name}_{task_name}",
            local_dir="./ray_results"
        )
        
        # Get best config
        best_trial = analysis.get_best_trial("score", "max", "last")
        best_config = self.create_config_from_params(best_trial.config)
        
        # Cache the result
        self.task_cache[cache_key] = best_config
        
        self.logger.info(f"Best config for {task_name}: {best_config}")
        return best_config
    
    def _objective_function(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Objective function for Ray Tune optimization."""
        # Create config instance
        attention_config = self.create_config_from_params(config)
        
        # TODO: Integrate with benchmark runner
        # For now, return random score - replace with actual benchmark
        import random
        score = random.random()
        
        return {"score": score}
    
    @property
    def config_type_name(self) -> str:
        """Get the name of the config type for caching."""
        return self._config_name


class SingleConfigOptimizer(SparseConfigOptimizer):
    """Optimizer for single masker configs."""
    
    def __init__(self, config_class: Type, config_name: str, overrides: Optional[Dict[str, Any]] = None):
        """Initialize single config optimizer.
        
        Args:
            config_class: The masker config class to optimize
            config_name: Name for caching purposes
            overrides: Optional manual overrides for specific fields
        """
        self.config_class = config_class
        self._config_name = config_name
        self.overrides = overrides or {}
        self.logger = logging.getLogger(__name__)
        
        # Validate that the config class has get_search_space method
        if not hasattr(config_class, 'get_search_space'):
            raise ValueError(f"Config class {config_class.__name__} must implement get_search_space() method")
        
        # Cache for task-specific best configs
        self.task_cache = {}
    
    def create_search_space(self, task_name: str) -> Dict[str, Any]:
        """Create search space from the config class for a specific task."""
        search_space = self.config_class.get_search_space(task_name)
        
        # Apply any overrides
        for key, value in self.overrides.items():
            search_space[key] = value
                
        return search_space
    
    def create_config_from_params(self, params: Dict[str, Any]) -> Any:
        """Create config instance from optimization parameters."""
        return self.config_class(**params)
    
    def optimize_for_task(self, task_name: str, num_samples: int = 10) -> Any:
        """Run optimization for a specific task and return best config."""
        # Check cache first
        cache_key = f"{task_name}_{num_samples}"
        if cache_key in self.task_cache:
            self.logger.info(f"Using cached best config for task {task_name}")
            return self.task_cache[cache_key]
        
        self.logger.info(f"Starting optimization for task {task_name} with {num_samples} samples")
        
        # Create search space for this task
        search_space = self.create_search_space(task_name)
        
        # Run Ray Tune optimization
        analysis = tune.run(
            self._objective_function,
            config=search_space,
            num_samples=num_samples,
            resources_per_trial={"cpu": 1, "gpu": 0.25},
            name=f"optimize_{self._config_name}_{task_name}",
            local_dir="./ray_results"
        )
        
        # Get best config
        best_trial = analysis.get_best_trial("score", "max", "last")
        best_config = self.create_config_from_params(best_trial.config)
        
        # Cache the result
        self.task_cache[cache_key] = best_config
        
        self.logger.info(f"Best config for {task_name}: {best_config}")
        return best_config
    
    def _objective_function(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Objective function for Ray Tune optimization."""
        # Create config instance
        attention_config = self.create_config_from_params(config)
        
        # TODO: Integrate with benchmark runner
        # For now, return random score - replace with actual benchmark
        import random
        score = random.random()
        
        return {"score": score}
    
    @property
    def config_type_name(self) -> str:
        """Get the name of the config type for caching."""
        return self._config_name


def create_optimizer_for_config(config_class: Type, config_name: str, overrides: Optional[Dict[str, Any]] = None) -> SingleConfigOptimizer:
    """Factory function to create a single config optimizer.
    
    Args:
        config_class: The masker config class to optimize
        config_name: Name for caching purposes
        overrides: Optional manual overrides for specific fields
        
    Returns:
        SingleConfigOptimizer instance
        
    Example:
        >>> from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import LocalMaskerConfig
        >>> optimizer = create_optimizer_for_config(
        ...     LocalMaskerConfig, 
        ...     "local_masker"
        ... )
        >>> best_config = optimizer.optimize_for_task("longbench_qasper", num_samples=20)
    """
    return SingleConfigOptimizer(config_class, config_name, overrides)


def auto_create_composite_optimizer(masker_configs: List[Type], config_name: str, overrides: Optional[Dict[str, Any]] = None) -> CompositeConfigOptimizer:
    """Factory function to create a composite optimizer with automatic search space discovery.
    
    This is similar to create_composite_optimizer but emphasizes that it uses auto-discovery.
    
    Args:
        masker_configs: List of masker config classes to optimize
        config_name: Name for caching purposes
        overrides: Optional manual overrides for specific fields (prefixed by masker name)
        
    Returns:
        CompositeConfigOptimizer instance
        
    Example:
        >>> from sparse_attention_hub.sparse_attention.research_attention.maskers import MagicPigConfig, LocalMaskerConfig
        >>> optimizer = auto_create_composite_optimizer(
        ...     [MagicPigConfig, LocalMaskerConfig], 
        ...     "magic_pig_local"
        ... )
        >>> best_config = optimizer.optimize_for_task("longbench_qasper", num_samples=20)
    """
    return create_composite_optimizer(masker_configs, config_name, overrides)


def create_composite_optimizer(masker_configs: List[Type], config_name: str, overrides: Optional[Dict[str, Any]] = None) -> CompositeConfigOptimizer:
    """Factory function to create a composite optimizer for ResearchAttentionConfig.
    
    Args:
        masker_configs: List of masker config classes to optimize
        config_name: Name for caching purposes
        overrides: Optional manual overrides for specific fields (prefixed by masker name)
        
    Returns:
        CompositeConfigOptimizer instance
        
    Example:
        >>> from sparse_attention_hub.sparse_attention.research_attention.maskers import MagicPigConfig, LocalMaskerConfig
        >>> optimizer = create_composite_optimizer(
        ...     [MagicPigConfig, LocalMaskerConfig], 
        ...     "magic_pig_local",
        ...     overrides={"magicpig_lsh_l": tune.choice([4, 8, 12])}
        ... )
        >>> best_config = optimizer.optimize_for_task("longbench_qasper", num_samples=20)
    """
    return CompositeConfigOptimizer(masker_configs, config_name, overrides)


# Task-specific optimization utilities
def optimize_configs_for_all_tasks(optimizer: CompositeConfigOptimizer, 
                                 tasks: List[str], 
                                 num_samples: int = 10) -> Dict[str, Any]:
    """Optimize configs for multiple tasks.
    
    Args:
        optimizer: CompositeConfigOptimizer instance
        tasks: List of task names to optimize for
        num_samples: Number of optimization samples per task
        
    Returns:
        Dictionary mapping task names to best configs
    """
    results = {}
    for task in tasks:
        results[task] = optimizer.optimize_for_task(task, num_samples)
    return results
