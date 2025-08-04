"""Generic config optimizer for automatic search space generation.

This module provides a generic optimizer that can introspect any dataclass config
and automatically generate Ray Tune search spaces, eliminating the need for 
config-specific optimizers.

Key Features:
- Automatic dataclass introspection and search space generation
- Type-aware search space creation (int, float, bool, string)
- Field name pattern recognition for domain-specific ranges
- Manual override support for custom search spaces
- Support for composite configs (ResearchAttentionConfig with multiple maskers)
- Extensible to any config class
"""

import logging
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional, Type, Union, get_type_hints, get_origin, get_args, List

from ray import tune


# SparseConfigOptimizer is defined in hyperparameter_optimization.py
from benchmark.optimizer.hyperparameter_optimization import SparseConfigOptimizer


class GenericConfigOptimizer(SparseConfigOptimizer):
    """Generic optimizer that can introspect any dataclass config and create search spaces."""

    def __init__(self, config_class: Type, config_name: str, search_space_overrides: Optional[Dict[str, Any]] = None):
        """Initialize generic optimizer.
        
        Args:
            config_class: The dataclass config type to optimize
            config_name: Name for caching purposes
            search_space_overrides: Optional manual overrides for specific fields
        """
        if not is_dataclass(config_class):
            raise ValueError(f"Config class {config_class} must be a dataclass")
        
        self.config_class = config_class
        self._config_name = config_name
        self.search_space_overrides = search_space_overrides or {}
        self.logger = logging.getLogger(__name__)
        
    def create_search_space(self) -> Dict[str, Any]:
        """Create Ray Tune search space by introspecting the config dataclass."""
        search_space = {}
        
        # First, check if config class defines its own default search space
        if hasattr(self.config_class, 'get_default_search_space'):
            try:
                default_search_space = self.config_class.get_default_search_space()
                if isinstance(default_search_space, dict):
                    search_space.update(default_search_space)
                    self.logger.info(f"Using default search space from {self.config_class.__name__}")
            except Exception as e:
                self.logger.warning(f"Failed to get default search space from {self.config_class.__name__}: {e}")
        
        # Get type hints for the config class
        type_hints = get_type_hints(self.config_class)
        
        # Iterate through dataclass fields for auto-generation
        for field_info in fields(self.config_class):
            field_name = field_info.name
            
            # Skip if manually overridden
            if field_name in self.search_space_overrides:
                search_space[field_name] = self.search_space_overrides[field_name]
                continue
            
            # Skip if already in default search space (unless overridden)
            if field_name in search_space:
                continue
                
            # Get field type and default value
            field_type = type_hints.get(field_name, field_info.type)
            default_value = field_info.default if field_info.default != field_info.default_factory else None
            
            # Generate search space based on type and name
            tune_space = self._create_tune_space_for_field(field_name, field_type, default_value)
            if tune_space is not None:
                search_space[field_name] = tune_space
            else:
                self.logger.debug(f"Skipping field {field_name} of type {field_type}")
        
        # Apply final overrides
        search_space.update(self.search_space_overrides)
        return search_space
    
    def _create_tune_space_for_field(self, field_name: str, field_type: Type, default_value: Any) -> Optional[Any]:
        """Create Ray Tune search space for a specific field based on its type and default value."""
        
        # Handle Union types (e.g., Union[int, float])
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # Filter out None type for Optional types
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                field_type = non_none_args[0]
            elif len(non_none_args) == 2 and (int in non_none_args and float in non_none_args):
                # Union[int, float] - treat as float
                field_type = float
            else:
                # Complex union, skip for now
                return None
        
        # Generate search space based on field type and default value
        if field_type is bool:
            return tune.choice([True, False])
        
        elif field_type is int:
            if default_value is not None and isinstance(default_value, int):
                # Create a reasonable range around the default value
                low = max(1, default_value // 2)
                high = default_value * 2
                return tune.randint(low, high + 1)
            else:
                # Generic integer choices if no default
                return tune.choice([1, 2, 4, 8, 16, 32, 64])
        
        elif field_type is float:
            if default_value is not None and isinstance(default_value, (int, float)):
                # Create a reasonable range around the default value
                low = max(0.01, float(default_value) * 0.5)
                high = float(default_value) * 2.0
                return tune.uniform(low, high)
            else:
                # Generic float range if no default
                return tune.uniform(0.1, 1.0)
        
        elif field_type is str:
            # For string fields, use the default value if available
            # Otherwise skip optimization for strings without clear choices
            if default_value is not None:
                return tune.choice([default_value])
            return None
            
        # For other types, return None to skip optimization
        return None
            
    def create_config_from_params(self, params: Dict[str, Any]) -> Any:
        """Create config instance from optimization parameters."""
        
        # Get default values from the config class
        config_kwargs = {}
        
        # Get field defaults
        for field_info in fields(self.config_class):
            field_name = field_info.name
            if field_name in params:
                config_kwargs[field_name] = params[field_name]
            elif field_info.default != field_info.default_factory:
                config_kwargs[field_name] = field_info.default
            elif field_info.default_factory != field_info.default_factory:
                config_kwargs[field_name] = field_info.default_factory()
        
        return self.config_class(**config_kwargs)
    
    @property
    def config_type_name(self) -> str:
        """Get the name of the config type for caching."""
        return self._config_name


def create_optimizer_for_config(config_class: Type, config_name: str, overrides: Optional[Dict[str, Any]] = None) -> GenericConfigOptimizer:
    """Factory function to create a generic optimizer for any config class.
    
    Args:
        config_class: The dataclass config type to optimize
        config_name: Name for caching purposes  
        overrides: Optional manual overrides for specific fields
        
    Returns:
        GenericConfigOptimizer instance
        
    Example:
        >>> from sparse_attention_hub.sparse_attention.research_attention.maskers import MagicPigConfig
        >>> optimizer = create_optimizer_for_config(
        ...     MagicPigConfig, 
        ...     "magic_pig",
        ...     overrides={"sampling_rate": tune.uniform(0.1, 0.9)}
        ... )
        >>> search_space = optimizer.create_search_space()
        >>> # Use with Ray Tune optimization
    """
    return GenericConfigOptimizer(config_class, config_name, overrides)


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
        
        # Create individual optimizers for each masker
        self.masker_optimizers = {}
        for i, masker_class in enumerate(masker_configs):
            masker_name = masker_class.__name__.lower().replace('config', '')
            # Extract overrides for this masker (prefixed with masker name)
            masker_overrides = {}
            prefix = f"{masker_name}_"
            for key, value in self.overrides.items():
                if key.startswith(prefix):
                    masker_overrides[key[len(prefix):]] = value
            
            self.masker_optimizers[masker_name] = GenericConfigOptimizer(
                masker_class, f"{masker_name}_{i}", masker_overrides
            )
    
    def create_search_space(self) -> Dict[str, Any]:
        """Create combined search space from all masker configs."""
        combined_space = {}
        
        for masker_name, optimizer in self.masker_optimizers.items():
            masker_space = optimizer.create_search_space()
            # Prefix each parameter with masker name to avoid conflicts
            for param_name, param_space in masker_space.items():
                combined_space[f"{masker_name}_{param_name}"] = param_space
                
        return combined_space
    
    def create_config_from_params(self, params: Dict[str, Any]) -> Any:
        """Create ResearchAttentionConfig from optimization parameters."""
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        
        masker_instances = []
        
        for masker_name, optimizer in self.masker_optimizers.items():
            # Extract parameters for this masker
            masker_params = {}
            prefix = f"{masker_name}_"
            for param_name, param_value in params.items():
                if param_name.startswith(prefix):
                    masker_params[param_name[len(prefix):]] = param_value
            
            # Create masker instance
            masker_instance = optimizer.create_config_from_params(masker_params)
            masker_instances.append(masker_instance)
        
        return ResearchAttentionConfig(masker_configs=masker_instances)
    
    @property
    def config_type_name(self) -> str:
        """Get the name of the config type for caching."""
        return self._config_name


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
        >>> search_space = optimizer.create_search_space()
        >>> # Creates search space with prefixed parameters: magicpig_lsh_l, localmasker_window_size, etc.
    """
    return CompositeConfigOptimizer(masker_configs, config_name, overrides)


def auto_create_composite_optimizer(masker_configs: List[Type], config_name: str) -> CompositeConfigOptimizer:
    """Auto-create composite optimizer using default search spaces from each config.
    
    This function automatically discovers and combines default search spaces from 
    each masker config without requiring manual override specification.
    
    Args:
        masker_configs: List of masker config classes to optimize
        config_name: Name for caching purposes
        
    Returns:
        CompositeConfigOptimizer instance with auto-discovered search spaces
        
    Example:
        >>> from sparse_attention_hub.sparse_attention.research_attention.maskers import MagicPigConfig, LocalMaskerConfig
        >>> # This will automatically use default search spaces from each config
        >>> optimizer = auto_create_composite_optimizer([MagicPigConfig, LocalMaskerConfig], "auto_composite")
        >>> search_space = optimizer.create_search_space()
    """
    return CompositeConfigOptimizer(masker_configs, config_name, overrides=None)


def auto_register_config(config_class: Type, config_name: Optional[str] = None) -> str:
    """Auto-register a config class for optimization with minimal setup.
    
    This function automatically creates and registers an optimizer for any config class.
    It will use the config's default search space if available, or fall back to 
    automatic field introspection.
    
    Args:
        config_class: The dataclass config type to register
        config_name: Optional name for registration (defaults to lowercase class name)
        
    Returns:
        The registered config name
        
    Example:
        >>> from sparse_attention_hub.sparse_attention.research_attention.maskers import MagicPigConfig
        >>> # Auto-register with default search space
        >>> name = auto_register_config(MagicPigConfig)  # Returns "magicpigconfig"
        >>> # Or with custom name
        >>> name = auto_register_config(MagicPigConfig, "magic_pig")  # Returns "magic_pig"
    """
    if config_name is None:
        config_name = config_class.__name__.lower()
    
    # This function will be imported by hyperparameter_optimization.py to register configs
    from benchmark.optimizer.hyperparameter_optimization import register_config_optimizer
    register_config_optimizer(config_class, config_name, overrides=None)
    return config_name
