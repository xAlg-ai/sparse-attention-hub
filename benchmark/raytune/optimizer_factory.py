"""
Optimizer Factory for Sparse Attention Configurations.

This module provides the core engine for creating optimizer objects that can
translate sparse attention masker configurations into Ray Tune search spaces.

The key design principle is that each masker's configuration class is responsible
for defining its own tunable parameters via a `get_search_space()` static method.
This factory then assembles these individual search spaces for optimization.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Optional

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)

class SparseConfigOptimizer(ABC):
    """
    Abstract Base Class for sparse attention config optimizers.

    An optimizer's main responsibilities are to create a search space for Ray Tune
    and to instantiate a valid attention configuration from a set of parameters
    produced by a Ray Tune trial.
    """

    @abstractmethod
    def create_search_space(self, task_name: str) -> Dict[str, Any]:
        """Creates the Ray Tune search space for a given task."""
        pass

    @abstractmethod
    def create_config_from_params(self, params: Dict[str, Any]) -> Any:
        """Creates an attention configuration instance from a dictionary of parameters."""
        pass

class SingleConfigOptimizer(SparseConfigOptimizer):
    """Optimizer for a single, non-composite masker configuration class."""

    def __init__(self, config_class: Type):
        if not hasattr(config_class, "get_search_space"):
            raise TypeError(
                f"Config class {config_class.__name__} must implement a "
                "`get_search_space(task_name)` static method."
            )
        self.config_class = config_class

    def create_search_space(self, task_name: str) -> Dict[str, Any]:
        return self.config_class.get_search_space(task_name)

    def create_config_from_params(self, params: Dict[str, Any]) -> Any:
        return self.config_class(**params)

class CompositeConfigOptimizer(SparseConfigOptimizer):
    """Optimizer for a `ResearchAttentionConfig` composed of multiple maskers."""

    def __init__(self, masker_configs: List[Type], template_config: Optional[ResearchAttentionConfig] = None):
        self.masker_configs = []
        self.template_config = template_config
        
        # Create a mapping from masker class to template instance if template is provided
        self.template_instances = {}
        if template_config:
            for template_masker in template_config.masker_configs:
                self.template_instances[type(template_masker)] = template_masker
        
        for masker_class in masker_configs:
            if not hasattr(masker_class, "get_search_space"):
                raise TypeError(
                    f"Masker config {masker_class.__name__} must implement a "
                    "`get_search_space(task_name)` static method."
                )
            self.masker_configs.append(masker_class)

    def create_search_space(self, task_name: str) -> Dict[str, Any]:
        """
        Creates a combined search space from all component masker configs.
        Each parameter is prefixed with its masker's name to prevent conflicts.
        """
        combined_space = {}
        for masker_class in self.masker_configs:
            masker_name = masker_class.__name__.lower().replace("config", "")
            masker_space = masker_class.get_search_space(task_name)
            for param_name, param_space in masker_space.items():
                combined_space[f"{masker_name}_{param_name}"] = param_space
        return combined_space

    def create_config_from_params(self, params: Dict[str, Any]) -> ResearchAttentionConfig:
        """Creates a ResearchAttentionConfig instance from the combined parameters."""
        masker_instances = []
        for masker_class in self.masker_configs:
            masker_name = masker_class.__name__.lower().replace("config", "")
            prefix = f"{masker_name}_"
            masker_params = {
                k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)
            }
            
            # If we have a template for this masker type, use its fixed parameters
            if masker_class in self.template_instances:
                template_masker = self.template_instances[masker_class]
                # Get all attributes from the template
                template_dict = {}
                for attr in dir(template_masker):
                    if not attr.startswith('_') and not callable(getattr(template_masker, attr)):
                        try:
                            value = getattr(template_masker, attr)
                            # Only include simple types that can be serialized
                            if isinstance(value, (int, float, str, bool, type(None))):
                                template_dict[attr] = value
                        except:
                            pass
                
                # Update template with search params (search params override template)
                template_dict.update(masker_params)
                masker_instances.append(masker_class(**template_dict))
            else:
                masker_instances.append(masker_class(**masker_params))
                
        return ResearchAttentionConfig(masker_configs=masker_instances)

def create_optimizer(masker_configs: List[Type], template_config: Optional[ResearchAttentionConfig] = None) -> SparseConfigOptimizer:
    """
    Factory function to create the appropriate optimizer.

    This function inspects the list of masker configurations and returns the
    correct optimizer type.
    
    Args:
        masker_configs: List of masker configuration classes to optimize
        template_config: Optional template configuration with fixed parameters
    """
    if not isinstance(masker_configs, list) or not masker_configs:
        raise ValueError("`masker_configs` must be a non-empty list of config classes.")

    logging.info(f"Creating optimizer for: {[c.__name__ for c in masker_configs]}")

    if len(masker_configs) == 1:
        return SingleConfigOptimizer(masker_configs[0])
    return CompositeConfigOptimizer(masker_configs, template_config)