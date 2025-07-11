"""Base classes for research maskers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type, TypeVar, cast

import torch

from ...utils.mask import Mask

# Type variables for the decorator
T = TypeVar('T', bound='ResearchMasker')
ConfigType = TypeVar('ConfigType', bound='MaskerConfig')


@dataclass
class MaskerConfig:
    """Base configuration class for all maskers."""


class MaskerRegistry:
    """Registry for masker classes using decorator-based auto-registration."""
    
    _registry: Dict[Type[MaskerConfig], Type['ResearchMasker']] = {}
    
    @classmethod
    def register(cls, config_type: Type[ConfigType]) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a masker class with its config type.
        
        Args:
            config_type: The configuration class type that this masker handles.
            
        Returns:
            Decorator function that registers the masker class.
            
        Example:
            @MaskerRegistry.register(LocalMaskerConfig)
            class LocalMasker(FixedMasker):
                ...
        """
        def decorator(masker_class: Type[T]) -> Type[T]:
            cls._registry[config_type] = masker_class
            return masker_class
        return decorator
    
    @classmethod
    def get_masker_class(cls, config_type: Type[MaskerConfig]) -> Type['ResearchMasker']:
        """Get the masker class for a given config type.
        
        Args:
            config_type: The configuration class type.
            
        Returns:
            The masker class that handles this config type.
            
        Raises:
            ValueError: If no masker class is registered for the config type.
        """
        masker_class = cls._registry.get(config_type)
        if masker_class is None:
            raise ValueError(f"No masker class registered for config type: {config_type}")
        return masker_class
    
    @classmethod
    def get_all_registered_types(cls) -> Dict[Type[MaskerConfig], Type['ResearchMasker']]:
        """Get all registered config types and their corresponding masker classes."""
        return cls._registry.copy()


class ResearchMasker(ABC):
    """Abstract base class for research maskers."""

    def __init__(self, config: MaskerConfig):
        """Initialize masker with configuration."""
        self.config = config

    @abstractmethod
    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[Any, Any], # want to keep it general here.
        previous_mask: Mask,
        **kwargs: Any,
    ) -> Mask:
        """Add mask to attention computation."""
        pass

    @classmethod
    @abstractmethod
    def create_from_config(cls, config: MaskerConfig) -> "ResearchMasker":
        """Create masker instance from configuration.

        Args:
            config: Configuration for the masker.

        Returns:
            Instance of the masker.
        """
        pass

    @classmethod
    def create_masker_from_config(cls, config: MaskerConfig) -> "ResearchMasker":
        """Create masker instance from configuration using the registry.

        Args:
            config: Configuration for the masker.

        Returns:
            Instance of the concrete masker class.

        Raises:
            ValueError: If no masker class is found for the config type.
                Make sure to import the masker module before calling this method.
        """
        # Get the masker class from the registry
        masker_class = MaskerRegistry.get_masker_class(type(config))
        
        # Cast to help mypy understand the type
        masker_class = cast(Type[ResearchMasker], masker_class)
        return masker_class.create_from_config(config)
