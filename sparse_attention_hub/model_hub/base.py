"""Base model hub interface."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class ModelHub(ABC):
    """Abstract base class for model hubs."""

    def __init__(self, api_token: Optional[str] = None) -> None:
        self.api_token = api_token

    @abstractmethod
    def addPreAttentionHooks(
        self, model: Any, hook_generator: Callable, hook_name: str
    ) -> None:
        """Add pre-attention hooks to the model.

        Args:
            model: The model to add hooks to
            hook_generator: Function that generates the hook
            hook_name: Name identifier for the hook
        """
        pass

    @abstractmethod
    def removePreAttentionHooks(self, model: Any, hook_name: str) -> None:
        """Remove pre-attention hooks from the model.

        Args:
            model: The model to remove hooks from
            hook_name: Name identifier for the hook to remove
        """
        pass

    @abstractmethod
    def replaceAttentionInterface(
        self,
        model: Any,
        attention_interface_callable: Callable,
        attention_interface_name: str,
    ) -> None:
        """Replace the attention interface in the model.

        Args:
            model: The model to modify
            attention_interface_callable: New attention interface function
            attention_interface_name: Name identifier for the interface
        """
        pass

    @abstractmethod
    def revertAttentionInterface(self, model: Any) -> None:
        """Revert the attention interface to original.

        Args:
            model: The model to revert
        """
        pass
