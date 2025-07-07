"""Base model hub interface (bare metal)."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class ModelHub(ABC):
    """Abstract base class for model hubs."""

    def __init__(self, api_token: Optional[str] = None) -> None:
        """Initialize model hub."""
        self.api_token = api_token

    @abstractmethod
    def addPreAttentionHooks(  # pylint: disable=invalid-name
        self, model: Any, hook_generator: Callable, hook_name: str
    ) -> None:
        """Add pre-attention hooks to the model.

        Args:
            model: The model to add hooks to
            hook_generator: Function that generates the hook
            hook_name: Name identifier for the hook
        """
        # Abstract method - implementation required in subclass

    @abstractmethod
    def removePreAttentionHooks(  # pylint: disable=invalid-name
        self, model: Any, hook_name: str
    ) -> None:
        """Remove pre-attention hooks from the model.

        Args:
            model: The model to remove hooks from
            hook_name: Name identifier for the hook to remove
        """
        # Abstract method - implementation required in subclass

    @abstractmethod
    def replaceAttentionInterface(  # pylint: disable=invalid-name
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
        # Abstract method - implementation required in subclass

    @abstractmethod
    def revertAttentionInterface(  # pylint: disable=invalid-name
        self, model: Any
    ) -> None:
        """Revert the attention interface to original.

        Args:
            model: The model to revert
        """
        # Abstract method - implementation required in subclass
