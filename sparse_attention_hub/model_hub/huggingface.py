"""HuggingFace model hub implementation (bare metal)."""

from typing import Any, Callable, Dict, Optional

from .base import ModelHub


class ModelHubHF(ModelHub):
    """HuggingFace-specific model hub implementation."""

    def __init__(self, api_token: Optional[str] = None) -> None:
        """Initialize HF model hub."""
        super().__init__(api_token)
        self._original_attention_interfaces: Dict[str, Any] = {}
        self._registered_hooks: Dict[str, Any] = {}

    def addPreAttentionHooks(
        self, model: Any, hook_generator: Callable, hook_name: str
    ) -> None:
        """Add pre-attention hooks to HuggingFace model.

        Args:
            model: HuggingFace model instance
            hook_generator: Function that generates the hook
            hook_name: Name identifier for the hook
        """
        # Bare metal implementation - no functionality
        pass

    def removePreAttentionHooks(self, model: Any, hook_name: str) -> None:
        """Remove pre-attention hooks from HuggingFace model.

        Args:
            model: HuggingFace model instance
            hook_name: Name identifier for the hook to remove
        """
        # Bare metal implementation - no functionality
        pass

    def replaceAttentionInterface(
        self,
        model: Any,
        attention_interface_callable: Callable,
        attention_interface_name: str,
    ) -> None:
        """Replace attention interface in HuggingFace model.

        Args:
            model: HuggingFace model instance
            attention_interface_callable: New attention interface function
            attention_interface_name: Name identifier for the interface
        """
        # Bare metal implementation - no functionality
        pass

    def revertAttentionInterface(self, model: Any) -> None:
        """Revert attention interface to original in HuggingFace model.

        Args:
            model: HuggingFace model instance
        """
        # Bare metal implementation - no functionality
        pass
