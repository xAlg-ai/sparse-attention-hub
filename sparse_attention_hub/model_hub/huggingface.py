"""HuggingFace model hub implementation."""

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
        if hook_name in self._registered_hooks:
            self.removePreAttentionHooks(model, hook_name)
        
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or hasattr(module, 'attention'):
                hook = module.register_forward_pre_hook(hook_generator())
                hooks.append((module, hook))
        
        self._registered_hooks[hook_name] = hooks

    def removePreAttentionHooks(self, model: Any, hook_name: str) -> None:
        """Remove pre-attention hooks from HuggingFace model.

        Args:
            model: HuggingFace model instance
            hook_name: Name identifier for the hook to remove
        """
        if hook_name in self._registered_hooks:
            for module, hook in self._registered_hooks[hook_name]:
                hook.remove()
            del self._registered_hooks[hook_name]

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
        if attention_interface_name not in self._original_attention_interfaces:
            self._original_attention_interfaces[attention_interface_name] = {}
        
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                # Store original forward method
                self._original_attention_interfaces[attention_interface_name][name] = module.forward
                # Replace with custom attention
                module.forward = attention_interface_callable

    def revertAttentionInterface(self, model: Any) -> None:
        """Revert attention interface to original in HuggingFace model.

        Args:
            model: HuggingFace model instance
        """
        for interface_name, modules in self._original_attention_interfaces.items():
            for name, module in model.named_modules():
                if name in modules:
                    module.forward = modules[name]
        self._original_attention_interfaces.clear()
