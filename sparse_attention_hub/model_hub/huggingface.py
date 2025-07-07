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

        Raises:
            ValueError: If model has no named_modules method or no attention
                modules found
            RuntimeError: If hook registration fails
        """
        if not hasattr(model, "named_modules"):
            raise ValueError(
                "Model must have 'named_modules' method (expected HuggingFace model)"
            )

        if hook_name in self._registered_hooks:
            self.removePreAttentionHooks(model, hook_name)

        hooks = []
        hooks_added = 0

        try:
            for name, module in model.named_modules():
                if self._is_attention_module(name, module):
                    if not hasattr(module, "register_forward_pre_hook"):
                        continue  # Skip modules that don't support hooks

                    hook = module.register_forward_pre_hook(hook_generator())
                    hooks.append((module, hook))
                    hooks_added += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Cleanup any hooks that were successfully registered
            for module, hook in hooks:
                try:
                    hook.remove()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass  # Ignore cleanup errors
            raise RuntimeError(f"Failed to add pre-attention hooks: {exc}") from exc

        if hooks_added == 0:
            raise ValueError("No attention modules found for hook registration")

        self._registered_hooks[hook_name] = hooks

    def removePreAttentionHooks(self, model: Any, hook_name: str) -> None:
        """Remove pre-attention hooks from HuggingFace model.

        Args:
            model: HuggingFace model instance
            hook_name: Name identifier for the hook to remove
        """
        if hook_name in self._registered_hooks:
            for _, hook in self._registered_hooks[hook_name]:
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

        Raises:
            ValueError: If model has no named_modules method or no attention
                modules found
            AttributeError: If attention modules don't have forward method
        """
        if not hasattr(model, "named_modules"):
            raise ValueError(
                "Model must have 'named_modules' method (expected HuggingFace model)"
            )

        if attention_interface_name not in self._original_attention_interfaces:
            self._original_attention_interfaces[attention_interface_name] = {}

        attention_modules_found = 0

        try:
            for name, module in model.named_modules():
                # More specific matching for attention modules
                if self._is_attention_module(name, module):
                    if not hasattr(module, "forward"):
                        raise AttributeError(
                            f"Attention module '{name}' does not have 'forward' method"
                        )

                    # Store original forward method
                    self._original_attention_interfaces[attention_interface_name][
                        name
                    ] = module.forward
                    # Replace with custom attention
                    module.forward = attention_interface_callable
                    attention_modules_found += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Cleanup partial replacements on error
            self._cleanup_partial_replacement(attention_interface_name)
            raise RuntimeError(f"Failed to replace attention interface: {exc}") from exc

        if attention_modules_found == 0:
            raise ValueError(
                "No attention modules found in model. Expected modules with "
                "'attention' in name and 'forward' method."
            )

    def _is_attention_module(self, name: str, module: Any) -> bool:
        """Check if a module is an attention module.

        Args:
            name: Module name
            module: Module instance

        Returns:
            True if module appears to be an attention module
        """
        name_lower = name.lower()
        # More specific patterns for attention modules
        attention_patterns = [
            "attention",
            "self_attn",
            "self_attention",
            "cross_attn",
            "cross_attention",
        ]

        # Check if name contains attention patterns
        has_attention_name = any(
            pattern in name_lower for pattern in attention_patterns
        )

        # Additional checks for module type/attributes that suggest it's attention
        has_attention_attributes = hasattr(module, "forward") and (
            hasattr(module, "query")
            or hasattr(module, "q_proj")
            or hasattr(module, "attention")
            or hasattr(module, "attn")
            or hasattr(module, "out_proj")
            or hasattr(module, "in_proj_weight")
            or hasattr(module, "q_proj_weight")
        )  # PyTorch MultiheadAttention attributes

        return has_attention_name and has_attention_attributes

    def _cleanup_partial_replacement(self, attention_interface_name: str) -> None:
        """Clean up partial replacements on error.

        Args:
            attention_interface_name: Name of the interface being replaced
        """
        if attention_interface_name in self._original_attention_interfaces:
            # Restore any modules that were already replaced
            for (
                _,
                _,
            ) in self._original_attention_interfaces[  # pylint: disable=unused-variable
                attention_interface_name
            ].items():
                # Note: We can't easily restore without the model reference
                # This is a limitation - in practice, the model would need to be
                # reloaded
                pass
            # Remove the failed replacement record
            del self._original_attention_interfaces[attention_interface_name]

    def revertAttentionInterface(self, model: Any) -> None:
        """Revert attention interface to original in HuggingFace model.

        Args:
            model: HuggingFace model instance

        Raises:
            ValueError: If model has no named_modules method
            RuntimeError: If reversion fails
        """
        if not hasattr(model, "named_modules"):
            raise ValueError(
                "Model must have 'named_modules' method (expected HuggingFace model)"
            )

        if not self._original_attention_interfaces:
            # No interfaces to revert - this is fine, just return
            return

        reverted_count = 0
        errors = []

        try:
            # Create a mapping of module names to modules for efficient lookup
            module_dict = dict(
                model.named_modules()
            )  # pylint: disable=unnecessary-comprehension

            for (
                _,
                modules,
            ) in self._original_attention_interfaces.items():
                for module_name, original_forward in modules.items():
                    if module_name in module_dict:
                        try:
                            module_dict[module_name].forward = original_forward
                            reverted_count += 1
                        except (
                            Exception
                        ) as exc:  # pylint: disable=broad-exception-caught
                            errors.append(
                                f"Failed to revert module '{module_name}': {exc}"
                            )
                    else:
                        errors.append(
                            f"Module '{module_name}' not found in model during "
                            "reversion"
                        )

            # Clear the stored interfaces after successful reversion
            self._original_attention_interfaces.clear()

            if errors:
                error_msg = (
                    f"Reverted {reverted_count} modules but encountered errors: "
                    f"{'; '.join(errors)}"
                )
                raise RuntimeError(error_msg)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not isinstance(exc, RuntimeError):
                raise RuntimeError(
                    f"Failed to revert attention interfaces: {exc}"
                ) from exc
            raise
