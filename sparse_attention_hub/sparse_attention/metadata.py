"""Metadata management for sparse attention mechanisms."""

from typing import Any, Dict


class SparseAttentionMetadata:
    """Manages metadata for sparse attention mechanisms."""

    def __init__(self):
        self.layer_wise_state: Dict[str, Any] = {}
        self.global_state_: Dict[str, Any] = {}

    def update_layer_state(self, layer_id: str, state: Dict[str, Any]) -> None:
        """Update state for a specific layer."""
        self.layer_wise_state[layer_id] = state

    def update_global_state(self, state: Dict[str, Any]) -> None:
        """Update global state."""
        self.global_state_.update(state)

    def get_layer_state(self, layer_id: str) -> Dict[str, Any]:
        """Get state for a specific layer."""
        return self.layer_wise_state.get(layer_id, {})

    def get_global_state(self) -> Dict[str, Any]:
        """Get global state."""
        return self.global_state_
