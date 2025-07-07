"""Utility functions for sparse attention mechanisms."""

from typing import Any, Optional, Tuple, Union

import torch


class Mask:
    """Simple mask class for sparse attention."""

    def __init__(self, mask_tensor: Optional[torch.Tensor] = None):
        """Initialize mask.

        Args:
            mask_tensor: Optional mask tensor. If None, creates empty mask.
        """
        self.mask_tensor = mask_tensor

    def is_empty(self) -> bool:
        """Check if mask is empty."""
        return self.mask_tensor is None or torch.all(self.mask_tensor == 0)

    def apply_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply mask to tensor."""
        if self.is_empty():
            return tensor
        return tensor * self.mask_tensor


def get_masked_attention_output(
    module: Any,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    return_attention_weights: bool = False,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute masked attention output.

    Args:
        module: The attention module
        queries: Query tensor of shape (b, h, sq, d)
        keys: Key tensor of shape (b, h, sk, d) - should already be expanded to match query heads
        values: Value tensor of shape (b, h, sk, d) - should already be expanded to match query heads
        attention_mask: Optional attention mask
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Sparse attention mask
        return_attention_weights: Whether to return attention weights
        **kwargs: Additional arguments

    Returns:
        Attention output tensor or tuple of (output, weights)
    """
    # Compute attention scores
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling

    # Apply attention mask if provided
    if attention_mask is not None:
        # Handle different attention mask shapes
        if attention_mask.dim() == 2:
            # Shape: (seq_len, seq_len) -> expand to (batch, heads, seq_len, seq_len)
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(scores.shape)
            )
        elif attention_mask.dim() == 3:
            # Shape: (batch, seq_len, seq_len) -> expand to (batch, heads, seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(1).expand(scores.shape)
        elif attention_mask.dim() == 4:
            # Already correct shape: (batch, heads, seq_len, seq_len)
            pass
        else:
            # Unsupported shape, skip mask
            attention_mask = None

        if attention_mask is not None:
            scores = scores + attention_mask

    # Apply sparse attention mask
    if not sparse_attention_mask.is_empty():
        scores = sparse_attention_mask.apply_mask(scores)

    # Compute attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # Apply dropout if in training mode
    if dropout > 0.0 and module is not None and module.training:
        attention_weights = torch.nn.functional.dropout(
            attention_weights, p=dropout, training=True
        )

    # Compute attention output
    attention_output = torch.matmul(attention_weights, values)

    if return_attention_weights:
        return attention_output, attention_weights
    return attention_output
