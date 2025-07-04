"""Utility functions for masked attention computation."""

from typing import Any, Optional

import torch

from .mask import Mask


def _compute_masked_exp_attention_weights(
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    sparse_attention_mask: Mask,
) -> torch.Tensor:
    """Compute masked attention weights (common logic for numerator and denominator).
    
    Args:
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h, sq, d)
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        sparse_attention_mask: Mask object for sparse attention
        
    Returns:
        Masked exponential attention weights tensor of shape (b, h, sq, sk)
    """
    raw_attention_weights = torch.matmul(queries, keys.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        raw_attention_weights = raw_attention_weights + attention_mask[:, :, :, : keys.shape[-2]]
    
    row_wise_max = torch.max(raw_attention_weights, dim=-1, keepdim=True)[0]
    raw_attention_weights = raw_attention_weights - row_wise_max
    exp_attention_weights = torch.exp(raw_attention_weights)
    
    if not sparse_attention_mask.is_empty():
        exp_attention_weights = sparse_attention_mask.apply_mask(exp_attention_weights)
    
    return exp_attention_weights


def get_attention_denominator(
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    **kwargs
) -> torch.Tensor:
    """Get masked attention denominator.
    
    Args:
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h, sq, d)
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments
        
    Returns:
        Denominator tensor of shape (b, h, sq, 1)
    """
    exp_attention_weights = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
    )
    
    return torch.sum(exp_attention_weights, dim=-1, keepdim=True)


def get_attention_numerator(
    module: Any,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    **kwargs
) -> torch.Tensor:
    """Get masked attention numerator.
    
    Args:
        module: The attention module (unused in this implementation)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h, sq, d)
        values: Value tensor of shape (b, h, sq, d)
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments
        
    Returns:
        Numerator tensor of shape (b, h, sq, d)
    """
    exp_attention_weights = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
    )
    
    return torch.matmul(exp_attention_weights, values)


def get_masked_attention_output(
    module: Any,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    **kwargs
) -> torch.Tensor:
    """Get masked attention output by dividing numerator by denominator.
    
    Args:
        module: The attention module (unused in this implementation)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h, sq, d)
        values: Value tensor of shape (b, h, sq, d)
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments
        
    Returns:
        Attention output tensor of shape (b, h, sq, d)
    """
    num = get_attention_numerator(
        module=module,
        queries=queries,
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_attention_mask=sparse_attention_mask,
        **kwargs
    )
    
    den = get_attention_denominator(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        sparse_attention_mask=sparse_attention_mask,
        **kwargs
    )
    
    return (num / den).transpose(1,2).contiguous() 