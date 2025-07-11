"""Utility functions for masked attention computation."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .kv_utils import _get_num_key_value_groups, repeat_kv
from .mask import Mask


def _compute_masked_exp_attention_weights(
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    sparse_attention_mask: Mask,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Compute masked attention weights (common logic for numerator and denominator).

    Args:
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        sparse_attention_mask: Mask object for sparse attention
        dropout: Dropout probability
        training: Whether the model is in training mode

    Returns:
        Masked exponential attention weights tensor of shape (b, h, sq, sk)
    """
    # Calculate num_key_value_groups from tensor shapes
    num_key_value_groups = _get_num_key_value_groups(queries, keys)

    # Apply key-value grouping if needed
    key_states = repeat_kv(keys, num_key_value_groups)

    raw_attention_weights = torch.matmul(queries, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        raw_attention_weights = (
            raw_attention_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        )

    row_wise_max = torch.max(raw_attention_weights, dim=-1, keepdim=True)[0]
    raw_attention_weights = raw_attention_weights - row_wise_max
    exp_attention_weights = torch.exp(raw_attention_weights)

    if not sparse_attention_mask.is_empty():
        exp_attention_weights = sparse_attention_mask.apply_inv_mask(
            exp_attention_weights
        )

    # Apply dropout to attention weights if specified
    if dropout > 0.0 and training:
        exp_attention_weights = torch.nn.functional.dropout(
            exp_attention_weights, p=dropout, training=training
        )

    return exp_attention_weights


def _get_attention_denominator(exp_attention_weights: torch.Tensor) -> torch.Tensor:
    """Get attention denominator from pre-computed exponential attention weights.

    Args:
        exp_attention_weights: Pre-computed exponential attention weights of shape (b, h, sq, sk)

    Returns:
        Denominator tensor of shape (b, h, sq, 1)
    """
    return torch.sum(exp_attention_weights, dim=-1, keepdim=True)


def _get_attention_numerator(
    exp_attention_weights: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Get attention numerator from pre-computed exponential attention weights and prepared values.

    Args:
        exp_attention_weights: Pre-computed exponential attention weights of shape (b, h, sq, sk)
        value_states: Prepared value tensor of shape (b, h, sq, d) - already grouped if needed

    Returns:
        Numerator tensor of shape (b, h, sq, d)
    """
    return torch.matmul(exp_attention_weights, value_states)


def get_attention_denominator(
    module: Optional[nn.Module],
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Get masked attention denominator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments

    Returns:
        Denominator tensor of shape (b, h, sq, 1)
    """
    training = module.training if module is not None else False
    exp_attention_weights = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
    )

    return _get_attention_denominator(exp_attention_weights)


def get_attention_numerator(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Get masked attention numerator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        values: Value tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments

    Returns:
        Numerator tensor of shape (b, h, sq, d)
    """
    training = module.training if module is not None else False
    exp_attention_weights = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
    )

    # Prepare values by applying key-value grouping
    num_key_value_groups = _get_num_key_value_groups(queries, values)
    value_states = repeat_kv(values, num_key_value_groups)

    return _get_attention_numerator(exp_attention_weights, value_states)


def get_masked_attention_output(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    return_attention_weights: bool = False,
    **kwargs: Dict[str, Any],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Get masked attention output by dividing numerator by denominator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        values: Value tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        return_attention_weights: Whether to return attention weights along with output
        **kwargs: Additional keyword arguments

    Returns:
        If return_attention_weights is False:
            Attention output tensor of shape (b, h, sq, d)
        If return_attention_weights is True:
            Tuple of (attention_output, attention_weights) where:
            - attention_output: tensor of shape (b, h, sq, d)
            - attention_weights: tensor of shape (b, h, sq, sk)
    """
    # Compute exponential attention weights once and reuse
    training = module.training if module is not None else False
    exp_attention_weights = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
    )

    # Prepare values by applying key-value grouping
    num_key_value_groups = _get_num_key_value_groups(queries, values)
    value_states = repeat_kv(values, num_key_value_groups)

    # Use internal helpers with pre-computed weights
    num = _get_attention_numerator(exp_attention_weights, value_states)
    den = _get_attention_denominator(exp_attention_weights)

    # Compute final attention output
    attention_output = (num / den).transpose(1, 2).contiguous()

    if return_attention_weights:
        # Normalize exponential weights to get attention probabilities
        attention_weights = exp_attention_weights / den
        return attention_output, attention_weights

    return attention_output
