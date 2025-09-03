"""Utility functions for masked attention computation."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .kv_utils import _get_num_key_value_groups, repeat_kv
from .mask import Mask


def get_true_attention_output(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Get the true (dense) attention output from the module.

    Args:
        module: The attention module (used for dropout training flag).
        queries: Query tensor of shape (..., seq_len_q, d_k).
        keys: Key tensor of shape (..., seq_len_k, d_k).
        values: Value tensor of shape (..., seq_len_k, d_v).
        attention_mask: Optional mask tensor to apply to attention weights.
        scaling: Scaling factor for attention logits.
        dropout: Dropout probability for attention weights.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        Tuple containing:
            - attention_output: Output tensor after applying attention.
            - attention_weights: Softmax-normalized attention weights.
    """
    num_key_value_groups: int = _get_num_key_value_groups(queries, keys)
    key_states = repeat_kv(keys, num_key_value_groups)
    value_states = repeat_kv(values, num_key_value_groups)

    attn_weights = torch.matmul(queries, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        queries.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_inv_mask_sum(input_tensor: torch.Tensor, mask: Mask) -> torch.Tensor:
    """Apply inverse mask to input tensor and sum along the last dimension.

    This function efficiently computes the sum of applying the inverse mask to an input tensor
    using sparse representation and scatter operations, avoiding the need to create dense tensors.

    Args:
        input_tensor: Input tensor of shape (..., n) where n is the last dimension
        mask: Mask object to apply inverse mask with

    Returns:
        Sum tensor of shape (..., 1) with the last dimension reduced

    Note:
        - For full masks: returns sum of all input values
        - For empty masks: returns zero tensor
        - For sparse masks: efficiently computes sum using sparse operations
    """
    if input_tensor.shape != mask.shape:
        raise ValueError(
            f"input_tensor.shape must be {mask.shape}, got {input_tensor.shape}"
        )

    # Handle special cases
    if mask.is_full_mask():
        # Full mask: sum all input values
        return input_tensor.sum(dim=-1, keepdim=True)
    elif mask.is_empty():
        # Empty mask: return zero tensor
        return torch.zeros(
            input_tensor.shape[:-1] + (1,),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    # Get sparse representation
    indices, ptr, data = mask.get_index_mask()

    if indices.numel() == 0:
        # No active indices: return zero tensor
        return torch.zeros(
            input_tensor.shape[:-1] + (1,),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    # Reshape input tensor to 1D for indexing
    input_flat = input_tensor.view(-1)  # (total_elements,)

    # Extract values at sparse indices and apply inverse mask
    input_at_indices = input_flat[indices]  # (num_active_indices,)
    inverse_data = 1.0 / data  # (num_active_indices,)
    weighted_input = input_at_indices * inverse_data  # (num_active_indices,)

    # Use scatter_add_ for vectorized row-wise summation
    num_rows = int(torch.prod(torch.tensor(input_tensor.shape[:-1])))

    # Create row indices for each sparse element
    # indices are flattened, so row_idx = indices // input_tensor.shape[-1]
    seq_len_last = input_tensor.shape[-1]
    row_indices = indices // seq_len_last  # (num_active_indices,)

    # Create output tensor for scatter operation
    result = torch.zeros(num_rows, device=input_tensor.device, dtype=input_tensor.dtype)

    # Use scatter_add_ to sum weighted values per row
    result.scatter_add_(0, row_indices, weighted_input)

    # Reshape back to original dimensions (except last dimension becomes 1)
    result = result.view(input_tensor.shape[:-1] + (1,))

    return result


def create_sampling_mask_with_per_head_budget(
    budgets: torch.Tensor,
    sampling_probability: torch.Tensor,
    seq_len_keys: int,
    start_idx: int,
    end_idx: int,
    dtype: torch.dtype = torch.float32,
) -> Mask:
    """Create a sampling mask with per-head budget using direct sparse construction.

    This function efficiently creates a sparse sampling mask by directly constructing
    the sparse representation without creating intermediate dense tensors.

    Args:
        budgets: Budget tensor of shape (b, h, q, 1) indicating how many elements to sample per row
        sampling_probability: Sampling probability tensor of shape (b, h, q, 1)
        seq_len_keys: Length of the key sequence dimension
        start_idx: Starting index for sampling range (inclusive)
        end_idx: Ending index for sampling range (exclusive)
        dtype: Data type for the mask

    Returns:
        Mask object with sparse sampling representation

    Note:
        - Uses direct sparse construction for memory efficiency
        - Generates random indices within [start_idx, end_idx) for each element
        - Creates proper ptr array for sparse representation
        - Assigns sampling probabilities as mask data values

    Important Note:
        - we use random sampling with replacement so the sampling probabilities might lead to be incorrect
    """
    batch_size, num_heads, seq_len_queries, _ = budgets.shape

    # Reshape budget to (num_rows,) for easier processing
    num_rows = batch_size * num_heads * seq_len_queries
    budgets_flat = budgets.view(num_rows)  # (num_rows,)

    # Calculate total number of elements to sample
    total_elements = int(budgets_flat.sum().item())

    # Create ptr array using cumulative sum of budgets
    ptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=budgets.device),
            torch.cumsum(budgets_flat, dim=0),
        ]
    )  # (num_rows + 1,)

    # Generate random indices within sampling range for each element
    idx_in_row = torch.randint(
        low=start_idx,
        high=end_idx,
        size=(total_elements,),
        device=budgets.device,
        dtype=torch.long,
    )  # (total_elements,)

    # Create row indices by repeating each row index according to its budget
    row_id = torch.repeat_interleave(
        torch.arange(num_rows, device=budgets.device), budgets_flat
    )  # (total_elements,)

    # Calculate global indices
    idx_global = idx_in_row + row_id * seq_len_keys  # (total_elements,)

    # Get sampling probabilities for each element
    sampling_prob_flat = sampling_probability.view(num_rows)  # (num_rows,)
    data_global = sampling_prob_flat[row_id]  # (total_elements,)

    # Create the sampling mask directly using sparse index construction
    sampling_mask = Mask.create_mask_from_indices(
        shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
        indices=idx_global,
        ptr=ptr,
        data=data_global,
        dtype=dtype,
    )

    return sampling_mask


def create_sampling_mask_with_per_head_budget_no_replacement(
    budgets: torch.Tensor,
    sampling_probability: torch.Tensor,
    seq_len_keys: int,
    start_idx: int,
    end_idx: int,
    dtype: torch.dtype = torch.float32,
) -> Mask:
    """Create a sampling mask with per-head budget without replacement using vectorization.

    This function creates a sparse sampling mask ensuring no duplicate indices within
    each row, providing more accurate sampling and better statistical guarantees.

    Args:
        budgets: Budget tensor of shape (b, h, q, 1) indicating how many elements to sample per row
        sampling_probability: Sampling probability tensor of shape (b, h, q, 1)
        seq_len_keys: Length of the key sequence dimension
        start_idx: Starting index for sampling range (inclusive)
        end_idx: Ending index for sampling range (exclusive)
        dtype: Data type for the mask

    Returns:
        Mask object with sparse sampling representation (no duplicates per row)

    Note:
        - Uses vectorized permutation generation for efficiency
        - When budget > sampling_range, effective budget is clamped to sampling_range
        - Each row gets unique indices within the sampling range
        - Sampling probabilities are adjusted based on effective budget
    """
    batch_size, num_heads, seq_len_queries, _ = budgets.shape
    sampling_range = end_idx - start_idx

    # Reshape for easier processing
    num_rows = batch_size * num_heads * seq_len_queries
    budgets_flat = budgets.view(num_rows)  # (num_rows,)
    sampling_prob_flat = sampling_probability.view(num_rows)  # (num_rows,)

    # Clamp budgets to sampling_range (handle edge case where budget > available positions)
    effective_budgets = torch.clamp(budgets_flat, max=sampling_range)

    # Vectorized permutation generation
    # Create a large permutation matrix for all rows at once
    max_budget = int(effective_budgets.max().item())
    if max_budget == 0:
        # Handle edge case: all budgets are 0
        return Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=dtype,
            mask_type="index",
        )

    # Generate permutations for each row using vectorized approach
    # Much more efficient: use argsort on random values to get permutations
    random_values = torch.rand(num_rows, sampling_range, device=budgets.device)
    all_perms = torch.argsort(
        random_values, dim=-1
    )  # Shape: (num_rows, sampling_range)

    # Fully vectorized approach to handle variable budgets

    if max_budget > 0:
        # Take indices for max budget from each permutation
        selected_indices = (
            all_perms[:, :max_budget] + start_idx
        )  # (num_rows, max_budget)

        # Create mask for valid budget per row
        budget_mask = torch.arange(max_budget, device=budgets.device).unsqueeze(
            0
        ) < effective_budgets.unsqueeze(1)

        # Filter valid indices and flatten
        valid_local_indices = selected_indices[budget_mask]  # (total_valid_elements,)

        # Create row indices for valid elements
        row_ids = (
            torch.arange(num_rows, device=budgets.device)
            .unsqueeze(1)
            .expand(-1, max_budget)[budget_mask]
        )

        # Convert to global indices
        final_indices = valid_local_indices + row_ids * seq_len_keys

        # Create data with sampling probabilities
        final_data = (
            sampling_prob_flat.unsqueeze(1)
            .expand(-1, max_budget)[budget_mask]
            .to(dtype)
        )
    else:
        # All budgets are 0
        final_indices = torch.empty(0, dtype=torch.long, device=budgets.device)
        final_data = torch.empty(0, dtype=dtype, device=budgets.device)

    # Create ptr array using cumulative sum (vectorized)
    final_ptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=budgets.device),
            torch.cumsum(effective_budgets, dim=0),
        ]
    )

    # Create the sampling mask
    sampling_mask = Mask.create_mask_from_indices(
        shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
        indices=final_indices,
        ptr=final_ptr,
        data=final_data,
        dtype=dtype,
    )

    return sampling_mask


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
    num_key_value_groups: int = _get_num_key_value_groups(queries, keys)

    # Apply key-value grouping if needed
    key_states: torch.Tensor = repeat_kv(keys, num_key_value_groups)

    raw_attention_weights: torch.Tensor = (
        torch.matmul(queries, key_states.transpose(2, 3)) * scaling
    )

    if attention_mask is not None:
        raw_attention_weights = (
            raw_attention_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        )

    row_wise_max: torch.Tensor = torch.max(raw_attention_weights, dim=-1, keepdim=True)[
        0
    ]
    raw_attention_weights = raw_attention_weights - row_wise_max
    exp_attention_weights: torch.Tensor = torch.exp(raw_attention_weights)

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
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
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
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
    )

    # Prepare values by applying key-value grouping
    num_key_value_groups: int = _get_num_key_value_groups(queries, values)
    value_states: torch.Tensor = repeat_kv(values, num_key_value_groups)

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
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
    )

    # Prepare values by applying key-value grouping
    num_key_value_groups: int = _get_num_key_value_groups(queries, values)
    value_states: torch.Tensor = repeat_kv(values, num_key_value_groups)

    # Use internal helpers with pre-computed weights
    num: torch.Tensor = _get_attention_numerator(exp_attention_weights, value_states)
    den: torch.Tensor = _get_attention_denominator(exp_attention_weights)

    # Compute final attention output
    attention_output: torch.Tensor = (num / den).transpose(1, 2).contiguous()

    if return_attention_weights:
        # Normalize exponential weights to get attention probabilities
        attention_weights: torch.Tensor = exp_attention_weights / den
        return attention_output, attention_weights

    return attention_output
