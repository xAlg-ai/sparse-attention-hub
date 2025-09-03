""" Utility functions for common kv manipulation. """

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch: int
    num_key_value_heads: int
    slen: int
    head_dim: int
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_num_key_value_groups(queries: torch.Tensor, kv_tensor: torch.Tensor) -> int:
    """
    Calculate the number of key-value groups based on tensor shapes.

    Args:
        queries: Query tensor of shape (b, h, sk, d)
        kv_tensor: Key or Value tensor of shape (b, h_kv, sq, d)

    Returns:
        Number of key-value groups (h // h_kv)

    Raises:
        ValueError: If num_attention_heads is not divisible by num_key_value_heads
    """
    num_attention_heads: int = queries.shape[1]
    num_key_value_heads: int = kv_tensor.shape[1]

    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
        )

    return num_attention_heads // num_key_value_heads
