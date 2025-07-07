"""Efficient attention implementations."""

from typing import Any, Optional, Tuple
import torch

from .base import EfficientAttention
from .utils import get_masked_attention_output, Mask


class DoubleSparsity(EfficientAttention):
    """Double sparsity attention mechanism."""

    def __init__(self, sparsity_ratio: float = 0.1):
        """Initialize double sparsity attention.
        
        Args:
            sparsity_ratio: Ratio of attention weights to keep (0.0 to 1.0)
        """
        super().__init__()
        self.sparsity_ratio = max(0.0, min(1.0, sparsity_ratio))

    def custom_attention(self, 
                        queries: Optional[torch.Tensor] = None,
                        keys: Optional[torch.Tensor] = None, 
                        values: Optional[torch.Tensor] = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        **kwargs) -> Tuple[Any, Optional[Any]]:
        """Compute double sparsity attention.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        if queries is None or keys is None or values is None:
            # Return dummy output for bare metal compatibility
            return torch.tensor([]), None
        
        # Handle grouped query attention by expanding keys/values to match query heads
        batch_size, num_heads, seq_len, head_dim = queries.shape
        _, num_kv_heads, _, _ = keys.shape
        
        if num_heads != num_kv_heads:
            repeat_factor = num_heads // num_kv_heads
            keys = keys.repeat_interleave(repeat_factor, dim=1)
            values = values.repeat_interleave(repeat_factor, dim=1)
        
        k = max(1, int(seq_len * self.sparsity_ratio))
        
        # Compute attention scores to determine sparsity pattern
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        _, top_indices = torch.topk(scores, k, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(scores)
        sparse_mask.scatter_(-1, top_indices, 1.0)
        mask = Mask(sparse_mask)
        
        return get_masked_attention_output(
            module=None,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0 / (head_dim ** 0.5),
            dropout=0.0,
            sparse_attention_mask=mask,
            **kwargs
        ), None


class HashAttention(EfficientAttention):
    """Hash-based attention mechanism."""

    def __init__(self, num_buckets: int = 64):
        """Initialize hash attention.
        
        Args:
            num_buckets: Number of hash buckets for attention
        """
        super().__init__()
        self.num_buckets = max(1, num_buckets)

    def custom_attention(self,
                        queries: Optional[torch.Tensor] = None,
                        keys: Optional[torch.Tensor] = None,
                        values: Optional[torch.Tensor] = None, 
                        attention_mask: Optional[torch.Tensor] = None,
                        **kwargs) -> Tuple[Any, Optional[Any]]:
        """Compute hash-based attention.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        if queries is None or keys is None or values is None:
            # Return dummy output for bare metal compatibility
            return torch.tensor([]), None
        
        # Handle grouped query attention by expanding keys/values to match query heads
        batch_size, num_heads, seq_len, head_dim = queries.shape
        _, num_kv_heads, _, _ = keys.shape
        
        if num_heads != num_kv_heads:
            repeat_factor = num_heads // num_kv_heads
            keys = keys.repeat_interleave(repeat_factor, dim=1)
            values = values.repeat_interleave(repeat_factor, dim=1)
        
        # Create hash buckets based on position
        hash_buckets = torch.arange(seq_len, device=queries.device) % self.num_buckets
        
        # Create sparse mask where tokens attend only within same bucket
        sparse_mask = torch.zeros(seq_len, seq_len, device=queries.device)
        for bucket in range(self.num_buckets):
            bucket_mask = (hash_buckets == bucket)
            bucket_indices = torch.where(bucket_mask)[0]
            if len(bucket_indices) > 0:
                sparse_mask[bucket_indices[:, None], bucket_indices] = 1.0
        
        # Expand mask to match attention dimensions
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        mask = Mask(sparse_mask)
        
        return get_masked_attention_output(
            module=None,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0 / (head_dim ** 0.5),
            dropout=0.0,
            sparse_attention_mask=mask,
            **kwargs
        ), None
