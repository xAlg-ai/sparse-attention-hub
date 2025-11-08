"""XAttention top-K masker ."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class XAttentionConfig(TopKMaskerConfig):
    """Configuration for XAttention masker."""
    block_size: int = 64
    importance_threshold: float = .9

 

@MaskerRegistry.register(XAttentionConfig)
class XAttention(TopKMasker):
    """XAttention masker."""

    def __init__(self, config: XAttentionConfig) -> None:
        """Initialize XAttention masker with configuration."""
        super().__init__(config)
        self.block_size = config.block_size
        self.importance_threshold = config.importance_threshold
        
    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        if previous_mask.is_full_mask():
            return previous_mask
        
        tensor_dims = self._extract_tensor_dimensions(keys, queries)

        #1. calculate block-level proxy scores
        block_importance_scores = self._compute_antidiagonal_proxy_scores(
            keys, queries, tensor_dims)

        #2. select top blocks based on threshold
        sorted_indices, k_per_head = self._select_top_blocks(
            block_importance_scores, self.importance_threshold
        )

        #3. create block-sparse mask from these indices
        block_mask = self._create_mask_from_block_indices(
            tensor_dims, sorted_indices, k_per_head, keys.device, previous_mask.dtype
        )

        #4. upsample block mask to full attention resolution
        full_res_mask = self._upsample_block_mask(
            block_mask, tensor_dims, previous_mask.dtype
        )

        return previous_mask.merge_mask(full_res_mask, inplace=False)

    
    def _compute_antidiagonal_proxy_scores(
        self, keys: torch.Tensor, queries: torch.Tensor, tensor_dims: AttentionTensorDimensions
    ) -> torch.Tensor:
        """Compute antidiagonal proxy scores.
        
        Supports Grouped Query Attention (GQA) where keys/values have fewer heads than queries.
        """
        #get query, key shape
        B, H_q, Nq, D = queries.shape
        B_k, H_k, Nk, D_k = keys.shape
        B_size = self.block_size
        
        # Validate dimensions
        if B != B_k:
            raise ValueError(f"Batch size mismatch: queries batch={B}, keys batch={B_k}")
        if D != D_k:
            raise ValueError(f"Head dimension mismatch: queries D={D}, keys D={D_k}")
        
        # Handle Grouped Query Attention (GQA)
        if H_k != H_q:
            if H_q % H_k != 0:
                raise ValueError(
                    f"GQA requires query heads ({H_q}) to be a multiple of key heads ({H_k})"
                )
            # Expand keys to match query heads by repeating each key head
            num_queries_per_kv = H_q // H_k
            keys = keys.repeat_interleave(num_queries_per_kv, dim=1)
            # Now keys has shape (B, H_q, Nk, D)
        
        H = H_q  # Use query heads as the number of heads

        #1 pad Q and K to be divisible by block_size
        pad_q  = (B_size - (Nq % B_size)) % B_size
        pad_k  = (B_size - (Nk % B_size)) % B_size

        #F.pad format (left, right, top, bottom) for last two dims
        queries_padded = F.pad(queries, (0, 0, 0, pad_q))
        keys_padded = F.pad(keys, (0, 0, 0, pad_k))

        Nq_p, Nk_p = queries_padded.shape[-2], keys_padded.shape[-2]
        Nq_b, Nk_b = Nq_p // B_size, Nk_p // B_size

        #reshape to make blocks
        #(B, H, Nq, D) -> (B, H, Nq_b, B_size, D)
        q_blocks = queries_padded.reshape(B, H, Nq_b, B_size, D)

        #(B, H, Nk, D) -> (B, H, Nk_b, B_size, D)
        k_blocks = keys_padded.reshape(B, H, Nk_b, B_size, D)

        #blockwise matmul
        #using einsum to compute (B, H, Nq_b, Nk_b, B_size, B_size)
        #b,h is batch, head
        #n = num_q_blocks, i = q_block_size
        #m = num_k_blocks, j = k_block_size
        #d = head_dim
        #result = (b, h, n, m, i, j)
        attn_blocks = torch.einsum("bhnid,bhmjd->bhnmij", q_blocks, k_blocks)
        #get sum of main antidiagonal
            #flip blocks left-to-right to get antidiagonal

        flipped_attn_blocks = torch.flip(attn_blocks, dims=[-1])
        #get the main diagonal (from the antidiagonal)

        #(b,h,n,m,i,j) -> (b,h,n,m,i) (j gets collapsed)
        main_antidiagonal = torch.diagonal(
            flipped_attn_blocks, offset=0, dim1=-2, dim2=-1
        )

        #sum diagonals to get one score per block
        #(b,h,n,m,i) -> (b,h,n,m)
        proxy_scores = torch.sum(main_antidiagonal, dim=-1)

        #crop back to the original number of blocks (pre-padding)

        orig_Nq_b = -(-Nq // B_size) 
        orig_Nk_b = -(-Nk // B_size)

        return proxy_scores[:, :, :orig_Nq_b, :orig_Nk_b]

    def _select_top_blocks(
        self, 
        proxy_scores: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top blocks based on threshold.
        
        Args:
            proxy_scores: Block importance scores with shape (B, H, Nq_b, Nk_b)
            threshold: Importance threshold for cumulative score selection
            
        Returns:
            Tuple of (sorted_indices, k_per_head) where:
            - sorted_indices: Indices of blocks sorted by importance (B*H, num_blocks)
            - k_per_head: Number of top blocks to select per head (B*H,)
        """
        B, H, num_q_blocks, num_k_blocks = proxy_scores.shape
        batch_heads = B * H
        num_total_blocks = num_q_blocks * num_k_blocks

        # Reshape to (B*H, num_total_blocks)
        flat_scores = proxy_scores.reshape(batch_heads, -1)

        sorted_scores, sort_indices = torch.sort(flat_scores, dim=-1, descending=True)

        cumulative_scores = torch.cumsum(sorted_scores, dim=-1)
        total_scores = cumulative_scores[:, -1].unsqueeze(-1)

        #small epsilon to avoid division by zero
        total_scores = total_scores + 1e-6

        is_above_threshold = cumulative_scores >= (threshold * total_scores)

        k_per_head = torch.argmax(is_above_threshold.int(), dim=-1) + 1        # Clamp to ensure we don't exceed available blocks
        k_per_head = torch.clamp(k_per_head, min=1, max=num_total_blocks)

        return sort_indices, k_per_head

    def _create_mask_from_block_indices(
        self, 
        dims: AttentionTensorDimensions,
        sorted_indices: torch.Tensor,
        k_per_head: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create mask from block indices.
        
        Args:
            dims: Tensor dimensions
            sorted_indices: Indices of blocks sorted by importance (B*H, num_blocks)
            k_per_head: Number of top blocks to select per head (B*H,)
            device: Device for tensor creation
            dtype: Data type for mask
            
        Returns:
            Block mask tensor with shape (B, H, Nq_b, Nk_b)
        """
        batch_heads, num_total_blocks = sorted_indices.shape
        B, H = dims.batch_size, dims.num_heads
        orig_Nq_b = -(-dims.seq_len_queries // self.block_size)
        orig_Nk_b = -(-dims.seq_len_keys // self.block_size)

        # Create flat mask and set selected blocks to 1
        block_mask_flat = torch.zeros(
            batch_heads, num_total_blocks, device=device, dtype=dtype
        )

        # For each head, set the selected block positions to 1
        # sorted_indices contains the original block positions sorted by importance
        # Select top k_per_head[i] blocks for each head
        for i in range(batch_heads):
            k = int(k_per_head[i].item())
            if k > 0:
                selected_indices = sorted_indices[i, :k]
                block_mask_flat[i, selected_indices] = 1.0

        # Reshape back to (B, H, Nq_b, Nk_b)
        block_mask = block_mask_flat.reshape(B, H, orig_Nq_b, orig_Nk_b)
        return block_mask

    
    def _upsample_block_mask(
        self, 
        block_mask_tensor: torch.Tensor,
        dims: AttentionTensorDimensions,
        dtype: torch.dtype
    ) -> Mask:
        """Upsample block mask to full attention resolution.
        
        Args:
            block_mask_tensor: Block mask with shape (B, H, Nq_b, Nk_b)
            dims: Tensor dimensions
            dtype: Data type for mask
            
        Returns:
            Upsampled mask at full attention resolution
        """
        full_res_mask = torch.repeat_interleave(
            block_mask_tensor, self.block_size, dim=2
        )

        full_res_mask = torch.repeat_interleave(
            full_res_mask, self.block_size, dim=3
        )

        full_res_mask = full_res_mask[
            :, :, :dims.seq_len_queries, :dims.seq_len_keys
        ]

        return Mask.create_mask_from_dense_mask(shape=full_res_mask.shape, mask=full_res_mask, dtype=dtype)

    
    

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "XAttention":
        """Create XAttention instance from configuration."""
        if not isinstance(config, XAttentionConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)


###
#what does the code do?
#1 attention matrix divided into bxb blocks
#softmax normalize antidiagonal sums, select minimal set of blocks who have high enough importance scores
#create mask from selected,
