"""XAttention top-K masker ."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import math

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
    stride: int = 8

 

@MaskerRegistry.register(XAttentionConfig)
class XAttention(TopKMasker):
    """XAttention masker."""

    def __init__(self, config: XAttentionConfig) -> None:
        """Initialize XAttention masker with configuration."""
        super().__init__(config)
        self.block_size = config.block_size
        self.importance_threshold = config.importance_threshold
        self.stride = config.stride
        
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
        """Compute block-level proxy scores using strided approximation.
        
        Uses downsampled blocks to compute cheap O(N^2/S^2) proxy scores instead of
        expensive O(N^2) full attention. This is A_approx from Algorithm 1.
        
        Supports Grouped Query Attention (GQA) where keys/values have fewer heads than queries.
        """
        # Get query, key shape
        B, H_q, Nq, D = queries.shape
        _, H_k, Nk, D_k = keys.shape
        B_size = self.block_size
        
        # --- GQA Handling (Keep this part) ---
        if H_k != H_q:
            if H_q % H_k != 0:
                raise ValueError(
                    f"GQA requires query heads ({H_q}) to be a multiple of key heads ({H_k})"
                )
            num_queries_per_kv = H_q // H_k
            keys = keys.repeat_interleave(num_queries_per_kv, dim=1).contiguous()
        
        H = H_q
        # --- End GQA Handling ---

        # Pad Q and K to be divisible by block_size
        pad_q  = (B_size - (Nq % B_size)) % B_size
        pad_k  = (B_size - (Nk % B_size)) % B_size

        queries_padded = F.pad(queries, (0, 0, 0, pad_q)).contiguous()
        keys_padded = F.pad(keys, (0, 0, 0, pad_k)).contiguous()

        Nq_p, Nk_p = queries_padded.shape[-2], keys_padded.shape[-2]
        Nq_b, Nk_b = Nq_p // B_size, Nk_p // B_size

        # Reshape into blocks
        # (B, H, Nq_p, D) -> (B, H, Nq_b, B_size, D)
        q_blocks = queries_padded.reshape(B, H, Nq_b, B_size, D)
        # (B, H, Nk_p, D) -> (B, H, Nk_b, B_size, D)
        k_blocks = keys_padded.reshape(B, H, Nk_b, B_size, D)

        # Downsample the blocks using the stride to create cheap approximation
        # (B, H, Nq_b, B_size, D) -> (B, H, Nq_b, B_size/S, D)
        q_blocks_downsampled = q_blocks[:, :, :, ::self.stride, :]
        
        # (B, H, Nk_b, B_size, D) -> (B, H, Nk_b, B_size/S, D)
        k_blocks_downsampled = k_blocks[:, :, :, ::self.stride, :]

        # Compute proxy scores by multiplying the downsampled blocks
        # This computes ONE score per block pair, not a full B_size x B_size matrix
        # (b)atch, (h)ead, (n)um_q_blocks, (i)_downsampled, (d)im
        # (b)atch, (h)ead, (m)um_k_blocks, (j)_downsampled, (d)im
        # Result: (b)atch, (h)ead, (n)um_q_blocks, (m)um_k_blocks


        scaling_factor = 1.0 / math.sqrt(D * self.stride)
        proxy_scores = torch.einsum(
            "bhnid,bhmjd->bhnm", q_blocks_downsampled, k_blocks_downsampled
        ) * scaling_factor

        # Crop back to the original number of blocks (pre-padding)
        orig_Nq_b = -(-Nq // B_size) 
        orig_Nk_b = -(-Nk // B_size)

        # Apply softmax normalization per query block (normalize over key dimension)
        proxy_scores = proxy_scores[:, :, :orig_Nq_b, :orig_Nk_b]
        proxy_scores = torch.softmax(proxy_scores, dim=-1)  # Normalize over key blocks
        
        return proxy_scores

    def _select_top_blocks(
        self, 
        proxy_scores: torch.Tensor,
        score_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top blocks based on fixed score threshold.
        
        This implements the find_blocks(A_approx, tau) logic from the paper.
        Blocks with scores > tau are selected, giving adaptive density behavior.
        
        Args:
            proxy_scores: Block importance scores with shape (B, H, Nq_b, Nk_b)
            score_threshold: Fixed score threshold (tau) to select blocks
            
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

        probabilities = torch.softmax(flat_scores, dim=-1)
        sorted_probs, sort_indices = torch.sort(probabilities, dim=-1, descending=True)

        #find minimum k where cumsum >= threshold
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        
        #first index where cumsum >= threshold
        mask = cumsum >= score_threshold
        k_per_head = torch.zeros(batch_heads, dtype=torch.long, device=flat_scores.device)

        for i in range(batch_heads):
            indices = torch.where(mask[i])[0]
            if len(indices) > 0:
                k_per_head[i] = indices[0] + 1
            else:
                k_per_head[i] = probabilities.shape[-1]
         
        #ensure at least 1 block
        k_per_head = torch.clamp(k_per_head, min=1, max=probabilities.shape[-1])


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
        ].contiguous()

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



#algorithm steps
#1
    #Divide length by block size, floor divide L/b
    #begin loop to process query one at a time
        #for each loop, extracts a slice of Q called Qslice which corresponds to current block

#2: reshaping logic
    #Takes Qslice and creates S (stride) different views of it
    #start at row i, take every s-th row from that starting point, add this new smaller matri to the Q reshaped list
    #compresses the query block

    #reshaping K same thing for keys

#approximate attention
    #intead of computing full attention, computes approxiate
    #softmax on compressed Q reshaped and K reshape
    #scaling factor is sqrt(dhS)

#finding and storing blocks
    #takes small matrix (A approx)and threshold tau
    #looks for any scores in A approx that are higher than the threshold
    #any region with high approximate score is important
    #crates a block mask that cselects this entire coarse grained block for real attention computation later

    #stitch blocks together to get full attention mask

#final result
    #output is sparse mask, 1 means it is important, 0 means it is not
