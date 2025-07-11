"""Hash attention top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.kv_utils import repeat_kv, _get_num_key_value_groups

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class HashAttentionTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for HashAttentionTopKMasker."""

    hat_bits: int
    hat_mlp_layers: int
    hat_mlp_hidden_size: int
    hat_mlp_activation: str  # activation to use relu / silu, etc
    hat_weights: Dict[
        int, Dict[str, List[torch.Tensor]]
    ]  # Dict of layer_idx to tensor lists


@MaskerRegistry.register(HashAttentionTopKMaskerConfig)
class HashAttentionTopKMasker(TopKMasker):
    """Hash attention top-K masker."""

    def __init__(self, config: HashAttentionTopKMaskerConfig):
        """Initialize hash attention top-K masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.hat_bits = config.hat_bits
        self.hat_mlp_layers = config.hat_mlp_layers
        self.hat_mlp_hidden_size = config.hat_mlp_hidden_size
        self.hat_mlp_activation = config.hat_mlp_activation
        self.hat_weights = config.hat_weights

    def _get_signatures(
        self,
        input_tensor: torch.Tensor,
        matrix_list: List[torch.Tensor],
        bias_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute signatures using MLP layers with activation and sign function.

        Args:
            input_tensor: Input tensor to compute signatures for
            matrix_list: List of weight matrices for MLP layers
            bias_list: List of bias vectors for MLP layers

        Returns:
            Signed signatures tensor
        """
        # Activation function mapping
        activation_map: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "relu": torch.nn.functional.relu,
            "silu": torch.nn.functional.silu,
            "gelu": torch.nn.functional.gelu,
            "tanh": torch.nn.functional.tanh,
        }

        if self.hat_mlp_activation not in activation_map:
            raise ValueError(
                f"Unsupported activation function: {self.hat_mlp_activation}"
            )

        activation_fn: Callable[[torch.Tensor], torch.Tensor] = activation_map[
            self.hat_mlp_activation
        ]
        signatures = input_tensor

        # Apply MLP layers except the last one with activation
        for i in range(len(matrix_list) - 1):
            # Use einsum for proper broadcasting: (B,H,s,d) x (H,d,d_out) -> (B,H,s,d_out)
            signatures = torch.einsum("bhsd,hde->bhse", signatures, matrix_list[i])
            # Add bias with proper broadcasting: (B,H,s,d_out) + (H,d_out) -> (B,H,s,d_out)
            signatures = signatures + bias_list[i].unsqueeze(0).unsqueeze(2)
            signatures = activation_fn(signatures)

        # Apply final layer without activation
        if len(matrix_list) > 0:
            signatures = torch.einsum("bhsd,hde->bhse", signatures, matrix_list[-1])
            signatures = signatures + bias_list[-1].unsqueeze(0).unsqueeze(2)

        # Apply sign function
        signatures = torch.sign(signatures)

        return signatures

    def _update_key_signatures(
        self,
        keys: torch.Tensor,
        sparse_meta_data: Dict[str, Dict[int, Optional[torch.Tensor]]],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Update key signatures in sparse_meta_data and return concatenated signatures.

        Args:
            keys: Key tensor of shape (B, H, #keys, dim)
            sparse_meta_data: Metadata dictionary for caching
            **kwargs: Additional arguments including layer_idx

        Returns:
            Concatenated key signatures tensor (B, H, #keys, hat_bits)
        """
        layer_idx = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")

        # Initialize sparse_meta_data structure if needed
        if "key" not in sparse_meta_data:
            sparse_meta_data["key"] = {}
        if layer_idx not in sparse_meta_data["key"]:
            sparse_meta_data["key"][layer_idx] = None

        # Get cached key signatures
        cached_key_signatures = sparse_meta_data["key"][layer_idx]

        if cached_key_signatures is None:
            # First run - all keys are new
            new_keys = keys
            cached_num_keys = 0
        else:
            # Determine how many keys are new
            cached_num_keys = cached_key_signatures.shape[2]
            current_num_keys = keys.shape[2]

            if current_num_keys < cached_num_keys:
                raise ValueError(
                    f"Current number of keys ({current_num_keys}) is less than cached number of keys ({cached_num_keys})"
                )
            elif current_num_keys > cached_num_keys:
                # We have new keys to process
                new_keys = keys[:, :, cached_num_keys:, :]
            else:
                # No new keys, return cached signatures
                return cached_key_signatures

        # Compute new key signatures
        key_weights = self.hat_weights[layer_idx]
        key_matrix_list = key_weights["key_matrix"]
        key_bias_list = key_weights["key_bias"]

        new_key_signatures: torch.Tensor = self._get_signatures(
            new_keys, key_matrix_list, key_bias_list
        )

        # Update sparse_meta_data with new signatures
        if cached_key_signatures is None:
            # First run - store new signatures
            sparse_meta_data["key"][layer_idx] = new_key_signatures
            return new_key_signatures
        else:
            # Concatenate cached and new signatures
            concatenated_signatures = torch.cat(
                [cached_key_signatures, new_key_signatures], dim=2
            )
            sparse_meta_data["key"][layer_idx] = concatenated_signatures
            return concatenated_signatures

    def _compute_hashattention_score(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        sparse_meta_data: Dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute hash attention scores using query and key signatures.

        Args:
            queries: Query tensor of shape (B, H, #queries, dim)
            keys: Key tensor of shape (B, H, #keys, dim)
            sparse_meta_data: Metadata dictionary for caching
            **kwargs: Additional arguments including layer_idx

        Returns:
            Hash attention scores tensor (B, H, #queries, #keys)
        """
        layer_idx = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")

        num_key_value_groups = _get_num_key_value_groups(queries, keys)
        keys = repeat_kv(keys, num_key_value_groups)
        # 1. Get key signatures
        key_signatures = self._update_key_signatures(keys, sparse_meta_data, **kwargs)

        # 2. Compute query signatures directly for input queries
        query_weights = self.hat_weights[layer_idx]
        query_matrix_list = query_weights["query_matrix"]
        query_bias_list = query_weights["query_bias"]

        query_signatures = self._get_signatures(
            queries, query_matrix_list, query_bias_list
        )

        # 3. Compute scores using raw attention inner product style computation
        # query_signatures: (B, H, #queries, hat_bits)
        # key_signatures: (B, H, #keys, hat_bits)
        # scores: (B, H, #queries, #keys)
        scores = torch.matmul(query_signatures, key_signatures.transpose(-2, -1))

        return scores

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Any,
    ) -> Mask:
        """Add hash attention top-K mask."""
        # Input validation
        if sparse_meta_data is None:
            raise ValueError("sparse_meta_data cannot be None")
        if "layer_idx" not in kwargs:
            raise ValueError("layer_idx must be provided in kwargs")

        # 1. Check if previous_mask is full mask, if so return full mask
        if previous_mask.is_full_mask():
            return previous_mask

        # Get tensor shapes
        batch_size = queries.shape[0]
        num_heads = queries.shape[1]
        seq_len_queries = queries.shape[2]
        seq_len_keys = keys.shape[2]

        # 2. Compute heavy_size: if int use as is, if float use heavy_size * #keys
        if isinstance(self.heavy_size, float):
            heavy_size = int(self.heavy_size * seq_len_keys)
        else:
            heavy_size = int(self.heavy_size)

        # 3. Check if # keys is smaller than heavy_size, if so return full mask
        if seq_len_keys <= heavy_size:
            mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
            return Mask.create_full_mask(mask_shape, dtype=previous_mask.dtype)

        # 4. Compute score using _compute_hashattention_score
        scores = self._compute_hashattention_score(
            queries, keys, sparse_meta_data, **kwargs
        )

        # 5. Extract row-wise top-k indices from inactive positions in previous_mask
        # Get the dense mask from previous_mask to identify inactive positions
        previous_dense_mask = previous_mask.get_dense_mask()

        # Mask out positions already active in previous_mask
        masked_scores = scores.clone()
        masked_scores[previous_dense_mask != 0] = float("-inf")

        # Get top-k indices from inactive positions
        _, top_k_indices = torch.topk(masked_scores, k=heavy_size, dim=-1, largest=True)
        data = torch.ones_like(top_k_indices, dtype=previous_mask.dtype)

        # 6. Use this row-wise idx to compute this_mask using Mask.create_row_wise_idx()
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        this_mask = Mask.create_from_row_wise_idx(
            mask_shape, top_k_indices, data, type="index", dtype=previous_mask.dtype
        )

        # 7. Merge this_mask with previous mask and return
        return previous_mask.merge_mask(this_mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "HashAttentionTopKMasker":
        """Create HashAttentionTopKMasker instance from configuration."""
        if not isinstance(config, HashAttentionTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
