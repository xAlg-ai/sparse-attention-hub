"""PQ cache top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class PQCacheConfig(TopKMaskerConfig):
    """Configuration for PQCache masker."""

    pq_sub_dim: int
    pq_bits: int


@MaskerRegistry.register(PQCacheConfig)
class PQCache(TopKMasker):
    """PQ cache-based top-K masker."""

    def __init__(self, config: PQCacheConfig) -> None:
        """Initialize PQ cache masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.pq_sub_dim = config.pq_sub_dim
        self.pq_bits = config.pq_bits
        self.num_centroids = 2**self.pq_bits



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
        """Add PQ cache mask."""
        
        layer_idx: int = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")
        
        #get current current layer index

        #sparse_meta_data to manage state across calls
        if "pq_structures" not in sparse_meta_data:
            sparse_meta_data["pq_structures"] = {}
        if "kv_cache" not in sparse_meta_data:
            sparse_meta_data["kv_cache"] = {}
        #prefilling vs decoding phase detect

        is_prefilling = keys.shape[2] > 1
        #assume prefilling phase is first call where sequence is > 1 

        if layer_idx not in sparse_meta_data["pq_structures"] and is_prefilling:
            #prefilling phase builds the pq structurs and offloads the kvcache

            ##offload uncompressed kvs to the cpu

            sparse_meta_data["kv_cache"][layer_idx] = {
                "keys": keys.detach().cpu(),
                "values": values.detach().cpu(),
            }
            
            centroids, codes = self._pq_construct(keys)
            #_pq_construct to be implemente

            sparse_meta_data["pq_structures"][layer_idx] = {
                "centroids": centroids,
                "codes": codes,
            }
            #set up pq structures for this layer\

            return previous_mask

        else:
            #decoding

            #evict oldest token
            current_query = queries[:, :, -1:, :]  # (B, num_heads, 1, head_dim)

            pq_centroids = sparse_meta_data["pq_structures"][layer_idx]["centroids"].to(queries.device)
            pq_codes = sparse_meta_data["pq_structures"][layer_idx]["codes"].to(queries.device)

            #perform pqsearch to get approx scores
            approx_scores = self._pq_search(current_query, pq_centroids, pq_codes)

            #pq_searhc need to implement

            #get indcies of topk tokens from full sequence
            topk_indices = torch.topk(approx_scores, k=self.heavy_size, dim=-1, sorted=False).indices

            #fetch full kv pairs for topk tokens
            full_keys_cache = sparse_meta_data["kv_cache"][layer_idx]["keys"]
            full_values_cache = sparse_meta_data["kv_cache"][layer_idx]["values"]


            max_idx = full_keys_cache.shape[2]
            topk_indices = torch.clamp(topk_indices, 0, max_idx - 1)


            #handle broadcasting issues so flatten indices for batch fetch then reshape
            topk_keys = full_keys_cache[:, :, topk_indices, :].to(queries.device)
            topk_values = full_values_cache[:, :, topk_indices, :].to(queries.device)

            #create dense mask for tokens
            mask_shape = (queries.shape[0], queries.shape[1], queries.shape[2], keys.shape[2])
            dense_mask = torch.zeros(mask_shape, dtype=torch.bool, device = queries.device)

            #current query only previous tokens
            #we want to map topk to full sequence

            for b in range(queries.shape[0]):
                for h in range(queries.shape[1]):
                    #get current batch and head index
                    head_indices = topk_indices[b, h, :]
                    dense_mask[b, h, -1, head_indices] = 1

            final_mask = previous_mask.merge_mask(
                Mask(None, None, dense_mask),
                inplace=False
            )

            return final_mask
        

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
    


    def _pq_construct(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs PQ construction on a batch of key tensors.

        Args:
            keys: key tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Returns:
            tuple of (centroids, codes)
                centroids: tensor of shape (num_heads, num_partitions, num_centroids, sub_dim)
                codes: tensor of shape (batch_size, num_heads, seq_len, num_partitions)
        """

        bs, h, s, dh = keys.shape
        num_partitions = dh // self.pq_sub_dim

        if dh % self.pq_sub_dim != 0:
            raise ValueError("head_dim must be divisible by pq_sub_dim")

        keys_cpu = keys.detach().cpu()

        all_heads_centroids = []
        all_heads_codes = []

        for head_idx in range(h):
            head_centroids = []
            head_codes = []

            # Partition keys for this head
            head_keys_partitions = keys_cpu[:, head_idx, :, :].reshape(bs * s, num_partitions, self.pq_sub_dim)

            for partition_idx in range(num_partitions):
                partition_data = head_keys_partitions[:, partition_idx, :].contiguous()
                
                # K-Means clustering
                centroids_tensor = partition_data[torch.randperm(partition_data.shape[0])[:self.num_centroids]]
                max_iters = 100
                for _ in range(max_iters):
                    distances = torch.cdist(partition_data, centroids_tensor)
                    nearest_centroid_indices = torch.argmin(distances, dim=1)
                    
                    new_centroids_tensor = torch.zeros_like(centroids_tensor)
                    counts = torch.zeros(self.num_centroids, dtype=torch.int)
                    
                    for j in range(self.num_centroids):
                        points_in_cluster = partition_data[nearest_centroid_indices == j]
                        if len(points_in_cluster) > 0:
                            new_centroids_tensor[j] = points_in_cluster.mean(dim=0)
                            counts[j] = len(points_in_cluster)
                        else:
                            new_centroids_tensor[j] = centroids_tensor[j]
                            counts[j] = 1
                    
                    if torch.allclose(centroids_tensor, new_centroids_tensor, atol=1e-4):
                        break
                    
                    centroids_tensor = new_centroids_tensor

                distances = torch.cdist(partition_data, centroids_tensor)
                codes_for_partition = torch.argmin(distances, dim=1)

                head_centroids.append(centroids_tensor.unsqueeze(0))
                head_codes.append(codes_for_partition.reshape(bs, s).unsqueeze(0))

            all_heads_centroids.append(torch.cat(head_centroids, dim=0).unsqueeze(0))
            all_heads_codes.append(torch.cat(head_codes, dim=0).unsqueeze(0))
        
        # Stack to get final shapes
        centroids_tensor = torch.cat(all_heads_centroids, dim=0)
        codes_tensor = torch.cat(all_heads_codes, dim=0).permute(1, 0, 2, 3)

        return centroids_tensor, codes_tensor