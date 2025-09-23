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
        
        layer_idx: int = kwargs.get("layer_idx") #get indx
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")

        #manage state across calls
        if "pq_structures" not in sparse_meta_data:
            sparse_meta_data["pq_structures"] = {}
        if "kv_cache" not in sparse_meta_data:
            sparse_meta_data["kv_cache"] = {}


        #check for prefilling/deconding phase
        is_prefill = keys.shape[2] > 1

        if layer_idx not in sparse_meta_data["pq_structures"] and is_prefill:
            #prefilling phase
            #offload KVcache to CPU
            sparse_meta_data["kv_cache"][layer_idx] = {
                "keys": keys.detach().cpu(),
                "values": values.detach().cpu(),
            }

            centroids, codes = self._pq_construct(keys) #to implement below, construct it though

            sparse_meta_data["pq_structures"][layer_idx] = {
                "centroids": centroids,
                "codes": codes,
            }

            return previous_mask

        else:
            #decoding perform pq search fetch topk tokens
            current_query = queries[:, :, -1:, :]  # (B, num_heads, 1, head_dim)
            pq_centroids = sparse_meta_data["pq_structures"][layer_idx]["centroids"].to(queries.device)
            pq_codes = sparse_meta_data["pq_structures"][layer_idx]["codes"].to(queries.device)

            approx_scores = self._pq_search(current_query, pq_centroids, pq_codes) #to implement below

            topk_indices = torch.topk(approx_scores, self.heavy_size, dim=-1, sorted=False).indices  # (B, num_heads, 1, heavy_size)

            #update full key cache
            full_keys_cache = sparse_meta_data["kv_cache"][layer_idx]["keys"]
            #handle case with sequence length changes
            topk_indices = torch.clamp(topk_indices, 0, full_keys_cache.shape[2]-1)

            #create dense mask to merge
            mask_shape = (queries.shape[0], queries.shape[1], queries.shape[2], keys[2])
            dense_mask = torch.zeros(mask_shape, dtype=torch.bool, device=queries.device)

            for p in range(queries.shape[0]):
                for d in range(queries.shape[1]):
                    head_indices = topk_indices[p, d, :]
                    dense_mask[p, d, -1, head_indices] = 1

            return previous_mask.merge_mask(Mask(None, None, dense_mask), inplace=False)
            #inplace due 


    def _pq_construct(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        performs pqconstruct on batch of keys
        args: 
        keys (tensor shape(batch size, num_heads, seq_len, head_dim)

        returns:
        tuple of centroids (num_heads, num_partitions, num_centroids, sub_dim)
        codes: tensor of shape (batch size, num_heads, seq_len, num_partitions)
        """
        B, H, L, D = keys.shape
        num_partitions = D // self.pq_sub_dim

        if D % self.pq_sub_dim != 0:
            raise ValueError("head_dim must be divisible by pq_sub_dim")

        #pq construct needs to ru non the CPU to offload GPU work
        keys_cpu = keys.detach().cpu()

        all_heads_centroids = []
        all_heads_codes = []


        #nonparallel for nwo
        for head_idx in range(h):
            head_centroids = []
            head_codes = []

            #reshape keys for this head B*L, D -> B*L, num_partitions, sub_dim
            head_keys_partitions = keys_cpu[:, head_idx, :, :].reshape(B*L, num_partitions, self.pq_sub_dim)

            for partition_idx in range(num_partitions):
                partition_data = head_keys_partitions[:, partition_idx, :].contiguous() #B*L, sub_dim


                #run kmeans on partition data
                kmeans = torch.kmeans(
                    partition_data,
                    num_clusters=self.num_centroids,
                    distance="euclidean",
                    device="cpu", #ensure on cpu
                    iter_limit=20,
                    verbose=False,
                )
                centroids = kmeans.centroids #num_centroids, sub_dim
                codes = kmeans.labels #B*L
                head_centroids.append(centroids)
                head_codes.append(codes)
            #stack centroids and codes for this head
            head_centroids_tensor = torch.stack(head_centroids, dim=0) #num_part
            head_codes_tensor = torch.stack(head_codes, dim=1).reshape(B, L, num_partitions) #B, L, num_partitions
            all_heads_centroids.append(head_centroids_tensor)
            all_heads_codes.append(head_codes_tensor)
        #stack all heads
        all_heads_centroids_tensor = torch.stack(all_heads_centroids, dim=0) #
        all_heads_codes_tensor = torch.stack(all_heads_codes, dim=0) #H, B, L, num_partitions
        all_heads_codes_tensor = all_heads_codes_tensor.permute(1,0,2,3).contiguous() #B, H, L, num_partitions  
        return all_heads_centroids_tensor, all_heads_codes_tensor

    def _pq_search(self, query: torch.Tensor, centroids: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """
        perform PQ search to approximate dot product between query and a set of keys
        args:
        query: current query tensor of shape (B, num_heads, 1, head_dim)
        centroids: tensor of shape (num_heads, num_partitions, num_centroids, sub_dim)
        codes: tensor of shape (B, num_heads, seq_len, num_partitions)

        returns:
        approximate scores: tensor of shape (B, num_heads, seq_len)

        """
        B, H, _, D = query.shape
        _, _, L, num_partitions = codes.shape

        #reshape query to B, H, num_partitions, sub_dim
        partitioned_query = query.reshape(B, H, num_partitions, self.pq_sub_dim)

        #reshape centroids to 1, H, num_partitions, num_centroids, sub_dim
        reshaped_centroids = centroids.reshape(1, H, num_partitions, self.num_centroids, self.pq_sub_dim)

        #compute dot product between query partitions and centroids
        #need to squeeze the sequence length dim from partitinoed query
        sim_to_centroids = torch.matmul(partitioned_query.squeeze(2), reshaped_centroids.transpose(-2, -1)) #B, H, num_partitions, num_centroids

        #gather scores using pq codes
        #reshape sim to centroids
        sim_to_centroids_reshaped = sim_to_centroids.unsqueeze(2)

        #codes tensor acts as indices
        gathered_sims = torch.gather(sim_to_centroids_reshaped, dim=-1, index=codes.unsqueeze(-1)).squeeze(-1)

        #sum scores across partitions
        approx_scores = gathered_sims.sum(dim=-1)

        return approx_scores


    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
    


