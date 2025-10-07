"""PQ cache top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    get_masked_attention_output,
    get_true_attention_output,
)

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class PQCacheConfig(TopKMaskerConfig):
    """Configuration for PQCache masker."""

    pq_sub_dim: int
    pq_bits: int
    kmeans_iters: int


@MaskerRegistry.register(PQCacheConfig)
class PQCache(TopKMasker):
    """PQ cache-based top-K masker."""

    def __init__(self, config: PQCacheConfig) -> None:
        """Initialize PQ cache masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.pq_sub_dim = config.pq_sub_dim
        self.pq_bits = config.pq_bits
        self.kmeans_iters = config.kmeans_iters

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

        #sequence length is number of keys in the cache
        n, h, s, dh = keys.shape
        _, _, lq, _ = queries.shape
        #given (n, h, s, dh)
        dm = self.pq_sub_dim #sub dim
        assert dh % dm == 0, "dh must be divisible by dm"

        #number of sub vectors is dh / dm
        num_sub_vectors = dh // dm
        #ex was dh = 4, dm = 1, num_sub_vectors = 4 in image

        num_centroids = 2 ** self.pq_bits
        #clustering 
        #iterate over num_sub_vectors, extract each sub_vector, cluster it
        #assign codes and store results

        #check if index built
        if "pq_centroids" in sparse_meta_data and "pq_codes" in sparse_meta_data:
            pass #skip to search

        else:
            #partition the keys
            keys_partitioned = keys.reshape(n, h, s, num_sub_vectors, dm)

            #cluster separately
            #tensors stored in shape (num_sub_vectors, num_centroids, dm)
            #codes in tensor with shape ((batch_size, num_heads, sequence length, num_sub_vectors))

            #cluster separately
            all_centroids = torch.zeros(num_sub_vectors, num_centroids, dm)
            all_codes = torch.zeros(n, h, s, num_sub_vectors)

            #reshape for clustering operation flatten first three dims
            keys_flat_for_clustering = keys_partitioned.reshape(-1, num_sub_vectors, dm)
            #number of vectors to cluster for each sub dim in the flattened size
            num_vectors_to_cluster = keys_flat_for_clustering.shape[0]

            #loop
            for i in range(num_sub_vectors):
                sub_vectors_to_cluster = keys_flat_for_clustering[:, i, :] #shape num_vectors_to_cluster, dm
                rand_indices = torch.randperm(num_vectors_to_cluster)[:num_centroids]
                centroids_i = sub_vectors_to_cluster[rand_indices]
                #initialize centroids

                #iterate for fixed number of epochs
                for _ in range(self.kmeans_iters):
                    #L2 distance betwen each vector to all centroids
                    distances = torch.cdist(sub_vectors_to_cluster, centroids_i, p=2)
                    codes_i = torch.argmin(distances, dim=1)

                    #closest centroid for each vector
                    new_centroids = torch.zeros_like(centroids_i)
                    counts = torch.zeros(num_centroids)
                    codes_i_expanded = codes_i.unsqueeze(1).expand(-1, dm)
                    
                    new_centroids.scatter_add_(0, codes_i_expanded, sub_vectors_to_cluster)
                    counts.scatter_add_(0, codes_i, torch.ones_like(codes_i))

                    counts[counts == 0] = 1
                    centroids_i = new_centroids / counts.unsqueeze(1)

                all_centroids[i, :, :] = centroids_i
                all_codes[:, :, :, i] = codes_i.reshape(n, h, s)
            

            sparse_meta_data["pq_centroids"] = all_centroids
            sparse_meta_data["pq_codes"] = all_codes

        #pq search phase
        centroids = sparse_meta_data["pq_centroids"]
        codes = sparse_meta_data["pq_codes"]
        #partitoin the query first
        queries_partitioned = queries.reshape(n, h, lq, num_sub_vectors, dm)
        #lq up in line 57

        ##compute similarity using matmul naive rn
        scores_per_centroid = torch.zeros(n, h, lq, num_sub_vectors, num_centroids)
        for i in range(num_sub_vectors):
            q_sub = queries_partitioned[:, :, :, i, :] #shape n, h, lq, dm
            #get centroids for this dimension
            c_set = centroids[i, :, :] #shape num_centroids, dm
            #calcualte dot product
            #reshape q_sub to (n*h*lq,dm) for batched matmul
            q_su_flat = q_sub.reshape(-1, dm)
            #transpose c_set to (dm, num_centroids) for matumul
            c_set_transposed = c_set.T
            #matmul result is (n*h*lq, num_centroids)
            scores = torch.matmul(q_su_flat, c_set_transposed)
            scores_per_centroid[:, :, :, i, :] = scores.reshape(n, h, lq, num_centroids)

        #gather and reduce
        total_scores = torch.zeros(n, h, lq, s)

        #loop over sub-vector dimensions m 
        for i in range(num_sub_vectors):
            #get scores for this sub-vector i for all queries
            #shape n,h,lq, num_centroids
            scores_for_sub_dim = scores_per_centroid[:, :, :, i, :]
            #get codes for this sub dim i for all keys #shape n, h, s
            codes_for_sub_dim = codes[:, :, :, i]

            #torch.gather to get scores for each for eahc keys code
            #output shaoe n, h, lq, s

            #input to gather needs to be expanded to match input shape
            scores_expanded = scores_for_sub_dim.unsqueeze(3).expand(n, h, lq, s, num_centroids)
            #gather expand to match input shape
            codes_expanded = codes_for_sub_dim.unsqueeze(2).expand(n, h, lq, s)
            #gather scores
            gathered_scores = torch.gather(scores_expanded, dim=4, index=codes_expanded.unsqueeze(4)).squeeze(4)
            #add scroes to tottal
            total_scores += gathered_scores


        #total scores now has shape n, h, lq, s 
        top_k_scores, top_k_indices = torch.topk(total_scores, self.heavy_size, dim=-1)
        #topk indices has shape n, h, lq, self.heavy_size
        #contains indices of topk keys for each query
        mask = torch.zeros_like(total_scores, dtype=torch.bool)
        #set topk indicies to true
        mask.scatter_(-1, top_k_indices, True)
        #combine with original mask
        final_mask = mask & attention_mask.to(torch.bool)

        return Mask(mask_qk = final_mask, heavy_size = self.heavy_size)


    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)

