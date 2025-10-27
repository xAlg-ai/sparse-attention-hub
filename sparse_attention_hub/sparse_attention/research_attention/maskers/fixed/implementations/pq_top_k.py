"""PQ cache top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from sklearn.cluster import KMeans
import numpy as np




import torch
#from torch_kmeans import KMeans

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
        

        #print(f"Keys shape: {keys.shape}")
        pq_s = sparse_meta_data.get("pq_sequence_length")

        #sequence length is number of keys in the cache
        n, h_kv, s, dh = keys.shape
        _, h_q, lq, _ = queries.shape

        dtype = keys.dtype
        #given (n, h, s, dh)
        dm = self.pq_sub_dim #sub dim
        assert dh % dm == 0, "dh must be divisible by dm"

        #number of sub vectors is dh / dm
        num_sub_vectors = dh // dm
        #ex was dh = 4, dm = 1, num_sub_vectors = 4 in image
        mask_shape = n, h_q, lq, s
        num_centroids = 2 ** self.pq_bits
        layer_idx = kwargs.get("layer_idx", 0)
        tensor_dims: AttentionTensorDimensions = self._extract_tensor_dimensions(
            keys, queries
        )

        #if torch.isnan(keys).any():
        #    print("NaN found in layer" + str(layer_idx))
        
        #if torch.isnan(queries).any():
        #    print("NaN found in queries layer" + str(layer_idx))

        queries_partitioned = self._partition_vectors(queries, num_sub_vectors, dm)
        keys_partitioned = self._partition_vectors(keys, num_sub_vectors, dm)
        #partition the keys, queries using helper





        

        if "pq_centroids" not in sparse_meta_data or "pq_codes" not in sparse_meta_data:
            if "pq_centroids" not in sparse_meta_data:
                sparse_meta_data["pq_centroids"] = {}
            if "pq_codes" not in sparse_meta_data:
                sparse_meta_data["pq_codes"] = {}

        if layer_idx not in sparse_meta_data["pq_centroids"] or layer_idx not in sparse_meta_data["pq_codes"]:
            codebook, centroids = self._cluster_keys(keys_partitioned)
            #if layer_idx == 0:
                #print(f"After clustering - centroids has NaN: {torch.isnan(centroids).any()}")
            sparse_meta_data["pq_centroids"][layer_idx] = centroids
            sparse_meta_data["pq_codes"][layer_idx] = codebook
        else:
            # Update PQ codes if cache has grown
            if layer_idx in sparse_meta_data["pq_codes"]:
                existing_s = sparse_meta_data["pq_codes"][layer_idx].shape[2]
                if s > existing_s:  # Cache has grown
                    self._update_sparse_meta_data(sparse_meta_data, keys, layer_idx=layer_idx)

        assert "pq_codes" in sparse_meta_data, "pq_codes must be present if pq_centroids are present"
        query_centroid_score = self._compute_query_centroid_score(queries_partitioned, centroids = sparse_meta_data["pq_centroids"][layer_idx], scaling = scaling)
        query_key_scores = self._compute_query_key_scores(query_centroid_score, key_codebook = sparse_meta_data["pq_codes"][layer_idx])
        #_scores, top_k_indices = torch.topk(query_key_scores, self.heavy_size, dim=-1)
        top_k_indices: torch.Tensor = self._get_topk_indices_from_inactive_positions(
            query_key_scores, previous_mask, self.heavy_size
        )
                
    
        # Check if shapes match
        expected_mask_shape = (n, h_q, lq, s)
        if previous_mask.shape != expected_mask_shape:
            previous_mask = Mask.create_full_mask(
                expected_mask_shape, dtype=dtype
            )
        
       

        this_mask = self._create_mask_from_rowise_indices(tensor_dims, top_k_indices, keys.device, previous_mask.dtype)
        #print("this mask shape after rowwise indices" + str(this_mask.shape))
        #if layer_idx == 0:
            #print(f"[DEBUG MASK] this_mask shape: {this_mask.data.shape if hasattr(this_mask, 'data') else 'no data attr'}, keys s={s}")
        #print("this mask shape" + str(this_mask.shape))
        new_mask =  previous_mask.merge_mask(this_mask, inplace=False)
        return new_mask

        
    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
    

    def _partition_vectors(self, vectors: torch.Tensor, num_sub_vectors: int, dm: int) -> torch.Tensor:
        """Partition vectors into sub-vectors

        Args: 
        vectors: input tensor of shape (n, h_kv, s, dh)
            n is batch size, h is # heads, s is sequence length, dh is dimension of each head
        num_sub_vectors: number of sub vectors to partition into, also = m
        dm: dimension of each sub vector (dm = dh/m)
        
        Returns:
        Partitioned tensor of shape (n, h, s, num_sub_vectors, dm)
    """
        
        n, h_kv, s, dh = vectors.shape
        assert dh == num_sub_vectors * dm, "dh must equal num_sub_vectors * dm"
        return vectors.reshape(n, h_kv, s, num_sub_vectors, dm)


    def _cluster_keys(self, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cluster keys using PQ and return codebook and centroids.

        Args:
            keys: Input keys tensor of shape (n, h, s, num_sub_vectors, dm)
        Returns:
            codebook: Tensor of shape (n, h, s, num_sub_vectors) with assigned codes
            centroids: Tensor of shape (num_sub_vectors, num_centroids, dm) with learned centroids
        """
        #take in keys already partitioned
        n, h_kv, s, num_sub_vectors, dm = keys.shape
        #keys_partitioned = keys
        device = keys.device
        dtype = keys.dtype
        num_centroids = 2 ** self.pq_bits
        keys_flat = keys.reshape(-1, num_sub_vectors, dm)
        N_flat = keys_flat.shape[0]

        all_centroids = torch.zeros((num_sub_vectors, num_centroids, dm), dtype=torch.float32, device=device)
        all_codes = torch.zeros((N_flat, num_sub_vectors), dtype=torch.long, device=device)

        for i in range(num_sub_vectors):
            sub_vectors = keys_flat[:, i, :]  # shape (N_flat, dm)
            x = sub_vectors.to(torch.float32).cpu().numpy()
            sub_vectors = torch.nan_to_num(sub_vectors, nan=0.0, posinf=0.0, neginf=0.0)
            x = sub_vectors.to(torch.float32).cpu().numpy()

            kmeans = KMeans(n_clusters=num_centroids, init='random', max_iter=self.kmeans_iters, random_state=0)
            kmeans.fit(x)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            #sanitize output to prevent nan centroids
            centers = np.nan_to_num(centers, nan=0.0, posinf=0.0, neginf=0.0)

            all_centroids[i] = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
            all_codes[:, i] = torch.from_numpy(labels).to(device=device, dtype=torch.long)

        codebook = all_codes.reshape(n, h_kv, s, num_sub_vectors) #reshape codebook back

        return codebook, all_centroids.to(device=device, dtype=dtype)
    
    def _compute_query_centroid_score(self, queries_partitioned: torch.Tensor, centroids: torch.Tensor, scaling: float) -> torch.Tensor:
        """Compute similarity between query sub-vectors and centroids.
        
        
        Args:
            queries_partitioned: tensor of shape (n, h, lq, num_sub_vectors, dm)
            centroids: tensor of shape (num_sub_vectors, num_centroids, dm)

            scaling : scaling factor for attention scores


        Returns:
            lookup table/scores: tensor of shape (n, h, lq, num_sub_vectors, num_centroids)
        """

        # Compute the dot product between queries and centroids

        scores = torch.einsum(
            "nhlmd,mcd->nhlmc",
            queries_partitioned,
            centroids
        )
        # Apply scaling
        scores *= scaling

        #inf debug
        #print("torch is inf" + str(torch.isinf(scores).any()))


        return scores

    def _compute_query_key_scores(self, query_centroid_scores: torch.Tensor, key_codebook: torch.Tensor) -> torch.Tensor:
        """Compute approximate query-key scores using centroid scores and key codes.

        Args:
            query_centroid_scores: tensor of shape (n, h, lq, num_sub_vectors, num_centroids)
            key_codebook: tensor of shape (n, h, s, num_sub_vectors)
        Returns:
            approximate query-key scores: tensor of shape (n, h, lq, s)

        """
        device = query_centroid_scores.device
        n, h_q, lq, num_sub_vectors, num_centroids = query_centroid_scores.shape
        _, h_kv, s, _ = key_codebook.shape

        # Initialize the final scores tensor

        if h_kv != h_q:
            head_ratio = h_q // h_kv
            #key_codebook_expanded = key_codebook.repeat_interleave
            key_codebook = key_codebook.repeat(1, head_ratio, 1, 1)


        # Loop over each sub-vector dimension to accumulate scores
        #create index tensors for indexing 
        n_idx = torch.arange(n, device=device)[:, None, None, None, None] #arange creates 1d tensor of n
        h_idx = torch.arange(h_q, device=device)[None, :, None, None, None]
        q_idx = torch.arange(lq, device=device)[None, None, :, None, None]
        m_idx = torch.arange(num_sub_vectors, device=device)[None, None, None, None, :]

        #pseudo for
        #for each (batch, head, query, key, partition):
            #centroid_id = codebook[batch, head, query, key, partition]
            #score = query_centroid_scores[batch, head, query, partition, centroid_id]
        key_codebook_expanded = key_codebook.unsqueeze(2)
        gathered_scores = query_centroid_scores[
            n_idx,
            h_idx,
            q_idx,
            m_idx,
            key_codebook_expanded
        ]  # shape (n, h_q, lq, s, num_sub_vectors)

        approx_scores = gathered_scores.sum(dim=-1)  # sum over num_sub_vectors
        return approx_scores

    def _update_sparse_meta_data(self, sparse_meta_data: Dict[Any, Any], keys: torch.Tensor, layer_idx: int = 0) -> None:   
        """Update PQ codes when new keys are added to cache."""
        if "pq_centroids" not in sparse_meta_data or layer_idx not in sparse_meta_data["pq_centroids"]:
            return
        
        n, h_kv, new_s, dh = keys.shape
        dm = self.pq_sub_dim
        num_sub_vectors = dh // dm
        
        centroids = sparse_meta_data["pq_centroids"][layer_idx]
        existing_codes = sparse_meta_data["pq_codes"][layer_idx]
        
        # Get existing sequence length
        _, _, existing_s, _ = existing_codes.shape
        
        # Only process new keys if cache has grown
        if new_s <= existing_s:
            return  # No new keys to process
        
        # Extract only the new keys
        new_keys_only = keys[:, :, existing_s:, :]  # Shape: (n, h_kv, new_s - existing_s, dh)
        
        # Partition only the new keys
        keys_partitioned = self._partition_vectors(
            new_keys_only, 
            num_sub_vectors, 
            dm
        )  # Shape: (n, h_kv, new_s - existing_s, num_sub_vectors, dm)
        
        # Assign codes to new keys
        new_codes = self._assign_codes_to_keys(keys_partitioned, centroids)
        
        # Append new codes to existing codes (concat on sequence length dim)
        updated_codes = torch.cat([existing_codes, new_codes], dim=2)
        sparse_meta_data["pq_codes"][layer_idx] = updated_codes


    def _assign_codes_to_keys( self, keys_partitioned: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Assign codes to keys based on nearest centroids.
        
        Args:
            keys_partitioned: tensor of shape (n, h, s, num_sub_vectors, dm)
            centroids: tensor of shape (num_sub_vectors, num_centroids, dm) or
                    (h, num_sub_vectors, num_centroids, dm) if cluster_per_head
        
        Returns:
            codes: tensor of shape (n, h, s, num_sub_vectors)
        """
        n, h_kv, s, num_sub_vectors, dm = keys_partitioned.shape
        device = keys_partitioned.device
        
        # Determine number of centroids and whether we have per-head clustering
        if centroids.dim() == 4:
            # Per-head centroids: (h, num_sub_vectors, num_centroids, dm)
            cluster_per_head = True
            num_centroids = centroids.shape[2]
        else:
            # Shared centroids: (num_sub_vectors, num_centroids, dm)
            cluster_per_head = False
            num_centroids = centroids.shape[1]
        
        codes = torch.zeros((n, h_kv, s, num_sub_vectors), dtype=torch.long, device=device)
        
        if cluster_per_head:
            # Handle per-head centroids
            for head_idx in range(h_kv):
                for i in range(num_sub_vectors):
                    sub_vectors = keys_partitioned[:, head_idx, :, i, :]  # (n, s, dm)
                    c_set = centroids[head_idx, i, :, :]  # (num_centroids, dm)
                    
                    # Flatten and compute distances
                    sub_vectors_flat = sub_vectors.reshape(-1, dm)  # (n*s, dm)
                    
                    # Use Euclidean distance (consistent with K-Means)
                    distances = torch.cdist(sub_vectors_flat, c_set)  # (n*s, num_centroids)
                    
                    # Assign to nearest centroid (argmin for distance)
                    assigned_codes = torch.argmin(distances, dim=1)  # (n*s)
                    codes[:, head_idx, :, i] = assigned_codes.reshape(n, s)
        else:
            # Handle shared centroids
            for i in range(num_sub_vectors):
                sub_vectors = keys_partitioned[:, :, :, i, :]  # (n, h_kv, s, dm)
                c_set = centroids[i, :, :]  # (num_centroids, dm)
                
                # Flatten and compute distances
                sub_vectors_flat = sub_vectors.reshape(-1, dm)  # (n*h_kv*s, dm)
                
                # Use Euclidean distance (consistent with K-Means)
                distances = torch.cdist(sub_vectors_flat, c_set)  # (n*h_kv*s, num_centroids)
                
                # Assign to nearest centroid (argmin for distance)
                assigned_codes = torch.argmin(distances, dim=1)  # (n*h_kv*s)
                codes[:, :, :, i] = assigned_codes.reshape(n, h_kv, s)
        
        return codes