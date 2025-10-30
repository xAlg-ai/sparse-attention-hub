"""PQ cache top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class PQCacheConfig(TopKMaskerConfig):
    """Configuration for PQCache masker."""
    pq_sub_dim: int
    pq_bits: int
    kmeans_iters: int
    sink_size: int


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
        self.sink_size = config.sink_size
        self._current_sink_size = 0

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
        """Add PQ cache mask using Product Quantization for efficient attention."""
        
        #get shapes
        n, h_kv, s, dh = keys.shape
        _, h_q, lq, _ = queries.shape
        
        #cehck if we should use full attention for small sequences
        if previous_mask.is_full_mask():
            return previous_mask
            
        #calculate effective heavy size
        tensor_dims = self._extract_tensor_dimensions(keys, queries)
        effective_heavy_size = self.calculate_effective_heavy_size(s)
        
        #if sequence is small enough, use full attention
        if s <= effective_heavy_size + self.sink_size:
            return self._create_full_mask(tensor_dims, previous_mask.dtype)
        
        dtype = keys.dtype
        dm = self.pq_sub_dim
        assert dh % dm == 0, "dh must be divisible by dm"

        num_sub_vectors = dh // dm
        num_centroids = 2 ** self.pq_bits
        layer_idx = kwargs.get("layer_idx", 0)

        #partition keys for clustering
        keys_partitioned = self._partition_vectors(keys, num_sub_vectors, dm)
        
        # init meta data if needed
        if "pq_centroids" not in sparse_meta_data:
            sparse_meta_data["pq_centroids"] = {}
        if "pq_codes" not in sparse_meta_data:
            sparse_meta_data["pq_codes"] = {}
        
        # get sink size
        self._current_sink_size = min(self.sink_size, s)
        
        # build or retrieve PQ structures
        if layer_idx not in sparse_meta_data["pq_centroids"] or layer_idx not in sparse_meta_data["pq_codes"]:
            codebook, centroids = self._cluster_keys(keys_partitioned, keys)
            sparse_meta_data["pq_centroids"][layer_idx] = centroids
            sparse_meta_data["pq_codes"][layer_idx] = codebook
        else:
            # Update if cache has grown
            if layer_idx in sparse_meta_data["pq_codes"]:
                existing_s = sparse_meta_data["pq_codes"][layer_idx].shape[2]
                if s > existing_s:
                    self._update_sparse_meta_data(sparse_meta_data, keys, layer_idx=layer_idx)

        # Compute approximate scores using hybrid approach
        query_key_scores = self._compute_approximate_scores_hybrid(
            queries,
            keys,
            sparse_meta_data["pq_centroids"][layer_idx],
            sparse_meta_data["pq_codes"][layer_idx],
            scaling
        )
        
        #apply attention mask if provided
        if attention_mask is not None:
            #make sure attention mask has correct shape
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
            if attention_mask.shape[-1] != s:
                attention_mask = attention_mask[:, :, :, :s]
            #apply mask (attention_mask is 0 for masked positions)
            query_key_scores = query_key_scores.masked_fill(attention_mask == 0, float("-inf"))
        
        #no compression of sink tokens
        if self._current_sink_size > 0:
            query_key_scores[:, :, :, :self._current_sink_size] = float("inf")
        
        #handle recent window
        recent_size = sparse_meta_data.get("recent_size", None)
        if recent_size is None:
            recent_ratio = sparse_meta_data.get("recent_ratio", 0.1)  # Default 10% recent
            effective_len = max(0, s - self._current_sink_size)
            recent_size = int(recent_ratio * effective_len)
        
        recent_size = min(recent_size, s - self._current_sink_size)
        if recent_size > 0:
            #include recent tokens (last tokens before query) perpaper
            recent_start = s - recent_size
            query_key_scores[:, :, :, recent_start:] = float("inf")
        
        #func to get  top-k indices
        top_k_indices = self._get_topk_indices_from_scores(
            query_key_scores, 
            previous_mask, 
            effective_heavy_size
        )
        
        #create mask from indices
        mask_shape = (n, h_q, lq, s)
        if previous_mask.shape != mask_shape:
            previous_mask = Mask.create_full_mask(mask_shape, dtype=dtype)
        
        this_mask = self._create_mask_from_rowise_indices(
            tensor_dims, 
            top_k_indices, 
            keys.device, 
            previous_mask.dtype
        )
        
        #new merge with previous mask
        new_mask = previous_mask.merge_mask(this_mask, inplace=False)
        return new_mask

    #helper
    def _get_topk_indices_from_scores(
        self, 
        scores: torch.Tensor, 
        previous_mask: Mask, 
        k: int
    ) -> torch.Tensor:
        """Get top-k indices from scores, handling inf values properly."""
        n, h, lq, s = scores.shape
        
        #replace -inf with very negative value for topk
        scores_for_topk = scores.clone()
        scores_for_topk[scores == float("-inf")] = -1e10
        
        # Get top-k values and indices
        _, top_k_indices = torch.topk(scores_for_topk, k, dim=-1, sorted=False)
        
        return top_k_indices

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)

    def _partition_vectors(self, vectors: torch.Tensor, num_sub_vectors: int, dm: int) -> torch.Tensor:
        """Partition vectors into sub-vectors.
        
        Args:
            vectors: input tensor of shape (n, h, s, dh)
            num_sub_vectors: number of sub vectors to partition into (m)
            dm: dimension of each sub vector
        
        Returns:
            Partitioned tensor of shape (n, h, s, num_sub_vectors, dm)
        """
        n, h, s, dh = vectors.shape
        assert dh == num_sub_vectors * dm, "dh must equal num_sub_vectors * dm"
        return vectors.reshape(n, h, s, num_sub_vectors, dm)

    def _cluster_keys(self, keys_partitioned: torch.Tensor, keys_original: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cluster keys using normalized PQ and return codebook and centroids.

        Args:
            keys_partitioned: Input keys tensor of shape (n, h, s, num_sub_vectors, dm)
            keys_original: Original keys tensor of shape (n, h, s, dh) for normalization

        Returns:
            codebook: Tensor of shape (n, h, s, num_sub_vectors) with assigned codes
            centroids: Tensor of shape (h_kv, num_sub_vectors, num_centroids, dm)
        """
        n, h_kv, s, num_sub_vectors, dm = keys_partitioned.shape
        device = keys_partitioned.device
        num_centroids = 2 ** self.pq_bits

        # Initialize tensors
        all_centroids = torch.zeros(
            (h_kv, num_sub_vectors, num_centroids, dm),
            dtype=torch.float32,
            device=device
        )
        all_codes = torch.zeros(
            (n, h_kv, s, num_sub_vectors),
            dtype=torch.long,
            device=device
        )

        sink_size = self._current_sink_size

        # Normalize keys for better clustering
        keys_norm = keys_original / (keys_original.norm(dim=-1, keepdim=True) + 1e-8)
        keys_partitioned_norm = keys_norm.reshape(n, h_kv, s, num_sub_vectors, dm)

        for head_idx in range(h_kv):
            for i in range(num_sub_vectors):
                # Only process non-sink region for PQ
                if sink_size < s:
                    sub_vectors = keys_partitioned_norm[:, head_idx, sink_size:, i, :]  # (n, s-sink_size, dm)
                    sub_vectors = torch.nan_to_num(sub_vectors, nan=0.0, posinf=0.0, neginf=0.0)

                    # Flatten and cluster
                    x = sub_vectors.reshape(-1, dm).to(torch.float32).cpu().numpy()

                    # Skip if all zeros or not enough samples
                    if np.all(x == 0) or len(x) < num_centroids:
                        continue

                    # Use MiniBatchKMeans for large datasets
                    if len(x) > 10000:
                        kmeans = MiniBatchKMeans(
                            n_clusters=num_centroids,
                            init='k-means++',
                            max_iter=self.kmeans_iters,
                            batch_size=min(1024, len(x)),
                            n_init=3,
                            random_state=42
                        )
                    else:
                        kmeans = KMeans(
                            n_clusters=num_centroids,
                            init='k-means++',
                            max_iter=self.kmeans_iters,
                            n_init=3,
                            random_state=42
                        )

                    try:
                        kmeans.fit(x)
                        labels = kmeans.labels_
                        centers = kmeans.cluster_centers_

                        centers = np.nan_to_num(centers, nan=0.0, posinf=0.0, neginf=0.0)

                        all_centroids[head_idx, i] = torch.from_numpy(centers).to(
                            device=device, dtype=torch.float32
                        )

                        codes_slice = torch.from_numpy(labels).reshape(n, s - sink_size).to(
                            device=device, dtype=torch.long
                        )
                        all_codes[:, head_idx, sink_size:, i] = codes_slice
                    except Exception as e:
                        print(f"K-means failed for head {head_idx}, sub {i}: {e}")
                        continue

        return all_codes, all_centroids

    def _compute_approximate_scores_hybrid(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        centroids: torch.Tensor,
        codes: torch.Tensor,
        scaling: float
    ) -> torch.Tensor:
        """Compute approximate attention scores using hybrid approach.

        For sink tokens: compute exact scores
        For middle tokens: use normalized PQ approximation with magnitude restoration

        Args:
            queries: tensor of shape (n, h_q, lq, dh)
            keys: tensor of shape (n, h_kv, s, dh)
            centroids: tensor of shape (h_kv, num_sub_vectors, num_centroids, dm)
            codes: tensor of shape (n, h_kv, s, num_sub_vectors)
            scaling: scaling factor for attention scores

        Returns:
            approximate scores: tensor of shape (n, h_q, lq, s)
        """
        n, h_q, lq, dh = queries.shape
        _, h_kv, s, _ = keys.shape
        device = queries.device
        sink_size = self._current_sink_size

        #init scores
        approx_scores = torch.zeros((n, h_q, lq, s), dtype=torch.float32, device=device)

        #handle GQA
        ngroups = h_q // h_kv if h_q != h_kv else 1

        #compute exact scores for sink tokens
        if sink_size > 0:
            sink_keys = keys[:, :, :sink_size, :]
            if h_q != h_kv:
                sink_keys = repeat_kv(sink_keys, ngroups)
            sink_scores = torch.matmul(queries * scaling, sink_keys.transpose(-2, -1))
            approx_scores[:, :, :, :sink_size] = sink_scores

        #compute PQ approximation for middle tokens (sink to end)
        if s > sink_size:
            #store magnitudes before normalization
            query_magnitudes = queries.norm(dim=-1, keepdim=True)  # (n, h_q, lq, 1)
            key_magnitudes = keys[:, :, sink_size:, :].norm(dim=-1)  # (n, h_kv, s-sink)

            #normalize
            queries_norm = queries / (query_magnitudes + 1e-8)
            keys_middle_norm = keys[:, :, sink_size:, :] / (key_magnitudes.unsqueeze(-1) + 1e-8)

            #partition normalized vectors
            dm = self.pq_sub_dim
            num_sub_vectors = dh // dm
            queries_partitioned = queries_norm.reshape(n, h_q, lq, num_sub_vectors, dm)

            #handle GQA for centroids
            centroids_expanded = centroids
            if h_q != h_kv:
                centroids_expanded = centroids.unsqueeze(1).repeat(1, ngroups, 1, 1, 1)
                centroids_expanded = centroids_expanded.reshape(h_q, num_sub_vectors, centroids.shape[2], dm)

            #compute query-centroid scores
            qf = queries_partitioned.float()
            cf = centroids_expanded.float()
            query_centroid_scores = torch.einsum("nhlmd,hmcd->nhlmc", qf, cf)

            #handle GQA for codes
            codes_middle = codes[:, :, sink_size:, :]
            if h_q != h_kv:
                codes_middle = codes_middle.repeat_interleave(ngroups, dim=1)

            #accumulate PQ scores
            middle_len = s - sink_size
            pq_scores = torch.zeros((n, h_q, lq, middle_len), dtype=torch.float32, device=device)

            for m in range(num_sub_vectors):
                centroid_ids = codes_middle[:, :, :, m]  # (n, h_q, middle_len)
                scores_m = query_centroid_scores[:, :, :, m, :]  # (n, h_q, lq, num_centroids)
                centroid_ids_expanded = centroid_ids.unsqueeze(2).expand(n, h_q, lq, middle_len)
                gathered = torch.gather(scores_m, dim=-1, index=centroid_ids_expanded)
                pq_scores += gathered

                        #restore magnitudes with proper GQA handling
            if h_q != h_kv:
                key_magnitudes_expanded = key_magnitudes.repeat_interleave(ngroups, dim=1)
            else:
                key_magnitudes_expanded = key_magnitudes

            key_magnitudes_expanded = key_magnitudes_expanded.unsqueeze(2)  # (n, h_q, 1, middle_len)

#get magnitudes and scale
            pq_scores = pq_scores * query_magnitudes * key_magnitudes_expanded * scaling

            approx_scores[:, :, :, sink_size:] = pq_scores

        return approx_scores

    def _update_sparse_meta_data(
        self, 
        sparse_meta_data: Dict[Any, Any], 
        keys: torch.Tensor, 
        layer_idx: int = 0
    ) -> None:
        """Update PQ codes when new keys are added to cache."""
        if "pq_centroids" not in sparse_meta_data or layer_idx not in sparse_meta_data["pq_centroids"]:
            return
        
        n, h_kv, new_s, dh = keys.shape
        dm = self.pq_sub_dim
        num_sub_vectors = dh // dm
        
        centroids = sparse_meta_data["pq_centroids"][layer_idx]
        existing_codes = sparse_meta_data["pq_codes"][layer_idx]
        
        _, _, existing_s, _ = existing_codes.shape
        
        if new_s <= existing_s:
            return
        
        #process only new keys
        new_keys_only = keys[:, :, existing_s:, :]
        keys_partitioned = self._partition_vectors(new_keys_only, num_sub_vectors, dm)
        
        #assign codes to new keys
        new_codes = self._assign_codes_to_keys(keys_partitioned, centroids)
        
        #add new codes
        updated_codes = torch.cat([existing_codes, new_codes], dim=2)
        sparse_meta_data["pq_codes"][layer_idx] = updated_codes

    def _assign_codes_to_keys(
        self,
        keys_partitioned: torch.Tensor,
        centroids: torch.Tensor
    ) -> torch.Tensor:
        """Assign codes to keys based on nearest centroids using cosine similarity.

        Args:
            keys_partitioned: tensor of shape (n, h, s, num_sub_vectors, dm)
            centroids: tensor of shape (h, num_sub_vectors, num_centroids, dm)

        Returns:
            codes: tensor of shape (n, h, s, num_sub_vectors)
        """
        n, h_kv, s, num_sub_vectors, dm = keys_partitioned.shape
        device = keys_partitioned.device
        num_centroids = centroids.shape[2]

        codes = torch.zeros(
            (n, h_kv, s, num_sub_vectors),
            dtype=torch.long,
            device=device
        )

        for head_idx in range(h_kv):
            for i in range(num_sub_vectors):
                sub_vectors = keys_partitioned[:, head_idx, :, i, :]  # (n, s, dm)
                c_set = centroids[head_idx, i, :, :]  # (num_centroids, dm)

                #flatten and normalize
                sub_vectors_flat = sub_vectors.reshape(-1, dm).float()
                c_set_flat = c_set.float()

                #normalize to do cosine similarity
                sub_vectors_norm = sub_vectors_flat / (sub_vectors_flat.norm(dim=-1, keepdim=True) + 1e-8)
                c_set_norm = c_set_flat / (c_set_flat.norm(dim=-1, keepdim=True) + 1e-8)

                #get cosine similarity (higher is better)
                similarities = torch.matmul(sub_vectors_norm, c_set_norm.t())

                #send to similar centroids to most similar centroid
                assigned_codes = torch.argmax(similarities, dim=1)
                codes[:, head_idx, :, i] = assigned_codes.reshape(n, s)

        return codes

    def calculate_effective_heavy_size(self, seq_len_keys: int) -> int:
        """Calculate effective heavy size based on configuration."""
        return self._calculate_effective_size(self.heavy_size, seq_len_keys)
