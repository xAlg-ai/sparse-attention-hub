"""PQ cache top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import multiprocessing as mp

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
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
        self.use_gpu_kmeans = torch.cuda.is_available()  # Use GPU K-means if available

    def _gpu_kmeans(self, data: torch.Tensor, n_clusters: int, max_iter: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized GPU K-means implementation for faster clustering."""
        n_samples, dim = data.shape
        device = data.device
        
        # K-means++ initialization for better convergence
        centers = torch.zeros(n_clusters, dim, device=device, dtype=data.dtype)
        centers[0] = data[torch.randint(n_samples, (1,), device=device)]
        
        # K-means++ initialization
        for i in range(1, n_clusters):
            # Compute distances to nearest center
            dists = torch.cdist(data, centers[:i]).min(dim=1)[0]
            # Sample next center with probability proportional to squared distance
            probs = dists ** 2
            probs = probs / probs.sum()
            cumprobs = torch.cumsum(probs, dim=0)
            centers[i] = data[(cumprobs > torch.rand(1, device=device)).nonzero()[0]]
        
        # Main K-means loop
        for _ in range(max_iter):
            # Assign points to nearest center
            distances = torch.cdist(data, centers, p=2)
            labels = torch.argmin(distances, dim=1)
            
            # Vectorized center update using scatter operations
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(n_clusters, device=device)
            
            # Use scatter_add for efficient aggregation
            labels_expanded = labels.unsqueeze(1).expand(-1, dim)
            new_centers.scatter_add_(0, labels_expanded, data)
            counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
            
            # Avoid division by zero
            counts = counts.clamp(min=1)
            centers = new_centers / counts.unsqueeze(1)
        
        return centers, labels

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
            # Use CUDA stream for overlapping computation
            with torch.cuda.stream(torch.cuda.Stream()):
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
            recent_ratio = sparse_meta_data.get("recent_ratio", 0.5)  # Default 50% recent (matching original PQCache)
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
        
        # Since we're using negative distances as scores, larger score = smaller distance
        # So we still want largest scores (which correspond to smallest distances)
        scores_for_topk = scores.clone()
        scores_for_topk.masked_fill_(scores == float("-inf"), -1e10)  # In-place operation
        
        # Get top-k values and indices (largest scores = smallest distances)
        _, top_k_indices = torch.topk(scores_for_topk, k, dim=-1, sorted=False, largest=True)
        
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
        """Cluster keys using parallel K-means and return codebook and centroids.

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
        if sink_size >= s:
            return all_codes, all_centroids

        # Prepare all data for clustering
        keys_to_cluster = keys_partitioned[:, :, sink_size:, :, :].reshape(
            n * h_kv * num_sub_vectors, s - sink_size, dm
        )
        
        # Use GPU K-means if available and sequence is large enough
        if self.use_gpu_kmeans and (s - sink_size) > 1000:
            # Process all groups on GPU in parallel
            all_centers = []
            all_labels = []
            
            for idx in range(n * h_kv * num_sub_vectors):
                data = keys_to_cluster[idx].float()
                
                # Skip if all zeros or not enough samples
                if torch.all(data == 0) or data.shape[0] < num_centroids:
                    all_centers.append(None)
                    all_labels.append(None)
                    continue
                
                try:
                    centers, labels = self._gpu_kmeans(data, num_centroids, max_iter=min(self.kmeans_iters, 10))
                    all_centers.append(centers)
                    all_labels.append(labels)
                except Exception as e:
                    print(f"GPU K-means failed for group {idx}: {e}")
                    all_centers.append(None)
                    all_labels.append(None)
            
            # Process results
            valid_indices = []
            centers_list = []
            labels_list = []
            
            for idx, (centers, labels) in enumerate(zip(all_centers, all_labels)):
                if centers is not None:
                    valid_indices.append(idx)
                    centers_list.append(centers)
                    labels_list.append(labels)
            
            # No CPU conversion needed - already on GPU
            if centers_list:
                for i, idx in enumerate(valid_indices):
                    head_idx = (idx // num_sub_vectors) % h_kv
                    sub_idx = idx % num_sub_vectors
                    
                    all_centroids[head_idx, sub_idx] = centers_list[i]
                    labels_reshaped = labels_list[i].reshape(n, s - sink_size)
                    all_codes[:, head_idx, sink_size:, sub_idx] = labels_reshaped
            
            return all_codes, all_centroids
        
        # Fall back to CPU K-means for smaller sequences
        keys_cpu = keys_to_cluster.to(torch.float32).cpu()
        
        def run_kmeans(group_idx):
            x = keys_cpu[group_idx].numpy()
            
            if np.all(x == 0) or len(x) < num_centroids:
                return None
                
            try:
                # Use simpler, faster K-means settings
                kmeans = MiniBatchKMeans(
                    n_clusters=num_centroids,
                    init='k-means++',
                    max_iter=min(self.kmeans_iters, 10),
                    batch_size=min(2048, len(x)),
                    n_init=1,
                    random_state=42,
                    compute_labels=True,
                    max_no_improvement=3
                )
                
                kmeans.fit(x)
                return (kmeans.cluster_centers_, kmeans.labels_)
            except Exception as e:
                print(f"K-means failed for group {group_idx}: {e}")
                return None
        
        # Run K-means in parallel
        num_workers = min(os.cpu_count() or 4, h_kv * num_sub_vectors)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all K-means tasks
            futures = {}
            for idx in range(n * h_kv * num_sub_vectors):
                future = executor.submit(run_kmeans, idx)
                futures[future] = idx
            
            # Collect results as they complete
            all_centers = []
            all_labels = []
            valid_indices = []
            
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                
                if result is not None:
                    centers, labels = result
                    all_centers.append(centers)
                    all_labels.append(labels)
                    valid_indices.append(idx)
        
        # Batch transfer all results to GPU
        if all_centers:
            # Convert to tensors and transfer in batch
            centers_tensor = torch.from_numpy(np.stack(all_centers)).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            labels_tensor = torch.from_numpy(np.stack(all_labels)).to(
                device=device, dtype=torch.long, non_blocking=True
            )
            
            # Assign results
            for i, idx in enumerate(valid_indices):
                # Decode original position
                head_idx = (idx // num_sub_vectors) % h_kv
                sub_idx = idx % num_sub_vectors
                
                all_centroids[head_idx, sub_idx] = centers_tensor[i]
                
                # Reshape labels back
                labels_reshaped = labels_tensor[i].reshape(n, s - sink_size)
                all_codes[:, head_idx, sink_size:, sub_idx] = labels_reshaped

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

        #compute PQ approximation for middle tokens (sink to end)
        if s > sink_size:
            # Partition queries into sub-vectors (no normalization to match original)
            dm = self.pq_sub_dim
            num_sub_vectors = dh // dm
            queries_partitioned = queries.reshape(n, h_q, lq, num_sub_vectors, dm)

            #handle GQA for centroids
            centroids_expanded = centroids
            if h_q != h_kv:
                centroids_expanded = centroids.unsqueeze(1).repeat(1, ngroups, 1, 1, 1)
                centroids_expanded = centroids_expanded.reshape(h_q, num_sub_vectors, centroids.shape[2], dm)

            # Compute query-centroid L2 distances
            # Distance = ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
            qf = queries_partitioned.float()
            cf = centroids_expanded.float()
            
            # Compute squared norms
            q_norm_sq = (qf ** 2).sum(dim=-1, keepdim=True)  # (n, h_q, lq, num_sub_vectors, 1)
            c_norm_sq = (cf ** 2).sum(dim=-1, keepdim=False)  # (h_q, num_sub_vectors, num_centroids)
            
            # Compute dot products
            qc_dot = torch.einsum("nhlmd,hmcd->nhlmc", qf, cf)
            
            # Compute L2 distances
            query_centroid_distances = q_norm_sq + c_norm_sq.unsqueeze(0).unsqueeze(2) - 2 * qc_dot

            #handle GQA for codes
            codes_middle = codes[:, :, sink_size:, :]
            if h_q != h_kv:
                codes_middle = codes_middle.repeat_interleave(ngroups, dim=1)

            #accumulate PQ distances
            middle_len = s - sink_size
            
            # Vectorized PQ distance accumulation
            # More memory efficient than one-hot encoding
            pq_distances = torch.zeros((n, h_q, lq, middle_len), dtype=torch.float32, device=device)
            
            # Process in a vectorized way
            for m in range(num_sub_vectors):
                # Get indices for this subvector
                indices = codes_middle[:, :, :, m].unsqueeze(2).expand(n, h_q, lq, middle_len)
                
                # Gather distances for this subvector
                distances_m = torch.gather(
                    query_centroid_distances[:, :, :, m, :],
                    dim=-1,
                    index=indices
                )
                
                # Accumulate
                pq_distances.add_(distances_m)

            # Convert distances to scores for selection
            # Use negative distance so larger score = smaller distance
            # Note: scaling is not applied here as PQ is only used for selection
            pq_scores = -pq_distances

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
        """Assign codes to keys based on nearest centroids using L2 distance.

        Args:
            keys_partitioned: tensor of shape (n, h, s, num_sub_vectors, dm)
            centroids: tensor of shape (h, num_sub_vectors, num_centroids, dm)

        Returns:
            codes: tensor of shape (n, h, s, num_sub_vectors)
        """
        n, h_kv, s, num_sub_vectors, dm = keys_partitioned.shape

        # Vectorized code assignment using broadcasting
        # Shape: keys_partitioned (n, h_kv, s, num_sub_vectors, dm)
        # Shape: centroids (h_kv, num_sub_vectors, num_centroids, dm)
        
        # Expand dimensions for broadcasting
        keys_expanded = keys_partitioned.unsqueeze(4).float()  # (n, h_kv, s, num_sub_vectors, 1, dm)
        centroids_expanded = centroids.unsqueeze(0).unsqueeze(2).float()  # (1, h_kv, 1, num_sub_vectors, num_centroids, dm)
        
        # Compute L2 distances using broadcasting
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        keys_norm_sq = (keys_expanded ** 2).sum(dim=-1, keepdim=True)  # (n, h_kv, s, num_sub_vectors, 1, 1)
        centroids_norm_sq = (centroids_expanded ** 2).sum(dim=-1, keepdim=True)  # (1, h_kv, 1, num_sub_vectors, num_centroids, 1)
        
        # Compute dot product
        keys_for_matmul = keys_expanded.squeeze(4)  # (n, h_kv, s, num_sub_vectors, dm)
        centroids_for_matmul = centroids_expanded.squeeze(4).transpose(-2, -1)  # (1, h_kv, 1, num_sub_vectors, dm, num_centroids)
        
        # Batch matrix multiplication
        dot_product = torch.matmul(keys_for_matmul.unsqueeze(4), centroids_for_matmul).squeeze(4)  # (n, h_kv, s, num_sub_vectors, num_centroids)
        
        # Compute distances
        distances = keys_norm_sq.squeeze(-1) + centroids_norm_sq.squeeze(-1) - 2 * dot_product
        
        # Get nearest centroid indices
        codes = torch.argmin(distances, dim=-1)  # (n, h_kv, s, num_sub_vectors)

        return codes

    def calculate_effective_heavy_size(self, seq_len_keys: int) -> int:
        """Calculate effective heavy size based on configuration."""
        return self._calculate_effective_size(self.heavy_size, seq_len_keys)
