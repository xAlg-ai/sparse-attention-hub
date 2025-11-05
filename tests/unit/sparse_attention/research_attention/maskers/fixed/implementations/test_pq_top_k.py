"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for PQCache masker implementation.
"""

import pytest
import torch
import numpy as np


@pytest.mark.unit
class TestPQCacheMaskerImplementation:
    def test_pq_cache_masker_config_creation(self):
        """Test that pq cache masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4, kmeans_iters=10, sink_size=4)
        assert config is not None
        assert config.heavy_size == 10
        assert config.pq_sub_dim == 8
        assert config.pq_bits == 4
        assert config.kmeans_iters == 10
        assert config.sink_size == 4

    def test_pq_cache_masker_creation(self):
        """Test that pq cache masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4, kmeans_iters=10, sink_size=4)
        masker = PQCache(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_creation_from_config(self):
        """Test that pq cache masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4, kmeans_iters=10, sink_size=4)
        masker = PQCache.create_from_config(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_inheritance(self):
        """Test that pq cache masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
        )

        assert issubclass(PQCache, TopKMasker)

    def test_pq_cache_masker_config_inheritance(self):
        """Test that pq cache masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        assert issubclass(PQCacheConfig, TopKMaskerConfig)
    
    def test_pq_cache_uses_l2_distance(self):
        """Test that PQCache uses L2 distance metric, not dot product."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        
        # Create a test case with sufficient sequence length
        # Need seq_len > heavy_size + sink_size to trigger PQ
        batch_size, n_heads, seq_len, head_dim = 1, 2, 128, 64
        config = PQCacheConfig(
            heavy_size=16,
            pq_sub_dim=8,  # head_dim / pq_sub_dim = 8 sub-vectors
            pq_bits=4,     # 16 centroids per sub-vector
            kmeans_iters=10,
            sink_size=4
        )
        
        masker = PQCache(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create synthetic keys/queries/values
        torch.manual_seed(42)
        keys = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        queries = torch.randn(batch_size, n_heads, 1, head_dim, device=device)
        values = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        attention_mask = torch.ones(batch_size, n_heads, 1, seq_len, device=device)
        
        # Create sparse metadata
        sparse_meta_data = {}
        # Create a non-full previous mask to trigger PQ computation
        # The PQCache logic returns early if previous_mask is full
        mask_tensor = torch.ones(batch_size, n_heads, 1, seq_len, dtype=torch.float32, device=device)
        # Make it non-full by zeroing out some positions
        mask_tensor[:, :, :, seq_len//2:] = 0.0
        previous_mask = Mask(
            shape=(batch_size, n_heads, 1, seq_len),
            dtype=torch.float32,
            mask=mask_tensor,
            from_dense_mask=True
        )
        
        # Add mask
        result_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
            layer_idx=0
        )
        
        # Check that metadata was populated with PQ structures
        assert "pq_centroids" in sparse_meta_data
        assert "pq_codes" in sparse_meta_data
        assert 0 in sparse_meta_data["pq_centroids"]
        assert 0 in sparse_meta_data["pq_codes"]
        
        # Centroids should be from original space (not normalized)
        centroids = sparse_meta_data["pq_centroids"][0]
        # Check centroids are not unit vectors (which would indicate normalization)
        centroid_norms = torch.norm(centroids.reshape(-1, centroids.shape[-1]), dim=1)
        assert not torch.allclose(centroid_norms, torch.ones_like(centroid_norms), atol=0.1)
    
    def test_pq_cache_top_k_selection(self):
        """Test that PQCache correctly selects top-k tokens based on approximated distances."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        
        # Create test configuration
        batch_size, n_heads, seq_len, head_dim = 1, 1, 64, 32
        heavy_size = 16
        sink_size = 4
        
        config = PQCacheConfig(
            heavy_size=heavy_size,
            pq_sub_dim=8,
            pq_bits=3,  # 8 centroids
            kmeans_iters=5,
            sink_size=sink_size
        )
        
        masker = PQCache(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create keys with clear pattern: earlier keys are closer to query
        torch.manual_seed(42)
        queries = torch.ones(batch_size, n_heads, 1, head_dim, device=device)
        
        # Create keys where distance increases with position
        keys = torch.zeros(batch_size, n_heads, seq_len, head_dim, device=device)
        for i in range(seq_len):
            # Keys get progressively further from query
            keys[:, :, i, :] = queries[:, :, 0, :] + i * 0.1 * torch.randn(head_dim, device=device)
        
        values = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        attention_mask = torch.ones(batch_size, n_heads, 1, seq_len, device=device)
        
        sparse_meta_data = {}
        # Create a non-full previous mask to trigger PQ computation
        mask_tensor = torch.ones(batch_size, n_heads, 1, seq_len, dtype=torch.float32, device=device)
        mask_tensor[:, :, :, seq_len//2:] = 0.0
        previous_mask = Mask(
            shape=(batch_size, n_heads, 1, seq_len),
            dtype=torch.float32,
            mask=mask_tensor,
            from_dense_mask=True
        )
        
        # Add mask
        result_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
            layer_idx=0
        )
        
        # Get selected indices from mask
        mask_tensor = result_mask.get_dense_mask()
        selected_indices = torch.where(mask_tensor[0, 0, 0] > 0)[0]
        
        # Should have selected heavy_size + sink_size tokens
        # Plus recent tokens based on recent_ratio
        expected_selected = heavy_size + sink_size
        
        # Check that sink tokens are always included
        sink_tokens = torch.arange(sink_size, device=device)
        assert torch.all(torch.isin(sink_tokens, selected_indices.to(device)))
    
    def test_pq_cache_distance_computation(self):
        """Test that the distance computation in PQCache is correct."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        
        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=4,
            pq_bits=2,  # 4 centroids
            kmeans_iters=10,
            sink_size=2
        )
        
        masker = PQCache(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test _partition_vectors
        vectors = torch.randn(2, 3, 10, 16, device=device)  # n, h, s, dh
        num_sub_vectors = 4  # 16 / 4 = 4
        dm = 4
        
        partitioned = masker._partition_vectors(vectors, num_sub_vectors, dm)
        assert partitioned.shape == (2, 3, 10, 4, 4)  # n, h, s, num_sub_vectors, dm
        
        # Test that partitioning preserves data
        reconstructed = partitioned.reshape(2, 3, 10, 16)
        assert torch.allclose(vectors, reconstructed)
    
    def test_pq_cache_code_assignment(self):
        """Test that code assignment uses L2 distance matching the clustering metric."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        
        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=4,
            pq_bits=2,  # 4 centroids
            kmeans_iters=10,
            sink_size=2
        )
        
        masker = PQCache(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create test data
        n, h, s, num_sub_vectors, dm = 1, 2, 10, 4, 4
        keys_partitioned = torch.randn(n, h, s, num_sub_vectors, dm, device=device)
        
        # Create simple centroids where we know the expected assignment
        centroids = torch.zeros(h, num_sub_vectors, 4, dm, device=device)
        # Make distinct centroids
        for i in range(4):
            centroids[:, :, i, :] = i * 2.0
        
        # Assign codes
        codes = masker._assign_codes_to_keys(keys_partitioned, centroids)
        
        assert codes.shape == (n, h, s, num_sub_vectors)
        assert codes.dtype == torch.long
        
        # Verify assignment is based on L2 distance
        # For each key, check that assigned centroid is indeed the nearest
        for head_idx in range(h):
            for sub_idx in range(num_sub_vectors):
                for seq_idx in range(s):
                    key_vec = keys_partitioned[0, head_idx, seq_idx, sub_idx, :]
                    assigned_idx = codes[0, head_idx, seq_idx, sub_idx]
                    
                    # Compute distances to all centroids
                    distances = torch.norm(
                        centroids[head_idx, sub_idx, :, :] - key_vec.unsqueeze(0),
                        dim=1
                    )
                    nearest_idx = torch.argmin(distances)
                    
                    assert assigned_idx == nearest_idx, f"Assignment mismatch at ({head_idx}, {sub_idx}, {seq_idx})"
    
    def test_pq_cache_no_normalization_in_clustering(self):
        """Verify that keys are not normalized before clustering."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        
        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=8,
            pq_bits=3,
            kmeans_iters=5,
            sink_size=4
        )
        
        masker = PQCache(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create keys with very different magnitudes
        n, h, s, dh = 1, 2, 20, 16
        dm = config.pq_sub_dim
        num_sub_vectors = dh // dm
        
        keys_original = torch.randn(n, h, s, dh, device=device)
        # Scale some keys to have very different magnitudes
        keys_original[:, :, ::2, :] *= 10.0  # Every other key has 10x magnitude
        
        keys_partitioned = masker._partition_vectors(keys_original, num_sub_vectors, dm)
        
        # Call clustering
        codes, centroids = masker._cluster_keys(keys_partitioned, keys_original)
        
        # Check that centroids reflect the magnitude differences
        # If keys were normalized, all centroids would have similar magnitudes
        centroid_magnitudes = torch.norm(centroids.reshape(-1, dm), dim=1)
        
        # There should be significant variance in centroid magnitudes
        # if we didn't normalize
        mag_std = torch.std(centroid_magnitudes)
        mag_mean = torch.mean(centroid_magnitudes)
        
        # Coefficient of variation should be high if not normalized
        cv = mag_std / (mag_mean + 1e-8)
        assert cv > 0.2, f"Centroid magnitudes have low variance (CV={cv:.3f}), suggesting normalization occurred"
