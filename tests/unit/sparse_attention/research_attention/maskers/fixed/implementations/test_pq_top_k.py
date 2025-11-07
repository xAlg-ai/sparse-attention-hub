"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for PQCache masker implementation.
"""

import pytest
import torch


@pytest.mark.unit
class TestPQCacheMaskerImplementation:
    def test_pq_cache_masker_config_creation(self):
        """Test that pq cache masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=8,
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        assert config is not None
        assert config.heavy_size == 10
        assert config.pq_sub_dim == 8
        assert config.pq_bits == 4
        assert config.kmeans_iter == 10
        assert config.init_offset == 4
        assert config.metric == "euclidean"

    def test_pq_cache_masker_creation(self):
        """Test that pq cache masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=8,
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker = PQCache(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_creation_from_config(self):
        """Test that pq cache masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=8,
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
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

    def test_perform_kmeans_clustering_euclidean(self):
        """Test K-means clustering on keys with Euclidean metric."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        # Setup config
        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=8,  # Each subvector has 8 dimensions
            pq_bits=4,  # 2^4 = 16 centroids per subvector
            kmeans_iter=10,
            init_offset=4,  # Skip first 4 tokens (sink)
            metric="euclidean",
        )
        masker = PQCache(config)

        # Create test keys: [bsz=1, num_heads=2, seq_len=20, head_dim=16]
        # head_dim=16 will be split into 2 subvectors of 8 dimensions each
        bsz = 1
        num_heads = 2
        seq_len_keys = 20
        head_dim = 16
        keys = torch.randn(bsz, num_heads, seq_len_keys, head_dim, dtype=torch.float32)

        # Create sparse_meta_data
        sparse_meta_data = {}
        layer_idx = 0

        # Initialize PQ cache
        masker._initialize_pq_cache(sparse_meta_data, layer_idx)

        # Perform K-means clustering
        centroids, codebook = masker._perform_kmeans_clustering(
            keys, layer_idx, sparse_meta_data
        )

        # Check output shapes
        n_subvec_per_head = head_dim // config.pq_sub_dim  # 16 // 8 = 2
        cent_cnt = 2 ** config.pq_bits  # 2^4 = 16
        n_quantized_keys = seq_len_keys - config.init_offset  # 20 - 4 = 16

        # Centroids: [bsz, num_heads, n_subvec, cent_cnt, subvec_d]
        assert centroids.shape == (
            bsz,
            num_heads,
            n_subvec_per_head,
            cent_cnt,
            config.pq_sub_dim,
        )
        assert centroids.shape == (1, 2, 2, 16, 8)

        # Codebook: [bsz, n_quantized_keys, num_heads, n_subvec]
        assert codebook.shape == (bsz, n_quantized_keys, num_heads, n_subvec_per_head)
        assert codebook.shape == (1, 16, 2, 2)

        # Check codebook values are within valid range [0, cent_cnt-1]
        assert codebook.min() >= 0
        assert codebook.max() < cent_cnt
        assert codebook.dtype == torch.int64

        # Check centroids are on the same device as keys
        assert centroids.device == keys.device
        assert codebook.device == keys.device

        # Check data is stored in sparse_meta_data
        assert sparse_meta_data["pq_centroids"][layer_idx] is not None
        assert sparse_meta_data["pq_codebook"][layer_idx] is not None
        assert torch.equal(sparse_meta_data["pq_centroids"][layer_idx], centroids)
        assert torch.equal(sparse_meta_data["pq_codebook"][layer_idx], codebook)

        # For Euclidean metric, phi should not be stored
        assert sparse_meta_data["pq_ip2l2_phi"][layer_idx] is None

        # Test reconstruction: compute errors using utility function
        from sparse_attention_hub.sparse_attention.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        original_keys = keys[:, :, config.init_offset :, :]
        errors = compute_reconstruction_errors(
            original_keys=original_keys,
            centroids=centroids,
            codebook=codebook,
            pq_sub_dim=config.pq_sub_dim,
            use_ip_metric=False,
        )

        print(f"\nEuclidean Metric Reconstruction Errors:")
        print(f"  MSE Error: {errors['mse_error']:.6f}")
        print(f"  L2 Error: {errors['l2_error']:.6f}")
        print(f"  Relative Error: {errors['relative_error']:.6f}")

        # Sanity check: error should be reasonable (not too large)
        assert errors["relative_error"] < 1.0, (
            f"Relative error {errors['relative_error']} is too large"
        )

    def test_perform_kmeans_clustering_ip(self):
        """Test K-means clustering on keys with IP metric."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        # Setup config with IP metric
        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=8,
            pq_bits=4,
            kmeans_iter=10,
            init_offset=4,
            metric="ip",
        )
        masker = PQCache(config)

        # Create test keys
        bsz = 1
        num_heads = 2
        seq_len_keys = 20
        head_dim = 16
        keys = torch.randn(bsz, num_heads, seq_len_keys, head_dim, dtype=torch.float32)

        # Create sparse_meta_data
        sparse_meta_data = {}
        layer_idx = 0

        # Initialize PQ cache
        masker._initialize_pq_cache(sparse_meta_data, layer_idx)

        # Perform K-means clustering
        centroids, codebook = masker._perform_kmeans_clustering(
            keys, layer_idx, sparse_meta_data
        )

        # Check output shapes
        n_subvec_per_head = head_dim // config.pq_sub_dim
        cent_cnt = 2 ** config.pq_bits
        n_quantized_keys = seq_len_keys - config.init_offset

        # For IP metric, centroids have augmented dimension (subvec_d + 1)
        assert centroids.shape == (
            bsz,
            num_heads,
            n_subvec_per_head,
            cent_cnt,
            config.pq_sub_dim + 1,  # +1 for augmentation
        )
        assert centroids.shape == (1, 2, 2, 16, 9)

        # Codebook shape remains the same
        assert codebook.shape == (bsz, n_quantized_keys, num_heads, n_subvec_per_head)
        assert codebook.shape == (1, 16, 2, 2)

        # For IP metric, phi should be stored
        assert sparse_meta_data["pq_ip2l2_phi"][layer_idx] is not None
        ip2l2_phi = sparse_meta_data["pq_ip2l2_phi"][layer_idx]

        # Phi shape: [n_groups, 1, 1] where n_groups = bsz * num_heads * n_subvec
        n_groups = bsz * num_heads * n_subvec_per_head
        assert ip2l2_phi.shape == (n_groups, 1, 1)
        assert ip2l2_phi.shape == (4, 1, 1)  # 1 * 2 * 2 = 4

        # Check phi values are positive (max squared norms)
        assert (ip2l2_phi > 0).all()

        # Test reconstruction: compute errors using utility function
        from sparse_attention_hub.sparse_attention.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        original_keys = keys[:, :, config.init_offset :, :]
        errors = compute_reconstruction_errors(
            original_keys=original_keys,
            centroids=centroids,
            codebook=codebook,
            pq_sub_dim=config.pq_sub_dim,
            use_ip_metric=True,  # IP metric: centroids are augmented
        )

        print(f"\nIP Metric Reconstruction Errors:")
        print(f"  MSE Error: {errors['mse_error']:.6f}")
        print(f"  L2 Error: {errors['l2_error']:.6f}")
        print(f"  Relative Error: {errors['relative_error']:.6f}")

        # Sanity check: error should be reasonable (not too large)
        assert errors["relative_error"] < 1.0, (
            f"Relative error {errors['relative_error']} is too large"
        )

    def test_perform_kmeans_clustering_different_init_offset(self):
        """Test that init_offset correctly skips keys from the front."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        # Test with different init_offsets
        # Using pq_bits=3 (8 centroids) and seq_len=50 to ensure enough samples
        for init_offset in [0, 5, 10]:
            config = PQCacheConfig(
                heavy_size=10,
                pq_sub_dim=8,
                pq_bits=3,  # 2^3 = 8 centroids (need at least 8 samples)
                kmeans_iter=10,
                init_offset=init_offset,
                metric="euclidean",
            )
            masker = PQCache(config)

            bsz = 1
            num_heads = 2
            seq_len_keys = 50  # Long enough for all init_offset values
            head_dim = 16
            keys = torch.randn(bsz, num_heads, seq_len_keys, head_dim, dtype=torch.float32)

            sparse_meta_data = {}
            layer_idx = 0
            masker._initialize_pq_cache(sparse_meta_data, layer_idx)

            centroids, codebook = masker._perform_kmeans_clustering(
                keys, layer_idx, sparse_meta_data
            )

            # Check that codebook only has codes for keys after init_offset
            n_quantized_keys = seq_len_keys - init_offset
            assert codebook.shape[1] == n_quantized_keys
            
            # Verify centroids count matches configuration
            cent_cnt = 2 ** config.pq_bits
            n_subvec_per_head = head_dim // config.pq_sub_dim
            assert centroids.shape[3] == cent_cnt  # Check number of centroids

            # Test reconstruction for this init_offset using utility function
            from sparse_attention_hub.sparse_attention.utils.pq_utils import (
                compute_reconstruction_errors,
            )

            original_keys = keys[:, :, init_offset:, :]
            errors = compute_reconstruction_errors(
                original_keys=original_keys,
                centroids=centroids,
                codebook=codebook,
                pq_sub_dim=config.pq_sub_dim,
                use_ip_metric=False,
            )
            
            print(
                f"\nInit Offset={init_offset} Reconstruction Error: "
                f"{errors['relative_error']:.6f}"
            )
            assert errors["relative_error"] < 1.0, (
                f"Relative error {errors['relative_error']} is too large"
            )

    def test_perform_kmeans_varying_centroids(self):
        """Test K-means clustering with varying number of centroids.
        
        This test demonstrates how reconstruction error decreases as the number
        of centroids increases (for both Euclidean and IP metrics).
        """
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        # Fixed parameters
        bsz, num_heads, seq_len_keys, head_dim = 1, 1, 128, 16
        init_offset = 0
        pq_sub_dim = 2
        kmeans_iter = 100  # More iterations for better convergence

        # Create data sample
        torch.manual_seed(42)  # For reproducibility
        keys = torch.randn(bsz, num_heads, seq_len_keys, head_dim, dtype=torch.float32)

        # Vary number of centroids from 2^1 to 2^7 (2 to 128)
        # Since we have 128 points, max centroids is 128
        pq_bits_range = [1,2,3,4,5,6,7]  # 2, 4, 8, 16, 32, 64, 128 centroids
        
        print("\n" + "=" * 80)
        print("Reconstruction Errors vs Number of Centroids")
        print("=" * 80)
        print(f"{'Centroids':<12} {'Euclidean Err':<20} {'IP Err':<20}")
        print("-" * 80)

        for pq_bits in pq_bits_range:
            n_centroids = 2 ** pq_bits
            
            # Test Euclidean metric
            config_euclidean = PQCacheConfig(
                heavy_size=10,
                pq_sub_dim=pq_sub_dim,
                pq_bits=pq_bits,
                kmeans_iter=kmeans_iter,
                init_offset=init_offset,
                metric="euclidean",
            )
            masker_euclidean = PQCache(config_euclidean)
            
            sparse_meta_data_euclidean = {}
            layer_idx = 0
            masker_euclidean._initialize_pq_cache(sparse_meta_data_euclidean, layer_idx)
            
            centroids_euc, codebook_euc = masker_euclidean._perform_kmeans_clustering(
                keys, layer_idx, sparse_meta_data_euclidean
            )
            
            original_keys = keys[:, :, init_offset:, :]
            errors_euc = compute_reconstruction_errors(
                original_keys=original_keys,
                centroids=centroids_euc,
                codebook=codebook_euc,
                pq_sub_dim=pq_sub_dim,
                use_ip_metric=False,
            )
            
            # Test IP metric
            config_ip = PQCacheConfig(
                heavy_size=10,
                pq_sub_dim=pq_sub_dim,
                pq_bits=pq_bits,
                kmeans_iter=kmeans_iter,
                init_offset=init_offset,
                metric="ip",
            )
            masker_ip = PQCache(config_ip)
            
            sparse_meta_data_ip = {}
            masker_ip._initialize_pq_cache(sparse_meta_data_ip, layer_idx)
            
            centroids_ip, codebook_ip = masker_ip._perform_kmeans_clustering(
                keys, layer_idx, sparse_meta_data_ip
            )
            
            errors_ip = compute_reconstruction_errors(
                original_keys=original_keys,
                centroids=centroids_ip,
                codebook=codebook_ip,
                pq_sub_dim=pq_sub_dim,
                use_ip_metric=True,
            )
            
            # Print results
            print(
                f"{n_centroids:<12} "
                f"{errors_euc['relative_error']:<20.6f} "
                f"{errors_ip['relative_error']:<20.6f}"
            )
        
        print("=" * 80)
        
        # Basic sanity checks: error should generally decrease as centroids increase
        # (though not strictly monotonic due to random initialization)
        # Just verify that error with 2 centroids > error with 128 centroids
        config_min = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=pq_sub_dim,
            pq_bits=1,  # 2 centroids
            kmeans_iter=kmeans_iter,
            init_offset=init_offset,
            metric="euclidean",
        )
        masker_min = PQCache(config_min)
        sparse_meta_data_min = {}
        masker_min._initialize_pq_cache(sparse_meta_data_min, 0)
        centroids_min, codebook_min = masker_min._perform_kmeans_clustering(
            keys, 0, sparse_meta_data_min
        )
        errors_min = compute_reconstruction_errors(
            original_keys=keys[:, :, init_offset:, :],
            centroids=centroids_min,
            codebook=codebook_min,
            pq_sub_dim=pq_sub_dim,
            use_ip_metric=False,
        )
        
        config_max = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=pq_sub_dim,
            pq_bits=7,  # 128 centroids
            kmeans_iter=kmeans_iter,
            init_offset=init_offset,
            metric="euclidean",
        )
        masker_max = PQCache(config_max)
        sparse_meta_data_max = {}
        masker_max._initialize_pq_cache(sparse_meta_data_max, 0)
        centroids_max, codebook_max = masker_max._perform_kmeans_clustering(
            keys, 0, sparse_meta_data_max
        )
        errors_max = compute_reconstruction_errors(
            original_keys=keys[:, :, init_offset:, :],
            centroids=centroids_max,
            codebook=codebook_max,
            pq_sub_dim=pq_sub_dim,
            use_ip_metric=False,
        )
        
        # Error with 2 centroids should be higher than with 128 centroids
        assert errors_min["relative_error"] > errors_max["relative_error"], (
            f"Expected error to decrease with more centroids: "
            f"2 centroids={errors_min['relative_error']:.6f}, "
            f"128 centroids={errors_max['relative_error']:.6f}"
        )

    def test_handle_incremental_keys(self):
        """Test handling of incremental keys during decoding."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        # Setup config
        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=8,
            pq_bits=4,  # 16 centroids
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",
        )
        masker = PQCache(config)

        # Create initial keys: [bsz=1, num_heads=2, seq_len=20, head_dim=16]
        bsz = 1
        num_heads = 2
        initial_seq_len = 20
        head_dim = 16
        initial_keys = torch.randn(
            bsz, num_heads, initial_seq_len, head_dim, dtype=torch.float32
        )

        # Initialize and perform initial clustering
        sparse_meta_data = {}
        layer_idx = 0
        masker._initialize_pq_cache(sparse_meta_data, layer_idx)

        initial_centroids, initial_codebook = masker._perform_kmeans_clustering(
            initial_keys, layer_idx, sparse_meta_data
        )

        # Store initial centroids for comparison
        initial_centroids_copy = initial_centroids.clone()
        n_initial_quantized = initial_seq_len - config.init_offset

        # Verify initial setup
        assert initial_codebook.shape[1] == n_initial_quantized
        assert sparse_meta_data["pq_centroids"][layer_idx] is not None
        assert sparse_meta_data["pq_codebook"][layer_idx] is not None

        # Create new keys (simulating incremental decoding - add 5 new tokens)
        n_new_keys = 5
        new_seq_len = initial_seq_len + n_new_keys
        combined_keys = torch.randn(
            bsz, num_heads, new_seq_len, head_dim, dtype=torch.float32
        )
        # Copy initial keys to maintain continuity
        combined_keys[:, :, :initial_seq_len, :] = initial_keys

        # Handle incremental keys
        updated_centroids, updated_codebook = masker._handle_incremental_keys(
            combined_keys, layer_idx, sparse_meta_data
        )

        # Verify centroids remain unchanged (not re-trained)
        assert torch.allclose(
            updated_centroids, initial_centroids_copy, rtol=1e-5, atol=1e-7
        ), "Centroids should not change during incremental update"

        # Verify codebook now includes codes for new keys
        n_total_quantized = new_seq_len - config.init_offset
        assert updated_codebook.shape[1] == n_total_quantized
        assert updated_codebook.shape == (bsz, n_total_quantized, num_heads, 2)

        # Verify codebook values are valid
        cent_cnt = 2 ** config.pq_bits
        assert updated_codebook.min() >= 0
        assert updated_codebook.max() < cent_cnt
        assert updated_codebook.dtype == torch.int64

        # Verify the first part of codebook matches initial codebook
        assert torch.equal(
            updated_codebook[:, :n_initial_quantized, :, :], initial_codebook
        ), "Initial codebook entries should remain unchanged"

        # Verify sparse_meta_data is updated
        assert torch.equal(
            sparse_meta_data["pq_centroids"][layer_idx], updated_centroids
        )
        assert torch.equal(
            sparse_meta_data["pq_codebook"][layer_idx], updated_codebook
        )

        # Test reconstruction for all quantized keys (including new ones)
        original_keys = combined_keys[:, :, config.init_offset :, :]
        errors = compute_reconstruction_errors(
            original_keys=original_keys,
            centroids=updated_centroids,
            codebook=updated_codebook,
            pq_sub_dim=config.pq_sub_dim,
            use_ip_metric=False,
        )

        print(f"\nIncremental Keys Reconstruction Errors:")
        print(f"  Total quantized keys: {n_total_quantized}")
        print(f"  Initial keys: {n_initial_quantized}")
        print(f"  New keys: {n_new_keys}")
        print(f"  MSE Error: {errors['mse_error']:.6f}")
        print(f"  L2 Error: {errors['l2_error']:.6f}")
        print(f"  Relative Error: {errors['relative_error']:.6f}")

        # Sanity check
        assert errors["relative_error"] < 1.0, (
            f"Relative error {errors['relative_error']} is too large"
        )

    def test_quantize_new_keys(self):
        """Test quantizing new keys using existing centroids."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        # Test with both Euclidean and IP metrics
        for metric in ["euclidean", "ip"]:
            print(f"\n{'='*60}")
            print(f"Testing _quantize_new_keys with {metric.upper()} metric")
            print(f"{'='*60}")

            # Setup config
            config = PQCacheConfig(
                heavy_size=10,
                pq_sub_dim=8,
                pq_bits=4,  # 16 centroids
                kmeans_iter=10,
                init_offset=4,
                metric=metric,
            )
            masker = PQCache(config)

            # Create initial keys and perform clustering
            bsz = 1
            num_heads = 2
            initial_seq_len = 20
            head_dim = 16
            initial_keys = torch.randn(
                bsz, num_heads, initial_seq_len, head_dim, dtype=torch.float32
            )

            # Initialize and perform initial clustering
            sparse_meta_data = {}
            layer_idx = 0
            masker._initialize_pq_cache(sparse_meta_data, layer_idx)

            centroids, initial_codebook = masker._perform_kmeans_clustering(
                initial_keys, layer_idx, sparse_meta_data
            )

            # Create new keys to quantize
            n_new_keys = 5
            new_keys = torch.randn(bsz, num_heads, n_new_keys, head_dim, dtype=torch.float32)

            # Quantize the new keys
            new_codes = masker._quantize_new_keys(
                new_keys, centroids, layer_idx, sparse_meta_data
            )

            # Verify output shape
            n_subvec_per_head = head_dim // config.pq_sub_dim  # 16 // 8 = 2
            assert new_codes.shape == (bsz, n_new_keys, num_heads, n_subvec_per_head)
            assert new_codes.shape == (1, 5, 2, 2)

            # Verify codes are valid (within centroid range)
            cent_cnt = 2 ** config.pq_bits  # 16
            assert new_codes.min() >= 0
            assert new_codes.max() < cent_cnt
            assert new_codes.dtype == torch.int64

            # Verify codes are on the same device
            assert new_codes.device == new_keys.device

            # Verify reconstruction quality for the new keys
            errors = compute_reconstruction_errors(
                original_keys=new_keys,
                centroids=centroids,
                codebook=new_codes,
                pq_sub_dim=config.pq_sub_dim,
                use_ip_metric=(metric == "ip"),
            )

            print(f"\nNew Keys Quantization Errors ({metric}):")
            print(f"  Number of new keys: {n_new_keys}")
            print(f"  MSE Error: {errors['mse_error']:.6f}")
            print(f"  L2 Error: {errors['l2_error']:.6f}")
            print(f"  Relative Error: {errors['relative_error']:.6f}")

            # Sanity check: error should be reasonable
            assert errors["relative_error"] < 1.0, (
                f"Relative error {errors['relative_error']} is too large for {metric} metric"
            )

            # Test that quantizing the same keys gives the same codes
            new_codes_repeat = masker._quantize_new_keys(
                new_keys, centroids, layer_idx, sparse_meta_data
            )
            assert torch.equal(new_codes, new_codes_repeat), (
                "Quantizing same keys should produce identical codes"
            )

            # Test with different new keys (should produce different codes)
            different_keys = torch.randn(
                bsz, num_heads, n_new_keys, head_dim, dtype=torch.float32
            )
            different_codes = masker._quantize_new_keys(
                different_keys, centroids, layer_idx, sparse_meta_data
            )
            assert different_codes.shape == new_codes.shape
            # Codes should likely be different (not guaranteed but very likely with random data)
            # Just verify the function runs without error

            print(f"{'='*60}\n")

    def test_compute_pq_scores(self):
        """Test computing PQ scores logic in isolation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        # Setup config
        config = PQCacheConfig(
            heavy_size=10,
            pq_sub_dim=4,  # 4 dimensions per subvector
            pq_bits=2,     # 4 centroids per subvector
            kmeans_iter=10,
            init_offset=4,
            metric="euclidean",  # Doesn't affect scoring logic
        )
        masker = PQCache(config)

        # Construct inputs manually (no clustering required)
        bsz = 1
        num_heads = 4  # Query heads
        kv_heads = 2   # Key/value heads (GQA)
        seq_len_queries = 2
        seq_len_keys = 10
        head_dim = 8  # Will be split into 2 subvectors of 4 dims each
        n_subvec = head_dim // config.pq_sub_dim  # 2
        cent_cnt = 2 ** config.pq_bits  # 4
        n_clustered = 5  # Number of quantized keys

        # Create queries: [bsz, num_heads, seq_len_queries, head_dim]
        queries = torch.randn(bsz, num_heads, seq_len_queries, head_dim, dtype=torch.float32)

        # Create keys (only used for shape reference in the function)
        keys = torch.randn(bsz, kv_heads, seq_len_keys, head_dim, dtype=torch.float32)

        # Manually construct centroids: [bsz, kv_heads, n_subvec, cent_cnt, subvec_d]
        centroids = torch.randn(
            bsz, kv_heads, n_subvec, cent_cnt, config.pq_sub_dim, dtype=torch.float32
        )

        # Manually construct codebook: [bsz, n_clustered, kv_heads, n_subvec]
        # Codebook contains indices [0, cent_cnt-1]
        codebook = torch.randint(0, cent_cnt, (bsz, n_clustered, kv_heads, n_subvec), dtype=torch.int64)

        # Call the function
        pq_scores = masker._compute_pq_scores(
            queries, keys, centroids, codebook
        )

        # Verify output shape: [bsz, num_heads, seq_len_queries, n_clustered]
        assert pq_scores.shape == (bsz, num_heads, seq_len_queries, n_clustered)
        assert pq_scores.shape == (1, 4, 2, 5)

        # Verify device
        assert pq_scores.device == queries.device

        # Verify the scoring logic manually for a single query-key pair
        # Let's verify the first query [0, 0, 0, :] against first key (index 0 in codebook)
        q = queries[0, 0, 0, :]  # [head_dim]
        
        # Split query into subvectors
        q_subvec = q.reshape(n_subvec, config.pq_sub_dim)  # [n_subvec, subvec_d]
        
        # For each subvector, get the centroid index from codebook
        expected_score = 0.0
        for s in range(n_subvec):
            # Get centroid index for key 0, head 0, subvector s
            # Due to GQA, query head 0 maps to kv_head 0
            kv_head = 0
            centroid_idx = codebook[0, 0, kv_head, s].item()
            
            # Get the centroid vector
            centroid_vec = centroids[0, kv_head, s, centroid_idx, :]  # [subvec_d]
            
            # Compute inner product for this subvector
            score_contrib = torch.dot(q_subvec[s], centroid_vec).item()
            expected_score += score_contrib
        
        # Compare with computed score
        computed_score = pq_scores[0, 0, 0, 0].item()
        
        print(f"\nManual verification:")
        print(f"  Expected score (manual): {expected_score:.6f}")
        print(f"  Computed score: {computed_score:.6f}")
        print(f"  Difference: {abs(expected_score - computed_score):.6f}")
        
        # Should be very close (allowing for floating point precision)
        assert torch.isclose(
            pq_scores[0, 0, 0, 0],
            torch.tensor(expected_score, dtype=torch.float32),
            rtol=1e-5,
            atol=1e-6
        ), f"Score mismatch: expected {expected_score}, got {computed_score}"

        # Test GQA: verify that query heads share the same kv_heads
        # Query heads 0,1 should use kv_head 0; query heads 2,3 should use kv_head 1
        num_key_value_groups = num_heads // kv_heads  # 2
        
        # For the same query and key index, heads in the same group should have same score
        # if they have the same query values (which they don't, so just verify shape is correct)
        
        print(f"\nShape verification:")
        print(f"  PQ Scores shape: {pq_scores.shape}")
        print(f"  Expected: ({bsz}, {num_heads}, {seq_len_queries}, {n_clustered})")
        print(f"  ✓ Shape matches!")

        # Test with IP metric (centroids with augmented dimension)
        print(f"\nTesting with augmented centroids (IP metric):")
        
        # Create augmented centroids: [bsz, kv_heads, n_subvec, cent_cnt, subvec_d + 1]
        centroids_aug = torch.randn(
            bsz, kv_heads, n_subvec, cent_cnt, config.pq_sub_dim + 1, dtype=torch.float32
        )
        
        # Call with augmented centroids (function automatically handles the extra dimension)
        pq_scores_aug = masker._compute_pq_scores(
            queries, keys, centroids_aug, codebook
        )
        
        # Should still work and produce same shape
        assert pq_scores_aug.shape == (bsz, num_heads, seq_len_queries, n_clustered)
        print(f"  Augmented centroids shape: {centroids_aug.shape}")
        print(f"  PQ Scores shape: {pq_scores_aug.shape}")
        print(f"  ✓ Correctly handles augmented dimension!")

        print(f"\n✓ All unit test checks passed!")
