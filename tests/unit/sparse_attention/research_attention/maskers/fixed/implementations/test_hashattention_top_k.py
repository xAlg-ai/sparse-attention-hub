"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for HashAttentionTopKMasker implementation.
"""

import pytest
import torch


@pytest.mark.unit
class TestHashAttentionTopKMaskerImplementation:
    def test_hash_attention_top_k_masker_config_creation(self):
        """Test that hash attention top k masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMaskerConfig,
        )
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 128, 64), torch.randn(2, 64, 4)],
                "key_bias": [torch.randn(2, 64), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 128, 64), torch.randn(2, 64, 4)],
                "query_bias": [torch.randn(2, 64), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=10, 
            hat_bits=4, 
            hat_mlp_hidden_size=128, 
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        assert config is not None
        assert config.heavy_size == 10
        assert config.hat_bits == 4
        assert config.hat_mlp_hidden_size == 128
        assert config.hat_mlp_layers == 2
        assert config.hat_mlp_activation == "relu"
        assert config.hat_weights == sample_weights

    def test_hash_attention_top_k_masker_creation(self):
        """Test that hash attention top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 128, 64), torch.randn(2, 64, 4)],
                "key_bias": [torch.randn(2, 64), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 128, 64), torch.randn(2, 64, 4)],
                "query_bias": [torch.randn(2, 64), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=10, 
            hat_bits=4, 
            hat_mlp_hidden_size=128, 
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker(config)
        assert type(masker) is HashAttentionTopKMasker
        assert masker.config == config

    def test_hash_attention_top_k_masker_creation_from_config(self):
        """Test that hash attention top k masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 128, 64), torch.randn(2, 64, 4)],
                "key_bias": [torch.randn(2, 64), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 128, 64), torch.randn(2, 64, 4)],
                "query_bias": [torch.randn(2, 64), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=10, 
            hat_bits=4, 
            hat_mlp_hidden_size=128, 
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker.create_from_config(config)
        assert type(masker) is HashAttentionTopKMasker
        assert masker.config == config

    def test_hash_attention_top_k_masker_inheritance(self):
        """Test that hash attention top k masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )

        assert issubclass(HashAttentionTopKMasker, TopKMasker)

    def test_hash_attention_top_k_masker_config_inheritance(self):
        """Test that hash attention top k masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMaskerConfig,
        )

        assert issubclass(HashAttentionTopKMaskerConfig, TopKMaskerConfig)

    def test_hash_attention_top_k_masker_add_mask_input_validation(self):
        """Test HashAttentionTopKMasker add_mask input validation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=2,
            hat_bits=4,
            hat_mlp_hidden_size=8,
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker(config)

        # Create test inputs
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 3, 5
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        # Test sparse_meta_data None
        with pytest.raises(ValueError, match="sparse_meta_data cannot be None"):
            masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                sparse_meta_data=None,
                previous_mask=empty_previous_mask,
            )

        # Test missing layer_idx
        with pytest.raises(ValueError, match="layer_idx must be provided in kwargs"):
            masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                sparse_meta_data={},
                previous_mask=empty_previous_mask,
            )

    def test_hash_attention_top_k_masker_add_mask_full_previous(self):
        """Test HashAttentionTopKMasker returns previous mask when it's full."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=2,
            hat_bits=4,
            hat_mlp_hidden_size=8,
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker(config)

        # Create test inputs
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 3, 5
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        full_previous_mask = Mask.create_full_mask(mask_shape)

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data={},
            previous_mask=full_previous_mask,
            layer_idx=0,
        )

        # Should return the same full mask
        assert result.is_full_mask()
        assert result.shape == mask_shape

    def test_hash_attention_top_k_masker_add_mask_small_sequence(self):
        """Test HashAttentionTopKMasker returns full mask when seq_len_keys <= heavy_size."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=8,  # heavy_size >= seq_len_keys
            hat_bits=4,
            hat_mlp_hidden_size=8,
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker(config)

        # Create test inputs with small sequence
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 3, 6
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        assert result.is_full_mask()
        assert result.shape == mask_shape

    def test_hash_attention_top_k_masker_add_mask_integer_heavy_size(self):
        """Test HashAttentionTopKMasker with integer heavy_size."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=2,
            hat_bits=4,
            hat_mlp_hidden_size=8,
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker(config)

        # Create test inputs - use num_heads=2 to match weight tensors
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 2, 5
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape
        
        # Each query should attend to exactly 2 keys (heavy_size=2)
        for h in range(num_heads):
            for q in range(seq_len_queries):
                num_attended = torch.sum(result_dense[0, h, q] != 0).item()
                assert num_attended == 2, f"Head {h}, Query {q} should attend to 2 keys, got {num_attended}"

    def test_hash_attention_top_k_masker_add_mask_float_heavy_size(self):
        """Test HashAttentionTopKMasker with float heavy_size (proportion of seq_len_keys)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=0.4,  # 0.4 * 5 = 2
            hat_bits=4,
            hat_mlp_hidden_size=8,
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker(config)

        # Create test inputs - use num_heads=2 to match weight tensors
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 2, 5
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape
        
        # Each query should attend to exactly 2 keys (0.4 * 5 = 2)
        for h in range(num_heads):
            for q in range(seq_len_queries):
                num_attended = torch.sum(result_dense[0, h, q] != 0).item()
                assert num_attended == 2, f"Head {h}, Query {q} should attend to 2 keys, got {num_attended}"

    def test_hash_attention_top_k_masker_add_mask_merge_with_previous(self):
        """Test HashAttentionTopKMasker merges correctly with non-empty previous mask."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=2,
            hat_bits=4,
            hat_mlp_hidden_size=8,
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker(config)

        # Create test inputs - use num_heads=2 to match weight tensors
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 2, 6
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        
        # Create previous mask with last 2 positions active
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        previous_mask_data = torch.zeros(mask_shape)
        previous_mask_data[:, :, :, -2:] = 1.0  # Last 2 positions
        previous_mask = Mask.create_mask_from_dense_mask(mask_shape, previous_mask_data)

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data={},
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape
        
        # Should have last 2 positions from previous mask + 2 new positions from top-K
        for h in range(num_heads):
            for q in range(seq_len_queries):
                # Check that last 2 positions are still active (from previous mask)
                assert result_dense[0, h, q, -2] == 1.0, f"Head {h}, Query {q} should attend to position -2"
                assert result_dense[0, h, q, -1] == 1.0, f"Head {h}, Query {q} should attend to position -1"
                
                # Check that exactly 2 additional positions are active (from top-K)
                additional_active = torch.sum(result_dense[0, h, q, :-2]).item()
                assert additional_active == 2, f"Head {h}, Query {q} should have 2 additional active positions, got {additional_active}"

    def test_hash_attention_top_k_masker_add_mask_signature_caching(self):
        """Test HashAttentionTopKMasker signature caching behavior."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        import torch

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=2,
            hat_bits=4,
            hat_mlp_hidden_size=8,
            hat_mlp_layers=2,
            hat_mlp_activation="relu",
            hat_weights=sample_weights
        )
        masker = HashAttentionTopKMasker(config)

        # Create test inputs
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 2, 4
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        # First call - should initialize cache
        sparse_meta_data = {}
        result1 = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Check that signatures were cached
        assert "key" in sparse_meta_data
        assert "query" in sparse_meta_data
        assert 0 in sparse_meta_data["key"]
        assert 0 in sparse_meta_data["query"]
        assert sparse_meta_data["key"][0] is not None
        assert sparse_meta_data["query"][0] is not None
        
        # Check cached signature shapes
        key_signatures = sparse_meta_data["key"][0]
        query_signatures = sparse_meta_data["query"][0]
        assert key_signatures.shape == (batch_size, num_heads, seq_len_keys, 4)  # hat_bits=4
        assert query_signatures.shape == (batch_size, num_heads, seq_len_queries, 4)  # hat_bits=4

        # Second call with same inputs - should use cached signatures
        result2 = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Results should be identical (same inputs, same cache)
        result1_dense = result1.get_dense_mask()
        result2_dense = result2.get_dense_mask()
        assert torch.equal(result1_dense, result2_dense)

    def test_hash_attention_top_k_masker_add_mask_different_activations(self):
        """Test HashAttentionTopKMasker with different activation functions."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask
        import torch

        activations = ["relu", "silu", "gelu", "tanh"]
        
        for activation in activations:
            # Create sample weight tensors
            sample_weights = {
                0: {
                    "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                    "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                    "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
                    "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
                }
            }

            config = HashAttentionTopKMaskerConfig(
                heavy_size=2,
                hat_bits=4,
                hat_mlp_hidden_size=8,
                hat_mlp_layers=2,
                hat_mlp_activation=activation,
                hat_weights=sample_weights
            )
            masker = HashAttentionTopKMasker(config)

            # Create test inputs - use num_heads=2 to match weight tensors
            batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 2, 5
            keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
            queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
            values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
            
            mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
            empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

            # Should work without errors for all activation functions
            result = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                sparse_meta_data={},
                previous_mask=empty_previous_mask,
                layer_idx=0,
            )

            # Convert to dense to check pattern
            result_dense = result.get_dense_mask()
            assert result_dense.shape == mask_shape
            
            # Each query should attend to exactly 2 keys
            for h in range(num_heads):
                for q in range(seq_len_queries):
                    num_attended = torch.sum(result_dense[0, h, q] != 0).item()
                    assert num_attended == 2, f"Head {h}, Query {q} with {activation} should attend to 2 keys, got {num_attended}" 