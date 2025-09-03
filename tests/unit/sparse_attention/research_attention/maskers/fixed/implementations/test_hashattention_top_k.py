"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for HashAttentionTopKMasker implementation.
"""

import os
import pickle
import tempfile

import mock
import pytest
import torch


@pytest.fixture
def sample_weights():
    """Create sample weight tensors for HashAttentionTopKMasker testing."""
    return {
        0: {
            "key_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
            "key_bias": [torch.randn(2, 8), torch.randn(2, 4)],
            "query_matrix": [torch.randn(2, 16, 8), torch.randn(2, 8, 4)],
            "query_bias": [torch.randn(2, 8), torch.randn(2, 4)],
        }
    }


@pytest.fixture
def large_sample_weights():
    """Create larger sample weight tensors for HashAttentionTopKMasker testing."""
    return {
        0: {
            "key_matrix": [torch.randn(2, 128, 64), torch.randn(2, 64, 4)],
            "key_bias": [torch.randn(2, 64), torch.randn(2, 4)],
            "query_matrix": [torch.randn(2, 128, 64), torch.randn(2, 64, 4)],
            "query_bias": [torch.randn(2, 64), torch.randn(2, 4)],
        }
    }


@pytest.fixture
def basic_config(sample_weights):
    """Create basic HashAttentionTopKMaskerConfig for testing."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        HashAttentionTopKMaskerConfig,
    )

    return HashAttentionTopKMaskerConfig(
        heavy_size=2,
        hat_bits=4,
        hat_mlp_hidden_size=8,
        hat_mlp_layers=2,
        hat_mlp_activation="relu",
        hat_weights=sample_weights,
    )


@pytest.fixture
def large_config(large_sample_weights):
    """Create HashAttentionTopKMaskerConfig with larger dimensions for testing."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        HashAttentionTopKMaskerConfig,
    )

    return HashAttentionTopKMaskerConfig(
        heavy_size=10,
        hat_bits=4,
        hat_mlp_hidden_size=128,
        hat_mlp_layers=2,
        hat_mlp_activation="relu",
        hat_weights=large_sample_weights,
    )


@pytest.fixture
def float_heavy_size_config(sample_weights):
    """Create HashAttentionTopKMaskerConfig with float heavy_size for testing."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        HashAttentionTopKMaskerConfig,
    )

    return HashAttentionTopKMaskerConfig(
        heavy_size=0.4,
        hat_bits=4,
        hat_mlp_hidden_size=8,
        hat_mlp_layers=2,
        hat_mlp_activation="relu",
        hat_weights=sample_weights,
    )


@pytest.fixture
def large_heavy_size_config(sample_weights):
    """Create HashAttentionTopKMaskerConfig with large heavy_size for testing."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        HashAttentionTopKMaskerConfig,
    )

    return HashAttentionTopKMaskerConfig(
        heavy_size=8,
        hat_bits=4,
        hat_mlp_hidden_size=8,
        hat_mlp_layers=2,
        hat_mlp_activation="relu",
        hat_weights=sample_weights,
    )


@pytest.fixture(params=["relu", "silu", "gelu", "tanh"])
def activation_config(request, sample_weights):
    """Create HashAttentionTopKMaskerConfig with different activation functions."""
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        HashAttentionTopKMaskerConfig,
    )

    return HashAttentionTopKMaskerConfig(
        heavy_size=2,
        hat_bits=4,
        hat_mlp_hidden_size=8,
        hat_mlp_layers=2,
        hat_mlp_activation=request.param,
        hat_weights=sample_weights,
    )


@pytest.fixture
def test_tensors():
    """Create common test tensors for HashAttentionTopKMasker testing."""
    batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 3, 5
    return {
        "keys": torch.randn(batch_size, num_heads, seq_len_keys, 16),
        "queries": torch.randn(batch_size, num_heads, seq_len_queries, 16),
        "values": torch.randn(batch_size, num_heads, seq_len_keys, 16),
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len_queries": seq_len_queries,
        "seq_len_keys": seq_len_keys,
    }


@pytest.fixture
def large_test_tensors():
    """Create larger test tensors for tests requiring more sequence length."""
    batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 3, 6
    return {
        "keys": torch.randn(batch_size, num_heads, seq_len_keys, 16),
        "queries": torch.randn(batch_size, num_heads, seq_len_queries, 16),
        "values": torch.randn(batch_size, num_heads, seq_len_keys, 16),
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len_queries": seq_len_queries,
        "seq_len_keys": seq_len_keys,
    }


@pytest.mark.unit
class TestHashAttentionTopKMaskerManual:
    def test_get_signatures(self, basic_config, test_tensors):
        """Test that get_signatures works."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )

        masker = HashAttentionTopKMasker(basic_config)
        signature = masker._get_signatures(
            input_tensor=test_tensors["keys"],
            matrix_list=basic_config.hat_weights[0]["key_matrix"],
            bias_list=basic_config.hat_weights[0]["key_bias"],
        )
        assert signature is not None
        assert signature.shape == (
            test_tensors["batch_size"],
            test_tensors["num_heads"],
            test_tensors["seq_len_keys"],
            basic_config.hat_bits,
        )

    def test_update_signatures(self, basic_config, test_tensors):
        """Test that update_signatures works with two subsequent calls correctly."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )

        masker = HashAttentionTopKMasker(basic_config)
        # phase 1 only 3 keys are processed
        sparse_meta_data = {}
        with mock.patch(
            "sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.HashAttentionTopKMasker._get_signatures"
        ) as mock_get_signatures:
            input_tensor = test_tensors["keys"][:, :, :3, :]
            mock_get_signatures.return_value = torch.randn(
                test_tensors["batch_size"],
                test_tensors["num_heads"],
                input_tensor.shape[2],
                basic_config.hat_bits,
            )
            partial_signature_1 = masker._update_key_signatures(
                keys=input_tensor,
                sparse_meta_data=sparse_meta_data,
                layer_idx=0,
            )
        assert sparse_meta_data["key"][0].shape[2] == 3

        with mock.patch(
            "sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.HashAttentionTopKMasker._get_signatures"
        ) as mock_get_signatures:
            rest_tensor = test_tensors["keys"][:, :, 3:, :]
            mock_get_signatures.return_value = torch.randn(
                test_tensors["batch_size"],
                test_tensors["num_heads"],
                rest_tensor.shape[2],
                basic_config.hat_bits,
            )
            partial_signature_2 = masker._update_key_signatures(
                keys=test_tensors["keys"],  # passing full tensor
                sparse_meta_data=sparse_meta_data,
                layer_idx=0,
            )

        assert sparse_meta_data["key"][0].shape[2] == test_tensors["seq_len_keys"]
        assert torch.allclose(partial_signature_1, partial_signature_2[:, :, :3, :])

    def test_compute_hashattetion_scores(self, basic_config, test_tensors):
        """Test that compute_hashattetion_scores works."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )

        masker = HashAttentionTopKMasker(basic_config)
        sparse_meta_data = {}
        with mock.patch(
            "sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.HashAttentionTopKMasker._update_key_signatures"
        ) as mock_update_key_signatures:
            with mock.patch(
                "sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.HashAttentionTopKMasker._get_signatures"
            ) as mock_get_signatures:
                key_signatures = torch.randn(
                    test_tensors["batch_size"],
                    test_tensors["num_heads"],
                    test_tensors["seq_len_keys"],
                    basic_config.hat_bits,
                )
                query_signatures = torch.randn(
                    test_tensors["batch_size"],
                    test_tensors["num_heads"],
                    test_tensors["seq_len_queries"],
                    basic_config.hat_bits,
                )
                mock_update_key_signatures.return_value = key_signatures
                mock_get_signatures.return_value = query_signatures
                scores = masker._compute_hashattention_score(
                    keys=test_tensors["keys"],
                    queries=test_tensors["queries"],
                    attention_mask=None,
                    previous_dense_mask=torch.zeros(
                        test_tensors["batch_size"],
                        test_tensors["num_heads"],
                        test_tensors["seq_len_queries"],
                        test_tensors["seq_len_keys"],
                    ),
                    sparse_meta_data=sparse_meta_data,
                    layer_idx=0,
                )
                true_scores = query_signatures @ key_signatures.transpose(2, 3)
        assert torch.allclose(scores, true_scores)


@pytest.mark.unit
class TestHashAttentionTopKMaskerImplementation:
    def test_hash_attention_top_k_masker_config_creation(self, large_config):
        """Test that hash attention top k masker config can be created."""
        assert large_config is not None
        assert large_config.heavy_size == 10
        assert large_config.hat_bits == 4
        assert large_config.hat_mlp_hidden_size == 128
        assert large_config.hat_mlp_layers == 2
        assert large_config.hat_mlp_activation == "relu"
        assert large_config.hat_weights is not None

    def test_hash_attention_top_k_masker_creation(self, large_config):
        """Test that hash attention top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )

        masker = HashAttentionTopKMasker(large_config)
        assert type(masker) is HashAttentionTopKMasker
        assert masker.config == large_config

    def test_hash_attention_top_k_masker_creation_from_config(self, large_config):
        """Test that hash attention top k masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )

        masker = HashAttentionTopKMasker.create_from_config(large_config)
        assert type(masker) is HashAttentionTopKMasker
        assert masker.config == large_config

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

    def test_hash_attention_top_k_masker_add_mask_input_validation(
        self, basic_config, test_tensors
    ):
        """Test HashAttentionTopKMasker add_mask input validation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        masker = HashAttentionTopKMasker(basic_config)

        mask_shape = (
            test_tensors["batch_size"],
            test_tensors["num_heads"],
            test_tensors["seq_len_queries"],
            test_tensors["seq_len_keys"],
        )
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        # Test sparse_meta_data None
        with pytest.raises(ValueError, match="sparse_meta_data cannot be None"):
            masker.add_mask(
                keys=test_tensors["keys"],
                queries=test_tensors["queries"],
                values=test_tensors["values"],
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data=None,
                previous_mask=empty_previous_mask,
            )

        # Test missing layer_idx
        with pytest.raises(ValueError, match="layer_idx must be provided in kwargs"):
            masker.add_mask(
                keys=test_tensors["keys"],
                queries=test_tensors["queries"],
                values=test_tensors["values"],
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                previous_mask=empty_previous_mask,
            )

    def test_hash_attention_top_k_masker_add_mask_full_previous(
        self, basic_config, test_tensors
    ):
        """Test HashAttentionTopKMasker returns previous mask when it's full."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        masker = HashAttentionTopKMasker(basic_config)

        mask_shape = (
            test_tensors["batch_size"],
            test_tensors["num_heads"],
            test_tensors["seq_len_queries"],
            test_tensors["seq_len_keys"],
        )
        full_previous_mask = Mask.create_full_mask(mask_shape)

        result = masker.add_mask(
            keys=test_tensors["keys"],
            queries=test_tensors["queries"],
            values=test_tensors["values"],
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=full_previous_mask,
            layer_idx=0,
        )

        # Should return the same full mask
        assert result.is_full_mask()
        assert result.shape == mask_shape

    def test_hash_attention_top_k_masker_add_mask_small_sequence(
        self, large_heavy_size_config, large_test_tensors
    ):
        """Test HashAttentionTopKMasker returns full mask when seq_len_keys <= heavy_size."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        masker = HashAttentionTopKMasker(large_heavy_size_config)

        mask_shape = (
            large_test_tensors["batch_size"],
            large_test_tensors["num_heads"],
            large_test_tensors["seq_len_queries"],
            large_test_tensors["seq_len_keys"],
        )
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=large_test_tensors["keys"],
            queries=large_test_tensors["queries"],
            values=large_test_tensors["values"],
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        assert result.is_full_mask()
        assert result.shape == mask_shape

    def test_hash_attention_top_k_masker_add_mask_integer_heavy_size(
        self, basic_config, test_tensors
    ):
        """Test HashAttentionTopKMasker with integer heavy_size."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        masker = HashAttentionTopKMasker(basic_config)

        mask_shape = (
            test_tensors["batch_size"],
            test_tensors["num_heads"],
            test_tensors["seq_len_queries"],
            test_tensors["seq_len_keys"],
        )
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=test_tensors["keys"],
            queries=test_tensors["queries"],
            values=test_tensors["values"],
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Each query should attend to exactly 2 keys (heavy_size=2)
        for h in range(test_tensors["num_heads"]):
            for q in range(test_tensors["seq_len_queries"]):
                num_attended = torch.sum(result_dense[0, h, q] != 0).item()
                assert (
                    num_attended == 2
                ), f"Head {h}, Query {q} should attend to 2 keys, got {num_attended}"

    def test_hash_attention_top_k_masker_add_mask_float_heavy_size(
        self, float_heavy_size_config, test_tensors
    ):
        """Test HashAttentionTopKMasker with float heavy_size (proportion of seq_len_keys)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        masker = HashAttentionTopKMasker(float_heavy_size_config)

        mask_shape = (
            test_tensors["batch_size"],
            test_tensors["num_heads"],
            test_tensors["seq_len_queries"],
            test_tensors["seq_len_keys"],
        )
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=test_tensors["keys"],
            queries=test_tensors["queries"],
            values=test_tensors["values"],
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Each query should attend to exactly 2 keys (0.4 * 5 = 2)
        for h in range(test_tensors["num_heads"]):
            for q in range(test_tensors["seq_len_queries"]):
                num_attended = torch.sum(result_dense[0, h, q] != 0).item()
                assert (
                    num_attended == 2
                ), f"Head {h}, Query {q} should attend to 2 keys, got {num_attended}"

    def test_hash_attention_top_k_masker_add_mask_merge_with_previous(
        self, basic_config, large_test_tensors
    ):
        """Test HashAttentionTopKMasker merges correctly with non-empty previous mask."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        masker = HashAttentionTopKMasker(basic_config)

        # Create previous mask with last 2 positions active
        mask_shape = (
            large_test_tensors["batch_size"],
            large_test_tensors["num_heads"],
            large_test_tensors["seq_len_queries"],
            large_test_tensors["seq_len_keys"],
        )
        previous_mask_data = torch.zeros(mask_shape)
        previous_mask_data[:, :, :, -2:] = 1.0  # Last 2 positions
        previous_mask = Mask.create_mask_from_dense_mask(mask_shape, previous_mask_data)

        result = masker.add_mask(
            keys=large_test_tensors["keys"],
            queries=large_test_tensors["queries"],
            values=large_test_tensors["values"],
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Should have last 2 positions from previous mask + 2 new positions from top-K
        for h in range(large_test_tensors["num_heads"]):
            for q in range(large_test_tensors["seq_len_queries"]):
                # Check that last 2 positions are still active (from previous mask)
                assert (
                    result_dense[0, h, q, -2] == 1.0
                ), f"Head {h}, Query {q} should attend to position -2"
                assert (
                    result_dense[0, h, q, -1] == 1.0
                ), f"Head {h}, Query {q} should attend to position -1"

                # Check that exactly 2 additional positions are active (from top-K)
                additional_active = torch.sum(result_dense[0, h, q, :-2]).item()
                assert (
                    additional_active == 2
                ), f"Head {h}, Query {q} should have 2 additional active positions, got {additional_active}"

    def test_hash_attention_top_k_masker_add_mask_signature_caching(
        self, basic_config, test_tensors
    ):
        """Test HashAttentionTopKMasker signature caching behavior."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        masker = HashAttentionTopKMasker(basic_config)

        mask_shape = (
            test_tensors["batch_size"],
            test_tensors["num_heads"],
            test_tensors["seq_len_queries"],
            test_tensors["seq_len_keys"],
        )
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        # First call - should initialize cache
        sparse_meta_data = {}
        result1 = masker.add_mask(
            keys=test_tensors["keys"],
            queries=test_tensors["queries"],
            values=test_tensors["values"],
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Check that only key signatures were cached (query signatures are computed fresh)
        assert "key" in sparse_meta_data
        assert 0 in sparse_meta_data["key"]
        assert sparse_meta_data["key"][0] is not None

        # Query signatures should not be cached
        assert "query" not in sparse_meta_data

        # Check cached key signature shapes
        key_signatures = sparse_meta_data["key"][0]
        assert key_signatures.shape == (
            test_tensors["batch_size"],
            test_tensors["num_heads"],
            test_tensors["seq_len_keys"],
            4,
        )  # hat_bits=4

        # Second call with same inputs - should use cached signatures
        result2 = masker.add_mask(
            keys=test_tensors["keys"],
            queries=test_tensors["queries"],
            values=test_tensors["values"],
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Results should be identical (same inputs, same cache)
        result1_dense = result1.get_dense_mask()
        result2_dense = result2.get_dense_mask()
        assert torch.equal(result1_dense, result2_dense)

    def test_hash_attention_top_k_masker_add_mask_different_activations(
        self, activation_config, test_tensors
    ):
        """Test HashAttentionTopKMasker with different activation functions."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        masker = HashAttentionTopKMasker(activation_config)

        mask_shape = (
            test_tensors["batch_size"],
            test_tensors["num_heads"],
            test_tensors["seq_len_queries"],
            test_tensors["seq_len_keys"],
        )
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        # Should work without errors for all activation functions
        result = masker.add_mask(
            keys=test_tensors["keys"],
            queries=test_tensors["queries"],
            values=test_tensors["values"],
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_previous_mask,
            layer_idx=0,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Each query should attend to exactly 2 keys
        for h in range(test_tensors["num_heads"]):
            for q in range(test_tensors["seq_len_queries"]):
                num_attended = torch.sum(result_dense[0, h, q] != 0).item()
                assert (
                    num_attended == 2
                ), f"Head {h}, Query {q} with {activation_config.hat_mlp_activation} should attend to 2 keys, got {num_attended}"

    def test_hash_attention_top_k_masker_config_with_hat_weight_file(
        self, sample_weights
    ):
        """Test HashAttentionTopKMaskerConfig with hat_weight_file parameter."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMaskerConfig,
        )

        # Create a temporary file with sample weights
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump(sample_weights, f)
            temp_file_path = f.name

        try:
            config = HashAttentionTopKMaskerConfig(
                heavy_size=2,
                hat_bits=4,
                hat_mlp_hidden_size=8,
                hat_mlp_layers=2,
                hat_mlp_activation="relu",
                hat_weight_file=temp_file_path,
            )

            assert config.hat_weight_file == temp_file_path
            assert config.hat_weights is None
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    def test_hash_attention_top_k_masker_creation_with_hat_weight_file(
        self, sample_weights
    ):
        """Test HashAttentionTopKMasker creation with hat_weight_file."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )

        # Create a temporary file with sample weights
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump(sample_weights, f)
            temp_file_path = f.name

        try:
            config = HashAttentionTopKMaskerConfig(
                heavy_size=2,
                hat_bits=4,
                hat_mlp_hidden_size=8,
                hat_mlp_layers=2,
                hat_mlp_activation="relu",
                hat_weight_file=temp_file_path,
            )

            masker = HashAttentionTopKMasker(config)
            assert isinstance(masker, HashAttentionTopKMasker)

            # Compare structure and shapes instead of direct equality
            assert len(masker.hat_weights) == len(sample_weights)
            for layer_idx in sample_weights:
                assert layer_idx in masker.hat_weights
                for key in sample_weights[layer_idx]:
                    assert key in masker.hat_weights[layer_idx]
                    assert len(masker.hat_weights[layer_idx][key]) == len(
                        sample_weights[layer_idx][key]
                    )
                    for i, tensor in enumerate(sample_weights[layer_idx][key]):
                        assert (
                            masker.hat_weights[layer_idx][key][i].shape == tensor.shape
                        )
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    def test_hash_attention_top_k_masker_config_validation_both_provided(
        self, sample_weights
    ):
        """Test that providing both hat_weights and hat_weight_file raises ValueError."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )

        with pytest.raises(
            ValueError,
            match="Only one of hat_weights or hat_weight_file should be provided",
        ):
            config = HashAttentionTopKMaskerConfig(
                heavy_size=2,
                hat_bits=4,
                hat_mlp_hidden_size=8,
                hat_mlp_layers=2,
                hat_mlp_activation="relu",
                hat_weights=sample_weights,
                hat_weight_file="some_file.pkl",
            )
            HashAttentionTopKMasker(config)

    def test_hash_attention_top_k_masker_config_validation_neither_provided(self):
        """Test that providing neither hat_weights nor hat_weight_file raises ValueError."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )

        with pytest.raises(
            ValueError, match="Either hat_weights or hat_weight_file must be provided"
        ):
            config = HashAttentionTopKMaskerConfig(
                heavy_size=2,
                hat_bits=4,
                hat_mlp_hidden_size=8,
                hat_mlp_layers=2,
                hat_mlp_activation="relu",
            )
            HashAttentionTopKMasker(config)
