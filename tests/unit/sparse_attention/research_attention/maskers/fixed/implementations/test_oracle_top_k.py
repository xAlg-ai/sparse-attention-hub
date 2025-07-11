"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for OracleTopK masker implementation.
"""

import pytest
import torch


@pytest.mark.unit
class TestOracleTopKMaskerImplementation:
    def test_oracle_top_k_masker_config_creation(self):
        """Test that oracle top k masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopKConfig,
        )

        config = OracleTopKConfig(heavy_size=10)
        assert config is not None
        assert config.heavy_size == 10

    def test_oracle_top_k_masker_creation(self):
        """Test that oracle top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        config = OracleTopKConfig(heavy_size=10)
        masker = OracleTopK(config)
        assert type(masker) is OracleTopK
        assert masker.config == config

    def test_oracle_top_k_masker_creation_from_config(self):
        """Test that oracle top k masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        config = OracleTopKConfig(heavy_size=10)
        masker = OracleTopK.create_from_config(config)
        assert type(masker) is OracleTopK
        assert masker.config == config

    def test_oracle_top_k_masker_inheritance(self):
        """Test that oracle top k masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
        )

        assert issubclass(OracleTopK, TopKMasker)

    def test_oracle_top_k_masker_config_inheritance(self):
        """Test that oracle top k masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopKConfig,
        )

        assert issubclass(OracleTopKConfig, TopKMaskerConfig)

    def test_oracle_top_k_masker_add_mask_full_previous(self):
        """Test that OracleTopKMasker returns full mask when previous mask is full."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = OracleTopKConfig(heavy_size=3)
        masker = OracleTopK(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 2, 4, 5, 8

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 64)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 64)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 64)

        # Create full mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        full_previous_mask = Mask.create_full_mask(mask_shape)

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=full_previous_mask,
        )

        assert result.is_full_mask()
        assert result.shape == mask_shape

    def test_oracle_top_k_masker_add_mask_small_sequence(self):
        """Test that OracleTopKMasker returns full mask when seq_len_keys <= heavy_size."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = OracleTopKConfig(heavy_size=8)  # heavy_size >= seq_len_keys
        masker = OracleTopK(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 2, 3, 6

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 32)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 32)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 32)

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=empty_previous_mask,
        )

        assert result.is_full_mask()
        assert result.shape == mask_shape

    def test_oracle_top_k_masker_add_mask_integer_heavy_size(self):
        """Test OracleTopKMasker with integer heavy_size selects top-K from inactive positions."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = OracleTopKConfig(heavy_size=2)
        masker = OracleTopK(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 5

        # Create deterministic inputs for predictable top-K selection
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=empty_previous_mask,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Each query should attend to exactly 2 keys (heavy_size=2)
        for q in range(seq_len_queries):
            num_attended = torch.sum(result_dense[0, 0, q] != 0).item()
            assert (
                num_attended == 2
            ), f"Query {q} should attend to 2 keys, got {num_attended}"

    def test_oracle_top_k_masker_add_mask_float_heavy_size(self):
        """Test OracleTopKMasker with float heavy_size (proportion of seq_len_keys)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = OracleTopKConfig(heavy_size=0.4)  # 0.4 * 5 = 2
        masker = OracleTopK(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 5

        # Create deterministic inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=empty_previous_mask,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Each query should attend to exactly 2 keys (0.4 * 5 = 2)
        for q in range(seq_len_queries):
            num_attended = torch.sum(result_dense[0, 0, q] != 0).item()
            assert (
                num_attended == 2
            ), f"Query {q} should attend to 2 keys, got {num_attended}"

    def test_oracle_top_k_masker_add_mask_avoids_previous_active(self):
        """Test that OracleTopKMasker avoids selecting already active positions."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = OracleTopKConfig(heavy_size=2)
        masker = OracleTopK(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 5

        # Create inputs with high scores for first 2 positions
        keys = torch.zeros(batch_size, num_heads, seq_len_keys, 4)
        queries = torch.zeros(batch_size, num_heads, seq_len_queries, 4)

        # Set up so that positions 0 and 1 would naturally have highest scores
        keys[0, 0, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
        keys[0, 0, 1] = torch.tensor([5.0, 0.0, 0.0, 0.0])
        keys[0, 0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        keys[0, 0, 3] = torch.tensor([0.5, 0.0, 0.0, 0.0])
        keys[0, 0, 4] = torch.tensor([0.1, 0.0, 0.0, 0.0])

        queries[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        queries[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])

        values = torch.randn(batch_size, num_heads, seq_len_keys, 4)

        # Create previous mask with positions 0 and 1 already active
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        previous_mask_data = torch.zeros(mask_shape)
        previous_mask_data[:, :, :, :2] = 1.0  # First 2 positions active
        previous_mask = Mask.create_mask_from_dense_mask(mask_shape, previous_mask_data)

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=previous_mask,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Should have positions 0,1 from previous mask + 2 new positions from top-K
        for q in range(seq_len_queries):
            # Check that positions 0 and 1 are still active (from previous mask)
            assert (
                result_dense[0, 0, q, 0] == 1.0
            ), f"Query {q} should attend to position 0"
            assert (
                result_dense[0, 0, q, 1] == 1.0
            ), f"Query {q} should attend to position 1"

            # Check that exactly 2 additional positions are active (from top-K)
            additional_active = torch.sum(result_dense[0, 0, q, 2:]).item()
            assert (
                additional_active == 2
            ), f"Query {q} should have 2 additional active positions, got {additional_active}"

    def test_oracle_top_k_masker_add_mask_merge_with_previous(self):
        """Test OracleTopKMasker merges correctly with non-empty previous mask."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = OracleTopKConfig(heavy_size=2)
        masker = OracleTopK(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 6

        # Create inputs
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
            sparse_meta_data=None,
            previous_mask=previous_mask,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Should have last 2 positions from previous mask + 2 new positions from top-K
        for q in range(seq_len_queries):
            # Check that last 2 positions are still active (from previous mask)
            assert (
                result_dense[0, 0, q, -2] == 1.0
            ), f"Query {q} should attend to position -2"
            assert (
                result_dense[0, 0, q, -1] == 1.0
            ), f"Query {q} should attend to position -1"

            # Check that exactly 2 additional positions are active (from top-K)
            additional_active = torch.sum(result_dense[0, 0, q, :-2]).item()
            assert (
                additional_active == 2
            ), f"Query {q} should have 2 additional active positions, got {additional_active}"

    def test_oracle_top_k_masker_add_mask_edge_case_heavy_size_zero(self):
        """Test OracleTopKMasker with heavy_size=0."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = OracleTopKConfig(heavy_size=0)
        masker = OracleTopK(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 5

        # Create inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 8)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 8)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 8)

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=empty_previous_mask,
        )

        # Should return empty mask (heavy_size=0 means no new attention)
        assert result.is_empty()

    def test_oracle_top_k_masker_add_mask_edge_case_heavy_size_one(self):
        """Test OracleTopKMasker with heavy_size=1."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = OracleTopKConfig(heavy_size=1)
        masker = OracleTopK(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 3, 5

        # Create inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 8)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 8)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 8)

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=empty_previous_mask,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Each query should attend to exactly 1 key (heavy_size=1)
        for q in range(seq_len_queries):
            num_attended = torch.sum(result_dense[0, 0, q] != 0).item()
            assert (
                num_attended == 1
            ), f"Query {q} should attend to 1 key, got {num_attended}"
