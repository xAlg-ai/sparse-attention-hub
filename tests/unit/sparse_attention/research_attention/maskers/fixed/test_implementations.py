"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for masker implementations. This file is part of the Sparse Attention Hub project.
"""

import pytest
import torch


@pytest.mark.unit
class TestImplementationsImports:
    def test_implementations_imports(self):
        """Test that all implementation classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            CausalMasker,
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
            LocalMasker,
            LocalMaskerConfig,
            OracleTopK,
            OracleTopKConfig,
            PQCache,
            PQCacheConfig,
            SinkMasker,
            SinkMaskerConfig,
        )

        # Test masker classes
        assert LocalMasker is not None
        assert CausalMasker is not None
        assert SinkMasker is not None
        assert OracleTopK is not None
        assert PQCache is not None
        assert HashAttentionTopKMasker is not None
        assert DoubleSparsityTopKMasker is not None

        # Test config classes
        assert LocalMaskerConfig is not None
        assert SinkMaskerConfig is not None
        assert OracleTopKConfig is not None
        assert PQCacheConfig is not None
        assert HashAttentionTopKMaskerConfig is not None
        assert DoubleSparsityTopKMaskerConfig is not None


@pytest.mark.unit
class TestLocalMaskerImplementation:
    def test_local_masker_config_creation(self):
        """Test that local masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
        )

        config = LocalMaskerConfig(window_size=10)
        assert config is not None
        assert config.window_size == 10

    def test_local_masker_creation(self):
        """Test that local masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )

        config = LocalMaskerConfig(window_size=10)
        masker = LocalMasker(config)
        assert type(masker) is LocalMasker
        assert masker.config == config

    def test_local_masker_creation_from_config(self):
        """Test that local masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )

        config = LocalMaskerConfig(window_size=10)
        masker = LocalMasker.create_from_config(config)
        assert type(masker) is LocalMasker
        assert masker.config == config

    def test_local_masker_inheritance(self):
        """Test that local masker inherits from FixedMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
        )

        assert issubclass(LocalMasker, FixedMasker)

    def test_local_masker_config_inheritance(self):
        """Test that local masker config inherits from FixedMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
        )

        assert issubclass(LocalMaskerConfig, FixedMaskerConfig)

    def test_local_masker_add_mask_full_previous(self):
        """Test that LocalMasker returns full mask when previous mask is full."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = LocalMaskerConfig(window_size=3)
        masker = LocalMasker(config)
        
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

    def test_local_masker_add_mask_small_sequence(self):
        """Test that LocalMasker returns full mask when seq_len_keys <= window_size + seq_len_queries."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = LocalMaskerConfig(window_size=4)  # 4 + 3 = 7, keys=6 < 7
        masker = LocalMasker(config)
        
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

    def test_local_masker_add_mask_integer_window(self):
        """Test LocalMasker with integer window_size creates correct local pattern."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = LocalMaskerConfig(window_size=2)
        masker = LocalMasker(config)
        
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 3, 8
        
        # Create mock inputs
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
        
        # Expected pattern based on corrected formula:
        # Query 0 → Keys [4, 5] (window_start = 8 - 3 - 2 + 0 + 1 = 4)
        # Query 1 → Keys [5, 6] (window_start = 8 - 3 - 2 + 1 + 1 = 5)  
        # Query 2 → Keys [6, 7] (window_start = 8 - 3 - 2 + 2 + 1 = 6)
        expected_pattern = torch.tensor([
            # Keys: 0  1  2  3  4  5  6  7
            [      [0, 0, 0, 0, 1, 1, 0, 0],  # Query 0
                   [0, 0, 0, 0, 0, 1, 1, 0],  # Query 1
                   [0, 0, 0, 0, 0, 0, 1, 1]], # Query 2
        ], dtype=torch.float32)
        
        assert torch.allclose(result_dense[0, 0], expected_pattern[0])

    def test_local_masker_add_mask_float_window(self):
        """Test LocalMasker with float window_size (proportion of seq_len_keys)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = LocalMaskerConfig(window_size=0.25)  # 0.25 * 8 = 2
        masker = LocalMasker(config)
        
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 3, 8
        
        # Create mock inputs
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
        
        # Should create same pattern as integer window_size=2
        expected_pattern = torch.tensor([
            # Keys: 0  1  2  3  4  5  6  7
            [      [0, 0, 0, 0, 1, 1, 0, 0],  # Query 0
                   [0, 0, 0, 0, 0, 1, 1, 0],  # Query 1
                   [0, 0, 0, 0, 0, 0, 1, 1]], # Query 2
        ], dtype=torch.float32)
        
        assert torch.allclose(result_dense[0, 0], expected_pattern[0])

    def test_local_masker_add_mask_merge_with_previous(self):
        """Test LocalMasker merges correctly with non-empty previous mask."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = LocalMaskerConfig(window_size=2)
        masker = LocalMasker(config)
        
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 3, 8
        
        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        
        # Create a previous mask with some attention on first 2 positions
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        previous_mask_data = torch.zeros(mask_shape)
        previous_mask_data[:, :, :, :2] = 1.0  # First 2 positions
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
        
        # Should merge previous mask (positions 0,1) with local mask (see positions 4-7)
        expected_pattern = torch.tensor([
            # Keys: 0  1  2  3  4  5  6  7
            [      [1, 1, 0, 0, 1, 1, 0, 0],  # Query 0: prev[0,1] + local[4,5]
                   [1, 1, 0, 0, 0, 1, 1, 0],  # Query 1: prev[0,1] + local[5,6]
                   [1, 1, 0, 0, 0, 0, 1, 1]], # Query 2: prev[0,1] + local[6,7]
        ], dtype=torch.float32)
        
        assert torch.allclose(result_dense[0, 0], expected_pattern[0])

    def test_local_masker_add_mask_edge_case_window_size_zero(self):
        """Test LocalMasker with window_size=0."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = LocalMaskerConfig(window_size=0)
        masker = LocalMasker(config)
        
        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 5
        
        # Create mock inputs
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
        
        # Should return empty mask (window_size=0 means no local attention)
        assert result.is_empty()

@pytest.mark.unit
class TestCausalMaskerImplementation:
    def test_causal_masker_config_creation(self):
        """Test that causal masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,  # default config
        )

        config = FixedMaskerConfig()
        assert config is not None

    def test_causal_masker_creation(self):
        """Test that causal masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            CausalMasker,
        )

        config = FixedMaskerConfig()
        masker = CausalMasker(config)
        assert type(masker) is CausalMasker
        assert masker.config == config

    def test_causal_masker_creation_from_config(self):
        """Test that local masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            CausalMasker,
        )

        config = FixedMaskerConfig()
        masker = CausalMasker.create_from_config(config)
        assert type(masker) is CausalMasker
        assert masker.config == config

    def test_causal_masker_inheritance(self):
        """Test that local masker inherits from FixedMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            CausalMasker,
        )

        assert issubclass(CausalMasker, FixedMasker)


@pytest.mark.unit
class TestSinkMaskerImplementation:
    def test_sink_masker_config_creation(self):
        """Test that sink masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMaskerConfig,
        )

        config = SinkMaskerConfig(sink_size=10)
        assert config is not None
        assert config.sink_size == 10

    def test_sink_masker_creation(self):
        """Test that sink masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker,
            SinkMaskerConfig,
        )

        config = SinkMaskerConfig(sink_size=10)
        masker = SinkMasker(config)
        assert type(masker) is SinkMasker
        assert masker.config == config

    def test_sink_masker_creation_from_config(self):
        """Test that sink masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker,
            SinkMaskerConfig,
        )

        config = SinkMaskerConfig(sink_size=10)
        masker = SinkMasker.create_from_config(config)
        assert type(masker) is SinkMasker
        assert masker.config == config

    def test_sink_masker_inheritance(self):
        """Test that sink masker inherits from FixedMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker,
        )

        assert issubclass(SinkMasker, FixedMasker)

    def test_sink_masker_config_inheritance(self):
        """Test that sink masker config inherits from FixedMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMaskerConfig,
        )

        assert issubclass(SinkMaskerConfig, FixedMaskerConfig)

    def test_sink_masker_add_mask(self):
        """Test that sink masker add_mask method works correctly."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker,
            SinkMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        # Test case 1: Previous mask is full - should return full mask
        config = SinkMaskerConfig(sink_size=4)
        masker = SinkMasker(config)
        
        batch_size, num_heads, seq_len_queries, seq_len_keys = 2, 8, 10, 16
        
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
        
        # Test case 2: seq_len_keys <= sink_size - should return full mask
        config = SinkMaskerConfig(sink_size=20)  # larger than seq_len_keys
        masker = SinkMasker(config)
        
        # Create empty mask as previous mask (to test seq_len_keys <= sink_size condition)
        empty_previous_mask = Mask.create_mask_from_dense_mask(
            mask_shape, torch.zeros(mask_shape)
        )
        
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
        
        # Test case 3: Normal case - should create sink mask and merge
        config = SinkMaskerConfig(sink_size=4)
        masker = SinkMasker(config)
        
        # Create a partial mask as previous mask
        partial_mask_data = torch.zeros(mask_shape)
        partial_mask_data[:, :, :, -2:] = 1.0  # Last 2 positions have attention
        partial_previous_mask = Mask.create_mask_from_dense_mask(
            mask_shape, partial_mask_data
        )
        
        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=partial_previous_mask,
        )
        
        # Result should contain sink positions (0, 1, 2, 3) and last 2 positions
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape
        
        # Check that sink positions (first 4 positions) are attended to
        assert torch.all(result_dense[:, :, :, :4] == 1.0)
        
        # Check that last 2 positions are still attended to (from merge)
        assert torch.all(result_dense[:, :, :, -2:] == 1.0)
        
        # Check that positions 4 to -3 are not attended to (should be 0)
        if seq_len_keys > 6:  # Only check if there are positions between sink and last 2
            assert torch.all(result_dense[:, :, :, 4:-2] == 0.0)

        # Test case 4: Edge case with sink_size = 0
        config = SinkMaskerConfig(sink_size=0)
        masker = SinkMasker(config)
        
        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            sparse_meta_data=None,
            previous_mask=partial_previous_mask,
        )
        
        result_dense = result.get_dense_mask()
        
        # Only position 0 should be attended to from sink
        assert torch.all(result_dense[:, :, :, 0] == 0.0)
        # Last 2 positions should still be attended to from merge
        assert torch.all(result_dense[:, :, :, -2:] == 1.0)


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
            assert num_attended == 2, f"Query {q} should attend to 2 keys, got {num_attended}"

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
            assert num_attended == 2, f"Query {q} should attend to 2 keys, got {num_attended}"

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
            assert result_dense[0, 0, q, 0] == 1.0, f"Query {q} should attend to position 0"
            assert result_dense[0, 0, q, 1] == 1.0, f"Query {q} should attend to position 1"
            
            # Check that exactly 2 additional positions are active (from top-K)
            additional_active = torch.sum(result_dense[0, 0, q, 2:]).item()
            assert additional_active == 2, f"Query {q} should have 2 additional active positions, got {additional_active}"

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
            assert result_dense[0, 0, q, -2] == 1.0, f"Query {q} should attend to position -2"
            assert result_dense[0, 0, q, -1] == 1.0, f"Query {q} should attend to position -1"
            
            # Check that exactly 2 additional positions are active (from top-K)
            additional_active = torch.sum(result_dense[0, 0, q, :-2]).item()
            assert additional_active == 2, f"Query {q} should have 2 additional active positions, got {additional_active}"

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
            assert num_attended == 1, f"Query {q} should attend to 1 key, got {num_attended}"

    def test_pq_cache_masker_config_inheritance(self):
        """Test that pq cache masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        assert issubclass(PQCacheConfig, TopKMaskerConfig)


@pytest.mark.unit
class TestPQCacheMaskerImplementation:
    def test_pq_cache_masker_config_creation(self):
        """Test that pq cache masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4)
        assert config is not None
        assert config.heavy_size == 10
        assert config.pq_sub_dim == 8
        assert config.pq_bits == 4

    def test_pq_cache_masker_creation(self):
        """Test that pq cache masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4)
        masker = PQCache(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_creation_from_config(self):
        """Test that pq cache masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4)
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


@pytest.mark.unit
class TestHashAttentionTopKMaskerImplementation:
    def test_hash_attention_top_k_masker_config_creation(self):
        """Test that hash attention top k masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMaskerConfig,
        )

        config = HashAttentionTopKMaskerConfig(
            heavy_size=10, hat_bits=4, hat_mlp_hidden_size=128, hat_mlp_layers=2
        )
        assert config is not None
        assert config.heavy_size == 10
        assert config.hat_bits == 4
        assert config.hat_mlp_hidden_size == 128
        assert config.hat_mlp_layers == 2

    def test_hash_attention_top_k_masker_creation(self):
        """Test that hash attention top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )

        config = HashAttentionTopKMaskerConfig(
            heavy_size=10, hat_bits=4, hat_mlp_hidden_size=128, hat_mlp_layers=2
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

        config = HashAttentionTopKMaskerConfig(
            heavy_size=10, hat_bits=4, hat_mlp_hidden_size=128, hat_mlp_layers=2
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


@pytest.mark.unit
class TestDoubleSparsityTopKMaskerImplementation:
    def test_double_sparsity_top_k_masker_config_creation(self):
        """Test that double sparsity top k masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMaskerConfig,
        )

        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=10, group_factor=4, label_bits=4, channel_config="auto"
        )
        assert config is not None
        assert config.heavy_size == 10
        assert config.group_factor == 4
        assert config.label_bits == 4
        assert config.channel_config == "auto"

    def test_double_sparsity_top_k_masker_creation(self):
        """Test that double sparsity top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
        )

        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=10, group_factor=4, label_bits=4, channel_config="auto"
        )
        masker = DoubleSparsityTopKMasker(config)
        assert type(masker) is DoubleSparsityTopKMasker
        assert masker.config == config

    def test_double_sparsity_top_k_masker_creation_from_config(self):
        """Test that double sparsity top k masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
        )

        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=10, group_factor=4, label_bits=4, channel_config="auto"
        )
        masker = DoubleSparsityTopKMasker.create_from_config(config)
        assert type(masker) is DoubleSparsityTopKMasker
        assert masker.config == config

    def test_double_sparsity_top_k_masker_inheritance(self):
        """Test that double sparsity top k masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker,
        )

        assert issubclass(DoubleSparsityTopKMasker, TopKMasker)

    def test_double_sparsity_top_k_masker_config_inheritance(self):
        """Test that double sparsity top k masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMaskerConfig,
        )

        assert issubclass(DoubleSparsityTopKMaskerConfig, TopKMaskerConfig)
