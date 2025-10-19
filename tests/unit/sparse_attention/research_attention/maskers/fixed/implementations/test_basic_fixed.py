"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for basic fixed masker implementations (LocalMasker, CausalMasker, SinkMasker).
"""

import pytest
import torch


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
        full_previous_mask = Mask.create_full_mask(mask_shape, dtype=torch.float32, device=torch.device("cpu"))

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
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
        empty_previous_mask = Mask.create_empty_mask(mask_shape, dtype=torch.float32, device=torch.device("cpu"))

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
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
        empty_previous_mask = Mask.create_empty_mask(mask_shape, dtype=torch.float32, device=torch.device("cpu"))

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
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
        expected_pattern = torch.tensor(
            [
                # Keys: 0  1  2  3  4  5  6  7
                [
                    [0, 0, 0, 0, 1, 1, 0, 0],  # Query 0
                    [0, 0, 0, 0, 0, 1, 1, 0],  # Query 1
                    [0, 0, 0, 0, 0, 0, 1, 1],
                ],  # Query 2
            ],
            dtype=torch.float32,
        )

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
        empty_previous_mask = Mask.create_empty_mask(mask_shape, dtype=torch.float32, device=torch.device("cpu"))

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=None,
            previous_mask=empty_previous_mask,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Should create same pattern as integer window_size=2
        expected_pattern = torch.tensor(
            [
                # Keys: 0  1  2  3  4  5  6  7
                [
                    [0, 0, 0, 0, 1, 1, 0, 0],  # Query 0
                    [0, 0, 0, 0, 0, 1, 1, 0],  # Query 1
                    [0, 0, 0, 0, 0, 0, 1, 1],
                ],  # Query 2
            ],
            dtype=torch.float32,
        )

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
        previous_mask = Mask.create_mask_from_dense_mask(mask_shape, previous_mask_data, dtype=torch.float32)

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=None,
            previous_mask=previous_mask,
        )

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Should merge previous mask (positions 0,1) with local mask (see positions 4-7)
        expected_pattern = torch.tensor(
            [
                # Keys: 0  1  2  3  4  5  6  7
                [
                    [1, 1, 0, 0, 1, 1, 0, 0],  # Query 0: prev[0,1] + local[4,5]
                    [1, 1, 0, 0, 0, 1, 1, 0],  # Query 1: prev[0,1] + local[5,6]
                    [1, 1, 0, 0, 0, 0, 1, 1],
                ],  # Query 2: prev[0,1] + local[6,7]
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(result_dense[0, 0], expected_pattern[0])

    def test_local_masker_add_mask_edge_case_window_size_zero(self):
        """Test LocalMasker with window_size=0 should raise ValueError."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
        )

        with pytest.raises(ValueError, match="window_size must be > 0, got 0"):
            config = LocalMaskerConfig(window_size=0)


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
        full_previous_mask = Mask.create_full_mask(mask_shape, dtype=torch.float32, device=torch.device("cpu"))

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
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
            mask_shape, torch.zeros(mask_shape), dtype=torch.float32
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
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
            mask_shape, partial_mask_data, dtype=torch.float32
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
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
        if (
            seq_len_keys > 6
        ):  # Only check if there are positions between sink and last 2
            assert torch.all(result_dense[:, :, :, 4:-2] == 0.0)

        # Test case 4: Edge case with sink_size = 0 should raise error
        with pytest.raises(ValueError, match="sink_size must be > 0, got 0"):
            config = SinkMaskerConfig(sink_size=0)
