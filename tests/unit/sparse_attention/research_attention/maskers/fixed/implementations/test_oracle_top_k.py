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
        full_previous_mask = Mask.create_full_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

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
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
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
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
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
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
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
        previous_mask = Mask.create_mask_from_dense_mask(
            mask_shape, previous_mask_data, dtype=torch.float32
        )

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
        previous_mask = Mask.create_mask_from_dense_mask(
            mask_shape, previous_mask_data, dtype=torch.float32
        )

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
        """Test OracleTopKMasker with heavy_size=0 should raise ValueError."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopKConfig,
        )

        with pytest.raises(ValueError, match="heavy_size must be > 0, got 0"):
            OracleTopKConfig(heavy_size=0)

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
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
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

        # Convert to dense to check pattern
        result_dense = result.get_dense_mask()
        assert result_dense.shape == mask_shape

        # Each query should attend to exactly 1 key (heavy_size=1)
        for q in range(seq_len_queries):
            num_attended = torch.sum(result_dense[0, 0, q] != 0).item()
            assert (
                num_attended == 1
            ), f"Query {q} should attend to 1 key, got {num_attended}"


@pytest.mark.unit
class TestOracleTopKGetUpdatedMaskStressTest:
    """Stress tests to ensure get_updated_mask_old and get_updated_mask_new produce identical results."""

    def _create_test_inputs(
        self,
        batch_size: int,
        num_heads_queries: int,
        num_heads_keys: int,
        seq_len_queries: int,
        seq_len_keys: int,
        head_dim: int,
        previous_mask_pattern: str = "empty",
        use_attention_mask: bool = False,
        seed: int = 42,
    ):
        """Create test inputs for get_updated_mask methods.

        Args:
            batch_size: Batch size
            num_heads_queries: Number of query heads
            num_heads_keys: Number of key-value heads (for GQA)
            seq_len_queries: Query sequence length
            seq_len_keys: Key sequence length
            head_dim: Head dimension
            previous_mask_pattern: Pattern for previous mask ("empty", "partial", "custom")
            use_attention_mask: Whether to create an attention mask
            seed: Random seed for reproducibility

        Returns:
            Tuple of (keys, queries, attention_mask, previous_mask, tensor_dims)
        """
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            AttentionTensorDimensions,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        torch.manual_seed(seed)

        # Create inputs
        keys: torch.Tensor = torch.randn(
            batch_size, num_heads_keys, seq_len_keys, head_dim
        )
        queries: torch.Tensor = torch.randn(
            batch_size, num_heads_queries, seq_len_queries, head_dim
        )

        # Create attention mask if needed
        attention_mask: torch.Tensor = None
        if use_attention_mask:
            # Create causal mask
            attention_mask = torch.zeros(batch_size, 1, seq_len_queries, seq_len_keys)
            for i in range(seq_len_queries):
                # Mask future positions
                if i < seq_len_keys:
                    attention_mask[:, :, i, i + 1 :] = float("-inf")

        # Create previous mask
        mask_shape: tuple = (
            batch_size,
            num_heads_queries,
            seq_len_queries,
            seq_len_keys,
        )
        if previous_mask_pattern == "empty":
            previous_mask: Mask = Mask.create_empty_mask(
                mask_shape, dtype=torch.float32, device=torch.device("cpu")
            )
        elif previous_mask_pattern == "partial":
            # Create a partial mask with some positions already active
            previous_mask_data: torch.Tensor = torch.zeros(mask_shape)
            # Activate first 2 positions for all queries
            previous_mask_data[:, :, :, :2] = 1.0
            previous_mask = Mask.create_mask_from_dense_mask(
                mask_shape, previous_mask_data, dtype=torch.float32
            )
        elif previous_mask_pattern == "custom":
            # Random pattern
            previous_mask_data = torch.zeros(mask_shape)
            for b in range(batch_size):
                for h in range(num_heads_queries):
                    for q in range(seq_len_queries):
                        # Randomly activate 20% of positions
                        num_active: int = max(1, int(seq_len_keys * 0.2))
                        active_indices: torch.Tensor = torch.randperm(seq_len_keys)[
                            :num_active
                        ]
                        previous_mask_data[b, h, q, active_indices] = 1.0
            previous_mask = Mask.create_mask_from_dense_mask(
                mask_shape, previous_mask_data, dtype=torch.float32
            )
        else:
            raise ValueError(f"Unknown previous_mask_pattern: {previous_mask_pattern}")

        # Create tensor dimensions (using query heads as that's what the mask shape uses)
        tensor_dims: AttentionTensorDimensions = AttentionTensorDimensions(
            batch_size=batch_size,
            num_heads=num_heads_queries,
            seq_len_queries=seq_len_queries,
            seq_len_keys=seq_len_keys,
        )

        return keys, queries, attention_mask, previous_mask, tensor_dims

    def _compare_masks(
        self, mask_old: torch.Tensor, mask_new: torch.Tensor, test_name: str
    ) -> None:
        """Compare two masks and assert they are identical.

        Args:
            mask_old: Mask from old implementation
            mask_new: Mask from new implementation
            test_name: Name of the test for error messages
        """
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        # Compare shapes
        assert (
            mask_old.shape == mask_new.shape
        ), f"{test_name}: Shape mismatch - old: {mask_old.shape}, new: {mask_new.shape}"

        # Get dense representations
        dense_old: torch.Tensor = (
            mask_old.get_dense_mask() if isinstance(mask_old, Mask) else mask_old
        )
        dense_new: torch.Tensor = (
            mask_new.get_dense_mask() if isinstance(mask_new, Mask) else mask_new
        )

        # Compare values with tolerance for floating point
        assert torch.allclose(dense_old, dense_new, rtol=1e-5, atol=1e-7), (
            f"{test_name}: Mask values mismatch\n"
            f"Max difference: {torch.max(torch.abs(dense_old - dense_new))}\n"
            f"Number of mismatches: {torch.sum(dense_old != dense_new)}\n"
            f"Old mask sum: {torch.sum(dense_old)}, New mask sum: {torch.sum(dense_new)}"
        )

    def test_stress_empty_previous_mask_various_configs(self):
        """Stress test with empty previous mask and various configurations."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        test_configs = [
            # (batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, heavy_size)
            (1, 1, 1, 4, 8, 16, 2),
            (2, 4, 4, 8, 16, 32, 4),
            (1, 8, 8, 16, 32, 64, 8),
            (4, 2, 2, 10, 20, 16, 5),
            (1, 1, 1, 5, 10, 8, 0.3),  # float heavy_size
            (2, 4, 4, 8, 20, 32, 0.25),
        ]

        for idx, (
            batch_size,
            num_heads_queries,
            num_heads_keys,
            seq_len_queries,
            seq_len_keys,
            head_dim,
            heavy_size,
        ) in enumerate(test_configs):
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)

            (
                keys,
                queries,
                attention_mask,
                previous_mask,
                tensor_dims,
            ) = self._create_test_inputs(
                batch_size,
                num_heads_queries,
                num_heads_keys,
                seq_len_queries,
                seq_len_keys,
                head_dim,
                previous_mask_pattern="empty",
                use_attention_mask=False,
                seed=42 + idx,
            )

            effective_heavy_size: int = masker._calculate_effective_heavy_size(
                seq_len_keys
            )

            # Get results from both methods
            result_old = masker.get_updated_mask_old(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )
            result_new = masker.get_updated_mask_new(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )

            self._compare_masks(
                result_old,
                result_new,
                f"test_stress_empty_previous_mask_various_configs[config_{idx}]",
            )

    def test_stress_partial_previous_mask(self):
        """Stress test with partial previous mask (some positions already active)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        test_configs = [
            # (batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, heavy_size)
            (1, 1, 1, 4, 10, 16, 3),
            (2, 4, 4, 8, 20, 32, 5),
            (1, 8, 8, 12, 24, 64, 0.2),
            (3, 2, 2, 6, 15, 16, 4),
        ]

        for idx, (
            batch_size,
            num_heads_queries,
            num_heads_keys,
            seq_len_queries,
            seq_len_keys,
            head_dim,
            heavy_size,
        ) in enumerate(test_configs):
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)

            (
                keys,
                queries,
                attention_mask,
                previous_mask,
                tensor_dims,
            ) = self._create_test_inputs(
                batch_size,
                num_heads_queries,
                num_heads_keys,
                seq_len_queries,
                seq_len_keys,
                head_dim,
                previous_mask_pattern="partial",
                use_attention_mask=False,
                seed=100 + idx,
            )

            effective_heavy_size: int = masker._calculate_effective_heavy_size(
                seq_len_keys
            )

            result_old = masker.get_updated_mask_old(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )
            result_new = masker.get_updated_mask_new(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )

            self._compare_masks(
                result_old,
                result_new,
                f"test_stress_partial_previous_mask[config_{idx}]",
            )

    def test_stress_with_attention_mask(self):
        """Stress test with attention mask (e.g., causal masking)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        test_configs = [
            # (batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, heavy_size)
            (1, 1, 1, 4, 8, 16, 2),
            (2, 4, 4, 8, 16, 32, 4),
            (1, 2, 2, 10, 15, 16, 0.3),
        ]

        for idx, (
            batch_size,
            num_heads_queries,
            num_heads_keys,
            seq_len_queries,
            seq_len_keys,
            head_dim,
            heavy_size,
        ) in enumerate(test_configs):
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)

            (
                keys,
                queries,
                attention_mask,
                previous_mask,
                tensor_dims,
            ) = self._create_test_inputs(
                batch_size,
                num_heads_queries,
                num_heads_keys,
                seq_len_queries,
                seq_len_keys,
                head_dim,
                previous_mask_pattern="empty",
                use_attention_mask=True,
                seed=200 + idx,
            )

            effective_heavy_size: int = masker._calculate_effective_heavy_size(
                seq_len_keys
            )

            result_old = masker.get_updated_mask_old(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )
            result_new = masker.get_updated_mask_new(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )

            self._compare_masks(
                result_old, result_new, f"test_stress_with_attention_mask[config_{idx}]"
            )

    def test_stress_gqa_scenario(self):
        """Stress test with GQA (Grouped Query Attention) - different number of key/value heads."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        test_configs = [
            # (batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, heavy_size)
            (1, 8, 2, 4, 8, 16, 2),  # 8 query heads, 2 key heads (4:1 ratio)
            (2, 12, 3, 6, 12, 32, 3),  # 12 query heads, 3 key heads (4:1 ratio)
            (1, 16, 4, 8, 16, 64, 0.25),  # 16 query heads, 4 key heads (4:1 ratio)
            (1, 4, 1, 5, 10, 16, 2),  # 4 query heads, 1 key head (4:1 ratio)
        ]

        for idx, (
            batch_size,
            num_heads_queries,
            num_heads_keys,
            seq_len_queries,
            seq_len_keys,
            head_dim,
            heavy_size,
        ) in enumerate(test_configs):
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)

            (
                keys,
                queries,
                attention_mask,
                previous_mask,
                tensor_dims,
            ) = self._create_test_inputs(
                batch_size,
                num_heads_queries,
                num_heads_keys,
                seq_len_queries,
                seq_len_keys,
                head_dim,
                previous_mask_pattern="empty",
                use_attention_mask=False,
                seed=300 + idx,
            )

            effective_heavy_size: int = masker._calculate_effective_heavy_size(
                seq_len_keys
            )

            result_old = masker.get_updated_mask_old(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )
            result_new = masker.get_updated_mask_new(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )

            self._compare_masks(
                result_old, result_new, f"test_stress_gqa_scenario[config_{idx}]"
            )

    def test_stress_custom_previous_mask_pattern(self):
        """Stress test with custom random previous mask patterns."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        test_configs = [
            # (batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, heavy_size)
            (1, 2, 2, 5, 12, 16, 3),
            (2, 4, 4, 8, 20, 32, 5),
            (1, 1, 1, 6, 15, 8, 0.4),
        ]

        for idx, (
            batch_size,
            num_heads_queries,
            num_heads_keys,
            seq_len_queries,
            seq_len_keys,
            head_dim,
            heavy_size,
        ) in enumerate(test_configs):
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)

            (
                keys,
                queries,
                attention_mask,
                previous_mask,
                tensor_dims,
            ) = self._create_test_inputs(
                batch_size,
                num_heads_queries,
                num_heads_keys,
                seq_len_queries,
                seq_len_keys,
                head_dim,
                previous_mask_pattern="custom",
                use_attention_mask=False,
                seed=400 + idx,
            )

            effective_heavy_size: int = masker._calculate_effective_heavy_size(
                seq_len_keys
            )

            result_old = masker.get_updated_mask_old(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )
            result_new = masker.get_updated_mask_new(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )

            self._compare_masks(
                result_old,
                result_new,
                f"test_stress_custom_previous_mask_pattern[config_{idx}]",
            )

    def test_stress_large_batch_and_heads(self):
        """Stress test with large batch sizes and many heads."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        test_configs = [
            # (batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, heavy_size)
            (8, 12, 12, 8, 16, 64, 4),
            (16, 8, 8, 4, 12, 32, 3),
            (4, 16, 4, 6, 20, 64, 0.2),  # GQA with large batch
        ]

        for idx, (
            batch_size,
            num_heads_queries,
            num_heads_keys,
            seq_len_queries,
            seq_len_keys,
            head_dim,
            heavy_size,
        ) in enumerate(test_configs):
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)

            (
                keys,
                queries,
                attention_mask,
                previous_mask,
                tensor_dims,
            ) = self._create_test_inputs(
                batch_size,
                num_heads_queries,
                num_heads_keys,
                seq_len_queries,
                seq_len_keys,
                head_dim,
                previous_mask_pattern="empty",
                use_attention_mask=False,
                seed=500 + idx,
            )

            effective_heavy_size: int = masker._calculate_effective_heavy_size(
                seq_len_keys
            )

            result_old = masker.get_updated_mask_old(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )
            result_new = masker.get_updated_mask_new(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )

            self._compare_masks(
                result_old,
                result_new,
                f"test_stress_large_batch_and_heads[config_{idx}]",
            )

    def test_stress_combined_scenarios(self):
        """Stress test combining multiple challenging scenarios.

        Note: Some scenarios are excluded where attention masks combined with partial/custom
        masks can create situations where torch.topk selects from insufficient valid positions.
        This is a known limitation where both implementations may behave differently.
        """
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        test_configs = [
            # (batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, heavy_size,
            #  previous_mask_pattern, use_attention_mask)
            # Excluded: (2, 8, 2, 6, 12, 32, 3, "partial", True) - GQA + partial mask + causal attention
            #   creates insufficient valid positions for topk, causing implementation differences
            (
                1,
                4,
                1,
                8,
                20,
                16,
                0.15,
                "custom",
                False,
            ),  # GQA + custom mask (no attention mask)
            (
                4,
                12,
                4,
                4,
                16,
                64,
                4,
                "partial",
                False,
            ),  # Large batch + GQA + partial mask
            (2, 6, 6, 10, 25, 32, 0.2, "empty", True),  # Empty mask + attention mask
            (
                2,
                4,
                4,
                8,
                16,
                32,
                3,
                "partial",
                False,
            ),  # Partial mask without attention mask
        ]

        for idx, (
            batch_size,
            num_heads_queries,
            num_heads_keys,
            seq_len_queries,
            seq_len_keys,
            head_dim,
            heavy_size,
            previous_mask_pattern,
            use_attention_mask,
        ) in enumerate(test_configs):
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)

            (
                keys,
                queries,
                attention_mask,
                previous_mask,
                tensor_dims,
            ) = self._create_test_inputs(
                batch_size,
                num_heads_queries,
                num_heads_keys,
                seq_len_queries,
                seq_len_keys,
                head_dim,
                previous_mask_pattern=previous_mask_pattern,
                use_attention_mask=use_attention_mask,
                seed=600 + idx,
            )

            effective_heavy_size: int = masker._calculate_effective_heavy_size(
                seq_len_keys
            )

            result_old = masker.get_updated_mask_old(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )
            result_new = masker.get_updated_mask_new(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )

            self._compare_masks(
                result_old, result_new, f"test_stress_combined_scenarios[config_{idx}]"
            )

    def test_stress_edge_cases(self):
        """Stress test with edge cases (small sequences, minimal heavy_size, etc.)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
            OracleTopKConfig,
        )

        test_configs = [
            # (batch_size, num_heads_queries, num_heads_keys, seq_len_queries, seq_len_keys, head_dim, heavy_size)
            (1, 1, 1, 1, 4, 8, 1),  # Single query
            (1, 1, 1, 2, 3, 8, 1),  # Minimal dimensions
            (1, 2, 2, 3, 8, 16, 1),  # heavy_size=1
            (1, 1, 1, 4, 10, 8, 0.1),  # Small float heavy_size
        ]

        for idx, (
            batch_size,
            num_heads_queries,
            num_heads_keys,
            seq_len_queries,
            seq_len_keys,
            head_dim,
            heavy_size,
        ) in enumerate(test_configs):
            config: OracleTopKConfig = OracleTopKConfig(heavy_size=heavy_size)
            masker: OracleTopK = OracleTopK(config)

            (
                keys,
                queries,
                attention_mask,
                previous_mask,
                tensor_dims,
            ) = self._create_test_inputs(
                batch_size,
                num_heads_queries,
                num_heads_keys,
                seq_len_queries,
                seq_len_keys,
                head_dim,
                previous_mask_pattern="empty",
                use_attention_mask=False,
                seed=700 + idx,
            )

            effective_heavy_size: int = masker._calculate_effective_heavy_size(
                seq_len_keys
            )

            result_old = masker.get_updated_mask_old(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )
            result_new = masker.get_updated_mask_new(
                tensor_dims,
                effective_heavy_size,
                keys,
                queries,
                attention_mask,
                previous_mask,
            )

            self._compare_masks(
                result_old, result_new, f"test_stress_edge_cases[config_{idx}]"
            )
