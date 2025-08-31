"""Tests for AdaptiveSamplingMasker implementation."""

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.adaptive_sampling import (
    AdaptiveSamplingMasker,
    AdaptiveSamplingMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.unit
class TestAdaptiveSamplingMaskerConfig:
    """Test AdaptiveSamplingMaskerConfig validation."""

    def test_valid_float_config(self):
        """Test valid configuration with float base_rate_sampling."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.5,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        assert config.base_rate_sampling == 0.5
        assert config.epsilon == 0.1
        assert config.delta == 0.05
        assert config.sample_without_replacement is False  # Default value

    def test_valid_config_with_sample_without_replacement(self):
        """Test valid configuration with sample_without_replacement enabled."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.3,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
            sample_without_replacement=True,
        )
        assert config.base_rate_sampling == 0.3
        assert config.sample_without_replacement is True

    def test_valid_int_config(self):
        """Test valid configuration with int base_rate_sampling."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=10,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        assert config.base_rate_sampling == 10

    def test_invalid_float_base_rate_sampling(self):
        """Test invalid float base_rate_sampling values."""
        with pytest.raises(
            ValueError, match="base_rate_sampling must be in \\(0, 1\\) if float"
        ):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.0,
                epsilon=0.1,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

        with pytest.raises(
            ValueError, match="base_rate_sampling must be in \\(0, 1\\) if float"
        ):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=1.0,
                epsilon=0.1,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

    def test_invalid_int_base_rate_sampling(self):
        """Test invalid int base_rate_sampling values."""
        with pytest.raises(
            ValueError, match="base_rate_sampling must be positive if int"
        ):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0,
                epsilon=0.1,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

        with pytest.raises(
            ValueError, match="base_rate_sampling must be positive if int"
        ):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=-1,
                epsilon=0.1,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

    def test_invalid_epsilon(self):
        """Test invalid epsilon values."""
        with pytest.raises(ValueError, match="epsilon must be in \\(0, 1\\)"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.0,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

        with pytest.raises(ValueError, match="epsilon must be in \\(0, 1\\)"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=1.0,
                delta=0.05,
                init_offset=0,
                local_offset=0,
            )

    def test_invalid_delta(self):
        """Test invalid delta values."""
        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.1,
                delta=0.0,
                init_offset=0,
                local_offset=0,
            )

        with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.1,
                delta=1.0,
                init_offset=0,
                local_offset=0,
            )

    def test_invalid_offsets(self):
        """Test invalid offset values."""
        with pytest.raises(ValueError, match="init_offset must be non-negative"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.1,
                delta=0.05,
                init_offset=-1,
                local_offset=0,
            )

        with pytest.raises(ValueError, match="local_offset must be non-negative"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.5,
                epsilon=0.1,
                delta=0.05,
                init_offset=0,
                local_offset=-1,
            )


@pytest.mark.unit
class TestAdaptiveSamplingMasker:
    """Test AdaptiveSamplingMasker implementation."""

    @pytest.fixture
    def config(self):
        """Create a valid configuration for testing."""
        return AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )

    @pytest.fixture
    def masker(self, config):
        """Create an AdaptiveSamplingMasker instance."""
        return AdaptiveSamplingMasker(config)

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing."""
        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 32

        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        attention_mask = torch.zeros(
            batch_size, num_heads, seq_len_queries, seq_len_keys
        )

        return keys, queries, values, attention_mask

    def test_init(self, config):
        """Test masker initialization."""
        masker = AdaptiveSamplingMasker(config)
        assert masker.base_rate_sampling == 0.1
        assert masker.epsilon == 0.1
        assert masker.delta == 0.05
        assert masker.init_offset == 0
        assert masker.local_offset == 0
        assert isinstance(masker.delta_ppf, float)
        assert masker.delta_ppf > 0

    def test_compute_exp_attention_scores(self, masker, sample_tensors):
        """Test exponential attention scores computation."""
        keys, queries, _, _ = sample_tensors

        exp_scores = masker._compute_exp_attention_scores(
            queries, keys, scaling=1.0, attention_mask=None
        )

        assert exp_scores.shape == (2, 4, 8, 16)
        assert torch.all(exp_scores >= 0)  # Exponential should be non-negative
        assert torch.all(torch.isfinite(exp_scores))  # Should be finite

    def test_get_sampling_range(self, masker):
        """Test sampling range calculation."""
        seq_len_keys = 16

        start_idx, end_idx, sampling_range = masker._get_sampling_range(seq_len_keys)

        assert start_idx == 0
        assert end_idx == 16
        assert sampling_range == 16

    def test_get_sampling_range_with_offsets(self):
        """Test sampling range with non-zero offsets."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,
            epsilon=0.1,
            delta=0.05,
            init_offset=2,
            local_offset=3,
        )
        masker = AdaptiveSamplingMasker(config)

        start_idx, end_idx, sampling_range = masker._get_sampling_range(16)

        assert start_idx == 2
        assert end_idx == 13
        assert sampling_range == 11

    def test_get_sampling_range_invalid(self):
        """Test invalid sampling range returns full mask."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,
            epsilon=0.1,
            delta=0.05,
            init_offset=10,
            local_offset=10,
        )
        masker = AdaptiveSamplingMasker(config)

        # Test that _get_sampling_range returns a negative sampling range
        start_idx, end_idx, sampling_range = masker._get_sampling_range(16)
        assert sampling_range == -4  # 6 - 10 = -4

        # Test that should_return_full_mask returns True for negative sampling range
        assert masker.should_return_full_mask(sampling_range) is True

    def test_get_base_sample_count_float(self, masker):
        """Test base sample count calculation with float."""
        sampling_range = 1000
        count = masker._get_base_sample_count(sampling_range)
        expected = int(0.1 * 1000)  # 0.1 * 1000 = 100 -> int(100) = 100
        assert count == expected

    def test_get_base_sample_count_int(self):
        """Test base sample count calculation with int."""
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=5,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
        )
        masker = AdaptiveSamplingMasker(config)

        sampling_range = 16
        count = masker._get_base_sample_count(sampling_range)
        assert count == 5

    def test_get_std_estimate_using_base_sample(self, masker, sample_tensors):
        """Test standard deviation estimation using base sampling."""
        batch_size, num_heads, seq_len_queries, seq_len_keys = 2, 4, 8, 1024
        expwts = torch.randn(batch_size, num_heads, seq_len_queries, seq_len_keys)

        start_idx, end_idx = 0, seq_len_keys
        num_base_samples = 5
        dtype = torch.float32

        base_mask, std_estimate, _ = masker._get_std_estimate_using_base_sample(
            expwts,
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
            start_idx,
            end_idx,
            num_base_samples,
            dtype,
        )

        assert isinstance(base_mask, Mask)
        assert base_mask.shape == (batch_size, num_heads, seq_len_queries, seq_len_keys)
        assert std_estimate.shape == (2, 4, 8, 1)
        assert torch.all(std_estimate >= 1e-8)  # Should be clamped to minimum

        dense_mask = base_mask.get_dense_mask()
        dense_mask_2d = dense_mask.view(-1, seq_len_keys)
        std_estimate_2d = std_estimate.view(-1, 1)
        expwts_2d = expwts.view(-1, seq_len_keys)

        for i in range(dense_mask_2d.shape[0]):
            true_std = torch.std(expwts_2d[i][dense_mask_2d[i] > 0])
            achieved_std = std_estimate_2d[i][0]
            # for this to be true repetitions should not happen. so set seq_lent ot large
            # and budget to small
            print(f"row: {i}, true_std: {true_std}, achieved_std: {achieved_std}")
            torch.testing.assert_close(true_std, achieved_std, rtol=0.1, atol=0.05)

    @pytest.mark.parametrize(
        "epsilon, delta", [(0.2, 0.2), (0.25, 0.25), (0.5, 0.5), (0.2, 0.1)]
    )
    def test_compute_adaptive_budget(self, masker, epsilon, delta):
        """Test adaptive budget computation."""
        std_estimate = torch.ones(1, 1)  # 1
        sampling_range = 100000
        data = torch.randn(1, sampling_range)
        static_denominator = 10000
        true_denominator = data.sum(dim=-1, keepdim=True) + static_denominator
        print(
            f"true_denominator: {true_denominator} = {data.sum(dim=-1, keepdim=True)} + {static_denominator}"
        )
        masker = AdaptiveSamplingMasker(
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.1,
                epsilon=epsilon,
                delta=delta,
                init_offset=0,
                local_offset=0,
            )
        )
        # i.e. assuming that data comes from a N(0,1) distribution
        budget = masker._compute_adaptive_budget(
            std_estimate, true_denominator, sampling_range
        )
        budget = int(budget.item())
        num_extreme_values = 0
        total_runs = 1000
        for i in range(total_runs):
            indices = torch.randperm(sampling_range)[:budget]
            data_sampled = data[:, indices]
            estimated_sum = (
                data_sampled.sum(dim=-1) * (sampling_range / budget)
            ).item() + static_denominator
            true_sum = true_denominator.item()
            extreme_value_present = (
                true_sum - estimated_sum
            ) > true_sum * masker.epsilon
            num_extreme_values += float(extreme_value_present)
        empirical_delta = num_extreme_values / total_runs
        print(
            f"budget: {budget}, empirical_delta: {empirical_delta} , masker.delta: {masker.delta}"
        )
        torch.testing.assert_close(empirical_delta, masker.delta, rtol=0.2, atol=0.05)

    def test_add_mask_early_exit(self, masker, sample_tensors):
        """Test early exit when previous mask is full."""
        keys, queries, values, attention_mask = sample_tensors

        # Create a full mask
        full_mask = Mask.create_full_mask((2, 4, 8, 16), dtype=torch.float32)

        result = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=full_mask,
        )

        assert result is full_mask

    def test_add_mask_basic(self, masker, sample_tensors):
        """Test basic add_mask functionality."""
        keys, queries, values, attention_mask = sample_tensors

        # Create an empty mask
        empty_mask = Mask.create_empty_mask((2, 4, 8, 16), dtype=torch.float32)

        result = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_mask,
        )

        assert isinstance(result, Mask)
        assert result.shape == (2, 4, 8, 16)
        assert not result.is_empty()

    def test_create_from_config(self, config):
        """Test create_from_config factory method."""
        masker = AdaptiveSamplingMasker.create_from_config(config)
        assert isinstance(masker, AdaptiveSamplingMasker)
        assert masker.base_rate_sampling == 0.1

    def test_create_from_config_invalid(self):
        """Test create_from_config with invalid config type."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            MaskerConfig,
        )

        invalid_config = MaskerConfig()

        with pytest.raises(ValueError, match="Invalid config type"):
            AdaptiveSamplingMasker.create_from_config(invalid_config)

    def test_device_consistency(self, masker, sample_tensors):
        """Test that all tensors are on the same device."""
        keys, queries, values, attention_mask = sample_tensors

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keys = keys.to(device)
        queries = queries.to(device)
        values = values.to(device)
        attention_mask = attention_mask.to(device)

        empty_mask = Mask.create_empty_mask((2, 4, 8, 16), dtype=torch.float32)

        result = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_mask,
        )

        # Check that result is on the same device
        assert result.get_dense_mask().device == keys.device

    def test_numerical_stability(self, masker, sample_tensors):
        """Test numerical stability with extreme values."""
        keys, queries, values, attention_mask = sample_tensors

        # Use very large values to test numerical stability
        keys = keys * 1000
        queries = queries * 1000

        empty_mask = Mask.create_empty_mask((2, 4, 8, 16), dtype=torch.float32)

        result = masker.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_mask,
        )

        # Should not have NaN or infinite values
        dense_mask = result.get_dense_mask()
        assert torch.all(torch.isfinite(dense_mask))
        assert not torch.any(torch.isnan(dense_mask))


@pytest.mark.unit
class TestGetBaseSamplesWithoutReplacement:
    """Test _get_base_samples_without_replacement method."""

    @pytest.fixture
    def config_no_replacement(self):
        """Create configuration with sample_without_replacement=True."""
        return AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.2,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
            sample_without_replacement=True,
        )

    @pytest.fixture
    def masker_no_replacement(self, config_no_replacement):
        """Create masker with sample_without_replacement=True."""
        return AdaptiveSamplingMasker(config_no_replacement)

    def test_unique_indices_generated(self, masker_no_replacement):
        """Test that generated indices are unique within each row."""
        batch_size, num_heads, seq_len_queries = 2, 3, 4
        start_idx, end_idx = 5, 25  # Sampling range of 20
        num_base_samples = 8
        device = torch.device("cpu")

        result = masker_no_replacement._get_base_samples_without_replacement(
            batch_size, num_heads, seq_len_queries, start_idx, end_idx, num_base_samples, device
        )

        assert result.shape == (batch_size, num_heads, seq_len_queries, num_base_samples)
        
        # Check uniqueness for each row
        for b in range(batch_size):
            for h in range(num_heads):
                for q in range(seq_len_queries):
                    row_indices = result[b, h, q, :]
                    unique_indices = torch.unique(row_indices)
                    assert len(unique_indices) == len(row_indices), f"Duplicates found in row [{b},{h},{q}]"

    def test_indices_within_range(self, masker_no_replacement):
        """Test that all generated indices are within the specified range."""
        batch_size, num_heads, seq_len_queries = 1, 2, 3
        start_idx, end_idx = 10, 30  # Range [10, 30)
        num_base_samples = 5
        device = torch.device("cpu")

        result = masker_no_replacement._get_base_samples_without_replacement(
            batch_size, num_heads, seq_len_queries, start_idx, end_idx, num_base_samples, device
        )

        # All indices should be in range [start_idx, end_idx)
        assert torch.all(result >= start_idx)
        assert torch.all(result < end_idx)

    def test_budget_exceeds_range(self, masker_no_replacement):
        """Test behavior when budget exceeds available sampling range."""
        batch_size, num_heads, seq_len_queries = 1, 1, 1
        start_idx, end_idx = 0, 5  # Small range of 5
        num_base_samples = 10  # Budget exceeds range
        device = torch.device("cpu")

        result = masker_no_replacement._get_base_samples_without_replacement(
            batch_size, num_heads, seq_len_queries, start_idx, end_idx, num_base_samples, device
        )

        # Should return effective_budget = min(num_base_samples, sampling_range) = 5
        expected_samples = min(num_base_samples, end_idx - start_idx)
        assert result.shape[-1] == expected_samples

        # All indices should still be unique
        row_indices = result[0, 0, 0, :]
        unique_indices = torch.unique(row_indices)
        assert len(unique_indices) == len(row_indices)

    def test_different_device(self, masker_no_replacement):
        """Test with different device (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len_queries = 1, 1, 2
        start_idx, end_idx = 0, 20
        num_base_samples = 6
        device = torch.device("cuda")

        result = masker_no_replacement._get_base_samples_without_replacement(
            batch_size, num_heads, seq_len_queries, start_idx, end_idx, num_base_samples, device
        )

        assert result.device.type == device.type
        assert result.shape == (batch_size, num_heads, seq_len_queries, num_base_samples)

    def test_large_tensors(self, masker_no_replacement):
        """Test with larger tensor dimensions."""
        batch_size, num_heads, seq_len_queries = 4, 8, 16
        start_idx, end_idx = 0, 128  # Large sampling range
        num_base_samples = 20
        device = torch.device("cpu")

        result = masker_no_replacement._get_base_samples_without_replacement(
            batch_size, num_heads, seq_len_queries, start_idx, end_idx, num_base_samples, device
        )

        assert result.shape == (batch_size, num_heads, seq_len_queries, num_base_samples)
        
        # Verify uniqueness for a few random rows
        total_rows = batch_size * num_heads * seq_len_queries
        result_flat = result.view(total_rows, num_base_samples)
        
        # Test first and last rows
        for row_idx in [0, total_rows - 1]:
            row_indices = result_flat[row_idx]
            unique_indices = torch.unique(row_indices)
            assert len(unique_indices) == len(row_indices), f"Duplicates in row {row_idx}"

    def test_minimal_sampling_range(self, masker_no_replacement):
        """Test with minimal sampling range (edge case)."""
        batch_size, num_heads, seq_len_queries = 1, 1, 1
        start_idx, end_idx = 0, 2  # Minimal range of 2
        num_base_samples = 2  # Use entire range
        device = torch.device("cpu")

        result = masker_no_replacement._get_base_samples_without_replacement(
            batch_size, num_heads, seq_len_queries, start_idx, end_idx, num_base_samples, device
        )

        assert result.shape[-1] == 2
        row_indices = result[0, 0, 0, :]
        
        # Should contain both available indices (0 and 1)
        sorted_indices = torch.sort(row_indices)[0]
        expected = torch.tensor([0, 1], dtype=torch.long)
        torch.testing.assert_close(sorted_indices, expected)

    def test_integration_with_std_estimation(self, masker_no_replacement):
        """Test integration with standard deviation estimation."""
        batch_size, num_heads, seq_len_queries, seq_len_keys = 2, 2, 4, 32
        expwts = torch.randn(batch_size, num_heads, seq_len_queries, seq_len_keys)
        start_idx, end_idx = 0, seq_len_keys
        num_base_samples = 8
        dtype = torch.float32

        base_mask, std_estimate, effective_samples = masker_no_replacement._get_std_estimate_using_base_sample(
            expwts, batch_size, num_heads, seq_len_queries, seq_len_keys,
            start_idx, end_idx, num_base_samples, dtype
        )

        # Verify return values
        assert isinstance(base_mask, Mask)
        assert base_mask.shape == (batch_size, num_heads, seq_len_queries, seq_len_keys)
        assert std_estimate.shape == (batch_size, num_heads, seq_len_queries, 1)
        assert effective_samples == num_base_samples
        assert torch.all(std_estimate >= 1e-8)  # Should be clamped

        # Verify mask has correct sparsity
        expected_density = num_base_samples / seq_len_keys
        actual_density = base_mask.get_density()
        torch.testing.assert_close(torch.tensor(actual_density), torch.tensor(expected_density), rtol=0.1, atol=0.05)


@pytest.mark.unit
class TestAdaptiveSamplingWithoutReplacement:
    """Test complete adaptive sampling functionality with sample_without_replacement=True."""

    @pytest.fixture
    def config_no_replacement(self):
        """Create configuration with sample_without_replacement=True."""
        return AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.15,
            epsilon=0.2,
            delta=0.2,
            init_offset=0,
            local_offset=0,
            sample_without_replacement=True,
        )

    @pytest.fixture
    def masker_no_replacement(self, config_no_replacement):
        """Create masker with sample_without_replacement=True."""
        return AdaptiveSamplingMasker(config_no_replacement)

    @pytest.fixture
    def large_sample_tensors(self):
        """Create larger sample tensors to avoid edge cases."""
        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 6, 64, 32

        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        attention_mask = torch.zeros(
            batch_size, num_heads, seq_len_queries, seq_len_keys
        )

        return keys, queries, values, attention_mask

    def test_end_to_end_no_replacement(self, masker_no_replacement, large_sample_tensors):
        """Test complete end-to-end functionality with no replacement."""
        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 6, 10240, 32

        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        attention_mask = torch.zeros(
            batch_size, num_heads, seq_len_queries, seq_len_keys
        )

        # Create empty mask
        empty_mask = Mask.create_empty_mask((2, 4, 6, 10240), dtype=torch.float32)

        result = masker_no_replacement.add_mask(
            keys,
            queries,
            values,
            attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=empty_mask,
        )

        assert isinstance(result, Mask)
        assert result.shape == (2, 4, 6, 10240)
        assert not result.is_empty()
        assert not result.is_full_mask()

        # Verify result is valid
        dense_mask = result.get_dense_mask()
        assert torch.all(torch.isfinite(dense_mask))
        assert not torch.any(torch.isnan(dense_mask))
        assert torch.all(dense_mask >= 0)

    def test_comparison_replacement_vs_no_replacement(self, large_sample_tensors):
        """Compare results between replacement and no-replacement modes."""
        keys, queries, values, attention_mask = large_sample_tensors

        # Config with replacement
        config_replacement = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.15,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
            sample_without_replacement=False,
        )
        masker_replacement = AdaptiveSamplingMasker(config_replacement)

        # Config without replacement
        config_no_replacement = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.15,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
            sample_without_replacement=True,
        )
        masker_no_replacement = AdaptiveSamplingMasker(config_no_replacement)

        # Same starting mask
        empty_mask = Mask.create_empty_mask((2, 4, 6, 64), dtype=torch.float32)

        # Test both modes
        result_replacement = masker_replacement.add_mask(
            keys, queries, values, attention_mask,
            scaling=1.0, dropout=0.0, sparse_meta_data={},
            previous_mask=empty_mask,
        )

        result_no_replacement = masker_no_replacement.add_mask(
            keys, queries, values, attention_mask,
            scaling=1.0, dropout=0.0, sparse_meta_data={},
            previous_mask=empty_mask,
        )

        # Both should produce valid results
        assert isinstance(result_replacement, Mask)
        assert isinstance(result_no_replacement, Mask)
        assert result_replacement.shape == result_no_replacement.shape

        # No-replacement might have slightly different density due to no duplicates
        density_replacement = result_replacement.get_density()
        density_no_replacement = result_no_replacement.get_density()
        assert density_no_replacement >= 0.0
        assert density_replacement >= 0.0

    def test_edge_case_small_sequences(self, masker_no_replacement):
        """Test with small sequence lengths that might trigger edge cases."""
        # Very small tensors
        keys = torch.randn(1, 2, 4, 16)
        queries = torch.randn(1, 2, 2, 16)
        values = torch.randn(1, 2, 4, 16)
        attention_mask = torch.zeros(1, 2, 2, 4)

        empty_mask = Mask.create_empty_mask((1, 2, 2, 4), dtype=torch.float32)

        result = masker_no_replacement.add_mask(
            keys, queries, values, attention_mask,
            scaling=1.0, dropout=0.0, sparse_meta_data={},
            previous_mask=empty_mask,
        )

        assert isinstance(result, Mask)
        assert result.shape == (1, 2, 2, 4)
        # Should handle small sequences gracefully

    def test_configuration_validation_with_new_parameter(self):
        """Test that configuration validation works with new parameter."""
        # Valid configuration
        config = AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.2,
            epsilon=0.1,
            delta=0.05,
            init_offset=0,
            local_offset=0,
            sample_without_replacement=True,
        )
        masker = AdaptiveSamplingMasker(config)
        assert masker.sample_without_replacement is True

        # Test that existing validation still works
        with pytest.raises(ValueError, match="epsilon must be in"):
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.2,
                epsilon=1.5,  # Invalid
                delta=0.05,
                init_offset=0,
                local_offset=0,
                sample_without_replacement=True,
            )
