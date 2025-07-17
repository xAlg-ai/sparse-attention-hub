"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for sampling masker implementations. This file is part of the Sparse Attention Hub project.
"""

import pytest
import torch


@pytest.mark.unit
class TestSamplingImplementationsImports:
    def test_implementations_imports(self):
        """Test that all sampling implementation classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        # Test masker classes
        assert RandomSamplingMasker is not None
        assert MagicPig is not None

        # Test config classes
        assert RandomSamplingMaskerConfig is not None
        assert MagicPigConfig is not None


@pytest.mark.unit
class TestRandomSamplingMaskerImplementation:
    def test_random_sampling_masker_config_creation(self):
        """Test that random sampling masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMaskerConfig,
        )

        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        assert config is not None
        assert config.sampling_rate == 0.5

    def test_random_sampling_masker_config_validation_valid_rates(self):
        """Test that random sampling masker config accepts valid sampling rates."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMaskerConfig,
        )

        # Test valid sampling rates
        valid_rates = [0.0, 0.1, 0.5, 0.9, 1.0]
        for rate in valid_rates:
            config = RandomSamplingMaskerConfig(sampling_rate=rate)
            assert config.sampling_rate == rate

    def test_random_sampling_masker_config_validation_invalid_rates(self):
        """Test that random sampling masker config rejects invalid sampling rates."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMaskerConfig,
        )

        # Test invalid sampling rates
        invalid_rates = [-0.1, 1.1, 2.0, -1.0]
        for rate in invalid_rates:
            with pytest.raises(ValueError, match=f"sampling_rate must be in range \\[0,1\\], got {rate}"):
                RandomSamplingMaskerConfig(sampling_rate=rate)

    def test_random_sampling_masker_creation(self):
        """Test that random sampling masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        masker = RandomSamplingMasker(config)
        assert type(masker) is RandomSamplingMasker
        assert masker.config == config
        assert masker.sampling_rate == 0.5

    def test_random_sampling_masker_creation_from_config(self):
        """Test that random sampling masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        masker = RandomSamplingMasker.create_from_config(config)
        assert type(masker) is RandomSamplingMasker
        assert masker.config == config

    def test_random_sampling_masker_creation_from_config_invalid_type(self):
        """Test that random sampling masker creation fails with invalid config type."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMaskerConfig,
        )

        config = SamplingMaskerConfig(sampling_rate=0.5)
        with pytest.raises(ValueError, match="Invalid config type"):
            RandomSamplingMasker.create_from_config(config)

    def test_random_sampling_masker_inheritance(self):
        """Test that random sampling masker inherits from SamplingMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
        )

        assert issubclass(RandomSamplingMasker, SamplingMasker)

    def test_random_sampling_masker_config_inheritance(self):
        """Test that random sampling masker config inherits from SamplingMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMaskerConfig,
        )

        assert issubclass(RandomSamplingMaskerConfig, SamplingMaskerConfig)

    def test_random_sampling_masker_add_mask_full_previous(self):
        """Test that RandomSamplingMasker returns full mask when previous mask is full."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        masker = RandomSamplingMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 2, 4, 5, 8

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 64)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 64)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 64)
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        # Create full mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        full_previous_mask = Mask.create_full_mask(mask_shape)

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=full_previous_mask,
        )

        assert result.is_full_mask()
        assert result.shape == mask_shape

    def test_random_sampling_masker_add_mask_basic_functionality(self):
        """Test that RandomSamplingMasker creates masks with correct sampling rate."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        sampling_rate = 0.25
        config = RandomSamplingMaskerConfig(sampling_rate=sampling_rate)
        masker = RandomSamplingMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 3, 100000

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 32)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 32)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 32)
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
        )

        # Check basic properties
        assert result.shape == mask_shape
        assert not result.is_full_mask()
        assert not result.is_empty()

        # Convert to dense to check sampling rate
        result_dense = result.get_dense_mask()
        
        # Check that approximately sampling_rate fraction of elements are non-zero
        # Allow some tolerance due to randomness
        non_zero_fraction = (result_dense > 0).float().mean(dim = -1).mean().item()
        assert 0.2 <= non_zero_fraction <= 0.3  # Allow 20% tolerance

    def test_random_sampling_masker_add_mask_sampling_rate_zero(self):
        """Test RandomSamplingMasker with sampling_rate=0.0."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = RandomSamplingMaskerConfig(sampling_rate=0.0)
        masker = RandomSamplingMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 5

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
        )

        # With sampling_rate=0.0, no indices should be sampled
        assert result.is_empty()

    def test_random_sampling_masker_add_mask_sampling_rate_one(self):
        """Test RandomSamplingMasker with sampling_rate=1.0."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = RandomSamplingMaskerConfig(sampling_rate=1.0)
        masker = RandomSamplingMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 5

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
        )

        # With sampling_rate=1.0, all indices should be sampled
        result_dense = result.get_dense_mask()
        assert torch.allclose(result_dense, torch.ones_like(result_dense) * 1.0)

    def test_random_sampling_masker_add_mask_merge_with_previous(self):
        """Test that RandomSamplingMasker correctly merges with previous mask."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        masker = RandomSamplingMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 6

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        # Create a previous mask with some positions already active
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        previous_mask_dense = torch.zeros(mask_shape)
        previous_mask_dense[0, 0, 0, 0] = 1.0  # Set one position to active
        previous_mask = Mask.create_mask_from_dense_mask(mask_shape, previous_mask_dense)

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
        )

        # Check that the result is not empty and has the correct shape
        assert result.shape == mask_shape
        assert not result.is_empty()

        # Convert to dense and check that the previous position is still active
        result_dense = result.get_dense_mask()
        assert result_dense[0, 0, 0, 0] > 0

    def test_random_sampling_masker_add_mask_data_values(self):
        """Test that RandomSamplingMasker sets data values to sampling_rate."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        sampling_rate = 0.3
        config = RandomSamplingMaskerConfig(sampling_rate=sampling_rate)
        masker = RandomSamplingMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 8

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
        )

        # Get the sparse representation to check data values
        indices, ptr, data = result.get_index_mask()
        
        # Check that all data values are equal to sampling_rate
        if data.numel() > 0:
            assert torch.allclose(data, torch.full_like(data, sampling_rate))

    def test_random_sampling_masker_add_mask_different_devices(self):
        """Test that RandomSamplingMasker works on different devices."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        masker = RandomSamplingMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 4

        # Test on CPU
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16, device="cpu")
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16, device="cpu")
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16, device="cpu")
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool, device="cpu")
        sparse_meta_data = {}

        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(mask_shape, mask_type="index")

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
        )

        assert result.shape == mask_shape
        assert not result.is_empty()

        # Test on CUDA if available
        if torch.cuda.is_available():
            keys = torch.randn(batch_size, num_heads, seq_len_keys, 16, device="cuda")
            queries = torch.randn(batch_size, num_heads, seq_len_queries, 16, device="cuda")
            values = torch.randn(batch_size, num_heads, seq_len_keys, 16, device="cuda")
            attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool, device="cuda")

            result = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=attention_mask,
                sparse_meta_data=sparse_meta_data,
                previous_mask=empty_previous_mask,
            )

            assert result.shape == mask_shape
            assert not result.is_empty()


@pytest.mark.unit
class TestMagicPigImplementation:
    def test_magic_pig_config_creation(self):
        """Test that magic pig config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )

        config = MagicPigConfig(sampling_rate=0.3, lsh_l=4, lsh_k=8)
        assert config is not None
        assert config.sampling_rate == 0.3
        assert config.lsh_l == 4
        assert config.lsh_k == 8

    def test_magic_pig_creation(self):
        """Test that magic pig can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(sampling_rate=0.3, lsh_l=4, lsh_k=8)
        masker = MagicPig(config)
        assert type(masker) is MagicPig
        assert masker.config == config

    def test_magic_pig_creation_from_config(self):
        """Test that magic pig can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(sampling_rate=0.3, lsh_l=4, lsh_k=8)
        masker = MagicPig.create_from_config(config)
        assert type(masker) is MagicPig
        assert masker.config == config

    def test_magic_pig_inheritance(self):
        """Test that magic pig inherits from SamplingMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
        )

        assert issubclass(MagicPig, SamplingMasker)

    def test_magic_pig_config_inheritance(self):
        """Test that magic pig config inherits from SamplingMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )

        assert issubclass(MagicPigConfig, SamplingMaskerConfig)
