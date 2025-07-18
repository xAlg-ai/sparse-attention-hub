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
            with pytest.raises(
                ValueError,
                match=f"sampling_rate must be in range \\[0, 1\\], got {rate}",
            ):
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
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
        )

        config = SamplingMaskerConfig()
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
        non_zero_fraction = (result_dense > 0).float().mean(dim=-1).mean().item()
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
        previous_mask = Mask.create_mask_from_dense_mask(
            mask_shape, previous_mask_dense
        )

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
        attention_mask = torch.ones(
            batch_size, seq_len_keys, dtype=torch.bool, device="cpu"
        )
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
            queries = torch.randn(
                batch_size, num_heads, seq_len_queries, 16, device="cuda"
            )
            values = torch.randn(batch_size, num_heads, seq_len_keys, 16, device="cuda")
            attention_mask = torch.ones(
                batch_size, seq_len_keys, dtype=torch.bool, device="cuda"
            )

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

        config = MagicPigConfig(lsh_l=4, lsh_k=8)
        assert config is not None
        assert config.lsh_l == 4
        assert config.lsh_k == 8

    def test_magic_pig_creation(self):
        """Test that magic pig can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=4, lsh_k=8)
        masker = MagicPig(config)
        assert type(masker) is MagicPig
        assert masker.config == config

    def test_magic_pig_creation_from_config(self):
        """Test that magic pig can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=4, lsh_k=8)
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

    def test_magic_pig_config_validation_invalid_lsh_l(self):
        """Test that magic pig config rejects invalid lsh_l parameter."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )

        # Test invalid lsh_l values
        invalid_lsh_l_values = [0, -1, -5]
        for lsh_l in invalid_lsh_l_values:
            with pytest.raises(ValueError, match="lsh_l must be positive"):
                MagicPigConfig(lsh_l=lsh_l, lsh_k=4)

    def test_magic_pig_config_validation_invalid_lsh_k(self):
        """Test that magic pig config rejects invalid lsh_k parameter."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )

        # Test invalid lsh_k values
        invalid_lsh_k_values = [0, -1, -5]
        for lsh_k in invalid_lsh_k_values:
            with pytest.raises(ValueError, match="lsh_k must be positive"):
                MagicPigConfig(lsh_l=5, lsh_k=lsh_k)

    def test_magic_pig_create_from_config_invalid_type(self):
        """Test create_from_config with invalid config type."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            RandomSamplingMaskerConfig,
        )

        invalid_config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        with pytest.raises(ValueError, match="Invalid config type"):
            MagicPig.create_from_config(invalid_config)

    def test_magic_pig_transform_for_mips(self):
        """Test MIPS transformation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=3, lsh_k=2)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)

        transformed_keys, transformed_queries = masker._transform_for_mips(
            keys, queries
        )

        # Check shapes - should have one extra dimension for augmentation
        assert transformed_keys.shape[-1] == keys.shape[-1] + 1
        assert transformed_queries.shape[-1] == queries.shape[-1] + 1

        # Check that queries are normalized
        query_norms = torch.norm(transformed_queries[..., :-1], dim=-1)
        assert torch.allclose(query_norms, torch.ones_like(query_norms), atol=1e-6)

        # check that keys are also norm 1
        transformed_key_norms = torch.norm(transformed_keys, dim=-1)
        assert torch.allclose(
            transformed_key_norms, torch.ones_like(transformed_key_norms), atol=1e-6
        )

        # Check that keys are scaled by max norm
        key_norms_original = torch.norm(keys, dim=-1)
        max_key_norm = torch.max(key_norms_original)
        key_norms_scaled = torch.norm(transformed_keys[..., :-1], dim=-1)
        assert torch.allclose(
            key_norms_scaled, key_norms_original / max_key_norm, atol=1e-6
        )

        # check if query last dimension is 0
        assert torch.allclose(
            transformed_queries[..., -1],
            torch.zeros_like(transformed_queries[..., -1]),
            atol=1e-6,
        )

        # check if query and transformed query have cosine similarity 1
        cosine_similarity = torch.nn.functional.cosine_similarity(
            transformed_queries[..., :-1], queries, dim=-1
        )
        assert torch.allclose(
            cosine_similarity, torch.ones_like(cosine_similarity), atol=1e-6
        )

        # check if keys and transformed keys have cosine similarity 1
        cosine_similarity = torch.nn.functional.cosine_similarity(
            transformed_keys[..., :-1], keys, dim=-1
        )
        assert torch.allclose(
            cosine_similarity, torch.ones_like(cosine_similarity), atol=1e-6
        )

    def test_magic_pig_compute_probabilities(self):
        """Test probability computation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=3, lsh_k=2)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)

        probabilities = masker._compute_probabilities(keys, queries)

        # Check shape
        assert probabilities.shape == (
            keys.shape[0],
            keys.shape[1],
            queries.shape[2],
            keys.shape[2],
        )

        # Check that probabilities are in [0, 1]
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 1, 1, 1, 1, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)

        # if query is parallel to key then prob is 1
        keys = queries.clone() * 1.5
        probabilities = masker._compute_probabilities(keys, queries)
        assert probabilities.shape == (1, 1, 1, 1)
        assert probabilities[0, 0, 0, 0] == 1

        # if query is anti parallel to key then prob is 0
        keys = queries.clone() * -1.5
        probabilities = masker._compute_probabilities(keys, queries)
        assert probabilities.shape == (1, 1, 1, 1)
        assert probabilities[0, 0, 0, 0] == 0

        # if query1 is close to key and query2 is far away then prob(query1) > prob(query2)
        keys = torch.randn(batch_size, num_heads, 1, head_dim)
        queries = torch.randn(batch_size, num_heads, 2, head_dim)
        queries[0, 0, 0] = keys[0, 0, 0] * 0.9 + torch.randn_like(keys[0, 0, 0]) * 0.1
        queries[0, 0, 1] = keys[0, 0, 0] * 0.1 + torch.randn_like(keys[0, 0, 0]) * 0.9
        probabilities = masker._compute_probabilities(keys, queries)
        assert probabilities[0, 0, 0, 0] > probabilities[0, 0, 1, 0]

    def test_magic_pig_compute_lsh_matches(self):
        """Test LSH match computation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=3, lsh_k=2)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)

        matches = masker._compute_lsh_matches(keys, queries)

        # Check shape
        assert matches.shape == (
            keys.shape[0],
            keys.shape[1],
            queries.shape[2],
            keys.shape[2],
        )

        # Check that matches are binary
        assert torch.all((matches == 0) | (matches == 1))

        # test with 1,1,1,1 since we want to avoid effect of transformation

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 1, 1, 1, 1, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)

        # Check that identical vectors match
        identical_keys = keys.clone()
        identical_queries = keys.clone()  # Use keys as queries for identical case

        identical_matches = masker._compute_lsh_matches(
            identical_keys, identical_queries
        )
        print(identical_matches[0, 0])
        # Diagonal should have matches (identical vectors)
        for b in range(identical_matches.shape[0]):
            for h in range(identical_matches.shape[1]):
                diagonal_matches = torch.diagonal(identical_matches[b, h])
                # Should have some matches on diagonal (allowing for LSH randomness)
                assert torch.sum(diagonal_matches) > 0

        # Check that antiparallel vectors dont match
        antiparallel_keys = keys.clone()
        antiparallel_queries = (
            -1 * keys.clone()
        )  # Use keys as queries for identical case

        antiparallel_matches = masker._compute_lsh_matches(
            antiparallel_keys, antiparallel_queries
        )

        # Diagonal should have no matches (antiparallel vectors)
        for b in range(antiparallel_matches.shape[0]):
            for h in range(antiparallel_matches.shape[1]):
                diagonal_matches = torch.diagonal(antiparallel_matches[b, h])
                # Should have no matches on diagonal (allowing for LSH randomness)
                assert torch.sum(diagonal_matches) == 0

    def test_magic_pig_add_mask_basic(self):
        """Test basic add_mask functionality."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = MagicPigConfig(lsh_l=3, lsh_k=2)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)

        # Create empty previous mask
        previous_mask = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
        )

        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
        )

        # Check that result is a Mask
        assert isinstance(result_mask, Mask)

        # Check shape
        assert result_mask.shape == (
            batch_size,
            num_heads,
            seq_len_queries,
            seq_len_keys,
        )

        # Check that result is not empty (should have some LSH matches)
        assert not result_mask.is_empty()

    def test_magic_pig_add_mask_full_previous(self):
        """Test add_mask with full previous mask."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = MagicPigConfig(lsh_l=3, lsh_k=2)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)

        # Create full previous mask
        previous_mask = Mask.create_full_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
        )

        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
        )

        # Should return the same full mask
        assert result_mask.is_full_mask()
        assert result_mask.shape == previous_mask.shape

    def test_magic_pig_add_mask_merge(self):
        """Test add_mask with existing mask for merging."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = MagicPigConfig(lsh_l=3, lsh_k=2)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)

        # Create a mask with some existing entries
        existing_mask = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
        )

        # Add some existing entries manually
        dense_existing = torch.zeros(
            batch_size, num_heads, seq_len_queries, seq_len_keys
        )
        dense_existing[0, 0, 0, 0] = 1.0  # Add one entry
        existing_mask = Mask.create_mask_from_dense_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            mask=dense_existing,
            dtype=torch.float32,
        )

        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=existing_mask,
        )

        # Check that result is a Mask
        assert isinstance(result_mask, Mask)

        # Check that the existing entry is preserved
        result_dense = result_mask.get_dense_mask()
        assert result_dense[0, 0, 0, 0] > 0  # Should still have the existing entry

        # Check that new entries are added
        assert torch.sum(result_dense) > torch.sum(dense_existing)

    def test_magic_pig_different_lsh_parameters(self):
        """Test different LSH parameter combinations."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)

        previous_mask = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        # Test different parameter combinations
        configs = [
            MagicPigConfig(lsh_l=2, lsh_k=2),
            MagicPigConfig(lsh_l=5, lsh_k=3),
            MagicPigConfig(lsh_l=10, lsh_k=1),
        ]

        results = []
        for config in configs:
            masker = MagicPig(config)
            result = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=attention_mask,
                sparse_meta_data=sparse_meta_data,
                previous_mask=previous_mask,
            )
            results.append(result)

        # All results should be valid masks
        for result in results:
            assert isinstance(result, Mask)
            assert result.shape == (
                batch_size,
                num_heads,
                seq_len_queries,
                seq_len_keys,
            )
            assert not result.is_empty()

    def test_magic_pig_device_consistency(self):
        """Test that masker works with different devices."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = MagicPigConfig(lsh_l=3, lsh_k=2)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64

        # Test on CPU
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        previous_mask = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_cpu = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
        )

        # Test on GPU if available
        if torch.cuda.is_available():
            keys_gpu = keys.cuda()
            queries_gpu = queries.cuda()
            values_gpu = values.cuda()
            previous_mask_gpu = Mask.create_empty_mask(
                shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
                dtype=torch.float32,
            )
            attention_mask_gpu = attention_mask.cuda()

            result_gpu = masker.add_mask(
                keys=keys_gpu,
                queries=queries_gpu,
                values=values_gpu,
                attention_mask=attention_mask_gpu,
                sparse_meta_data=sparse_meta_data,
                previous_mask=previous_mask_gpu,
            )

            # Both should be valid masks
            assert isinstance(result_cpu, Mask)
            assert isinstance(result_gpu, Mask)
            assert result_cpu.shape == result_gpu.shape

    def test_magic_pig_dtype_consistency(self):
        """Test that masker works with different dtypes."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = MagicPigConfig(lsh_l=3, lsh_k=2)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64

        # Test with float32
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        previous_mask_f32 = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_f32 = masker.add_mask(
            keys=keys.float(),
            queries=queries.float(),
            values=values.float(),
            attention_mask=attention_mask,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask_f32,
        )

        # Test with float16 if available
        if hasattr(torch, "float16"):
            previous_mask_f16 = Mask.create_empty_mask(
                shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
                dtype=torch.float16,
            )

            result_f16 = masker.add_mask(
                keys=keys.half(),
                queries=queries.half(),
                values=values.half(),
                attention_mask=attention_mask,
                sparse_meta_data=sparse_meta_data,
                previous_mask=previous_mask_f16,
            )

            # Both should be valid masks
            assert isinstance(result_f32, Mask)
            assert isinstance(result_f16, Mask)
            assert result_f32.shape == result_f16.shape
