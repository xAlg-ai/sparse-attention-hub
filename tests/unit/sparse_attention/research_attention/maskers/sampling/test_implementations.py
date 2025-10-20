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

        # Test valid sampling rates - must be in range (0, 1] (exclusive of 0)
        valid_rates = [0.1, 0.5, 0.9, 1.0]
        for rate in valid_rates:
            config = RandomSamplingMaskerConfig(sampling_rate=rate)
            assert config.sampling_rate == rate

    def test_random_sampling_masker_config_validation_invalid_rates(self):
        """Test that random sampling masker config rejects invalid sampling rates."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMaskerConfig,
        )

        # Test invalid sampling rates - 0.0 is also invalid (must be > 0)
        invalid_rates = [0.0, -0.1, 1.1, 2.0, -1.0]
        for rate in invalid_rates:
            with pytest.raises(
                ValueError,
                match=f"sampling_rate must be in range \\(0, 1\\], got {rate}",
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
        full_previous_mask = Mask.create_full_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
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
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
        )

        # Check basic properties
        assert result.shape == mask_shape
        assert not result.is_full_mask()
        assert not result.is_empty

        # Convert to dense to check sampling rate
        result_dense = result.get_dense_mask()

        # Check that approximately sampling_rate fraction of elements are non-zero
        # Allow some tolerance due to randomness
        non_zero_fraction = (result_dense > 0).float().mean(dim=-1).mean().item()
        assert 0.2 <= non_zero_fraction <= 0.3  # Allow 20% tolerance

    def test_random_sampling_masker_add_mask_sampling_rate_zero(self):
        """Test RandomSamplingMasker with sampling_rate=0.0 should raise ValueError."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMaskerConfig,
        )

        # sampling_rate=0.0 is not valid (must be > 0)
        with pytest.raises(
            ValueError, match="sampling_rate must be in range \\(0, 1\\], got 0.0"
        ):
            RandomSamplingMaskerConfig(sampling_rate=0.0)

    def test_random_sampling_masker_add_mask_sampling_rate_one(self):
        """Test RandomSamplingMasker with sampling_rate=1.0.

        Note: Sampling is performed with replacement, so not all positions
        are guaranteed to be covered even with sampling_rate=1.0.
        """
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        config = RandomSamplingMaskerConfig(sampling_rate=1.0)
        masker = RandomSamplingMasker(config)

        batch_size, num_heads, seq_len_queries, seq_len_keys = 1, 1, 2, 100

        # Create mock inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, 16)
        values = torch.randn(batch_size, num_heads, seq_len_keys, 16)
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        # Create empty mask as previous mask
        mask_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
        )

        # With sampling_rate=1.0 and large seq_len, most positions should be covered
        # Sampling is with replacement, so coverage should be ~63% (1 - 1/e)
        result_dense = result.get_dense_mask()
        assert not result.is_empty
        # Check that a high fraction of positions are covered (relaxed check)
        non_zero_fraction = (result_dense > 0).float().mean().item()
        assert (
            non_zero_fraction > 0.5
        )  # Should have good coverage with sampling_rate=1.0

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
            mask_shape, previous_mask_dense, dtype=torch.float32
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
        )

        # Check that the result is not empty and has the correct shape
        assert result.shape == mask_shape
        assert not result.is_empty

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
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
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
        empty_previous_mask = Mask.create_empty_mask(
            mask_shape, dtype=torch.float32, device=torch.device("cpu")
        )

        result = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=empty_previous_mask,
        )

        assert result.shape == mask_shape
        assert not result.is_empty

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
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data=sparse_meta_data,
                previous_mask=empty_previous_mask,
            )

            assert result.shape == mask_shape
            assert not result.is_empty


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

        # if query is anti parallel to key then prob is close to 0
        keys = queries.clone() * -1.5
        probabilities = masker._compute_probabilities(keys, queries)
        assert probabilities.shape == (1, 1, 1, 1)
        assert torch.allclose(probabilities[0, 0, 0, 0], torch.tensor(0.0), atol=1e-6)

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
        sparse_meta_data = {}
        layer_idx = 0

        matches = masker._compute_lsh_matches(
            keys, queries, sparse_meta_data, layer_idx
        )

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
        sparse_meta_data_identical = {}  # Use fresh metadata for this test

        identical_matches = masker._compute_lsh_matches(
            identical_keys, identical_queries, sparse_meta_data_identical, layer_idx
        )
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
        sparse_meta_data_antiparallel = {}  # Use fresh metadata for this test

        antiparallel_matches = masker._compute_lsh_matches(
            antiparallel_keys,
            antiparallel_queries,
            sparse_meta_data_antiparallel,
            layer_idx,
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
            device=torch.device("cpu"),
        )

        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
            layer_idx=0,
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
        assert not result_mask.is_empty

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
            device=torch.device("cpu"),
        )

        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
            layer_idx=0,
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
            device=torch.device("cpu"),
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
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=existing_mask,
            layer_idx=0,
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
            device=torch.device("cpu"),
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)

        # Test different parameter combinations
        configs = [
            MagicPigConfig(lsh_l=2, lsh_k=2),
            MagicPigConfig(lsh_l=5, lsh_k=3),
            MagicPigConfig(lsh_l=10, lsh_k=1),
        ]

        results = []
        for config in configs:
            masker = MagicPig(config)
            # Use separate metadata for each configuration to avoid conflicts
            config_sparse_meta_data = {}
            result = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=attention_mask,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data=config_sparse_meta_data,
                previous_mask=previous_mask,
                layer_idx=0,
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
            assert not result.is_empty

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
            device=torch.device("cpu"),
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_cpu = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Test on GPU if available
        if torch.cuda.is_available():
            keys_gpu = keys.cuda()
            queries_gpu = queries.cuda()
            values_gpu = values.cuda()
            previous_mask_gpu = Mask.create_empty_mask(
                shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
                dtype=torch.float32,
                device=torch.device("cuda"),
            )
            attention_mask_gpu = attention_mask.cuda()
            sparse_meta_data_gpu = (
                {}
            )  # Use separate metadata for GPU to avoid device conflicts

            result_gpu = masker.add_mask(
                keys=keys_gpu,
                queries=queries_gpu,
                values=values_gpu,
                attention_mask=attention_mask_gpu,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data=sparse_meta_data_gpu,
                previous_mask=previous_mask_gpu,
                layer_idx=0,
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
            device=torch.device("cpu"),
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)
        sparse_meta_data = {}

        result_f32 = masker.add_mask(
            keys=keys.float(),
            queries=queries.float(),
            values=values.float(),
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data,
            previous_mask=previous_mask_f32,
            layer_idx=0,
        )

        # Test with float16 if available
        if hasattr(torch, "float16"):
            previous_mask_f16 = Mask.create_empty_mask(
                shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
                dtype=torch.float16,
                device=torch.device("cpu"),
            )

            result_f16 = masker.add_mask(
                keys=keys.half(),
                queries=queries.half(),
                values=values.half(),
                attention_mask=attention_mask,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data=sparse_meta_data,
                previous_mask=previous_mask_f16,
                layer_idx=0,
            )

            # Both should be valid masks
            assert isinstance(result_f32, Mask)
            assert isinstance(result_f16, Mask)
            assert result_f32.shape == result_f16.shape

    # Tests for centering feature
    def test_centering_config_creation(self):
        """Test that MagicPigConfig can be created with centering enabled/disabled."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )

        # Test with centering enabled (default)
        config_centered = MagicPigConfig(lsh_l=3, lsh_k=2)
        assert config_centered.center is True

        # Test with centering disabled
        config_not_centered = MagicPigConfig(lsh_l=3, lsh_k=2, center=False)
        assert config_not_centered.center is False

        # Test explicitly enabled
        config_explicit_true = MagicPigConfig(lsh_l=3, lsh_k=2, center=True)
        assert config_explicit_true.center is True

    def test_center_kq_disabled(self):
        """Test _center_KQ method when centering is disabled."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=3, lsh_k=2, center=False)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_keys, seq_len_queries, head_dim = 2, 4, 16, 8, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        sparse_meta_data = {}
        layer_idx = 0

        centered_keys, centered_queries = masker._center_KQ(
            keys, queries, sparse_meta_data, layer_idx
        )

        # When centering is disabled, input tensors should be returned unchanged
        assert torch.equal(centered_keys, keys)
        assert torch.equal(centered_queries, queries)

    def test_center_kq_enabled_first_call(self):
        """Test _center_KQ method when centering is enabled for the first time."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=3, lsh_k=2, center=True)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_keys, seq_len_queries, head_dim = 2, 4, 16, 8, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        sparse_meta_data = {}
        layer_idx = 0

        centered_keys, centered_queries = masker._center_KQ(
            keys, queries, sparse_meta_data, layer_idx
        )

        # Check that key mean is cached
        assert "key_mean" in sparse_meta_data
        assert layer_idx in sparse_meta_data["key_mean"]
        assert sparse_meta_data["key_mean"][layer_idx] is not None

        # Check that the key mean has correct shape
        key_mean = sparse_meta_data["key_mean"][layer_idx]
        assert key_mean.shape == (batch_size, num_heads, 1, head_dim)

        # Check that centering was applied correctly
        expected_key_mean = torch.mean(keys, dim=2, keepdim=True)
        assert torch.allclose(key_mean, expected_key_mean, atol=1e-6)

        expected_centered_keys = keys - key_mean
        expected_centered_queries = queries - key_mean

        assert torch.allclose(centered_keys, expected_centered_keys, atol=1e-6)
        assert torch.allclose(centered_queries, expected_centered_queries, atol=1e-6)

        # Check that centered keys have mean close to zero along sequence dimension
        centered_key_mean = torch.mean(centered_keys, dim=2, keepdim=True)
        assert torch.allclose(
            centered_key_mean, torch.zeros_like(centered_key_mean), atol=1e-6
        )

    def test_center_kq_cached_mean(self):
        """Test _center_KQ method when using cached mean from previous call."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=3, lsh_k=2, center=True)
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_keys, seq_len_queries, head_dim = 2, 4, 16, 8, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        sparse_meta_data = {}
        layer_idx = 0

        # First call - should compute and cache mean
        centered_keys_1, centered_queries_1 = masker._center_KQ(
            keys, queries, sparse_meta_data, layer_idx
        )
        cached_mean = sparse_meta_data["key_mean"][layer_idx].clone()

        # Second call with different queries (simulating inference step)
        new_queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        centered_keys_2, centered_queries_2 = masker._center_KQ(
            keys, new_queries, sparse_meta_data, layer_idx
        )

        # Check that the same cached mean is used
        assert torch.equal(sparse_meta_data["key_mean"][layer_idx], cached_mean)

        # Check that keys are centered using cached mean
        assert torch.allclose(centered_keys_2, keys - cached_mean, atol=1e-6)
        assert torch.allclose(centered_queries_2, new_queries - cached_mean, atol=1e-6)

    def test_centering_integration_with_add_mask(self):
        """Test that centering works correctly when integrated with add_mask."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        # Test both centered and non-centered versions
        config_centered = MagicPigConfig(lsh_l=3, lsh_k=2, center=True)
        config_not_centered = MagicPigConfig(lsh_l=3, lsh_k=2, center=False)

        masker_centered = MagicPig(config_centered)
        masker_not_centered = MagicPig(config_not_centered)

        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)

        previous_mask = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)

        sparse_meta_data_centered = {}
        sparse_meta_data_not_centered = {}

        result_centered = masker_centered.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data_centered,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        result_not_centered = masker_not_centered.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data_not_centered,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Both should produce valid masks
        assert isinstance(result_centered, Mask)
        assert isinstance(result_not_centered, Mask)
        assert result_centered.shape == result_not_centered.shape

        # Check that key_mean is initialized in both cases but only used for centered version
        assert "key_mean" in sparse_meta_data_centered
        assert "key_mean" in sparse_meta_data_not_centered
        # The key_mean should be None for non-centered case and a tensor for centered case
        assert sparse_meta_data_not_centered["key_mean"][0] is None
        assert sparse_meta_data_centered["key_mean"][0] is not None

        # Results may be different due to centering affecting LSH
        # But both should be valid sparse masks
        assert not result_centered.is_empty
        assert not result_not_centered.is_empty

    # Tests for packing feature
    def test_packing_config_creation(self):
        """Test that MagicPigConfig can be created with different packing modes."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )

        # Test with int64 packing (default)
        config_int64 = MagicPigConfig(lsh_l=3, lsh_k=2)
        assert config_int64.packing == "int64"

        # Test explicitly with int64
        config_explicit_int64 = MagicPigConfig(lsh_l=3, lsh_k=2, packing="int64")
        assert config_explicit_int64.packing == "int64"

        # Test with float32 packing
        config_float32 = MagicPigConfig(lsh_l=3, lsh_k=2, packing="float32")
        assert config_float32.packing == "float32"

    def test_seed_config_creation(self):
        """Test that MagicPigConfig can be created with seed parameter."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )

        # Test with default seed (42)
        config_default_seed = MagicPigConfig(lsh_l=3, lsh_k=2)
        assert config_default_seed.seed == 42

        # Test with custom seed specified
        config_with_seed = MagicPigConfig(lsh_l=3, lsh_k=2, seed=123)
        assert config_with_seed.seed == 123

        # Test that None seed is rejected
        with pytest.raises(ValueError, match="seed cannot be None"):
            MagicPigConfig(lsh_l=3, lsh_k=2, seed=None)

        # Test that seed is properly passed to masker
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
        )

        masker = MagicPig(config_with_seed)
        assert masker.seed == 123

    def test_packing_config_validation(self):
        """Test validation for packing configuration."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )

        # Test invalid packing mode
        with pytest.raises(ValueError, match="packing must be 'int64' or 'float32'"):
            MagicPigConfig(lsh_l=3, lsh_k=2, packing="invalid")

        # Test int64 with too many bits
        with pytest.raises(
            ValueError, match="For 'int64' packing, lsh_k must be <= 64"
        ):
            MagicPigConfig(lsh_l=1, lsh_k=65, packing="int64")

    def test_pack_bits_basic(self):
        """Test basic functionality of _pack_bits method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=2, lsh_k=3, packing="int64")
        masker = MagicPig(config)

        # Test with simple signatures
        batch_size, num_heads, seq_len, total_bits = (
            2,
            4,
            8,
            6,
        )  # 2 tables * 3 bits = 6 total bits
        signatures = torch.ones(batch_size, num_heads, seq_len, total_bits)
        signatures[..., ::2] = -1  # Alternate pattern: [-1, 1, -1, 1, -1, 1]

        packed = masker._pack_bits(signatures)

        # Check shape - should have lsh_l dimensions for each table
        expected_shape = (batch_size, num_heads, seq_len, config.lsh_l)
        assert packed.shape == expected_shape

        # Check dtype
        assert packed.dtype == torch.int64

        # Verify specific packing for known pattern
        # First table: [-1, 1, -1] -> [0, 1, 0] -> binary 010 = 2
        # Second table: [1, -1, 1] -> [1, 0, 1] -> binary 101 = 5
        assert packed[0, 0, 0, 0].item() == 2  # First table
        assert packed[0, 0, 0, 1].item() == 5  # Second table

    def test_pack_bits_edge_cases(self):
        """Test _pack_bits with edge cases."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=1, lsh_k=1, packing="int64")
        masker = MagicPig(config)

        # Test with single bit
        signatures = torch.tensor(
            [[[[1.0]]]], dtype=torch.float32
        )  # Shape: (1, 1, 1, 1)
        packed = masker._pack_bits(signatures)

        assert packed.shape == (1, 1, 1, 1)
        assert packed[0, 0, 0, 0].item() == 1  # 1 -> 1

        # Test with all negative values
        signatures_neg = torch.tensor([[[[-1.0]]]], dtype=torch.float32)
        packed_neg = masker._pack_bits(signatures_neg)
        assert packed_neg[0, 0, 0, 0].item() == 0  # -1 -> 0

    def test_compute_signatures_int64_packing(self):
        """Test that _compute_signatures uses int64 packing when configured."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=2, lsh_k=3, packing="int64")
        masker = MagicPig(config)

        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 64
        vectors = torch.randn(batch_size, num_heads, seq_len, head_dim)

        sparse_meta_data = {
            "projections": {0: torch.randn(head_dim, config.lsh_l * config.lsh_k)}
        }
        layer_idx = 0

        signatures = masker._compute_signatures(vectors, sparse_meta_data, layer_idx)

        # Should be packed into int64 format
        expected_shape = (batch_size, num_heads, seq_len, config.lsh_l)
        assert signatures.shape == expected_shape
        assert signatures.dtype == torch.int64

    def test_compute_signatures_float32_packing(self):
        """Test that _compute_signatures uses float32 when configured."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=2, lsh_k=3, packing="float32")
        masker = MagicPig(config)

        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 64
        vectors = torch.randn(batch_size, num_heads, seq_len, head_dim)

        sparse_meta_data = {
            "projections": {0: torch.randn(head_dim, config.lsh_l * config.lsh_k)}
        }
        layer_idx = 0

        signatures = masker._compute_signatures(vectors, sparse_meta_data, layer_idx)

        # Should be float format
        expected_shape = (batch_size, num_heads, seq_len, config.lsh_l * config.lsh_k)
        assert signatures.shape == expected_shape
        assert signatures.dtype == torch.float32

    # Tests for efficient matching feature
    def test_compute_matches_int64_basic(self):
        """Test basic functionality of _compute_matches_int64 method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=2, lsh_k=3, packing="int64")
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_keys, seq_len_queries = 2, 4, 8, 8

        # Create packed signatures - identical signatures should match
        keys_signatures = torch.randint(
            0, 8, (batch_size, num_heads, seq_len_keys, config.lsh_l), dtype=torch.int64
        )
        queries_signatures = keys_signatures.clone()  # Make queries identical to keys

        matches = masker._compute_matches_int64(keys_signatures, queries_signatures)

        # Check shape
        expected_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        assert matches.shape == expected_shape

        # Check dtype
        assert matches.dtype in [torch.int32, torch.int]

        # When keys and queries are identical, diagonal should be 1
        diagonal_matches = torch.diagonal(matches, dim1=-2, dim2=-1)
        assert torch.all(diagonal_matches == 1)

    def test_compute_matches_int64_no_matches(self):
        """Test _compute_matches_int64 when there should be no matches."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=2, lsh_k=3, packing="int64")
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_keys, seq_len_queries = 1, 1, 1, 1

        # Create different signatures that shouldn't match
        keys_signatures = torch.zeros(
            (batch_size, num_heads, seq_len_keys, config.lsh_l), dtype=torch.int64
        )
        queries_signatures = torch.ones(
            (batch_size, num_heads, seq_len_queries, config.lsh_l), dtype=torch.int64
        )

        matches = masker._compute_matches_int64(keys_signatures, queries_signatures)

        # Should have no matches since signatures are completely different
        assert torch.all(matches == 0)

    def test_compute_matches_float32_basic(self):
        """Test basic functionality of _compute_matches_float32 method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )

        config = MagicPigConfig(lsh_l=2, lsh_k=3, packing="float32")
        masker = MagicPig(config)

        batch_size, num_heads, seq_len_keys, seq_len_queries = 2, 4, 8, 8
        total_bits = config.lsh_l * config.lsh_k

        # Create float signatures with known pattern
        keys_signatures = torch.ones(batch_size, num_heads, seq_len_keys, total_bits)
        queries_signatures = torch.ones(
            batch_size, num_heads, seq_len_queries, total_bits
        )

        matches = masker._compute_matches_float32(keys_signatures, queries_signatures)

        # Check shape and dtype
        expected_shape = (batch_size, num_heads, seq_len_queries, seq_len_keys)
        assert matches.shape == expected_shape
        assert matches.dtype in [torch.int32, torch.int]

        # Identical signatures should match for both tables
        assert torch.all(matches == 1)

    def test_matching_methods_consistency(self):
        """Test that different matching methods produce consistent results."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        batch_size, num_heads, seq_len_keys, seq_len_queries, head_dim = 2, 4, 8, 8, 64
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)

        # Test int64 packing
        config_int64 = MagicPigConfig(lsh_l=3, lsh_k=4, packing="int64")
        masker_int64 = MagicPig(config_int64)

        # Test float32 packing
        config_float32 = MagicPigConfig(lsh_l=3, lsh_k=4, packing="float32")
        masker_float32 = MagicPig(config_float32)

        previous_mask = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)

        sparse_meta_data_int64 = {}
        sparse_meta_data_float32 = {}

        result_int64 = masker_int64.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data_int64,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        result_float32 = masker_float32.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data_float32,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Both should produce valid masks
        assert isinstance(result_int64, Mask)
        assert isinstance(result_float32, Mask)
        assert result_int64.shape == result_float32.shape

        # Both should produce sparse masks
        assert not result_int64.is_empty
        assert not result_float32.is_empty

    def test_packing_integration_with_add_mask(self):
        """Test that different packing modes work correctly when integrated with add_mask."""
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
            device=torch.device("cpu"),
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)

        # Test with int64 packing
        config_int64 = MagicPigConfig(lsh_l=3, lsh_k=4, packing="int64")
        masker_int64 = MagicPig(config_int64)
        sparse_meta_data_int64 = {}

        result_int64 = masker_int64.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data_int64,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Test with float32 packing
        config_float32 = MagicPigConfig(lsh_l=3, lsh_k=4, packing="float32")
        masker_float32 = MagicPig(config_float32)
        sparse_meta_data_float32 = {}

        result_float32 = masker_float32.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data_float32,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Both should produce valid masks
        assert isinstance(result_int64, Mask)
        assert isinstance(result_float32, Mask)
        assert result_int64.shape == result_float32.shape
        assert not result_int64.is_empty
        assert not result_float32.is_empty

    def test_packing_deterministic_with_seed(self):
        """Test that int64 and float32 packing produce exactly the same results with the same seed."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        # Use a fixed seed for reproducibility
        seed = 42
        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64

        # Create identical inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)

        previous_mask = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)

        # Test with int64 packing and seed
        config_int64 = MagicPigConfig(lsh_l=3, lsh_k=4, packing="int64", seed=seed)
        masker_int64 = MagicPig(config_int64)
        sparse_meta_data_int64 = {}

        result_int64 = masker_int64.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data_int64,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Test with float32 packing and same seed
        config_float32 = MagicPigConfig(lsh_l=3, lsh_k=4, packing="float32", seed=seed)
        masker_float32 = MagicPig(config_float32)
        sparse_meta_data_float32 = {}

        result_float32 = masker_float32.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data_float32,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Both should produce valid masks
        assert isinstance(result_int64, Mask)
        assert isinstance(result_float32, Mask)
        assert result_int64.shape == result_float32.shape

        # The key test: with the same seed, both packing methods should produce identical results
        result_int64_dense = result_int64.get_dense_mask()
        result_float32_dense = result_float32.get_dense_mask()

        # Check that the masks are exactly the same
        assert torch.equal(
            result_int64_dense, result_float32_dense
        ), "int64 and float32 packing should produce identical results with the same seed"

        # Verify that the projections used are identical
        proj_int64 = sparse_meta_data_int64["projections"][0]
        proj_float32 = sparse_meta_data_float32["projections"][0]
        assert torch.equal(
            proj_int64, proj_float32
        ), "Both packing methods should use identical projection matrices with the same seed"

    def test_seed_reproducibility(self):
        """Test that using the same seed produces reproducible results."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.utils.mask import Mask

        seed = 123
        batch_size, num_heads, seq_len_queries, seq_len_keys, head_dim = 2, 4, 8, 16, 64

        # Create identical inputs
        keys = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)
        queries = torch.randn(batch_size, num_heads, seq_len_queries, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len_keys, head_dim)

        previous_mask = Mask.create_empty_mask(
            shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        attention_mask = torch.ones(batch_size, seq_len_keys, dtype=torch.bool)

        # First run with seed
        config1 = MagicPigConfig(lsh_l=3, lsh_k=4, packing="int64", seed=seed)
        masker1 = MagicPig(config1)
        sparse_meta_data1 = {}

        result1 = masker1.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data1,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Second run with same seed
        config2 = MagicPigConfig(lsh_l=3, lsh_k=4, packing="int64", seed=seed)
        masker2 = MagicPig(config2)
        sparse_meta_data2 = {}

        result2 = masker2.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data=sparse_meta_data2,
            previous_mask=previous_mask,
            layer_idx=0,
        )

        # Results should be identical
        result1_dense = result1.get_dense_mask()
        result2_dense = result2.get_dense_mask()
        assert torch.equal(
            result1_dense, result2_dense
        ), "Same seed should produce identical results across multiple runs"

        # Projections should be identical
        proj1 = sparse_meta_data1["projections"][0]
        proj2 = sparse_meta_data2["projections"][0]
        assert torch.equal(
            proj1, proj2
        ), "Same seed should produce identical projection matrices"
