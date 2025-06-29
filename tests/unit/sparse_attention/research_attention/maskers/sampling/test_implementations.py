"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for sampling masker implementations. This file is part of the Sparse Attention Hub project.
"""

import pytest

@pytest.mark.unit
class TestSamplingImplementationsImports:
    def test_implementations_imports(self):
        """Test that all sampling implementation classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker, RandomSamplingMaskerConfig,
            MagicPig, MagicPigConfig,
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

    def test_random_sampling_masker_creation(self):
        """Test that random sampling masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker, RandomSamplingMaskerConfig,
        )
        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        masker = RandomSamplingMasker(config)
        assert type(masker) is RandomSamplingMasker
        assert masker.config == config

    def test_random_sampling_masker_creation_from_config(self):
        """Test that random sampling masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker, RandomSamplingMaskerConfig,
        )
        config = RandomSamplingMaskerConfig(sampling_rate=0.5)
        masker = RandomSamplingMasker.create_from_config(config)
        assert type(masker) is RandomSamplingMasker
        assert masker.config == config

    def test_random_sampling_masker_inheritance(self):
        """Test that random sampling masker inherits from SamplingMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMasker,
        )
        assert issubclass(RandomSamplingMasker, SamplingMasker)

    def test_random_sampling_masker_config_inheritance(self):
        """Test that random sampling masker config inherits from SamplingMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            RandomSamplingMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMaskerConfig,
        )
        assert issubclass(RandomSamplingMaskerConfig, SamplingMaskerConfig)


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
            MagicPig, MagicPigConfig,
        )
        config = MagicPigConfig(sampling_rate=0.3, lsh_l=4, lsh_k=8)
        masker = MagicPig(config)
        assert type(masker) is MagicPig
        assert masker.config == config

    def test_magic_pig_creation_from_config(self):
        """Test that magic pig can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig, MagicPigConfig,
        )
        config = MagicPigConfig(sampling_rate=0.3, lsh_l=4, lsh_k=8)
        masker = MagicPig.create_from_config(config)
        assert type(masker) is MagicPig
        assert masker.config == config

    def test_magic_pig_inheritance(self):
        """Test that magic pig inherits from SamplingMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMasker,
        )
        assert issubclass(MagicPig, SamplingMasker)

    def test_magic_pig_config_inheritance(self):
        """Test that magic pig config inherits from SamplingMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
            MagicPigConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMaskerConfig,
        )
        assert issubclass(MagicPigConfig, SamplingMaskerConfig)
