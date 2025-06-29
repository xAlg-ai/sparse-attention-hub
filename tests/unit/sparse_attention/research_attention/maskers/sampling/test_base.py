''' 
Author: Aditya Desai
:copyright: 2025 Sparse Attention hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for sampling masker base classes, configs, and create_from_config methods. This file is part of the Sparse Attention Hub project.
'''

import pytest

@pytest.mark.unit
class TestSamplingMaskerImports:
    """Test class for sampling masker imports."""

    def test_base_sampling_masker_imports(self):
        """Test that all base sampling masker classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMasker, SamplingMaskerConfig,
            RandomSamplingMasker, RandomSamplingMaskerConfig,
            MagicPig, MagicPigConfig,
        )
        assert SamplingMasker is not None
        assert RandomSamplingMasker is not None
        assert MagicPig is not None

        assert SamplingMaskerConfig is not None
        assert RandomSamplingMaskerConfig is not None
        assert MagicPigConfig is not None

@pytest.mark.unit
class TestSamplingMaskerConfigCreation:
    """Test class for sampling masker config creation."""

    def test_sampling_masker_config_creation(self):
        """Test that sampling masker configs can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMaskerConfig,
        )
        config = SamplingMaskerConfig(sampling_rate=0.5)
        assert config is not None


@pytest.mark.unit
class TestSamplingInheritance:
    """Test class for sampling masker inheritance."""

    def test_sampling_masker_inheritance(self):
        """Test that sampling masker inherits from ResearchMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMasker
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker
        )
        assert issubclass(SamplingMasker, ResearchMasker)
