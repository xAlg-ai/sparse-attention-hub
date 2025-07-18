"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-07-03
:summary: Tests for research attention. This file is part of the Sparse Attention Hub project.
"""

import pytest


@pytest.mark.unit
class TestImports:
    """Test class for imports."""

    def test_imports(self):
        """Test that all imports are working."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
        )

        assert ResearchAttention is not None


@pytest.mark.unit
class TestResearchAttentionAndConfigCreation:
    """Test class for research attention and config creation."""

    def test_research_attention_creation(self):
        """Test that research attention can be created."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker,
            LocalMaskerConfig,
            OracleTopK,
            OracleTopKConfig,
            SinkMasker,
            SinkMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        masker_configs = [
            SinkMaskerConfig(sink_size=10),
            LocalMaskerConfig(window_size=10),
            OracleTopKConfig(heavy_size=10),
            RandomSamplingMaskerConfig(sampling_rate=0.5),
        ]

        config = ResearchAttentionConfig(masker_configs=masker_configs)
        assert config is not None
        attention = ResearchAttention.create_from_config(config)
        assert attention is not None
        assert len(attention.maskers) == len(masker_configs)
        assert isinstance(attention.maskers[0], SinkMasker)
        assert isinstance(attention.maskers[1], LocalMasker)
        assert isinstance(attention.maskers[2], OracleTopK)
        assert isinstance(attention.maskers[3], RandomSamplingMasker)


@pytest.mark.unit
class TestInheritance:
    """Test class for inheritance."""

    def test_inheritance(self):
        """Test that research attention inherits from sparse attention."""
        from sparse_attention_hub.sparse_attention import (
            SparseAttention,
            SparseAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )

        assert issubclass(ResearchAttention, SparseAttention)
        assert issubclass(ResearchAttentionConfig, SparseAttentionConfig)


@pytest.mark.unit
class TestSamplingMaskerValidation:
    """Test class for sampling masker validation."""

    def test_single_sampling_masker_allowed(self):
        """Test that a single sampling masker is allowed."""
        from sparse_attention_hub.sparse_attention import SparseAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker,
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        # Create maskers: one fixed, one sampling
        local_masker = LocalMasker.create_from_config(LocalMaskerConfig(window_size=10))
        sampling_masker = RandomSamplingMasker.create_from_config(
            RandomSamplingMaskerConfig(sampling_rate=0.5)
        )

        # This should not raise an error
        config = SparseAttentionConfig()
        attention = ResearchAttention(config, [local_masker, sampling_masker])
        assert attention is not None
        assert len(attention.maskers) == 2

    def test_multiple_sampling_maskers_rejected(self):
        """Test that multiple sampling maskers are rejected."""
        from sparse_attention_hub.sparse_attention import SparseAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            MagicPig,
            MagicPigConfig,
            RandomSamplingMasker,
            RandomSamplingMaskerConfig,
        )

        # Create two sampling maskers
        random_masker = RandomSamplingMasker.create_from_config(
            RandomSamplingMaskerConfig(sampling_rate=0.5)
        )
        magic_pig_masker = MagicPig.create_from_config(
            MagicPigConfig(lsh_l=4, lsh_k=16)
        )

        # This should raise an error
        config = SparseAttentionConfig()
        with pytest.raises(
            ValueError,
            match="Only one sampling masker supported for efficiency; consider implementing all sampling logic in one masker",
        ):
            ResearchAttention(config, [random_masker, magic_pig_masker])

    def test_multiple_sampling_maskers_via_config(self):
        """Test that multiple sampling maskers are rejected when created via config."""
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            MagicPigConfig,
            RandomSamplingMaskerConfig,
        )

        # Create config with two sampling maskers
        masker_configs = [
            RandomSamplingMaskerConfig(sampling_rate=0.5),
            MagicPigConfig(lsh_l=4, lsh_k=16),
        ]
        config = ResearchAttentionConfig(masker_configs=masker_configs)

        # This should raise an error
        with pytest.raises(
            ValueError,
            match="Only one sampling masker supported for efficiency; consider implementing all sampling logic in one masker",
        ):
            ResearchAttention.create_from_config(config)

    def test_no_sampling_maskers_allowed(self):
        """Test that no sampling maskers is allowed."""
        from sparse_attention_hub.sparse_attention import SparseAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention import (
            ResearchAttention,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker,
            LocalMaskerConfig,
            SinkMasker,
            SinkMaskerConfig,
        )

        # Create only fixed maskers
        local_masker = LocalMasker.create_from_config(LocalMaskerConfig(window_size=10))
        sink_masker = SinkMasker.create_from_config(SinkMaskerConfig(sink_size=5))

        # This should not raise an error
        config = SparseAttentionConfig()
        attention = ResearchAttention(config, [local_masker, sink_masker])
        assert attention is not None
        assert len(attention.maskers) == 2
