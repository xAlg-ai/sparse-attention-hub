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
            ResearchAttentionConfig,
        )

        assert ResearchAttention is not None
        assert ResearchAttentionConfig is not None


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
