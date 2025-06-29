''' 
Author: Aditya Desai
:copyright: 2025 Sparse Attention hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for masker base classes, configs, and create_from_config methods. This file is part of the Sparse Attention Hub project.
'''

import pytest
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
    FixedMasker, FixedMaskerConfig,
    LocalMasker, LocalMaskerConfig,
    CausalMasker,
    SinkMasker, SinkMaskerConfig,
    OracleTopK, OracleTopKConfig,
    PQCache, PQCacheConfig,
    HashAttentionTopKMasker, HashAttentionTopKMaskerConfig,
    DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig,
)

@pytest.mark.unit
class TestMaskerImports:
    """Test class for masker imports."""

    def test_base_masker_imports(self):
        """Test that all base masker classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker, FixedMaskerConfig,
            LocalMasker, LocalMaskerConfig,
            CausalMasker,
            SinkMasker, SinkMaskerConfig,
            OracleTopK, OracleTopKConfig,
            PQCache, PQCacheConfig,
            HashAttentionTopKMasker, HashAttentionTopKMaskerConfig,
            DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig,
        )
        assert FixedMasker is not None
        assert LocalMasker is not None
        assert CausalMasker is not None
        assert SinkMasker is not None
        assert OracleTopK is not None
        assert PQCache is not None
        assert HashAttentionTopKMasker is not None
        assert DoubleSparsityTopKMasker is not None

        assert FixedMaskerConfig is not None
        assert LocalMaskerConfig is not None
        assert SinkMaskerConfig is not None
        assert OracleTopKConfig is not None
        assert PQCacheConfig is not None
        assert HashAttentionTopKMaskerConfig is not None
        assert DoubleSparsityTopKMaskerConfig is not None

@pytest.mark.unit
class TestMaskerConfigCreation:
    """Test class for masker config creation."""

    def test_fixed_masker_config_creation(self):
        """Test that fixed masker configs can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        config = FixedMaskerConfig()
        assert config is not None

    def test_top_k_masker_config_creation(self):
        """Test that all top k masker configs can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        config = TopKMaskerConfig(heavy_size=100)
        assert config is not None
        
    def test_top_p_masker_config_creation(self):
        """Test that top p masker configs can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopPMaskerConfig,
        )
        config = TopPMaskerConfig(top_p=0.5)
        assert config is not None

@pytest.mark.unit
class TestInheritance:
    """Test class for masker inheritance."""

    def test_fixed_masker_inheritance(self):
        """Test that fixed masker inherits from ResearchMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker
        )
        assert issubclass(FixedMasker, ResearchMasker)

    def test_top_k_masker_inheritance(self):
        """Test that top k masker inherits from FixedMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
                TopKMasker, FixedMasker
        )
        assert issubclass(TopKMasker, FixedMasker)
    

    def test_top_p_masker_inheritance(self):
        """Test that top p masker inherits from FixedMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopPMasker, FixedMasker
        )
        assert issubclass(TopPMasker, FixedMasker)