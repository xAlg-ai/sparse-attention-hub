"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for PQCache masker implementation.
"""

import pytest


@pytest.mark.unit
class TestPQCacheMaskerImplementation:
    def test_pq_cache_masker_config_creation(self):
        """Test that pq cache masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4)
        assert config is not None
        assert config.heavy_size == 10
        assert config.pq_sub_dim == 8
        assert config.pq_bits == 4

    def test_pq_cache_masker_creation(self):
        """Test that pq cache masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4)
        masker = PQCache(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_creation_from_config(self):
        """Test that pq cache masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
            PQCacheConfig,
        )

        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4)
        masker = PQCache.create_from_config(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_inheritance(self):
        """Test that pq cache masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
        )

        assert issubclass(PQCache, TopKMasker)

    def test_pq_cache_masker_config_inheritance(self):
        """Test that pq cache masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )

        assert issubclass(PQCacheConfig, TopKMaskerConfig)
