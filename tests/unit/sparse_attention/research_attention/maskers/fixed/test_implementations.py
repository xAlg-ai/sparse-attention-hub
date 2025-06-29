"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for masker implementations. This file is part of the Sparse Attention Hub project.
"""

import pytest

@pytest.mark.unit
class TestImplementationsImports:
    def test_implementations_imports(self):
        """Test that all implementation classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker, LocalMaskerConfig,
            CausalMasker,
            SinkMasker, SinkMaskerConfig,
            OracleTopK, OracleTopKConfig,
            PQCache, PQCacheConfig,
            HashAttentionTopKMasker, HashAttentionTopKMaskerConfig,
            DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig,
        )
        
        # Test masker classes
        assert LocalMasker is not None
        assert CausalMasker is not None
        assert SinkMasker is not None
        assert OracleTopK is not None
        assert PQCache is not None
        assert HashAttentionTopKMasker is not None
        assert DoubleSparsityTopKMasker is not None
        
        # Test config classes
        assert LocalMaskerConfig is not None
        assert SinkMaskerConfig is not None
        assert OracleTopKConfig is not None
        assert PQCacheConfig is not None
        assert HashAttentionTopKMaskerConfig is not None
        assert DoubleSparsityTopKMaskerConfig is not None


@pytest.mark.unit
class TestLocalMaskerImplementation:

    def test_local_masker_config_creation(self):
        """Test that local masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
        )
        config = LocalMaskerConfig(window_size=10)
        assert config is not None
        assert config.window_size == 10

    def test_local_masker_creation(self):
        """Test that local masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker, LocalMaskerConfig,
        )
        config = LocalMaskerConfig(window_size=10)
        masker = LocalMasker(config)
        assert type(masker) is LocalMasker
        assert masker.config == config

    def test_local_masker_creation_from_config(self):
        """Test that local masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker, LocalMaskerConfig,
        )
        config = LocalMaskerConfig(window_size=10)
        masker = LocalMasker.create_from_config(config)
        assert type(masker) is LocalMasker
        assert masker.config == config

    def test_local_masker_inheritance(self):
        """Test that local masker inherits from FixedMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker,
        )
        assert issubclass(LocalMasker, FixedMasker)

    def test_local_masker_config_inheritance(self):
        """Test that local masker config inherits from FixedMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        assert issubclass(LocalMaskerConfig, FixedMaskerConfig)


@pytest.mark.unit
class TestCausalMaskerImplementation:


    def test_causal_masker_config_creation(self):
        """Test that causal masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig, # default config
        )
        config = FixedMaskerConfig()
        assert config is not None

    def test_causal_masker_creation(self):
        """Test that causal masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            CausalMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        config = FixedMaskerConfig()
        masker = CausalMasker(config)
        assert type(masker) is CausalMasker
        assert masker.config == config

    def test_causal_masker_creation_from_config(self):
        """Test that local masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            CausalMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        config = FixedMaskerConfig()
        masker = CausalMasker.create_from_config(config)
        assert type(masker) is CausalMasker
        assert masker.config == config

    def test_causal_masker_inheritance(self):
        """Test that local masker inherits from FixedMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            CausalMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker,
        )
        assert issubclass(CausalMasker, FixedMasker)


    
@pytest.mark.unit
class TestSinkMaskerImplementation:


    def test_sink_masker_config_creation(self):
        """Test that sink masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMaskerConfig,
        )
        config = SinkMaskerConfig(sink_size=10)
        assert config is not None
        assert config.sink_size == 10

    def test_sink_masker_creation(self):
        """Test that sink masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker, SinkMaskerConfig,
        )

        config = SinkMaskerConfig(sink_size=10)
        masker = SinkMasker(config)
        assert type(masker) is SinkMasker
        assert masker.config == config

    def test_sink_masker_creation_from_config(self):
        """Test that sink masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker, SinkMaskerConfig,
        )
        config = SinkMaskerConfig(sink_size=10)
        masker = SinkMasker.create_from_config(config)
        assert type(masker) is SinkMasker
        assert masker.config == config

    def test_sink_masker_inheritance(self):
        """Test that sink masker inherits from FixedMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker,
        )
        assert issubclass(SinkMasker, FixedMasker)

    def test_sink_masker_config_inheritance(self):
        """Test that sink masker config inherits from FixedMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            SinkMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig,
        )
        assert issubclass(SinkMaskerConfig, FixedMaskerConfig)

@pytest.mark.unit
class TestOracleTopKMaskerImplementation:

    def test_oracle_top_k_masker_config_creation(self):
        """Test that oracle top k masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopKConfig,
        )
        config = OracleTopKConfig(heavy_size=10)
        assert config is not None
        assert config.heavy_size == 10

    def test_oracle_top_k_masker_creation(self):
        """Test that oracle top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK, OracleTopKConfig,
        )
        config = OracleTopKConfig(heavy_size=10)
        masker = OracleTopK(config)
        assert type(masker) is OracleTopK
        assert masker.config == config

    def test_oracle_top_k_masker_creation_from_config(self):
        """Test that oracle top k masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK, OracleTopKConfig,
        )
        config = OracleTopKConfig(heavy_size=10)
        masker = OracleTopK.create_from_config(config)
        assert type(masker) is OracleTopK
        assert masker.config == config

    def test_oracle_top_k_masker_inheritance(self):
        """Test that oracle top k masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            OracleTopK,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        assert issubclass(OracleTopK, TopKMasker)

    def test_pq_cache_masker_config_inheritance(self):
        """Test that pq cache masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        assert issubclass(PQCacheConfig, TopKMaskerConfig)


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
            PQCache, PQCacheConfig,
        )
        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4)
        masker = PQCache(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_creation_from_config(self):
        """Test that pq cache masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache, PQCacheConfig,
        )
        config = PQCacheConfig(heavy_size=10, pq_sub_dim=8, pq_bits=4)
        masker = PQCache.create_from_config(config)
        assert type(masker) is PQCache
        assert masker.config == config

    def test_pq_cache_masker_inheritance(self):
        """Test that pq cache masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCache,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        assert issubclass(PQCache, TopKMasker)

    def test_pq_cache_masker_config_inheritance(self):
        """Test that pq cache masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            PQCacheConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        assert issubclass(PQCacheConfig, TopKMaskerConfig)


@pytest.mark.unit
class TestHashAttentionTopKMaskerImplementation:

    def test_hash_attention_top_k_masker_config_creation(self):
        """Test that hash attention top k masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMaskerConfig,
        )
        config = HashAttentionTopKMaskerConfig(heavy_size=10, hat_bits=4, hat_mlp_hidden_size=128, hat_mlp_layers=2)
        assert config is not None
        assert config.heavy_size == 10
        assert config.hat_bits == 4
        assert config.hat_mlp_hidden_size == 128
        assert config.hat_mlp_layers == 2


    def test_hash_attention_top_k_masker_creation(self):
        """Test that hash attention top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker, HashAttentionTopKMaskerConfig,
        )
        config = HashAttentionTopKMaskerConfig(heavy_size=10, hat_bits=4, hat_mlp_hidden_size=128, hat_mlp_layers=2)
        masker = HashAttentionTopKMasker(config)
        assert type(masker) is HashAttentionTopKMasker
        assert masker.config == config

    def test_hash_attention_top_k_masker_creation_from_config(self):
        """Test that hash attention top k masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker, HashAttentionTopKMaskerConfig,
        )
        config = HashAttentionTopKMaskerConfig(heavy_size=10, hat_bits=4, hat_mlp_hidden_size=128, hat_mlp_layers=2)
        masker = HashAttentionTopKMasker.create_from_config(config)
        assert type(masker) is HashAttentionTopKMasker
        assert masker.config == config

    def test_hash_attention_top_k_masker_inheritance(self):
        """Test that hash attention top k masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        assert issubclass(HashAttentionTopKMasker, TopKMasker)

    def test_hash_attention_top_k_masker_config_inheritance(self):
        """Test that hash attention top k masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            HashAttentionTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        assert issubclass(HashAttentionTopKMaskerConfig, TopKMaskerConfig)


@pytest.mark.unit
class TestDoubleSparsityTopKMaskerImplementation:

    def test_double_sparsity_top_k_masker_config_creation(self):
        """Test that double sparsity top k masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMaskerConfig,
        )
        config = DoubleSparsityTopKMaskerConfig(heavy_size=10, group_factor=4, label_bits=4, channel_config="auto")
        assert config is not None
        assert config.heavy_size == 10
        assert config.group_factor == 4
        assert config.label_bits == 4
        assert config.channel_config == "auto"

    def test_double_sparsity_top_k_masker_creation(self):
        """Test that double sparsity top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig,
        )
        config = DoubleSparsityTopKMaskerConfig(heavy_size=10, group_factor=4, label_bits=4, channel_config="auto")
        masker = DoubleSparsityTopKMasker(config)
        assert type(masker) is DoubleSparsityTopKMasker
        assert masker.config == config

    def test_double_sparsity_top_k_masker_creation_from_config(self):
        """Test that double sparsity top k masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig,
        )
        config = DoubleSparsityTopKMaskerConfig(heavy_size=10, group_factor=4, label_bits=4, channel_config="auto")
        masker = DoubleSparsityTopKMasker.create_from_config(config)
        assert type(masker) is DoubleSparsityTopKMasker
        assert masker.config == config

    def test_double_sparsity_top_k_masker_inheritance(self):
        """Test that double sparsity top k masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        assert issubclass(DoubleSparsityTopKMasker, TopKMasker)

    def test_double_sparsity_top_k_masker_config_inheritance(self):
        """Test that double sparsity top k masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        assert issubclass(DoubleSparsityTopKMaskerConfig, TopKMaskerConfig)

