"""Tests for masker base classes, configs, and create_from_config methods."""

import pytest
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    ResearchMasker, MaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.base import (
    FixedMasker, FixedMaskerConfig,
    LocalMasker, LocalMaskerConfig,
    CausalMasker,
    SinkMasker, SinkMaskerConfig,
    OracleTopK, OracleTopKConfig,
    PQCache, PQCacheConfig,
    HashAttentionTopKMasker, HashAttentionTopKMaskerConfig,
    DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.base import (
    SamplingMasker, SamplingMaskerConfig
)


@pytest.mark.unit
class TestMaskerConfigsAndCreation:
    """Test class for masker configs and create_from_config methods."""

    def test_base_masker_imports(self):
        """Test that all base masker classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker, MaskerConfig
        )
        assert ResearchMasker is not None
        assert MaskerConfig is not None

    def test_fixed_masker_imports(self):
        """Test that all fixed masker classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker, LocalMaskerConfig,
            CausalMasker, FixedMaskerConfig,
            SinkMasker, SinkMaskerConfig,
            OracleTopK, OracleTopKConfig,
            PQCache, PQCacheConfig,
            HashAttentionTopKMasker, HashAttentionTopKMaskerConfig,
            DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig,
            FixedMaskerConfig, TopKMaskerConfig, TopPMaskerConfig
        )
        
        # Verify all classes are imported
        assert LocalMasker is not None
        assert LocalMaskerConfig is not None
        assert CausalMasker is not None
        assert SinkMasker is not None
        assert SinkMaskerConfig is not None
        assert OracleTopK is not None
        assert OracleTopKConfig is not None
        assert PQCache is not None
        assert PQCacheConfig is not None
        assert HashAttentionTopKMasker is not None
        assert HashAttentionTopKMaskerConfig is not None
        assert DoubleSparsityTopKMasker is not None
        assert DoubleSparsityTopKMaskerConfig is not None

    def test_sampling_masker_imports(self):
        """Test that all sampling masker classes can be imported."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            RandomSamplingMasker, RandomSamplingMaskerConfig,
            MagicPig, MagicPigConfig,
            SamplingMaskerConfig
        )
        
        assert RandomSamplingMasker is not None
        assert RandomSamplingMaskerConfig is not None
        assert MagicPig is not None
        assert MagicPigConfig is not None
        assert SamplingMaskerConfig is not None

    def test_local_masker_config_and_creation(self):
        """Test LocalMasker config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMasker, LocalMaskerConfig
        )
        
        # Test with float window_size
        config = LocalMaskerConfig(window_size=0.5)
        masker = LocalMasker.create_from_config(config)
        assert masker.window_size == 0.5
        assert masker.config == config
        
        # Test with integer window_size
        config = LocalMaskerConfig(window_size=10)
        masker = LocalMasker.create_from_config(config)
        assert masker.window_size == 10

    def test_causal_masker_config_and_creation(self):
        """Test CausalMasker config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            CausalMasker, FixedMaskerConfig
        )
        
        config = FixedMaskerConfig()
        masker = CausalMasker.create_from_config(config)
        assert masker.config == config

    def test_sink_masker_config_and_creation(self):
        """Test SinkMasker config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            SinkMasker, SinkMaskerConfig
        )
        
        # Test with float sink_size
        config = SinkMaskerConfig(sink_size=0.3)
        masker = SinkMasker.create_from_config(config)
        assert masker.sink_size == 0.3
        assert masker.config == config
        
        # Test with integer sink_size
        config = SinkMaskerConfig(sink_size=5)
        masker = SinkMasker.create_from_config(config)
        assert masker.sink_size == 5

    def test_oracle_top_k_config_and_creation(self):
        """Test OracleTopK config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            OracleTopK, OracleTopKConfig
        )
        
        config = OracleTopKConfig(heavy_size=0.4)
        masker = OracleTopK.create_from_config(config)
        assert masker.heavy_size == 0.4
        assert masker.config == config

    def test_random_sampling_masker_config_and_creation(self):
        """Test RandomSamplingMasker config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            RandomSamplingMasker, RandomSamplingMaskerConfig
        )
        
        config = RandomSamplingMaskerConfig(sampling_rate=0.8)
        masker = RandomSamplingMasker.create_from_config(config)
        assert masker.sampling_rate == 0.8
        assert masker.config == config

    def test_hash_attention_config_and_creation(self):
        """Test HashAttention config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            HashAttentionTopKMasker, HashAttentionTopKMaskerConfig
        )
        
        config = HashAttentionTopKMaskerConfig(
            heavy_size=0.4,
            hat_bits=8,
            hat_mlp_layers=2,
            hat_mlp_hidden_size=64
        )
        masker = HashAttentionTopKMasker.create_from_config(config)
        
        assert masker.heavy_size == 0.4
        assert masker.hat_bits == 8
        assert masker.hat_mlp_layers == 2
        assert masker.hat_mlp_hidden_size == 64
        assert masker.config == config

    def test_double_sparsity_config_and_creation(self):
        """Test DoubleSparsity config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            DoubleSparsityTopKMasker, DoubleSparsityTopKMaskerConfig
        )
        
        channel_config = {'stats': 'required', 'channels': 64}
        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=0.3,
            group_factor=4,
            label_bits=8,
            channel_config=channel_config
        )
        masker = DoubleSparsityTopKMasker.create_from_config(config)
        
        assert masker.heavy_size == 0.3
        assert masker.group_factor == 4
        assert masker.label_bits == 8
        assert masker.channel_config == channel_config
        assert masker.config == config

    def test_pq_cache_config_and_creation(self):
        """Test PQCache config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            PQCache, PQCacheConfig
        )
        
        config = PQCacheConfig(
            heavy_size=0.5,
            pq_sub_dim=8,
            pq_bits=4
        )
        masker = PQCache.create_from_config(config)
        
        assert masker.heavy_size == 0.5
        assert masker.pq_sub_dim == 8
        assert masker.pq_bits == 4
        assert masker.config == config

    def test_magic_pig_config_and_creation(self):
        """Test MagicPig config and create_from_config method."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            MagicPig, MagicPigConfig
        )
        
        config = MagicPigConfig(
            sampling_rate=0.7,
            lsh_l=10,
            lsh_k=8
        )
        masker = MagicPig.create_from_config(config)
        
        assert masker.sampling_rate == 0.7
        assert masker.lsh_l == 10
        assert masker.lsh_k == 8
        assert masker.config == config

    def test_masker_inheritance(self):
        """Test that maskers properly inherit from their base classes."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import ResearchMasker
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMasker, TopKMasker, LocalMasker, CausalMasker, SinkMasker,
            OracleTopK, PQCache, HashAttentionTopKMasker, DoubleSparsityTopKMasker
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMasker, RandomSamplingMasker, MagicPig
        )
        
        # Test inheritance hierarchy
        assert issubclass(FixedMasker, ResearchMasker)
        assert issubclass(TopKMasker, FixedMasker)
        assert issubclass(SamplingMasker, ResearchMasker)
        
        # Test specific maskers
        assert issubclass(LocalMasker, FixedMasker)
        assert issubclass(CausalMasker, FixedMasker)
        assert issubclass(SinkMasker, FixedMasker)
        assert issubclass(OracleTopK, TopKMasker)
        assert issubclass(PQCache, TopKMasker)
        assert issubclass(HashAttentionTopKMasker, TopKMasker)
        assert issubclass(DoubleSparsityTopKMasker, TopKMasker)
        assert issubclass(RandomSamplingMasker, SamplingMasker)
        assert issubclass(MagicPig, SamplingMasker)

    def test_config_inheritance(self):
        """Test that configs properly inherit from their base classes."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers import MaskerConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            FixedMaskerConfig, TopKMaskerConfig, LocalMaskerConfig, SinkMaskerConfig,
            OracleTopKConfig, PQCacheConfig, HashAttentionTopKMaskerConfig, DoubleSparsityTopKMaskerConfig
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
            SamplingMaskerConfig, RandomSamplingMaskerConfig, MagicPigConfig
        )
        
        # Test inheritance hierarchy
        assert issubclass(FixedMaskerConfig, MaskerConfig)
        assert issubclass(TopKMaskerConfig, FixedMaskerConfig)
        assert issubclass(SamplingMaskerConfig, MaskerConfig)
        
        # Test specific configs
        assert issubclass(LocalMaskerConfig, FixedMaskerConfig)
        assert issubclass(SinkMaskerConfig, FixedMaskerConfig)
        assert issubclass(OracleTopKConfig, TopKMaskerConfig)
        assert issubclass(PQCacheConfig, TopKMaskerConfig)
        assert issubclass(HashAttentionTopKMaskerConfig, TopKMaskerConfig)
        assert issubclass(DoubleSparsityTopKMaskerConfig, TopKMaskerConfig)
        assert issubclass(RandomSamplingMaskerConfig, SamplingMaskerConfig)
        assert issubclass(MagicPigConfig, SamplingMaskerConfig)


class TestResearchMasker:
    """Test ResearchMasker base class."""

    def test_research_masker_is_abstract(self):
        """Test that ResearchMasker is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ResearchMasker()

    def test_research_masker_has_required_methods(self):
        """Test that ResearchMasker has required abstract methods."""
        # Check that the class has the required abstract methods
        assert hasattr(ResearchMasker, 'create_mask')
        assert hasattr(ResearchMasker, 'create_masker_from_config')


class TestMaskerConfig:
    """Test MaskerConfig base class."""

    def test_masker_config_is_dataclass(self):
        """Test that MaskerConfig is a dataclass."""
        # This should work since MaskerConfig is a dataclass
        config = MaskerConfig()
        assert isinstance(config, MaskerConfig)


class TestFixedMasker:
    """Test FixedMasker base class."""

    def test_fixed_masker_is_abstract(self):
        """Test that FixedMasker is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            FixedMasker()

    def test_fixed_masker_has_required_methods(self):
        """Test that FixedMasker has required abstract methods."""
        # Check that the class has the required abstract methods
        assert hasattr(FixedMasker, 'create_mask')


class TestFixedMaskerConfig:
    """Test FixedMaskerConfig base class."""

    def test_fixed_masker_config_is_dataclass(self):
        """Test that FixedMaskerConfig is a dataclass."""
        # This should work since FixedMaskerConfig is a dataclass
        config = FixedMaskerConfig()
        assert isinstance(config, FixedMaskerConfig)


class TestSamplingMasker:
    """Test SamplingMasker base class."""

    def test_sampling_masker_is_abstract(self):
        """Test that SamplingMasker is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            SamplingMasker()

    def test_sampling_masker_has_required_methods(self):
        """Test that SamplingMasker has required abstract methods."""
        # Check that the class has the required abstract methods
        assert hasattr(SamplingMasker, 'create_mask')


class TestSamplingMaskerConfig:
    """Test SamplingMaskerConfig base class."""

    def test_sampling_masker_config_is_dataclass(self):
        """Test that SamplingMaskerConfig is a dataclass."""
        # This should work since SamplingMaskerConfig is a dataclass
        config = SamplingMaskerConfig()
        assert isinstance(config, SamplingMaskerConfig)


class TestMaskerInheritance:
    """Test masker inheritance relationships."""

    def test_fixed_masker_inheritance(self):
        """Test that FixedMasker inherits from ResearchMasker."""
        # FixedMasker should inherit from ResearchMasker
        assert issubclass(FixedMasker, ResearchMasker)

    def test_sampling_masker_inheritance(self):
        """Test that SamplingMasker inherits from ResearchMasker."""
        # SamplingMasker should inherit from ResearchMasker
        assert issubclass(SamplingMasker, ResearchMasker)

    def test_config_inheritance(self):
        """Test that config classes inherit correctly."""
        # FixedMaskerConfig and SamplingMaskerConfig should inherit from MaskerConfig
        assert issubclass(FixedMaskerConfig, MaskerConfig)
        assert issubclass(SamplingMaskerConfig, MaskerConfig)
