"""Tests for sparse attention config classes and create_from_config methods."""
import pytest


@pytest.mark.unit
class TestSparseAttentionConfigsAndFactories:
    """Test class for sparse attention configs and create_from_config methods."""

    def test_sparse_attention_config_import(self):
        """Test that SparseAttentionConfig can be imported."""
        from sparse_attention_hub.sparse_attention import SparseAttentionConfig

        assert SparseAttentionConfig is not None

    def test_efficient_attention_config_import(self):
        """Test that EfficientAttentionConfig can be imported."""
        from sparse_attention_hub.sparse_attention import EfficientAttentionConfig

        assert EfficientAttentionConfig is not None

    def test_research_attention_config_import(self):
        """Test that ResearchAttentionConfig can be imported."""
        from sparse_attention_hub.sparse_attention import ResearchAttentionConfig

        assert ResearchAttentionConfig is not None

    def test_double_sparsity_config_import(self):
        """Test that DoubleSparsityConfig can be imported."""
        from sparse_attention_hub.sparse_attention import (
            ChannelConfig,
            DoubleSparsityConfig,
        )

        assert DoubleSparsityConfig is not None
        assert ChannelConfig is not None

    def test_hash_attention_config_import(self):
        """Test that HashAttentionConfig can be imported."""
        from sparse_attention_hub.sparse_attention import HashAttentionConfig

        assert HashAttentionConfig is not None

    def test_double_sparsity_top_k_masker_config_import(self):
        """Test that DoubleSparsityTopKMaskerConfig can be imported."""
        from sparse_attention_hub.sparse_attention import DoubleSparsityTopKMaskerConfig

        assert DoubleSparsityTopKMaskerConfig is not None

    def test_hash_attention_top_k_masker_config_import(self):
        """Test that HashAttentionTopKMaskerConfig can be imported."""
        from sparse_attention_hub.sparse_attention import HashAttentionTopKMaskerConfig

        assert HashAttentionTopKMaskerConfig is not None

    def test_sparse_attention_config_creation(self):
        """Test SparseAttentionConfig creation and usage."""
        from sparse_attention_hub.sparse_attention import (
            SparseAttention,
            SparseAttentionConfig,
        )

        config = SparseAttentionConfig()
        assert config is not None

        # Test that we can create a concrete subclass with this config
        class ConcreteSparseAttention(SparseAttention):
            def custom_attention(self):
                return None, None

        attention = ConcreteSparseAttention(config)
        assert attention.sparse_attention_config == config

    def test_efficient_attention_config_creation(self):
        """Test EfficientAttentionConfig creation and usage."""
        from sparse_attention_hub.sparse_attention import (
            EfficientAttention,
            EfficientAttentionConfig,
        )

        config = EfficientAttentionConfig()
        assert config is not None

        # Test that we can create a concrete subclass with this config
        class ConcreteEfficientAttention(EfficientAttention):
            def custom_attention(self):
                return None, None

        attention = ConcreteEfficientAttention(config)
        assert attention.sparse_attention_config == config

    def test_double_sparsity_config_and_creation(self):
        """Test DoubleSparsityConfig and create_from_config method."""
        from sparse_attention_hub.sparse_attention import (
            DoubleSparsity,
            DoubleSparsityConfig,
        )

        config = DoubleSparsityConfig(
            heavy_size=0.3,
            sink_size=10,
            local_size=5,
            ds_channel_config="config_file.json",
            ds_bits=8,
            ds_group_factor=4.0,
        )

        assert config.heavy_size == 0.3
        assert config.sink_size == 10
        assert config.local_size == 5
        assert config.ds_channel_config == "config_file.json"
        assert config.ds_bits == 8
        assert config.ds_group_factor == 4.0

        # Test create_from_config
        double_sparsity = DoubleSparsity.create_from_config(config)
        assert isinstance(double_sparsity, DoubleSparsity)
        assert double_sparsity.group_factor == 4.0
        assert double_sparsity.label_bits == 8
        assert double_sparsity.channel_config == "config_file.json"

    def test_hash_attention_config_and_creation(self):
        """Test HashAttentionConfig and create_from_config method."""
        from sparse_attention_hub.sparse_attention import (
            HashAttention,
            HashAttentionConfig,
        )

        config = HashAttentionConfig(
            heavy_size=0.4,
            sink_size=10,
            local_size=5,
            hat_weights="weights.pt",
            hat_bits=8,
            hat_mlp_layers=2,
            hat_mlp_hidden_size=64,
        )

        assert config.heavy_size == 0.4
        assert config.sink_size == 10
        assert config.local_size == 5
        assert config.hat_weights == "weights.pt"
        assert config.hat_bits == 8
        assert config.hat_mlp_layers == 2
        assert config.hat_mlp_hidden_size == 64

        # Test create_from_config
        hash_attention = HashAttention.create_from_config(config)
        assert isinstance(hash_attention, HashAttention)
        assert hash_attention.hat_bits == 8
        assert hash_attention.hat_mlp_layers == 2
        assert hash_attention.hat_mlp_hidden_size == 64

    def test_double_sparsity_top_k_masker_config_and_creation(self):
        """Test DoubleSparsityTopKMaskerConfig and create_from_config method."""
        import os
        import tempfile

        from sparse_attention_hub.sparse_attention import (
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
        )

        # Create a temporary file for sorted_channel_file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write('{"test": "data"}')
            temp_file_path = temp_file.name

        try:
            config = DoubleSparsityTopKMaskerConfig(
                heavy_size=0.3,
                group_factor=4,
                label_bits=8,
                sorted_channel_file=temp_file_path,
            )

            assert config.heavy_size == 0.3
            assert config.group_factor == 4
            assert config.label_bits == 8
            assert config.sorted_channel_file == temp_file_path

            # Test create_from_config
            double_sparsity = DoubleSparsityTopKMasker.create_from_config(config)
            assert isinstance(double_sparsity, DoubleSparsityTopKMasker)
            assert double_sparsity.group_factor == 4
            assert double_sparsity.label_bits == 8
            assert double_sparsity.config.sorted_channel_file == temp_file_path
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_hash_attention_top_k_masker_config_and_creation(self):
        """Test HashAttentionTopKMaskerConfig and create_from_config method."""
        import torch

        from sparse_attention_hub.sparse_attention import (
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
        )

        # Create sample weight tensors
        sample_weights = {
            0: {
                "key_matrix": [torch.randn(2, 64, 32), torch.randn(2, 32, 8)],
                "key_bias": [torch.randn(2, 32), torch.randn(2, 8)],
                "query_matrix": [torch.randn(2, 64, 32), torch.randn(2, 32, 8)],
                "query_bias": [torch.randn(2, 32), torch.randn(2, 8)],
            }
        }

        config = HashAttentionTopKMaskerConfig(
            heavy_size=0.4,
            hat_bits=8,
            hat_mlp_layers=2,
            hat_mlp_hidden_size=64,
            hat_mlp_activation="relu",
            hat_weights=sample_weights,
        )

        assert config.heavy_size == 0.4
        assert config.hat_bits == 8
        assert config.hat_mlp_layers == 2
        assert config.hat_mlp_hidden_size == 64
        assert config.hat_mlp_activation == "relu"
        assert config.hat_weights == sample_weights

        # Test create_from_config
        hash_attention = HashAttentionTopKMasker.create_from_config(config)
        assert isinstance(hash_attention, HashAttentionTopKMasker)
        assert hash_attention.hat_bits == 8
        assert hash_attention.hat_mlp_layers == 2
        assert hash_attention.hat_mlp_hidden_size == 64
        assert hash_attention.hat_mlp_activation == "relu"
        assert hash_attention.hat_weights == sample_weights

    def test_research_attention_direct_instantiation(self):
        """Test that ResearchAttention can be instantiated directly since it's not abstract."""
        from sparse_attention_hub.sparse_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMaskerConfig,
        )

        # Create a simple config
        masker_configs = [LocalMaskerConfig(window_size=5)]
        sparse_config = ResearchAttentionConfig(masker_configs=masker_configs)

        # Test direct instantiation
        research_attention = ResearchAttention(sparse_config, [])
        assert isinstance(research_attention, ResearchAttention)
        assert research_attention.sparse_attention_config == sparse_config
        assert research_attention.maskers == []

        # Test that custom_attention method works
        import torch

        module = None  # Mock module
        queries = torch.randn(2, 4, 10, 64)  # (b, h, sk, d)
        keys = torch.randn(2, 4, 12, 64)  # (b, h, sq, d)
        values = torch.randn(2, 4, 12, 64)  # (b, h, sq, d)
        attention_mask = None
        scaling = 1.0
        dropout = 0.1

        result = research_attention.custom_attention(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data={},
        )
        assert result[0] is not None  # attention_output should not be None

    def test_research_attention_config_and_creation(self):
        """Test ResearchAttentionConfig and create_from_config method."""
        from sparse_attention_hub.sparse_attention import (
            ResearchAttention,
            ResearchAttentionConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            LocalMaskerConfig,
            OracleTopKConfig,
        )

        # Create masker configs
        masker_configs = [
            LocalMaskerConfig(window_size=5),
            OracleTopKConfig(heavy_size=10),
        ]

        config = ResearchAttentionConfig(masker_configs=masker_configs)
        assert config.masker_configs == masker_configs

        # Test create_from_config - ResearchAttention is now concrete
        research_attention = ResearchAttention.create_from_config(config)
        assert isinstance(research_attention, ResearchAttention)
        assert len(research_attention.maskers) == 2
        # The maskers should be concrete instances created from the configs
        assert research_attention.maskers[0].config == masker_configs[0]
        assert research_attention.maskers[1].config == masker_configs[1]

    def test_efficient_attention_factory_method(self):
        """Test that EfficientAttention.create_from_config works with different configs."""
        from sparse_attention_hub.sparse_attention import (
            DoubleSparsity,
            DoubleSparsityConfig,
            EfficientAttention,
            HashAttention,
            HashAttentionConfig,
        )

        # Test with DoubleSparsityConfig
        double_sparsity_config = DoubleSparsityConfig(
            heavy_size=0.3,
            sink_size=10,
            local_size=5,
            ds_channel_config="config_file.json",
            ds_bits=8,
            ds_group_factor=4.0,
        )

        double_sparsity = EfficientAttention.create_from_config(double_sparsity_config)
        assert isinstance(double_sparsity, DoubleSparsity)

        # Test with HashAttentionConfig
        hash_attention_config = HashAttentionConfig(
            heavy_size=0.4,
            sink_size=10,
            local_size=5,
            hat_weights="weights.pt",
            hat_bits=8,
            hat_mlp_layers=2,
            hat_mlp_hidden_size=64,
        )

        hash_attention = EfficientAttention.create_from_config(hash_attention_config)
        assert isinstance(hash_attention, HashAttention)

    def test_config_inheritance_hierarchy(self):
        """Test that config inheritance hierarchy is correct."""
        from sparse_attention_hub.sparse_attention import (
            DoubleSparsityConfig,
            EfficientAttentionConfig,
            HashAttentionConfig,
            ResearchAttentionConfig,
            SparseAttentionConfig,
        )

        # Test inheritance hierarchy
        assert issubclass(EfficientAttentionConfig, SparseAttentionConfig)
        assert issubclass(ResearchAttentionConfig, SparseAttentionConfig)
        assert issubclass(DoubleSparsityConfig, EfficientAttentionConfig)
        assert issubclass(HashAttentionConfig, EfficientAttentionConfig)
