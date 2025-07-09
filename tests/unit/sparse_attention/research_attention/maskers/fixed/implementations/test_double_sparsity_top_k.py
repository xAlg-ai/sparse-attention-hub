"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for DoubleSparsityTopKMasker implementation.
"""

import pytest
import torch


@pytest.mark.unit
class TestDoubleSparsityTopKMaskerImplementation:
    def test_double_sparsity_top_k_masker_config_creation(self):
        """Test that double sparsity top k masker config can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMaskerConfig,
        )

        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=10, group_factor=4, label_bits=4, channel_config="auto"
        )
        assert config is not None
        assert config.heavy_size == 10
        assert config.group_factor == 4
        assert config.label_bits == 4
        assert config.channel_config == "auto"

    def test_double_sparsity_top_k_masker_creation(self):
        """Test that double sparsity top k masker can be created."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
        )

        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=10, group_factor=4, label_bits=4, channel_config="auto"
        )
        masker = DoubleSparsityTopKMasker(config)
        assert type(masker) is DoubleSparsityTopKMasker
        assert masker.config == config

    def test_double_sparsity_top_k_masker_creation_from_config(self):
        """Test that double sparsity top k masker can be created from a config."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
        )

        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=10, group_factor=4, label_bits=4, channel_config="auto"
        )
        masker = DoubleSparsityTopKMasker.create_from_config(config)
        assert type(masker) is DoubleSparsityTopKMasker
        assert masker.config == config

    def test_double_sparsity_top_k_masker_inheritance(self):
        """Test that double sparsity top k masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMasker,
        )

        assert issubclass(DoubleSparsityTopKMasker, TopKMasker)

    def test_double_sparsity_top_k_masker_config_inheritance(self):
        """Test that double sparsity top k masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            DoubleSparsityTopKMaskerConfig,
        )

        assert issubclass(DoubleSparsityTopKMaskerConfig, TopKMaskerConfig) 