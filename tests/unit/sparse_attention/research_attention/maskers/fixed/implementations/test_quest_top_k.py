"""Tests for Quest Top-K masker implementation."""

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.quest_top_k import (
    QuestTopKMasker,
    QuestTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.common_utils import (
    pseudo_quantize,
)


@pytest.mark.unit
class TestQuestTopKMaskerImplementation:
    """Test class for QuestTopKMasker implementation."""

    def test_quest_top_k_masker_config_creation(self):
        """Test that quest top k masker config can be created."""
        config = QuestTopKMaskerConfig(
            heavy_size=0.05,
            page_size=128,
        )
        assert config is not None
        assert config.heavy_size == 0.05
        assert config.page_size == 128
        assert config.label_bits == 16  # Default value

    def test_quest_top_k_masker_config_with_custom_label_bits(self):
        """Test config creation with custom label_bits."""
        config = QuestTopKMaskerConfig(
            heavy_size=0.05,
            page_size=128,
            label_bits=4,
        )
        assert config.label_bits == 4

    def test_quest_top_k_masker_config_validation(self):
        """Test config validation raises appropriate errors."""
        # Test invalid label_bits
        with pytest.raises(ValueError, match="label_bits must be in range \\(0, 16\\]"):
            QuestTopKMaskerConfig(
                heavy_size=0.05,
                page_size=128,
                label_bits=17,
            )

        with pytest.raises(ValueError, match="label_bits must be in range \\(0, 16\\]"):
            QuestTopKMaskerConfig(
                heavy_size=0.05,
                page_size=128,
                label_bits=0,
            )

    def test_quest_top_k_masker_creation(self):
        """Test that quest top k masker can be created."""
        config = QuestTopKMaskerConfig(
            heavy_size=0.05,
            page_size=128,
        )
        masker = QuestTopKMasker.create_from_config(config)
        assert isinstance(masker, QuestTopKMasker)
        assert masker.config == config

    def test_quest_top_k_masker_pseudo_quantization(self):
        """Test pseudo-quantization functionality."""
        # Create test tensor with shape [B, H, P, D] similar to page_min/page_max
        test_tensor = torch.randn(2, 8, 16, 64)
        quantized_tensor = pseudo_quantize(test_tensor, 4)

        # Shape should remain the same
        assert quantized_tensor.shape == test_tensor.shape

        # Values should be quantized (different from original)
        assert not torch.allclose(quantized_tensor, test_tensor)

        # Test with different quantization bits
        quantized_tensor_8 = pseudo_quantize(test_tensor, 8)
        assert quantized_tensor_8.shape == test_tensor.shape
        assert not torch.allclose(quantized_tensor_8, test_tensor)

        # With label_bits=16, quantization should not be applied in the masker
        # but we can test the function directly
        quantized_tensor_16 = pseudo_quantize(test_tensor, 16)
        assert quantized_tensor_16.shape == test_tensor.shape

    def test_quest_top_k_masker_with_label_bits_16(self):
        """Test that masker works correctly with label_bits=16 (no quantization)."""
        config = QuestTopKMaskerConfig(
            heavy_size=0.05,
            page_size=128,
            label_bits=16,
        )
        masker = QuestTopKMasker.create_from_config(config)
        assert masker.label_bits == 16

    def test_quest_top_k_masker_with_label_bits_less_than_16(self):
        """Test that masker works correctly with label_bits < 16 (with quantization)."""
        config = QuestTopKMaskerConfig(
            heavy_size=0.05,
            page_size=128,
            label_bits=4,
        )
        masker = QuestTopKMasker.create_from_config(config)
        assert masker.label_bits == 4
