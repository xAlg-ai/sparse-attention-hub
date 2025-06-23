"""Unit tests for sparse attention base classes."""

from unittest.mock import Mock

import pytest

from sparse_attention_hub.sparse_attention.base import (
    EfficientAttention,
    ResearchAttention,
    SparseAttention,
)
from sparse_attention_hub.sparse_attention.metadata import SparseAttentionMetadata


class TestSparseAttentionMetadata:
    """Test cases for SparseAttentionMetadata."""

    def test_initialization(self):
        """Test metadata initialization."""
        metadata = SparseAttentionMetadata()
        assert isinstance(metadata.layer_wise_state, dict)
        assert isinstance(metadata.global_state_, dict)
        assert len(metadata.layer_wise_state) == 0
        assert len(metadata.global_state_) == 0

    def test_layer_state_operations(self):
        """Test layer state operations."""
        metadata = SparseAttentionMetadata()

        # Test update and get
        test_state = {"param1": "value1", "param2": 42}
        metadata.update_layer_state("layer_1", test_state)

        retrieved_state = metadata.get_layer_state("layer_1")
        assert retrieved_state == test_state

        # Test non-existent layer
        empty_state = metadata.get_layer_state("non_existent")
        assert empty_state == {}

    def test_global_state_operations(self):
        """Test global state operations."""
        metadata = SparseAttentionMetadata()

        # Test update and get
        test_state = {"global_param": "global_value"}
        metadata.update_global_state(test_state)

        retrieved_state = metadata.get_global_state()
        assert retrieved_state == test_state

        # Test multiple updates
        additional_state = {"another_param": 123}
        metadata.update_global_state(additional_state)

        final_state = metadata.get_global_state()
        assert "global_param" in final_state
        assert "another_param" in final_state


class ConcreteSparseAttention(SparseAttention):
    """Concrete implementation for testing."""

    def custom_attention(self):
        return Mock(), None


class ConcreteEfficientAttention(EfficientAttention):
    """Concrete implementation for testing."""

    def custom_attention(self):
        return Mock(), None


class ConcreteResearchAttention(ResearchAttention):
    """Concrete implementation for testing."""

    def custom_attention(self):
        return Mock(), None


class TestSparseAttentionBase:
    """Test cases for base sparse attention classes."""

    def test_sparse_attention_initialization(self):
        """Test SparseAttention initialization."""
        attention = ConcreteSparseAttention()
        assert hasattr(attention, "metadata")
        assert isinstance(attention.metadata, SparseAttentionMetadata)

    def test_sparse_attention_hook_generator(self):
        """Test pre_attention_hook_generator method."""
        attention = ConcreteSparseAttention()
        # Should not raise an exception
        attention.pre_attention_hook_generator()

    def test_efficient_attention_inheritance(self):
        """Test EfficientAttention inheritance."""
        attention = ConcreteEfficientAttention()
        assert isinstance(attention, SparseAttention)
        assert hasattr(attention, "metadata")

    def test_research_attention_initialization(self):
        """Test ResearchAttention initialization."""
        attention = ConcreteResearchAttention()
        assert isinstance(attention, SparseAttention)
        assert hasattr(attention, "masks")
        assert isinstance(attention.masks, (list, tuple))
        assert len(attention.masks) == 0
