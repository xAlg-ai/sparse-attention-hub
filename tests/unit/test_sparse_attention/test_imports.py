"""Test imports for the restructured sparse attention module."""

import pytest

# Test wildcard import at module level
from sparse_attention_hub.sparse_attention import *

@pytest.mark.unit
class TestSparseAttentionImports:
    """Test that all sparse attention classes can be imported correctly."""

    def test_base_classes_import(self):
        """Test that base classes can be imported."""
        from sparse_attention_hub.sparse_attention import (
            SparseAttention,
            EfficientAttention,
            ResearchAttention,
        )
        
        assert SparseAttention is not None
        assert EfficientAttention is not None
        assert ResearchAttention is not None

    def test_efficient_attention_implementations_import(self):
        """Test that efficient attention implementations can be imported."""
        from sparse_attention_hub.sparse_attention import (
            DoubleSparsity,
            HashAttention,
        )
        
        assert DoubleSparsity is not None
        assert HashAttention is not None

    def test_research_masker_base_classes_import(self):
        """Test that research masker base classes can be imported."""
        from sparse_attention_hub.sparse_attention import (
            ResearchMasker,
            SamplingMasker,
            FixedMasker,
            TopKMasker,
            TopPMasker,
        )
        
        assert ResearchMasker is not None
        assert SamplingMasker is not None
        assert FixedMasker is not None
        assert TopKMasker is not None
        assert TopPMasker is not None

    def test_fixed_masker_implementations_import(self):
        """Test that fixed masker implementations can be imported."""
        from sparse_attention_hub.sparse_attention import (
            LocalMasker,
            CausalMasker,
            SinkMasker,
            OracleTopK,
            PQCache,
            RHashAttention,
            RDoubleSparsity,
        )
        
        assert LocalMasker is not None
        assert CausalMasker is not None
        assert SinkMasker is not None
        assert OracleTopK is not None
        assert PQCache is not None
        assert RHashAttention is not None
        assert RDoubleSparsity is not None

    def test_sampling_masker_implementations_import(self):
        """Test that sampling masker implementations can be imported."""
        from sparse_attention_hub.sparse_attention import (
            RandomSamplingMasker,
            MagicPig,
        )
        
        assert RandomSamplingMasker is not None
        assert MagicPig is not None

    def test_generator_and_integration_import(self):
        """Test that generator and integration classes can be imported."""
        from sparse_attention_hub.sparse_attention import (
            SparseAttentionGen,
            SparseAttentionHF,
        )
        
        assert SparseAttentionGen is not None
        assert SparseAttentionHF is not None

    def test_utility_classes_import(self):
        """Test that utility classes can be imported."""
        from sparse_attention_hub.sparse_attention import (
            Mask,
            SparseAttentionMetadata,
        )
        
        assert Mask is not None
        assert SparseAttentionMetadata is not None

    def test_all_imports_together(self):
        """Test that all classes can be imported together."""
        from sparse_attention_hub.sparse_attention import (
            # Base classes
            SparseAttention,
            EfficientAttention,
            ResearchAttention,
            # Efficient implementations
            DoubleSparsity,
            HashAttention,
            # Research masker base classes
            ResearchMasker,
            SamplingMasker,
            FixedMasker,
            TopKMasker,
            TopPMasker,
            # Fixed masker implementations
            LocalMasker,
            CausalMasker,
            SinkMasker,
            OracleTopK,
            PQCache,
            RHashAttention,
            RDoubleSparsity,
            # Sampling masker implementations
            RandomSamplingMasker,
            MagicPig,
            # Generator and integration
            SparseAttentionGen,
            SparseAttentionHF,
            # Utility classes
            Mask,
            SparseAttentionMetadata,
        )
        
        # Verify all classes are imported
        classes = [
            SparseAttention, EfficientAttention, ResearchAttention,
            DoubleSparsity, HashAttention,
            ResearchMasker, SamplingMasker, FixedMasker, TopKMasker, TopPMasker,
            LocalMasker, CausalMasker, SinkMasker, OracleTopK, PQCache, 
            RHashAttention, RDoubleSparsity,
            RandomSamplingMasker, MagicPig,
            SparseAttentionGen, SparseAttentionHF,
            Mask, SparseAttentionMetadata,
        ]
        
        for cls in classes:
            assert cls is not None

    def test_wildcard_import(self):
        """Test that wildcard import works correctly."""
        # Check that key classes are available from module-level import
        expected_classes = [
            'SparseAttention', 'EfficientAttention', 'ResearchAttention',
            'DoubleSparsity', 'HashAttention',
            'ResearchMasker', 'SamplingMasker', 'FixedMasker', 'TopKMasker', 'TopPMasker',
            'LocalMasker', 'CausalMasker', 'SinkMasker', 'OracleTopK', 'PQCache',
            'RHashAttention', 'RDoubleSparsity',
            'RandomSamplingMasker', 'MagicPig',
            'SparseAttentionGen', 'SparseAttentionHF',
            'Mask', 'SparseAttentionMetadata',
        ]
        
        for class_name in expected_classes:
            assert class_name in globals(), f"Class {class_name} not found in wildcard import"

    def test_inheritance_hierarchy(self):
        """Test that the inheritance hierarchy is correct."""
        from sparse_attention_hub.sparse_attention import (
            SparseAttention,
            EfficientAttention,
            ResearchAttention,
            FixedMasker,
            TopKMasker,
            TopPMasker,
            SamplingMasker,
        )
        
        # Test inheritance hierarchy
        assert issubclass(EfficientAttention, SparseAttention)
        assert issubclass(ResearchAttention, SparseAttention)
        assert issubclass(FixedMasker, ResearchMasker)
        assert issubclass(TopKMasker, FixedMasker)
        assert issubclass(TopPMasker, FixedMasker)
        assert issubclass(SamplingMasker, ResearchMasker)

    def test_module_structure(self):
        """Test that the module structure is correct."""
        import sparse_attention_hub.sparse_attention as sa
        
        # Test that submodules exist
        assert hasattr(sa, 'efficient_attention')
        assert hasattr(sa, 'research_attention')
        assert hasattr(sa, 'integrations')
        assert hasattr(sa, 'utils')
        
        # Test that research_attention has maskers
        assert hasattr(sa.research_attention, 'maskers')
        assert hasattr(sa.research_attention.maskers, 'fixed')
        assert hasattr(sa.research_attention.maskers, 'sampling') 