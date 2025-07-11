"""Test imports for the restructured sparse attention module."""

import pytest


@pytest.mark.unit
class TestSparseAttentionImports:
    """Test that all sparse attention classes can be imported correctly."""

    def test_base_classes_import(self):
        """Test that base classes can be imported."""
        from sparse_attention_hub.sparse_attention import (
            EfficientAttention,
            ResearchAttention,
            SparseAttention,
        )

        assert SparseAttention is not None
        assert EfficientAttention is not None
        assert ResearchAttention is not None

    def test_config_classes_import(self):
        """Test that config classes can be imported."""
        from sparse_attention_hub.sparse_attention import (
            ChannelConfig,
            DoubleSparsityConfig,
            DoubleSparsityTopKMaskerConfig,
            EfficientAttentionConfig,
            HashAttentionConfig,
            HashAttentionTopKMaskerConfig,
            ResearchAttentionConfig,
            SparseAttentionConfig,
        )

        assert SparseAttentionConfig is not None
        assert EfficientAttentionConfig is not None
        assert ResearchAttentionConfig is not None
        assert DoubleSparsityConfig is not None
        assert HashAttentionConfig is not None
        assert DoubleSparsityTopKMaskerConfig is not None
        assert HashAttentionTopKMaskerConfig is not None
        assert ChannelConfig is not None

    def test_efficient_attention_implementations_import(self):
        """Test that efficient attention implementations can be imported."""
        from sparse_attention_hub.sparse_attention import DoubleSparsity, HashAttention

        assert DoubleSparsity is not None
        assert HashAttention is not None

    def test_research_masker_base_classes_import(self):
        """Test that research masker base classes can be imported."""
        from sparse_attention_hub.sparse_attention import (
            FixedMasker,
            ResearchMasker,
            SamplingMasker,
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
            CausalMasker,
            DoubleSparsityTopKMasker,
            HashAttentionTopKMasker,
            LocalMasker,
            OracleTopK,
            PQCache,
            SinkMasker,
        )

        assert LocalMasker is not None
        assert CausalMasker is not None
        assert SinkMasker is not None
        assert OracleTopK is not None
        assert PQCache is not None
        assert HashAttentionTopKMasker is not None
        assert DoubleSparsityTopKMasker is not None

    def test_sampling_masker_implementations_import(self):
        """Test that sampling masker implementations can be imported."""
        from sparse_attention_hub.sparse_attention import MagicPig, RandomSamplingMasker

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
        from sparse_attention_hub.sparse_attention import Mask, SparseAttentionMetadata

        assert Mask is not None
        assert SparseAttentionMetadata is not None

    def test_all_imports_together(self):
        """Test that all classes can be imported together."""
        from sparse_attention_hub.sparse_attention import (  # Base classes; Config classes; Efficient implementations; Research masker base classes; Fixed masker implementations; Sampling masker implementations; Generator and integration; Utility classes
            CausalMasker,
            ChannelConfig,
            DoubleSparsity,
            DoubleSparsityConfig,
            DoubleSparsityTopKMasker,
            DoubleSparsityTopKMaskerConfig,
            EfficientAttention,
            EfficientAttentionConfig,
            FixedMasker,
            HashAttention,
            HashAttentionConfig,
            HashAttentionTopKMasker,
            HashAttentionTopKMaskerConfig,
            LocalMasker,
            MagicPig,
            Mask,
            OracleTopK,
            PQCache,
            RandomSamplingMasker,
            ResearchAttention,
            ResearchAttentionConfig,
            ResearchMasker,
            SamplingMasker,
            SinkMasker,
            SparseAttention,
            SparseAttentionConfig,
            SparseAttentionGen,
            SparseAttentionHF,
            SparseAttentionMetadata,
            TopKMasker,
            TopPMasker,
        )

        # Verify all classes are imported
        classes = [
            SparseAttention,
            EfficientAttention,
            ResearchAttention,
            SparseAttentionConfig,
            EfficientAttentionConfig,
            ResearchAttentionConfig,
            DoubleSparsityConfig,
            HashAttentionConfig,
            DoubleSparsityTopKMaskerConfig,
            HashAttentionTopKMaskerConfig,
            ChannelConfig,
            DoubleSparsity,
            HashAttention,
            ResearchMasker,
            SamplingMasker,
            FixedMasker,
            TopKMasker,
            TopPMasker,
            LocalMasker,
            CausalMasker,
            SinkMasker,
            OracleTopK,
            PQCache,
            HashAttentionTopKMasker,
            DoubleSparsityTopKMasker,
            RandomSamplingMasker,
            MagicPig,
            SparseAttentionGen,
            SparseAttentionHF,
            Mask,
            SparseAttentionMetadata,
        ]

        for cls in classes:
            assert cls is not None

    def test_inheritance_hierarchy(self):
        """Test that the inheritance hierarchy is correct."""
        from sparse_attention_hub.sparse_attention import (
            EfficientAttention,
            EfficientAttentionConfig,
            FixedMasker,
            ResearchAttention,
            ResearchAttentionConfig,
            SamplingMasker,
            SparseAttention,
            SparseAttentionConfig,
            TopKMasker,
            TopPMasker,
        )
        from sparse_attention_hub.sparse_attention.research_attention.maskers import (
            ResearchMasker,
        )

        # Test inheritance hierarchy
        assert issubclass(EfficientAttention, SparseAttention)
        assert issubclass(ResearchAttention, SparseAttention)
        assert issubclass(FixedMasker, ResearchMasker)
        assert issubclass(TopKMasker, FixedMasker)
        assert issubclass(TopPMasker, FixedMasker)
        assert issubclass(SamplingMasker, ResearchMasker)

        # Test config inheritance hierarchy
        assert issubclass(EfficientAttentionConfig, SparseAttentionConfig)
        assert issubclass(ResearchAttentionConfig, SparseAttentionConfig)

    def test_module_structure(self):
        """Test that the module structure is correct."""
        import sparse_attention_hub.sparse_attention as sa

        # Test that submodules exist
        assert hasattr(sa, "efficient_attention")
        assert hasattr(sa, "research_attention")
        assert hasattr(sa, "integrations")
        assert hasattr(sa, "utils")

        # Test that research_attention has maskers
        assert hasattr(sa.research_attention, "maskers")
        assert hasattr(sa.research_attention.maskers, "fixed")
        assert hasattr(sa.research_attention.maskers, "sampling")
