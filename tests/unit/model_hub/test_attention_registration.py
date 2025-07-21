"""Tests for attention registration functionality in ModelHubHF."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Optional, Tuple
import torch

from sparse_attention_hub.model_hub import (
    ModelHubHF,
    BaseAttentionFunction,
    SparseAttentionAdapter,
    ResearchAttentionAdapter,
    EfficientAttentionAdapter,
)
from sparse_attention_hub.sparse_attention import (
    ResearchAttention,
    EfficientAttention,
    SparseAttention,
    SparseAttentionConfig,
    ResearchAttentionConfig,
    EfficientAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMasker,
    LocalMaskerConfig,
)


class MockSparseAttention(SparseAttention):
    """Mock implementation of SparseAttention for testing."""
    
    def custom_attention(
        self,
        module: Any,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Mock attention implementation."""
        # Simple mock attention computation
        batch_size, num_heads, seq_len, head_dim = queries.shape
        if scaling is None:
            scaling = 1.0 / (head_dim ** 0.5)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)
        
        return attn_output, attn_weights


@pytest.mark.unit
class TestModelHubHFAttentionRegistration:
    """Test class for ModelHubHF attention registration functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.model_hub = ModelHubHF()

    def test_model_hub_hf_initialization(self):
        """Test that ModelHubHF initializes correctly."""
        assert self.model_hub is not None
        assert self.model_hub._registered_attention_functions == {}
        assert self.model_hub._model_attention_configs == {}

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_register_research_attention(self):
        """Test registering a ResearchAttention implementation."""
        # Create a simple research attention
        local_config = LocalMaskerConfig(window_size=4)
        research_config = ResearchAttentionConfig(masker_configs=[local_config])
        research_attention = ResearchAttention.create_from_config(research_config)

        # Register the attention
        attention_name = self.model_hub.register_sparse_attention(
            research_attention, 
            attention_name="test_research_attention"
        )

        assert attention_name == "test_research_attention"
        assert attention_name in self.model_hub._registered_attention_functions
        
        # Check that the adapter is of the correct type
        adapter = self.model_hub._registered_attention_functions[attention_name]
        assert isinstance(adapter, ResearchAttentionAdapter)

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_register_efficient_attention(self):
        """Test registering an EfficientAttention implementation."""
        # Create a simple efficient attention
        efficient_config = EfficientAttentionConfig()
        efficient_attention = EfficientAttention(efficient_config)

        # Register the attention
        attention_name = self.model_hub.register_sparse_attention(
            efficient_attention,
            attention_name="test_efficient_attention"
        )

        assert attention_name == "test_efficient_attention"
        assert attention_name in self.model_hub._registered_attention_functions
        
        # Check that the adapter is of the correct type
        adapter = self.model_hub._registered_attention_functions[attention_name]
        assert isinstance(adapter, EfficientAttentionAdapter)

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_register_base_sparse_attention(self):
        """Test registering a base SparseAttention implementation."""
        # Create a simple sparse attention
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)

        # Register the attention
        attention_name = self.model_hub.register_sparse_attention(
            sparse_attention,
            attention_name="test_sparse_attention"
        )

        assert attention_name == "test_sparse_attention"
        assert attention_name in self.model_hub._registered_attention_functions
        
        # Check that the adapter is of the correct type
        adapter = self.model_hub._registered_attention_functions[attention_name]
        assert isinstance(adapter, SparseAttentionAdapter)

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_register_with_default_name(self):
        """Test registering attention with default name."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)

        # Register without specifying name
        attention_name = self.model_hub.register_sparse_attention(sparse_attention)

        # Should use the default name from the adapter
        assert attention_name == "sparse_attention_adapter"
        assert attention_name in self.model_hub._registered_attention_functions

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_register_duplicate_name_fails(self):
        """Test that registering with duplicate name fails."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)

        # Register first time
        self.model_hub.register_sparse_attention(
            sparse_attention, 
            attention_name="duplicate_name"
        )

        # Try to register again with same name
        with pytest.raises(ValueError, match="already registered"):
            self.model_hub.register_sparse_attention(
                sparse_attention, 
                attention_name="duplicate_name"
            )

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_unregister_attention_function(self):
        """Test unregistering an attention function."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)

        # Register first
        attention_name = self.model_hub.register_sparse_attention(
            sparse_attention, 
            attention_name="test_unregister"
        )

        # Verify it's registered
        assert attention_name in self.model_hub._registered_attention_functions

        # Unregister
        self.model_hub.unregister_attention_function(attention_name)

        # Verify it's removed
        assert attention_name not in self.model_hub._registered_attention_functions

    def test_unregister_nonexistent_function_fails(self):
        """Test that unregistering non-existent function fails."""
        with pytest.raises(ValueError, match="is not registered"):
            self.model_hub.unregister_attention_function("nonexistent")

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_configure_model_attention(self):
        """Test configuring a model to use registered attention."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)

        # Register attention
        attention_name = self.model_hub.register_sparse_attention(
            sparse_attention, 
            attention_name="test_configure"
        )

        # Create mock model
        mock_model = MagicMock()
        mock_model.config.attn_implementation = None

        # Configure model
        self.model_hub.configure_model_attention(mock_model, attention_name)

        # Verify configuration
        assert mock_model.config.attn_implementation == attention_name
        model_id = str(id(mock_model))
        assert self.model_hub._model_attention_configs[model_id] == attention_name

    def test_configure_model_with_unregistered_attention_fails(self):
        """Test that configuring model with unregistered attention fails."""
        mock_model = MagicMock()
        
        with pytest.raises(ValueError, match="is not registered"):
            self.model_hub.configure_model_attention(mock_model, "unregistered")

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_list_registered_attention_functions(self):
        """Test listing registered attention functions."""
        # Initially empty
        registered = self.model_hub.list_registered_attention_functions()
        assert len(registered) == 0

        # Register some functions
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)
        
        local_config = LocalMaskerConfig(window_size=4)
        research_config = ResearchAttentionConfig(masker_configs=[local_config])
        research_attention = ResearchAttention.create_from_config(research_config)

        self.model_hub.register_sparse_attention(sparse_attention, "sparse_test")
        self.model_hub.register_sparse_attention(research_attention, "research_test")

        # Check listing
        registered = self.model_hub.list_registered_attention_functions()
        assert len(registered) == 2
        assert "sparse_test" in registered
        assert "research_test" in registered
        assert registered["sparse_test"] == "SparseAttentionAdapter"
        assert registered["research_test"] == "ResearchAttentionAdapter"

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_get_attention_adapter(self):
        """Test getting attention adapter by name."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)

        # Register attention
        attention_name = self.model_hub.register_sparse_attention(
            sparse_attention, 
            attention_name="test_get_adapter"
        )

        # Get adapter
        adapter = self.model_hub.get_attention_adapter(attention_name)
        assert isinstance(adapter, SparseAttentionAdapter)
        assert adapter.sparse_attention is sparse_attention

    def test_get_nonexistent_adapter_fails(self):
        """Test that getting non-existent adapter fails."""
        with pytest.raises(ValueError, match="is not registered"):
            self.model_hub.get_attention_adapter("nonexistent")

    def test_transformers_import_error_handling(self):
        """Test handling of transformers import error."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)

        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'transformers.modeling_utils': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'transformers'")):
                with pytest.raises(ImportError, match="transformers library is required"):
                    self.model_hub.register_sparse_attention(sparse_attention)

    @patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', {})
    def test_register_with_config(self):
        """Test registering attention with custom config."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)
        
        custom_config = {"custom_param": "custom_value"}

        # Register with config
        attention_name = self.model_hub.register_sparse_attention(
            sparse_attention,
            attention_name="test_with_config",
            config=custom_config
        )

        # Verify config is stored in adapter
        adapter = self.model_hub.get_attention_adapter(attention_name)
        assert adapter.config == custom_config

    def test_legacy_methods_show_deprecation_warnings(self):
        """Test that legacy methods show deprecation warnings."""
        mock_model = MagicMock()
        
        with patch('sparse_attention_hub.model_hub.huggingface.logger') as mock_logger:
            self.model_hub.addPreAttentionHooks(mock_model, lambda: None, "test")
            mock_logger.warning.assert_called_with(
                "addPreAttentionHooks is deprecated. Use register_sparse_attention instead."
            )

        with patch('sparse_attention_hub.model_hub.huggingface.logger') as mock_logger:
            self.model_hub.removePreAttentionHooks(mock_model, "test")
            mock_logger.warning.assert_called_with(
                "removePreAttentionHooks is deprecated. Use unregister_attention_function instead."
            )

        with patch('sparse_attention_hub.model_hub.huggingface.logger') as mock_logger:
            self.model_hub.replaceAttentionInterface(mock_model, lambda: None, "test")
            mock_logger.warning.assert_called_with(
                "replaceAttentionInterface is deprecated. Use register_sparse_attention "
                "and configure_model_attention instead."
            )

        with patch('sparse_attention_hub.model_hub.huggingface.logger') as mock_logger:
            self.model_hub.revertAttentionInterface(mock_model)
            mock_logger.warning.assert_called_with(
                "revertAttentionInterface is deprecated. Use configure_model_attention "
                "with a standard attention implementation instead."
            )


@pytest.mark.unit
class TestAttentionAdapters:
    """Test class for attention adapter functionality."""

    def test_sparse_attention_adapter_initialization(self):
        """Test SparseAttentionAdapter initialization."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)
        config = {"test": "value"}

        adapter = SparseAttentionAdapter(sparse_attention, config)

        assert adapter.sparse_attention is sparse_attention
        assert adapter.config == config

    def test_research_attention_adapter_name(self):
        """Test ResearchAttentionAdapter name."""
        assert ResearchAttentionAdapter.get_attention_name() == "research_attention"

    def test_efficient_attention_adapter_name(self):
        """Test EfficientAttentionAdapter name."""
        assert EfficientAttentionAdapter.get_attention_name() == "efficient_attention"

    def test_sparse_attention_adapter_name(self):
        """Test SparseAttentionAdapter name."""
        assert SparseAttentionAdapter.get_attention_name() == "sparse_attention_adapter"

    def test_adapter_modify_model_default(self):
        """Test that adapter modify_model returns False by default."""
        sparse_config = SparseAttentionConfig()
        sparse_attention = MockSparseAttention(sparse_config)
        adapter = SparseAttentionAdapter(sparse_attention, {})

        mock_model = MagicMock()
        result = adapter.modify_model(mock_model)
        assert result is False

    def test_adapter_validate_config_default(self):
        """Test that adapter validate_config returns True by default."""
        assert SparseAttentionAdapter.validate_config(None) is True
        assert ResearchAttentionAdapter.validate_config(None) is True
        assert EfficientAttentionAdapter.validate_config(None) is True