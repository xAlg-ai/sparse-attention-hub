"""Unit tests for the HuggingFace ModelServer implementation."""

import gc
from typing import Any, Dict, Optional
from unittest.mock import Mock, call, patch

import pytest
import torch

from sparse_attention_hub.adapters.model_servers.base import ModelServer
from sparse_attention_hub.adapters.model_servers.huggingface import ModelServerHF
from sparse_attention_hub.adapters.utils.config import ModelServerConfig
from sparse_attention_hub.adapters.utils.exceptions import (
    ModelCreationError,
    ResourceCleanupError,
    TokenizerCreationError,
)


@pytest.mark.unit
class TestModelServerHF:
    """Test the ModelServerHF class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Reset singleton before each test
        ModelServer._instance = None

    def teardown_method(self) -> None:
        """Clean up after each test."""
        # Reset singleton after each test
        ModelServer._instance = None

    def test_inheritance(self) -> None:
        """Test that ModelServerHF properly inherits from ModelServer."""
        server = ModelServerHF()
        
        assert isinstance(server, ModelServer)
        assert isinstance(server, ModelServerHF)

    def test_singleton_behavior(self) -> None:
        """Test singleton behavior of ModelServerHF."""
        server1 = ModelServerHF()
        server2 = ModelServerHF()
        
        assert server1 is server2

    def test_initialization_with_config(self) -> None:
        """Test initialization with custom config."""
        config = ModelServerConfig(
            delete_on_zero_reference=True,
            enable_stats_logging=False
        )
        server = ModelServerHF(config)
        
        assert server.config == config


@pytest.mark.unit
class TestModelCreation:
    """Test HuggingFace model creation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ModelServerHF()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    @patch('torch.cuda.is_available')
    def test_create_model_gpu(self, mock_cuda_available, mock_auto_model) -> None:
        """Test creating model on GPU."""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Test model creation
        kwargs = {"torch_dtype": torch.float16}
        model = self.server._create_model("gpt2", 0, kwargs)
        
        # Verify calls
        mock_auto_model.from_pretrained.assert_called_once_with("gpt2", torch_dtype=torch.float16)
        mock_model.to.assert_called_once_with("cuda:0")
        assert model == mock_model

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    def test_create_model_cpu(self, mock_auto_model) -> None:
        """Test creating model on CPU."""
        # Setup mocks
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Test model creation with CPU (None gpu_id)
        kwargs = {"torch_dtype": torch.float32}
        model = self.server._create_model("bert-base", None, kwargs)
        
        # Verify calls
        mock_auto_model.from_pretrained.assert_called_once_with("bert-base", torch_dtype=torch.float32)
        mock_model.to.assert_called_once_with("cpu")
        assert model == mock_model

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    @patch('torch.cuda.is_available')
    def test_create_model_fallback_to_cpu(self, mock_cuda_available, mock_auto_model) -> None:
        """Test fallback to CPU when CUDA not available."""
        # Setup mocks
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Test model creation - should fallback to CPU
        kwargs = {"torch_dtype": torch.float16}
        model = self.server._create_model("gpt2", 0, kwargs)  # Request GPU 0
        
        # Should place on CPU instead
        mock_model.to.assert_called_once_with("cpu")
        assert model == mock_model

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    def test_create_model_failure(self, mock_auto_model) -> None:
        """Test model creation failure handling."""
        # Setup mock to raise exception
        mock_auto_model.from_pretrained.side_effect = ValueError("Model not found")
        
        # Test that exception is properly wrapped
        with pytest.raises(ModelCreationError) as exc_info:
            self.server._create_model("nonexistent-model", 0, {})
        
        assert exc_info.value.model_name == "nonexistent-model"
        assert exc_info.value.gpu_id == 0
        assert isinstance(exc_info.value.original_error, ValueError)
        assert "Model not found" in str(exc_info.value.original_error)

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    def test_create_model_empty_kwargs(self, mock_auto_model) -> None:
        """Test creating model with empty kwargs."""
        # Setup mocks
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Test with empty kwargs
        model = self.server._create_model("gpt2", None, {})
        
        # Should still work
        mock_auto_model.from_pretrained.assert_called_once_with("gpt2")
        mock_model.to.assert_called_once_with("cpu")

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    def test_create_model_complex_kwargs(self, mock_auto_model) -> None:
        """Test creating model with complex kwargs."""
        # Setup mocks
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Test with complex kwargs
        kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "use_cache": True,
            "attn_implementation": "flash_attention_2"
        }
        model = self.server._create_model("llama-7b", None, kwargs)
        
        # Verify all kwargs passed through
        mock_auto_model.from_pretrained.assert_called_once_with("llama-7b", **kwargs)


@pytest.mark.unit
class TestTokenizerCreation:
    """Test HuggingFace tokenizer creation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ModelServerHF()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer')
    def test_create_tokenizer_basic(self, mock_auto_tokenizer) -> None:
        """Test basic tokenizer creation."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<EOS>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Test tokenizer creation
        kwargs = {"padding_side": "left"}
        tokenizer = self.server._create_tokenizer("gpt2", kwargs)
        
        # Verify calls
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("gpt2", padding_side="left")
        assert tokenizer == mock_tokenizer
        assert tokenizer.pad_token == "<EOS>"  # Should be set to eos_token

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer')
    def test_create_tokenizer_existing_pad_token(self, mock_auto_tokenizer) -> None:
        """Test tokenizer creation when pad_token already exists."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<PAD>"
        mock_tokenizer.eos_token = "<EOS>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Test tokenizer creation
        tokenizer = self.server._create_tokenizer("bert-base", {})
        
        # pad_token should remain unchanged
        assert tokenizer.pad_token == "<PAD>"

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer')
    def test_create_tokenizer_failure(self, mock_auto_tokenizer) -> None:
        """Test tokenizer creation failure handling."""
        # Setup mock to raise exception
        mock_auto_tokenizer.from_pretrained.side_effect = OSError("Tokenizer not found")
        
        # Test that exception is properly wrapped
        with pytest.raises(TokenizerCreationError) as exc_info:
            self.server._create_tokenizer("nonexistent-tokenizer", {})
        
        assert exc_info.value.tokenizer_name == "nonexistent-tokenizer"
        assert isinstance(exc_info.value.original_error, OSError)
        assert "Tokenizer not found" in str(exc_info.value.original_error)

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer')
    def test_create_tokenizer_complex_kwargs(self, mock_auto_tokenizer) -> None:
        """Test creating tokenizer with complex kwargs."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<PAD>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Test with complex kwargs
        kwargs = {
            "padding_side": "left",
            "truncation": True,
            "max_length": 512,
            "use_fast": True
        }
        tokenizer = self.server._create_tokenizer("gpt2", kwargs)
        
        # Verify all kwargs passed through
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("gpt2", **kwargs)


@pytest.mark.unit
class TestModelDeletion:
    """Test HuggingFace model deletion functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ModelServerHF()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.cleanup_gpu_memory')
    @patch('sparse_attention_hub.adapters.model_servers.huggingface.gc.collect')
    @patch('torch.cuda.is_available')
    def test_delete_model_gpu(self, mock_cuda_available, mock_gc_collect, mock_cleanup_gpu) -> None:
        """Test deleting model from GPU."""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_model = Mock()
        
        # Test model deletion
        self.server._delete_model(mock_model, 0)
        
        # Verify calls
        mock_model.to.assert_called_once_with("cpu")  # Move to CPU first
        mock_gc_collect.assert_called_once()
        mock_cleanup_gpu.assert_called_once_with(0)

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.gc.collect')
    def test_delete_model_cpu(self, mock_gc_collect) -> None:
        """Test deleting model from CPU."""
        mock_model = Mock()
        
        # Test model deletion (CPU - gpu_id is None)
        self.server._delete_model(mock_model, None)
        
        # Should only call garbage collection, no GPU cleanup
        mock_gc_collect.assert_called_once()
        # Model should not be moved (already on CPU)
        mock_model.to.assert_not_called()

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.cleanup_gpu_memory')
    @patch('sparse_attention_hub.adapters.model_servers.huggingface.gc.collect')
    @patch('torch.cuda.is_available')
    def test_delete_model_cuda_not_available(self, mock_cuda_available, mock_gc_collect, mock_cleanup_gpu) -> None:
        """Test deleting model when CUDA not available."""
        # Setup mocks
        mock_cuda_available.return_value = False
        mock_model = Mock()
        
        # Test model deletion
        self.server._delete_model(mock_model, 0)  # Request GPU cleanup but CUDA not available
        
        # Should still do garbage collection but no GPU cleanup
        mock_gc_collect.assert_called_once()
        mock_cleanup_gpu.assert_not_called()

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.cleanup_gpu_memory')
    @patch('sparse_attention_hub.adapters.model_servers.huggingface.gc.collect')
    @patch('torch.cuda.is_available')
    def test_delete_model_cleanup_failure(self, mock_cuda_available, mock_gc_collect, mock_cleanup_gpu) -> None:
        """Test handling of GPU cleanup failure."""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_cleanup_gpu.side_effect = RuntimeError("GPU cleanup failed")
        mock_model = Mock()
        
        # Should handle cleanup failure gracefully (just log warning)
        self.server._delete_model(mock_model, 0)
        
        # Other cleanup should still happen
        mock_gc_collect.assert_called_once()

    def test_delete_model_to_cpu_failure(self) -> None:
        """Test handling when model.to(cpu) fails."""
        mock_model = Mock()
        mock_model.to.side_effect = RuntimeError("Failed to move to CPU")
        
        # Should handle the failure and continue with deletion
        with patch('torch.cuda.is_available', return_value=True):
            with patch('sparse_attention_hub.adapters.model_servers.huggingface.gc.collect'):
                # Should not raise exception
                self.server._delete_model(mock_model, 0)

    def test_delete_model_generic_failure(self) -> None:
        """Test handling of generic deletion failure."""
        mock_model = Mock()
        
        # Mock gc.collect to raise exception
        with patch('sparse_attention_hub.adapters.model_servers.huggingface.gc.collect', side_effect=MemoryError("Out of memory")):
            with pytest.raises(ResourceCleanupError) as exc_info:
                self.server._delete_model(mock_model, 0)
            
            assert exc_info.value.resource_type == "model"
            assert isinstance(exc_info.value.original_error, MemoryError)


@pytest.mark.unit
class TestTokenizerDeletion:
    """Test HuggingFace tokenizer deletion functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ModelServerHF()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.gc.collect')
    def test_delete_tokenizer_basic(self, mock_gc_collect) -> None:
        """Test basic tokenizer deletion."""
        mock_tokenizer = Mock()
        
        # Test tokenizer deletion
        self.server._delete_tokenizer(mock_tokenizer)
        
        # Verify garbage collection called
        mock_gc_collect.assert_called_once()

    def test_delete_tokenizer_failure(self) -> None:
        """Test handling of tokenizer deletion failure."""
        mock_tokenizer = Mock()
        
        # Mock gc.collect to raise exception
        with patch('sparse_attention_hub.adapters.model_servers.huggingface.gc.collect', side_effect=RuntimeError("GC failed")):
            with pytest.raises(ResourceCleanupError) as exc_info:
                self.server._delete_tokenizer(mock_tokenizer)
            
            assert exc_info.value.resource_type == "tokenizer"
            assert isinstance(exc_info.value.original_error, RuntimeError)


@pytest.mark.unit
class TestHuggingFaceSpecificMethods:
    """Test HuggingFace-specific utility methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ModelServerHF()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    def test_get_model_with_device_info(self, mock_auto_model) -> None:
        """Test getting model with device information."""
        # Setup mocks
        mock_model = Mock()
        mock_model.device = "cuda:0"
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Test getting model with device info
        kwargs = {"torch_dtype": torch.float16}
        model, device_info = self.server.get_model_with_device_info("gpt2", 0, kwargs)
        
        assert model == mock_model
        assert device_info == "cuda:0"

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    @patch('torch.cuda.is_available')
    def test_get_model_with_device_info_fallback(self, mock_cuda_available, mock_auto_model) -> None:
        """Test device info when model doesn't have device attribute."""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_model = Mock(spec=['to'])  # Include 'to' method in spec
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Test getting model with device info
        kwargs = {"torch_dtype": torch.float16}
        model, device_info = self.server.get_model_with_device_info("gpt2", 0, kwargs)
        
        assert model == mock_model
        assert device_info == "cuda:0"  # Should infer from gpu_id

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_validate_gpu_availability_success(self, mock_device_count, mock_cuda_available) -> None:
        """Test successful GPU validation."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 4
        
        with patch('torch.cuda.device'), patch('torch.zeros') as mock_zeros, patch('torch.cuda.empty_cache'):
            mock_zeros.return_value = Mock()
            result = self.server.validate_gpu_availability(0)
        
        assert result is True

    @patch('torch.cuda.is_available')
    def test_validate_gpu_availability_no_cuda(self, mock_cuda_available) -> None:
        """Test GPU validation when CUDA not available."""
        mock_cuda_available.return_value = False
        
        result = self.server.validate_gpu_availability(0)
        
        assert result is False

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_validate_gpu_availability_invalid_gpu(self, mock_device_count, mock_cuda_available) -> None:
        """Test GPU validation with invalid GPU ID."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2  # Only 2 GPUs
        
        result = self.server.validate_gpu_availability(5)  # Request GPU 5
        
        assert result is False

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_validate_gpu_availability_access_failure(self, mock_device_count, mock_cuda_available) -> None:
        """Test GPU validation when access fails."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 4
        
        with patch('torch.cuda.device', side_effect=RuntimeError("GPU access failed")):
            result = self.server.validate_gpu_availability(0)
        
        assert result is False

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    def test_get_gpu_memory_info_success(self, mock_get_props, mock_memory_reserved, mock_memory_allocated, mock_cuda_available) -> None:
        """Test successful GPU memory info retrieval."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 1024**3  # 1 GB
        mock_memory_reserved.return_value = 2 * 1024**3  # 2 GB
        
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024**3  # 8 GB
        mock_get_props.return_value = mock_props
        
        info = self.server.get_gpu_memory_info(0)
        
        assert info["allocated_gb"] == 1.0
        assert info["reserved_gb"] == 2.0
        assert info["total_gb"] == 8.0
        assert info["free_gb"] == 7.0

    @patch('torch.cuda.is_available')
    def test_get_gpu_memory_info_no_cuda(self, mock_cuda_available) -> None:
        """Test GPU memory info when CUDA not available."""
        mock_cuda_available.return_value = False
        
        info = self.server.get_gpu_memory_info(0)
        
        assert "error" in info
        assert info["error"] == "CUDA not available"

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_get_gpu_memory_info_failure(self, mock_memory_allocated, mock_cuda_available) -> None:
        """Test GPU memory info when access fails."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.side_effect = RuntimeError("GPU memory access failed")
        
        info = self.server.get_gpu_memory_info(0)
        
        assert "error" in info
        assert "GPU memory access failed" in info["error"]


@pytest.mark.unit
class TestIntegration:
    """Test integration scenarios with ModelServerHF."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer')
    def test_full_model_lifecycle(self, mock_auto_tokenizer, mock_auto_model) -> None:
        """Test complete model lifecycle with ModelServerHF."""
        # Setup mocks
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<EOS>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        server = ModelServerHF()
        
        # Create model and tokenizer
        model1 = server.get_model("gpt2", 0, {"torch_dtype": torch.float16})
        model2 = server.get_model("gpt2", 0, {"torch_dtype": torch.float16})  # Same model
        tokenizer = server.get_tokenizer("gpt2", {"padding_side": "left"})
        
        # Verify same instances
        assert model1 is model2
        
        # Check stats
        model_stats = server.get_model_stats()
        assert model_stats["total_models"] == 1
        assert model_stats["total_references"] == 2
        
        tokenizer_stats = server.get_tokenizer_stats()
        assert tokenizer_stats["total_tokenizers"] == 1
        assert tokenizer_stats["total_references"] == 1
        
        # Clear references
        server.clear_model("gpt2", 0, {"torch_dtype": torch.float16})
        server.clear_model("gpt2", 0, {"torch_dtype": torch.float16})
        server.clear_tokenizer("gpt2", {"padding_side": "left"})
        
        # Clean up
        cleanup_stats = server.cleanup_unused()
        assert cleanup_stats["models"]["deleted"] == 1
        assert cleanup_stats["tokenizers"]["deleted"] == 1

    @patch('sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM')
    def test_config_behavior_integration(self, mock_auto_model) -> None:
        """Test configuration behavior in practice."""
        # Setup mocks
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Test aggressive deletion
        config = ModelServerConfig(delete_on_zero_reference=True)
        server = ModelServerHF(config)
        
        # Create and immediately clear model
        server.get_model("test-model", 0, {})
        server.clear_model("test-model", 0, {})
        
        # Should be immediately deleted
        assert len(server._models) == 0
        
        # Reset for lazy deletion test
        ModelServer._instance = None
        config = ModelServerConfig(delete_on_zero_reference=False)
        server = ModelServerHF(config)
        
        # Create and clear model
        server.get_model("test-model", 0, {})
        server.clear_model("test-model", 0, {})
        
        # Should still exist with zero references
        assert len(server._models) == 1
        model_stats = server.get_model_stats()
        assert model_stats["zero_reference_models"] == 1
