"""Unit tests for GPU utility functions."""

from unittest.mock import patch

import pytest

from sparse_attention_hub.adapters.utils.gpu_utils import cleanup_gpu_memory


@pytest.mark.unit
class TestGPUUtils:
    """Test GPU utility functions."""

    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.is_available')
    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.empty_cache')
    def test_cleanup_gpu_memory_no_gpu_id(self, mock_empty_cache, mock_is_available) -> None:
        """Test GPU memory cleanup without specific GPU ID."""
        mock_is_available.return_value = True
        
        cleanup_gpu_memory()
        
        mock_is_available.assert_called_once()
        mock_empty_cache.assert_called_once()

    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.is_available')
    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.device')
    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.empty_cache')
    def test_cleanup_gpu_memory_with_gpu_id(self, mock_empty_cache, mock_device, mock_is_available) -> None:
        """Test GPU memory cleanup with specific GPU ID."""
        mock_is_available.return_value = True
        
        cleanup_gpu_memory(gpu_id=0)
        
        mock_is_available.assert_called_once()
        mock_device.assert_called_once_with(0)
        mock_empty_cache.assert_called_once()

    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.is_available')
    def test_cleanup_gpu_memory_cuda_not_available(self, mock_is_available) -> None:
        """Test GPU memory cleanup when CUDA is not available."""
        mock_is_available.return_value = False
        
        # Should not raise any exception
        cleanup_gpu_memory()
        cleanup_gpu_memory(gpu_id=0)
        
        mock_is_available.assert_called()

    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.is_available')
    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.device')
    @patch('sparse_attention_hub.adapters.utils.gpu_utils.torch.cuda.empty_cache')
    def test_cleanup_gpu_memory_with_exception(self, mock_empty_cache, mock_device, mock_is_available) -> None:
        """Test GPU memory cleanup handles exceptions gracefully."""
        mock_is_available.return_value = True
        mock_device.side_effect = RuntimeError("GPU error")
        
        # Should not raise an exception - the function should handle GPU errors gracefully
        try:
            cleanup_gpu_memory(gpu_id=0)
        except Exception:
            pytest.fail("cleanup_gpu_memory should handle GPU errors gracefully")
        
        mock_is_available.assert_called_once()
        mock_device.assert_called_once_with(0)
