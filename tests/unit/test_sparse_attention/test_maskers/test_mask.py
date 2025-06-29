import torch
from sparse_attention_hub.sparse_attention.utils.mask import Mask
import numpy as np  
import pytest
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import LocalMaskerConfig

def test_create_mask_from_dense_mask():
    shape = (3, 5)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]])
    mask_object = Mask.create_mask_from_dense_mask(shape, mask)
    assert mask_object.shape == shape
    assert torch.allclose(mask_object.mask, mask)
    assert mask_object.from_dense_mask

def test_get_index_mask():
    shape = (3, 5)
    ptr = torch.tensor([0, 5, 7, 9])
    indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3])
    data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    index_mask_object = Mask.create_mask_from_indices(shape, indices, ptr, data)
    index_mask, index_ptr, index_data = index_mask_object.get_index_mask()
    assert torch.allclose(index_mask, indices)
    assert torch.allclose(index_ptr, ptr)
    assert torch.allclose(index_data, data)
    assert index_mask_object.from_index

def test_getters_dense_index():
    shape = (3, 5)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]])
    dense_mask_object = Mask.create_mask_from_dense_mask(shape, mask)
    indices, ptr, data = dense_mask_object.get_index_mask()
    
    index_mask_object = Mask.create_mask_from_indices(shape, indices, ptr, data)
    dense_mask = index_mask_object.get_dense_mask()
    assert torch.allclose(dense_mask, mask)

def test_getters_index_dense():
    shape = (3, 5)
    ptr = torch.tensor([0, 5, 7, 9])
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13])
    data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    index_mask_object = Mask.create_mask_from_indices(shape, indices, ptr, data)

    dense_mask = index_mask_object.get_dense_mask()
    dense_mask_object = Mask.create_mask_from_dense_mask(shape, dense_mask)
    _indices, _ptr, _data = dense_mask_object.get_index_mask()
    assert _indices.shape == indices.shape
    assert _ptr.shape == ptr.shape
    assert _data.shape == data.shape
    assert torch.allclose(indices, _indices)
    assert torch.allclose(_ptr, ptr)
    assert torch.allclose(_data, data)



def test_getters_dense_index_n_dims(n_dims=3):
    shape = tuple([5] * n_dims)
    mask = (torch.rand(shape) > 0.5) * torch.rand(shape)
    dense_mask_object = Mask.create_mask_from_dense_mask(shape, mask)
    indices, ptr, data = dense_mask_object.get_index_mask()
    
    for i in range(np.prod(shape[:-1])):
        start = ptr[i]
        end = ptr[i + 1]
        data = data[start:end]
        indices = indices[start:end]
        assert torch.allclose(data, mask.view(-1)[indices])

class TestLocalMaskerConfig:
    """Test LocalMaskerConfig functionality."""

    def test_local_masker_config_creation(self):
        """Test LocalMaskerConfig can be created with valid parameters."""
        config = LocalMaskerConfig(window_size=5)
        assert config.window_size == 5

    def test_local_masker_config_defaults(self):
        """Test LocalMaskerConfig has correct default values."""
        config = LocalMaskerConfig()
        assert config.window_size == 1  # Assuming default is 1

    def test_local_masker_config_validation(self):
        """Test LocalMaskerConfig validates input parameters."""
        # Test with valid window_size
        config = LocalMaskerConfig(window_size=10)
        assert config.window_size == 10

        # Test with zero window_size (should be valid for local attention)
        config = LocalMaskerConfig(window_size=0)
        assert config.window_size == 0
