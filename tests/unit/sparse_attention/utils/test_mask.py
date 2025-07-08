"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-07-03
:summary: Tests for mask. This file is part of the Sparse Attention Hub project.
"""

import numpy as np
import pytest
import torch

from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.unit
class TestMask:
    def test_create_mask_from_dense_mask(self):
        shape = (3, 5)
        mask = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        mask_object = Mask.create_mask_from_dense_mask(shape, mask)
        assert mask_object.shape == shape
        assert torch.allclose(mask_object.mask, mask)
        assert mask_object.from_dense_mask

    def test_get_index_mask(self):
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

    def test_getters_dense_index(self):
        shape = (3, 5)
        mask = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        dense_mask_object = Mask.create_mask_from_dense_mask(shape, mask)
        indices, ptr, data = dense_mask_object.get_index_mask()

        index_mask_object = Mask.create_mask_from_indices(shape, indices, ptr, data)
        dense_mask = index_mask_object.get_dense_mask()
        assert torch.allclose(dense_mask, mask)

    def test_getters_index_dense(self):
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

    def test_getters_dense_index_n_dims(self):
        n_dims = 3
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

    def test_create_empty_mask_dense(self):
        shape = (3, 5, 7)
        mask_object = Mask.create_empty_mask(shape, mask_type="dense")
        assert mask_object.shape == shape
        assert mask_object.mask.shape == shape
        assert mask_object.mask.sum() == 0
        assert mask_object.from_dense_mask
        assert not mask_object.from_index

    def test_create_empty_mask_index(self):
        shape = (3, 5, 7)
        mask_object = Mask.create_empty_mask(shape, mask_type="index")
        mask = mask_object.get_dense_mask()
        assert mask.shape == shape
        assert mask.sum() == 0
        assert not mask_object.from_dense_mask
        assert mask_object.from_index

    def test_apply_mask(self):
        shape = (3, 5, 7)
        input = torch.rand(shape)
        dense_mask = (torch.rand(shape) > 0.5).float()

        true_answer = input * dense_mask

        dense_mask_object = Mask.create_mask_from_dense_mask(shape, dense_mask)
        indices, ptr, data = dense_mask_object.get_index_mask()
        index_mask_object = Mask.create_mask_from_indices(shape, indices, ptr, data)

        dense_masked_data = dense_mask_object.apply_mask(input)
        index_masked_data = index_mask_object.apply_mask(input)

        assert torch.allclose(dense_masked_data, true_answer)
        assert torch.allclose(index_masked_data, true_answer)

    def test_apply_inv_mask(self):
        shape = (3, 5, 7)
        input = torch.rand(shape)
        # Create a mask with non-zero values to test inverse operation
        dense_mask = torch.rand(shape) * 0.5 + 0.5  # Values in [0.5, 1.0]
        
        # Set some values to zero to test zero handling
        zero_mask = (torch.rand(shape) > 0.3).float()
        dense_mask = dense_mask * zero_mask

        # Calculate expected result
        true_answer = torch.zeros_like(input)
        non_zero_mask = dense_mask != 0
        true_answer[non_zero_mask] = input[non_zero_mask] * (1.0 / dense_mask[non_zero_mask])

        dense_mask_object = Mask.create_mask_from_dense_mask(shape, dense_mask)
        indices, ptr, data = dense_mask_object.get_index_mask()
        index_mask_object = Mask.create_mask_from_indices(shape, indices, ptr, data)

        dense_inv_masked_data = dense_mask_object.apply_inv_mask(input)
        index_inv_masked_data = index_mask_object.apply_inv_mask(input)

        assert torch.allclose(dense_inv_masked_data, true_answer)
        assert torch.allclose(index_inv_masked_data, true_answer)

    def test_apply_inv_mask_empty(self):
        shape = (2, 3)
        input = torch.rand(shape)
        
        empty_mask = Mask.create_empty_mask(shape, mask_type="dense")
        result = empty_mask.apply_inv_mask(input)
        
        assert torch.allclose(result, torch.zeros_like(input))

    def test_apply_inv_mask_all_ones(self):
        shape = (2, 3)
        input = torch.rand(shape)
        
        ones_mask = torch.ones(shape)
        mask_object = Mask.create_mask_from_dense_mask(shape, ones_mask)
        result = mask_object.apply_inv_mask(input)
        
        # When mask is all ones, inverse should be the same as input
        assert torch.allclose(result, input)

    def test_create_from_row_wise_idx_basic_dense(self):
        """Test basic functionality for dense mode."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]])
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])
        
        mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="dense")
        
        assert mask.shape == shape
        assert mask.from_dense_mask
        assert not mask.from_index
        
        # Check values are set correctly
        dense_mask = mask.get_dense_mask()
        expected = torch.tensor([
            [1.0, 0.0, 0.5, 0.0, 0.8],
            [0.0, 0.6, 0.0, 0.9, 0.3]
        ])
        assert torch.allclose(dense_mask, expected)

    def test_create_from_row_wise_idx_basic_sparse(self):
        """Test basic functionality for sparse mode."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]])
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])
        
        mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="index")
        
        assert mask.shape == shape
        assert not mask.from_dense_mask
        assert mask.from_index
        
        # Check sparse representation
        indices, ptr, values = mask.get_index_mask()
        expected_indices = torch.tensor([0, 2, 4, 6, 8, 9])  # Flat indices
        expected_ptr = torch.tensor([0, 3, 6])
        expected_values = torch.tensor([1.0, 0.5, 0.8, 0.6, 0.9, 0.3])
        
        assert torch.allclose(indices, expected_indices)
        assert torch.allclose(ptr, expected_ptr)
        assert torch.allclose(values, expected_values)

    def test_create_from_row_wise_idx_single_element(self):
        """Test with single element per row."""
        shape = (3, 4)
        row_wise_idx = torch.tensor([[1], [0], [3]])
        data = torch.tensor([[0.5], [0.8], [0.2]])
        
        mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="dense")
        
        dense_mask = mask.get_dense_mask()
        expected = torch.tensor([
            [0.0, 0.5, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2]
        ])
        assert torch.allclose(dense_mask, expected)

    def test_create_from_row_wise_idx_with_invalid_indices(self):
        """Test handling of invalid indices (e.g., -1 padding)."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, -1], [1, -1, -1]])  # -1 as padding
        data = torch.tensor([[1.0, 0.5, 0.0], [0.6, 0.0, 0.0]])
        
        mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="dense")
        
        dense_mask = mask.get_dense_mask()
        expected = torch.tensor([
            [1.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.0, 0.0]
        ])
        assert torch.allclose(dense_mask, expected)

    def test_create_from_row_wise_idx_multidimensional(self):
        """Test with multi-dimensional batch shape."""
        shape = (2, 3, 4)  # batch_size=2, seq_len=3, feature_dim=4
        row_wise_idx = torch.tensor([[[0, 2], [1, 3], [0, 1]], 
                                    [[1, 3], [0, 2], [2, 3]]])
        data = torch.tensor([[[1.0, 0.5], [0.8, 0.3], [0.2, 0.7]], 
                           [[0.4, 0.9], [0.6, 0.1], [0.8, 0.4]]])
        
        mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="dense")
        
        dense_mask = mask.get_dense_mask()
        
        # Check first batch
        assert torch.allclose(dense_mask[0, 0], torch.tensor([1.0, 0.0, 0.5, 0.0]))
        assert torch.allclose(dense_mask[0, 1], torch.tensor([0.0, 0.8, 0.0, 0.3]))
        assert torch.allclose(dense_mask[0, 2], torch.tensor([0.2, 0.7, 0.0, 0.0]))
        
        # Check second batch
        assert torch.allclose(dense_mask[1, 0], torch.tensor([0.0, 0.4, 0.0, 0.9]))
        assert torch.allclose(dense_mask[1, 1], torch.tensor([0.6, 0.0, 0.1, 0.0]))
        assert torch.allclose(dense_mask[1, 2], torch.tensor([0.0, 0.0, 0.8, 0.4]))

    def test_create_from_row_wise_idx_consistency(self):
        """Test that dense and sparse modes produce equivalent results."""
        shape = (3, 6)
        row_wise_idx = torch.tensor([[0, 2, 5], [1, 3, 4], [0, 1, 2]])
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3], [0.4, 0.7, 0.2]])
        
        dense_mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="dense")
        sparse_mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="index")
        
        # Convert both to dense representation and compare
        dense_from_dense = dense_mask.get_dense_mask()
        dense_from_sparse = sparse_mask.get_dense_mask()
        
        assert torch.allclose(dense_from_dense, dense_from_sparse)

    def test_create_from_row_wise_idx_error_invalid_shape(self):
        """Test error handling for invalid shapes."""
        shape = ()  # Empty shape
        row_wise_idx = torch.tensor([[0, 1]])
        data = torch.tensor([[1.0, 0.5]])
        
        with pytest.raises(ValueError, match="shape must have at least one dimension"):
            Mask.create_from_row_wise_idx(shape, row_wise_idx, data)

    def test_create_from_row_wise_idx_error_mismatched_shapes(self):
        """Test error handling for mismatched input shapes."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4]])  # Wrong batch size
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])
        
        with pytest.raises(ValueError, match="row_wise_idx.shape must be"):
            Mask.create_from_row_wise_idx(shape, row_wise_idx, data)

    def test_create_from_row_wise_idx_error_data_shape_mismatch(self):
        """Test error handling when data shape doesn't match row_wise_idx shape."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]])
        data = torch.tensor([[1.0, 0.5]])  # Wrong shape
        
        with pytest.raises(ValueError, match="data.shape must match row_wise_idx.shape"):
            Mask.create_from_row_wise_idx(shape, row_wise_idx, data)

    def test_create_from_row_wise_idx_error_out_of_bounds(self):
        """Test error handling for out-of-bounds indices."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 5], [1, 3, 4]])  # 5 is out of bounds
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])
        
        with pytest.raises(ValueError, match="All valid indices in row_wise_idx must be in range"):
            Mask.create_from_row_wise_idx(shape, row_wise_idx, data)

    def test_create_from_row_wise_idx_error_invalid_type(self):
        """Test error handling for invalid type parameter."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]])
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]])
        
        with pytest.raises(ValueError, match="type must be 'index' or 'dense'"):
            Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="invalid")

    def test_create_from_row_wise_idx_empty_mask(self):
        """Test creating empty mask with all -1 indices."""
        shape = (2, 5)
        row_wise_idx = torch.tensor([[-1, -1, -1], [-1, -1, -1]])
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="dense")
        
        assert mask.is_empty()
        dense_mask = mask.get_dense_mask()
        assert torch.allclose(dense_mask, torch.zeros(shape))

    def test_create_from_row_wise_idx_device_consistency(self):
        """Test that output tensors are on the same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        shape = (2, 5)
        row_wise_idx = torch.tensor([[0, 2, 4], [1, 3, 4]], device=device)
        data = torch.tensor([[1.0, 0.5, 0.8], [0.6, 0.9, 0.3]], device=device)
        
        mask = Mask.create_from_row_wise_idx(shape, row_wise_idx, data, type="dense")
        assert mask.get_dense_mask().device == row_wise_idx.device

    def test_merge_mask_basic(self):
        """Test basic merge functionality."""
        shape = (2, 5)
        
        # Create first mask
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [1.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0, 0.3]
            ])
        )
        
        # Create second mask
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [0.0, 0.2, 0.5, 0.0, 0.4],
                [0.1, 0.0, 0.0, 0.6, 0.0]
            ])
        )
        
        # Merge masks
        merged = mask1.merge_mask(mask2, inplace=False)
        
        # Expected result: union of indices with data addition
        expected_dense = torch.tensor([
            [1.0, 0.2, 1.0, 0.0, 0.4],  # 0.5 + 0.5 = 1.0
            [0.1, 0.8, 0.0, 0.6, 0.3]
        ])
        
        assert torch.allclose(merged.get_dense_mask(), expected_dense)
        assert merged.from_index
        assert not merged.from_dense_mask

    def test_merge_mask_with_capping(self):
        """Test merge with capping (always [0.0, 1.0])."""
        shape = (2, 5)
        
        # Create first mask
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [1.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0, 0.3]
            ])
        )
        
        # Create second mask
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [0.0, 0.2, 0.8, 0.0, 0.4],
                [0.1, 0.0, 0.0, 0.6, 0.0]
            ])
        )
        
        # Merge masks with capping
        merged = mask1.merge_mask(mask2, inplace=False)
        
        # Expected result with capping (now always [0.0, 1.0])
        expected_dense = torch.tensor([
            [1.0, 0.2, 1.0, 0.0, 0.4],  # 0.5 + 0.8 = 1.3 -> capped to 1.0
            [0.1, 0.8, 0.0, 0.6, 0.3]
        ])
        
        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_inplace(self):
        """Test in-place merge."""
        shape = (2, 5)
        
        # Create first mask
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [1.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0, 0.3]
            ])
        )
        
        # Create second mask
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [0.0, 0.2, 0.5, 0.0, 0.4],
                [0.1, 0.0, 0.0, 0.6, 0.0]
            ])
        )
        
        # Merge masks in-place
        original_id = id(mask1)
        result = mask1.merge_mask(mask2, inplace=True)
        
        # Check that the same object is returned
        assert id(result) == original_id
        assert result is mask1
        
        # Check that mask1 is updated
        assert mask1.from_index
        assert not mask1.from_dense_mask
        
        # Expected result
        expected_dense = torch.tensor([
            [1.0, 0.2, 1.0, 0.0, 0.4],
            [0.1, 0.8, 0.0, 0.6, 0.3]
        ])
        
        assert torch.allclose(mask1.get_dense_mask(), expected_dense)

    def test_merge_mask_empty_masks(self):
        """Test merging with empty masks."""
        shape = (2, 3)
        
        # Create empty mask
        empty_mask = Mask.create_empty_mask(shape, mask_type="index")
        
        # Create non-empty mask
        non_empty_mask = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [1.0, 0.0, 0.5],
                [0.0, 0.8, 0.0]
            ])
        )
        
        # Merge empty with non-empty
        merged1 = empty_mask.merge_mask(non_empty_mask, inplace=False)
        merged2 = non_empty_mask.merge_mask(empty_mask, inplace=False)
        
        # Both should be equal to the non-empty mask
        assert torch.allclose(merged1.get_dense_mask(), non_empty_mask.get_dense_mask())
        assert torch.allclose(merged2.get_dense_mask(), non_empty_mask.get_dense_mask())
        
        # Merge two empty masks
        merged_empty = empty_mask.merge_mask(Mask.create_empty_mask(shape, mask_type="index"), inplace=False)
        assert merged_empty.is_empty()

    def test_merge_mask_multidimensional(self):
        """Test merge with multi-dimensional masks."""
        shape = (2, 3, 4)
        
        # Create masks with known patterns
        mask1_dense = torch.tensor([
            [[1.0, 0.0, 0.5, 0.0], [0.0, 0.8, 0.0, 0.3], [0.2, 0.0, 0.0, 0.0]],
            [[0.0, 0.4, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0]]
        ])
        
        mask2_dense = torch.tensor([
            [[0.0, 0.2, 0.5, 0.0], [0.1, 0.0, 0.0, 0.3], [0.0, 0.0, 0.4, 0.0]],
            [[0.3, 0.0, 0.0, 0.7], [0.0, 0.0, 0.5, 0.0], [0.0, 0.8, 0.0, 0.0]]
        ])
        
        mask1 = Mask.create_mask_from_dense_mask(shape, mask1_dense)
        mask2 = Mask.create_mask_from_dense_mask(shape, mask2_dense)
        
        # Merge
        merged = mask1.merge_mask(mask2, inplace=False)
        
        # Expected: element-wise addition with capping to [0.0, 1.0]
        expected_dense = torch.clamp(mask1_dense + mask2_dense, min=0.0, max=1.0)
        
        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_sparse_representations(self):
        """Test merge with sparse representations."""
        shape = (2, 5)
        
        # Create sparse mask 1
        indices1 = torch.tensor([0, 2, 6, 8])
        ptr1 = torch.tensor([0, 2, 4])
        data1 = torch.tensor([1.0, 0.5, 0.8, 0.3])
        mask1 = Mask.create_mask_from_indices(shape, indices1, ptr1, data1)
        
        # Create sparse mask 2
        indices2 = torch.tensor([1, 2, 5, 7])
        ptr2 = torch.tensor([0, 2, 4])
        data2 = torch.tensor([0.2, 0.5, 0.1, 0.6])
        mask2 = Mask.create_mask_from_indices(shape, indices2, ptr2, data2)
        
        # Merge
        merged = mask1.merge_mask(mask2, inplace=False)
        
        # Expected dense representation (capped to [0.0, 1.0])
        expected_dense = torch.clamp(mask1.get_dense_mask() + mask2.get_dense_mask(), min=0.0, max=1.0)
        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_shape_mismatch(self):
        """Test error handling for shape mismatch."""
        shape1 = (2, 5)
        shape2 = (3, 5)
        
        mask1 = Mask.create_empty_mask(shape1, mask_type="index")
        mask2 = Mask.create_empty_mask(shape2, mask_type="index")
        
        with pytest.raises(ValueError, match="Cannot merge masks with different shapes"):
            mask1.merge_mask(mask2, inplace=False)

    def test_merge_mask_negative_capping(self):
        """Test merge with negative values and capping (always [0.0, 1.0])."""
        shape = (2, 3)
        
        # Create masks with negative values
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [1.0, -0.5, 0.0],
                [0.0, 0.8, -0.3]
            ])
        )
        
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [0.0, 0.2, -0.4],
                [-0.1, 0.0, 0.6]
            ])
        )
        
        # Merge with capping (now always [0.0, 1.0])
        merged = mask1.merge_mask(mask2, inplace=False)
        
        # Expected result (capped to [0.0, 1.0])
        expected_dense = torch.clamp(mask1.get_dense_mask() + mask2.get_dense_mask(), min=0.0, max=1.0)
        
        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_extreme_capping(self):
        """Test merge with large values and capping (always [0.0, 1.0])."""
        shape = (2, 3)
        
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ])
        )
        
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([
                [0.5, 1.5, 2.5],
                [3.5, 4.5, 5.5]
            ])
        )
        
        # Test with capping (now always [0.0, 1.0])
        merged = mask1.merge_mask(mask2, inplace=False)
        
        # Expected: all values clamped to [0.0, 1.0]
        expected_dense = torch.clamp(mask1.get_dense_mask() + mask2.get_dense_mask(), min=0.0, max=1.0)
        
        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_single_element(self):
        """Test merge with single element masks."""
        shape = (1, 1)
        
        mask1 = Mask.create_mask_from_dense_mask(shape, torch.tensor([[0.5]]))
        mask2 = Mask.create_mask_from_dense_mask(shape, torch.tensor([[0.3]]))
        
        merged = mask1.merge_mask(mask2, inplace=False)
        
        expected_dense = torch.tensor([[0.8]])  # 0.5 + 0.3 = 0.8
        assert torch.allclose(merged.get_dense_mask(), expected_dense)

    def test_merge_mask_device_consistency(self):
        """Test that merged masks maintain device consistency."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        shape = (2, 3)
        
        mask1 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.8, 0.0]], device=device)
        )
        true_device = mask1.get_dense_mask().device
        mask2 = Mask.create_mask_from_dense_mask(
            shape,
            torch.tensor([[0.0, 0.2, 0.5], [0.1, 0.0, 0.6]], device=device)
        )
    
        
        merged = mask1.merge_mask(mask2, inplace=False)
        
        assert merged.get_dense_mask().device == true_device
        indices, ptr, data = merged.get_index_mask()
        assert indices.device == true_device
        assert ptr.device == true_device
        assert data.device == true_device
