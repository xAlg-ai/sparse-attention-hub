"""Mask class for representing attention masks in both dense and sparse formats."""

from typing import Optional, Tuple

import numpy as np
import torch


class Mask:
    """
    Mask object represents a mask over a tensor of shape (..., n).

    There are three representations of a Mask Object:
    1. mask: mask.shape == shape (dense representation)
    2. sparse matrix format: stores indices and ptr of mask (sparse representation)
    3. full mask: all elements are 1.0 (optimized representation)

    Both representations store floating point values. A value of zero means that the
    element is not active. A non-zero value represents the weight assigned to this
    element in tensor.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        mask: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        from_dense_mask: bool = False,
        from_index: bool = False,
        is_full: bool = False,
    ):
        """
        Initialize a Mask object.

        Args:
            shape: Shape of the mask (*, n)
            dtype: Data type for the mask values
            mask: Dense mask tensor (if from_dense_mask=True)
            indices: Indices for sparse representation (if from_index=True)
            ptr: Pointer array for sparse representation (if from_index=True)
            data: Data for sparse representation (if from_index=True)
            from_dense_mask: Whether creating from dense mask
            from_index: Whether creating from sparse indices
            is_full: Whether this is a full mask (all elements are 1.0)
        """
        self.shape = shape
        self.dtype = dtype
        self.from_dense_mask = from_dense_mask
        self.from_index = from_index
        self.is_full = is_full

        if is_full:
            # Full mask optimization - don't store any actual data
            self.mask = None
            self.indices = None
            self.ptr = None
            self.data = None
        elif from_dense_mask and mask is not None:
            self.mask = mask.to(dtype)
            self.indices = None
            self.ptr = None
            self.data = None
            # Check if this is actually a full mask
            if self._detect_full_mask():
                self.is_full = True
                self.mask = None  # Free memory
        elif from_index and indices is not None and ptr is not None:
            self.mask = None
            self.indices = indices
            self.ptr = ptr
            self.data = data
            # Check if this is actually a full mask
            if self._detect_full_mask():
                self.is_full = True
                self.indices = None
                self.ptr = None
                self.data = None
        else:
            raise ValueError(
                "Must specify either from_dense_mask with mask, from_index with indices and ptr, or is_full=True"
            )

    def _detect_full_mask(self) -> bool:
        """
        Detect if the current mask represents a full mask (all elements are 1.0).

        Returns:
            True if all elements are 1.0, False otherwise
        """
        if self.is_full:
            return True

        if self.from_dense_mask and self.mask is not None:
            return bool(torch.all(self.mask == 1.0).item())
        elif self.from_index and self.indices is not None and self.ptr is not None:
            # Check if sparse representation has all positions filled with 1.0
            expected_size = int(np.prod(self.shape))
            if self.indices.numel() != expected_size:
                return False
            if self.data is not None:
                return bool(torch.all(self.data == 1.0).item())
            else:
                # If no data is provided, assume 1.0 values
                return True

        return False

    @classmethod
    def create_full_mask(
        cls,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> "Mask":
        """
        Create a full mask (all elements are 1.0) with optimized representation.

        Args:
            shape: Shape of the mask (*, n)
            dtype: Data type for the mask values

        Returns:
            Mask object with is_full=True
        """
        return cls(shape=shape, dtype=dtype, is_full=True)

    def is_full_mask(self) -> bool:
        """
        Check if this is a full mask (all elements are 1.0).

        Returns:
            True if all elements are 1.0, False otherwise
        """
        return self.is_full

    @classmethod
    def create_mask_from_indices(
        cls,
        shape: Tuple[int, ...],
        indices: torch.Tensor,
        ptr: torch.Tensor,
        data: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Mask":
        """
        Create a Mask from sparse indices and pointer representation.

        Args:
            shape: Shape of the mask (*, n)
            indices: Indices for sparse representation
            ptr: Pointer array for sparse representation
            data: Data for sparse representation
            dtype: Data type for the mask values

        Returns:
            Mask object with from_index=True
        """
        # For multi-dimensional tensors, ptr should have size equal to the product of all dimensions except the last one, plus 1
        expected_ptr_size = np.prod(shape[:-1]) + 1
        if ptr.numel() != expected_ptr_size:
            raise ValueError(
                f"ptr.numel() must be {expected_ptr_size}, got {ptr.numel()}"
            )

        return cls(
            shape=shape,
            dtype=dtype,
            indices=indices,
            ptr=ptr,
            data=data,
            from_index=True,
        )

    @classmethod
    def create_mask_from_dense_mask(
        cls,
        shape: Tuple[int, ...],
        mask: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> "Mask":
        """
        Create a Mask from dense mask tensor.

        Args:
            shape: Shape of the mask (*, n)
            mask: Dense mask tensor of shape (*, n)
            dtype: Data type for the mask values

        Returns:
            Mask object with from_dense_mask=True
        """
        if mask.shape != shape:
            raise ValueError(f"mask.shape must be {shape}, got {mask.shape}")

        return cls(shape=shape, dtype=dtype, mask=mask, from_dense_mask=True)

    @classmethod
    def create_from_row_wise_idx(
        cls,
        shape: Tuple[int, ...],
        row_wise_idx: torch.Tensor,
        data: torch.Tensor,
        type: str = "index",
        dtype: torch.dtype = torch.float32,
    ) -> "Mask":
        """
        Create a Mask from row-wise indices.

        Args:
            shape: Shape of the mask (*, n)
            row_wise_idx: Row-wise indices tensor of shape (*, k) -ve values are padding
            data: Data tensor of shape (*, k) corresponding to the indices
            type: Type of representation ("index" or "dense")
            dtype: Data type for the mask values

        Returns:
            Mask object with the specified representation
        """
        if len(shape) < 1:
            raise ValueError("shape must have at least one dimension")

        # Validate input shapes
        expected_row_wise_shape = shape[:-1] + (row_wise_idx.shape[-1],)
        if row_wise_idx.shape != expected_row_wise_shape:
            raise ValueError(
                f"row_wise_idx.shape must be {expected_row_wise_shape}, got {row_wise_idx.shape}"
            )

        if data.shape != row_wise_idx.shape:
            raise ValueError(
                f"data.shape must match row_wise_idx.shape {row_wise_idx.shape}, got {data.shape}"
            )

        # Validate indices are within bounds (allow -1 as padding)
        n = shape[-1]
        valid_indices_mask = row_wise_idx >= 0
        if torch.any((row_wise_idx >= n) & valid_indices_mask):
            raise ValueError(
                f"All valid indices in row_wise_idx must be in range [0, {n-1}]"
            )

        # Always compute sparse representation first
        batch_dims = shape[:-1]
        batch_size = int(np.prod(batch_dims))
        k = row_wise_idx.shape[-1]

        flat_row_wise_idx = row_wise_idx.reshape(batch_size, k)
        flat_data = data.reshape(batch_size, k)

        row_offsets = (
            torch.arange(batch_size, device=flat_row_wise_idx.device).unsqueeze(1) * n
        )
        flat_indices_with_offset = flat_row_wise_idx + row_offsets

        # Filter out invalid indices (e.g., -1 padding)
        valid_mask = flat_row_wise_idx >= 0
        valid_flat_indices = flat_indices_with_offset[valid_mask]
        valid_values = flat_data[valid_mask]

        valid_counts_per_row = valid_mask.sum(dim=1)
        ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=flat_row_wise_idx.device),
                torch.cumsum(valid_counts_per_row, dim=0),
            ]
        )

        # Create sparse mask
        sparse_mask = cls.create_mask_from_indices(
            shape, valid_flat_indices, ptr, valid_values, dtype
        )

        if type == "dense":
            # Convert to dense representation
            dense_mask = sparse_mask.get_dense_mask()
            return cls.create_mask_from_dense_mask(shape, dense_mask, dtype)
        elif type == "index":
            return sparse_mask
        else:
            raise ValueError(f"type must be 'index' or 'dense', got {type}")

    def get_index_mask(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the sparse index representation.

        Returns:
            Tuple of (indices, ptr, data) for sparse representation
        """
        if self.is_full:
            # For full masks, generate all indices with value 1.0
            total_size = int(np.prod(self.shape))
            indices = torch.arange(total_size, dtype=torch.long)
            data = torch.ones(total_size, dtype=self.dtype)

            # Create ptr array
            n_cols = self.shape[-1]
            ptr = torch.arange(0, total_size + 1, n_cols, dtype=torch.long)

            return indices, ptr, data
        elif self.from_index:
            if self.indices is None or self.ptr is None:
                raise RuntimeError("Sparse mask indices or ptr is None")
            if self.data is None:
                # If no data is provided, assume 1.0 values
                data = torch.ones(
                    self.indices.numel(), dtype=self.dtype, device=self.indices.device
                )
            else:
                data = self.data
            return self.indices, self.ptr, data
        elif self.from_dense_mask:
            if self.mask is None:
                raise RuntimeError("Dense mask is None")
            # Convert dense representation to sparse representation
            # Find non-zero indices
            non_zero_indices = torch.nonzero(
                self.mask.view(-1), as_tuple=False
            ).squeeze(-1)
            data = self.mask.view(-1)[non_zero_indices]

            # Create ptr array with size equal to product of all dimensions except the last one, plus 1
            # Vectorized version: count nonzero per row and cumsum
            counts = torch.count_nonzero(self.mask.view(-1, self.shape[-1]), dim=1)
            ptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=self.mask.device),
                    torch.cumsum(counts, dim=0),
                ]
            )

            return non_zero_indices, ptr, data
        else:
            raise RuntimeError("Mask object is in an invalid state")

    def get_dense_mask(self) -> torch.Tensor:
        """
        Get the dense mask representation.

        Returns:
            Dense mask tensor
        """
        if self.is_full:
            # For full masks, return tensor of ones
            return torch.ones(self.shape, dtype=self.dtype)
        elif self.from_dense_mask:
            if self.mask is None:
                raise RuntimeError("Dense mask is None")
            return self.mask
        elif self.from_index:
            if self.indices is None or self.ptr is None:
                raise RuntimeError("Sparse mask indices or ptr is None")
            # Convert sparse representation to dense mask
            # Ensure dense mask is on the same device as indices/data
            device = self.indices.device
            dense_mask = torch.zeros(self.shape, dtype=self.dtype, device=device)
            if len(self.indices) > 0:
                if self.data is not None:
                    dense_mask.view(-1)[self.indices] = self.data
                else:
                    dense_mask.view(-1)[self.indices] = 1.0
            return dense_mask
        else:
            raise RuntimeError("Mask object is in an invalid state")

    def apply_mask(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the mask to an input tensor.

        Args:
            if mask is empty or full, return input_tensor # as if no mask is applied
            if mask is not empty, then 0=> inactive and >0 => active with the corresponding weight

            input_tensor: Input tensor with shape same as the mask

        Returns:
            Output tensor after applying the mask
        """
        if input_tensor.shape != self.shape:
            raise ValueError(
                f"input_tensor.shape must be {self.shape}, got {input_tensor.shape}"
            )

        if self.is_full:
            # Full mask optimization - return input tensor directly (no-op)
            return input_tensor
        elif self.is_empty():
            return input_tensor

        if self.from_dense_mask:
            if self.mask is None:
                raise RuntimeError("Dense mask is None")
            return input_tensor * self.mask
        elif self.from_index:
            dense_mask = self.get_dense_mask()
            return input_tensor * dense_mask
        else:
            raise RuntimeError("Mask object is in an invalid state")

    def apply_inv_mask(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse mask to an input tensor.

        Args:
            input_tensor: Input tensor with shape same as the mask

        Returns:
            if mask is empty or full, return input_tensor # as if no mask is applied

            Output tensor after applying the inverse mask:
            - output[IDX] = 0 if mask[IDX] == 0
            - output[IDX] = input[IDX] * 1.0 / mask[IDX] if mask[IDX] != 0
        """
        if input_tensor.shape != self.shape:
            raise ValueError(
                f"input_tensor.shape must be {self.shape}, got {input_tensor.shape}"
            )

        if self.is_full:
            # Full mask optimization - return input tensor directly (1.0 / 1.0 = 1.0)
            return input_tensor
        elif self.is_empty():
            return input_tensor

        if self.from_dense_mask:
            if self.mask is None:
                raise RuntimeError("Dense mask is None")
            mask = self.mask
        elif self.from_index:
            mask = self.get_dense_mask()
        else:
            raise RuntimeError("Mask object is in an invalid state")

        # Create output tensor
        output = torch.zeros_like(input_tensor)

        # Apply inverse mask logic
        non_zero_mask = mask != 0
        output[non_zero_mask] = input_tensor[non_zero_mask] * (
            1.0 / mask[non_zero_mask]
        )

        return output

    @classmethod
    def create_empty_mask(
        cls,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        mask_type: str = "dense",
    ) -> "Mask":
        """
        Create a mask object with all values set to zero.

        Args:
            shape: Shape of the mask (*, n)
            dtype: Data type for the mask values
            mask_type: Type of representation ("dense" or "index")

        Returns:
            Mask object with all zeros in the specified representation
        """

        if mask_type == "dense":
            empty_mask = torch.zeros(shape, dtype=dtype)
            return cls.create_mask_from_dense_mask(shape, empty_mask, dtype)
        elif mask_type == "index":
            # For an empty mask, we have no non-zero indices
            empty_indices = torch.empty(0, dtype=torch.long)

            # Create ptr array with size equal to product of all dimensions except the last one, plus 1
            # All rows have 0 non-zero elements, so ptr is just zeros
            ptr_size = int(np.prod(shape[:-1]) + 1)
            ptr = torch.zeros(ptr_size, dtype=torch.long)

            # No data since all values are zero
            data = torch.empty(0, dtype=dtype)

            return cls.create_mask_from_indices(shape, empty_indices, ptr, data, dtype)
        else:
            raise ValueError(f"mask_type must be 'dense' or 'index', got {mask_type}")

    def is_empty(self) -> bool:
        """
        Check if the mask is empty.
        """
        if self.is_full:
            # Full masks are never empty
            return False
        elif self.from_dense_mask:
            if self.mask is None:
                raise RuntimeError("Dense mask is None")
            return bool(torch.all(self.mask == 0).item())
        elif self.from_index:
            if self.indices is None:
                raise RuntimeError("Sparse mask indices is None")
            return self.indices.numel() == 0
        else:
            raise RuntimeError("Mask object is in an invalid state")

    def __repr__(self) -> str:
        """String representation of the Mask object."""
        return f"Mask(shape={self.shape}, dtype={self.dtype}, from_dense_mask={self.from_dense_mask}, from_index={self.from_index}, is_full={self.is_full})"

    def merge_mask(
        self,
        other_mask: "Mask",
        inplace: bool = False,
    ) -> "Mask":
        """
        Merge this mask with another mask in sparse format.

        Args:
            other_mask: The other mask to merge with
            inplace: Whether to update the current mask in place

        Returns:
            Merged mask (new instance if inplace=False, self if inplace=True)
        """

        if self.shape != other_mask.shape:
            raise ValueError(
                f"Cannot merge masks with different shapes: {self.shape} vs {other_mask.shape}"
            )

        # Full mask optimizations
        if self.is_full or other_mask.is_full:
            # If either mask is full, the result is full
            if inplace:
                self.is_full = True
                self.from_dense_mask = False
                self.from_index = False
                self.mask = None
                self.indices = None
                self.ptr = None
                self.data = None
                return self
            else:
                return Mask.create_full_mask(self.shape, self.dtype)

        self_indices, self_ptr, self_data = self.get_index_mask()
        other_indices, other_ptr, other_data = other_mask.get_index_mask()

        device = (
            self_indices.device
            if self_indices.numel() > 0
            else (
                other_indices.device
                if other_indices.numel() > 0
                else torch.device("cpu")
            )
        )

        if self_indices.numel() == 0 and other_indices.numel() == 0:
            final_indices = torch.empty(0, dtype=torch.long, device=device)
            final_data = torch.empty(0, dtype=self.dtype, device=device)
            num_rows = int(np.prod(self.shape[:-1]))
            final_ptr = torch.zeros(num_rows + 1, dtype=torch.long, device=device)
        elif self_indices.numel() == 0:
            final_indices = other_indices
            final_data = torch.clamp(other_data, 0.0, 1.0)
            final_ptr = other_ptr
        elif other_indices.numel() == 0:
            final_indices = self_indices
            final_data = torch.clamp(self_data, 0.0, 1.0)
            final_ptr = self_ptr
        else:
            all_indices = torch.cat([self_indices, other_indices])
            all_data = torch.cat([self_data, other_data])

            sorted_indices, sort_order = torch.sort(all_indices)
            sorted_data = all_data[sort_order]

            unique_indices, inverse_indices = torch.unique_consecutive(
                sorted_indices, return_inverse=True
            )

            summed_data = torch.zeros(
                unique_indices.numel(), dtype=self.dtype, device=device
            )
            summed_data.scatter_add_(0, inverse_indices, sorted_data)

            final_data = torch.clamp(summed_data, 0.0, 1.0)
            final_indices = unique_indices

            # Reconstruct ptr array by counting elements per row
            num_rows = int(np.prod(self.shape[:-1]))
            n_cols = self.shape[-1]
            row_indices = final_indices // n_cols
            row_counts = torch.bincount(row_indices, minlength=num_rows)
            final_ptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.cumsum(row_counts, dim=0),
                ]
            )

        # Check if the result is a full mask
        expected_size = int(np.prod(self.shape))
        if final_indices.numel() == expected_size and torch.all(final_data == 1.0):
            # Result is a full mask, use optimization
            if inplace:
                self.is_full = True
                self.from_dense_mask = False
                self.from_index = False
                self.mask = None
                self.indices = None
                self.ptr = None
                self.data = None
                return self
            else:
                return Mask.create_full_mask(self.shape, self.dtype)

        if inplace:
            self.indices = final_indices
            self.ptr = final_ptr
            self.data = final_data
            self.from_index = True
            self.from_dense_mask = False
            self.is_full = False
            self.mask = None
            return self
        else:
            return Mask.create_mask_from_indices(
                shape=self.shape,
                indices=final_indices,
                ptr=final_ptr,
                data=final_data,
                dtype=self.dtype,
            )
