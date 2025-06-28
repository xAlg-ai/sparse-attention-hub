"""Mask class for representing attention masks in both dense and sparse formats."""

import torch
import numpy as np
from typing import Tuple, Optional, Union


class Mask:
    """
    Mask object represents a mask over a tensor of shape (..., n).
    
    There are two representations of a Mask Object:
    1. mask: mask.shape == shape (dense representation)
    2. sparse matrix format: stores indices and ptr of mask (sparse representation)
    
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
        from_index: bool = False
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
        """
        self.shape = shape
        self.dtype = dtype
        self.from_dense_mask = from_dense_mask
        self.from_index = from_index
        
        if from_dense_mask and mask is not None:
            self.mask = mask.to(dtype)
            self.indices = None
            self.ptr = None
            self.data = None
        elif from_index and indices is not None and ptr is not None:
            self.mask = None
            self.indices = indices
            self.ptr = ptr
            self.data = data
        else:
            raise ValueError("Must specify either from_dense_mask with mask or from_index with indices and ptr")
    
    @classmethod
    def create_mask_from_indices(
        cls,
        shape: Tuple[int, ...],
        indices: torch.Tensor,
        ptr: torch.Tensor,
        data: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32
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
            raise ValueError(f"ptr.numel() must be {expected_ptr_size}, got {ptr.numel()}")
        
        return cls(
            shape=shape,
            dtype=dtype,
            indices=indices,
            ptr=ptr,
            data=data,
            from_index=True
        )
    
    @classmethod
    def create_mask_from_dense_mask(
        cls,
        shape: Tuple[int, ...],
        mask: torch.Tensor,
        dtype: torch.dtype = torch.float32
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
        
        return cls(
            shape=shape,
            dtype=dtype,
            mask=mask,
            from_dense_mask=True
        )
    
    def get_index_mask(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the sparse index representation.
        
        Returns:
            Tuple of (indices, ptr, data) for sparse representation
        """
        if self.from_index:
            return self.indices, self.ptr, self.data
        elif self.from_dense_mask:
            # Convert dense representation to sparse representation
            # Find non-zero indices
            non_zero_indices = torch.nonzero(self.mask.view(-1), as_tuple=False).squeeze(-1)
            data = self.mask.view(-1)[non_zero_indices]
            
            # Create ptr array with size equal to product of all dimensions except the last one, plus 1
            ptr_size = np.prod(self.shape[:-1]) + 1
            # Vectorized version: count nonzero per row and cumsum
            counts = torch.count_nonzero(self.mask.view(-1, self.shape[-1]), dim=1)
            ptr = torch.cat([
                torch.zeros(1, dtype=torch.long),
                torch.cumsum(counts, dim=0)
            ])
            
            return non_zero_indices, ptr, data
        else:
            raise RuntimeError("Mask object is in an invalid state")
    
    def get_dense_mask(self) -> torch.Tensor:
        """
        Get the dense mask representation.
        
        Returns:
            Dense mask tensor
        """
        if self.from_dense_mask:
            return self.mask
        elif self.from_index:
            # Convert sparse representation to dense mask
            dense_mask = torch.zeros(self.shape, dtype=self.dtype)
            if self.indices is not None and len(self.indices) > 0:
                if self.data is not None:
                    dense_mask.view(-1)[self.indices] = self.data
                else:
                    dense_mask.view(-1)[self.indices] = 1.0
            return dense_mask
        else:
            raise RuntimeError("Mask object is in an invalid state")
    
    def __repr__(self) -> str:
        """String representation of the Mask object."""
        return f"Mask(shape={self.shape}, dtype={self.dtype}, from_dense_mask={self.from_dense_mask}, from_index={self.from_index})"
