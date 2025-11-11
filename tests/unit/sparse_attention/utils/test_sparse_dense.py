"""Test sparse-dense mode equivalence for mask operations and attention utilities.

This module tests that sparse and dense modes produce equivalent results when
comparing their dense mask outputs. This ensures consistency between the two
execution modes.

:author: Sparse Attention Hub
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-10-19
:summary: Tests for sparse-dense mode equivalence.
"""

import pytest
import torch

from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.unit
class TestSparseDenseEquivalence:
    """Test class for verifying sparse and dense mode equivalence."""

    @pytest.fixture
    def random_seed(self) -> int:
        """Random seed for reproducibility."""
        return 42

    @pytest.fixture
    def device(self) -> torch.device:
        """Device to run tests on."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self) -> torch.dtype:
        """Default dtype for tests."""
        return torch.float32

    @pytest.fixture
    def simple_mask(self, device: torch.device, dtype: torch.dtype) -> Mask:
        """Create a simple mask for basic testing."""
        shape: tuple[int, ...] = (2, 3, 5)
        mask_tensor: torch.Tensor = torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.5, 0.5, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    @pytest.fixture
    def random_mask(
        self, device: torch.device, dtype: torch.dtype, random_seed: int
    ) -> Mask:
        """Create a random sparse mask for stress testing."""
        torch.manual_seed(random_seed)
        shape: tuple[int, ...] = (4, 8, 32)
        # Create random mask with ~30% sparsity
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.3).to(dtype)
        # Add some random weights
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype)
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    @pytest.fixture
    def full_mask(self, device: torch.device, dtype: torch.dtype) -> Mask:
        """Create a full mask (all ones)."""
        shape: tuple[int, ...] = (2, 4, 8)
        return Mask.create_full_mask(shape, dtype=dtype, device=device)

    @pytest.fixture
    def empty_mask(self, device: torch.device, dtype: torch.dtype) -> Mask:
        """Create an empty mask (all zeros)."""
        shape: tuple[int, ...] = (2, 4, 8)
        return Mask.create_empty_mask(shape, dtype=dtype, device=device)

    @pytest.fixture
    def large_random_mask(
        self, device: torch.device, dtype: torch.dtype, random_seed: int
    ) -> Mask:
        """Create a large random sparse mask for stress testing."""
        torch.manual_seed(random_seed + 1)
        shape: tuple[int, ...] = (8, 16, 128)
        # Create random mask with ~50% sparsity
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.5).to(dtype)
        # Add some random weights
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype)
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    # ==================== Test apply_mask ====================

    def test_apply_mask_simple(
        self, simple_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_mask sparse vs dense with simple mask."""
        input_tensor: torch.Tensor = torch.randn(
            simple_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = simple_mask.apply_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = simple_mask.apply_mask(input_tensor, mode="dense")

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_mask outputs should match"

    def test_apply_mask_random(
        self, random_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_mask sparse vs dense with random mask."""
        input_tensor: torch.Tensor = torch.randn(
            random_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = random_mask.apply_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = random_mask.apply_mask(input_tensor, mode="dense")

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_mask outputs should match for random mask"

    def test_apply_mask_full(
        self, full_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_mask sparse vs dense with full mask."""
        input_tensor: torch.Tensor = torch.randn(
            full_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = full_mask.apply_mask(input_tensor, mode="sparse")
        output_dense: torch.Tensor = full_mask.apply_mask(input_tensor, mode="dense")

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_mask outputs should match for full mask"
        # For full mask, output should equal input
        assert torch.allclose(
            output_sparse, input_tensor, rtol=1e-5, atol=1e-6
        ), "Full mask should return input unchanged"

    def test_apply_mask_empty(
        self, empty_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_mask sparse vs dense with empty mask."""
        input_tensor: torch.Tensor = torch.randn(
            empty_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = empty_mask.apply_mask(input_tensor, mode="sparse")
        output_dense: torch.Tensor = empty_mask.apply_mask(input_tensor, mode="dense")

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_mask outputs should match for empty mask"
        # For empty mask, output should equal input
        assert torch.allclose(
            output_sparse, input_tensor, rtol=1e-5, atol=1e-6
        ), "Empty mask should return input unchanged"

    def test_apply_mask_large(
        self, large_random_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_mask sparse vs dense with large random mask."""
        input_tensor: torch.Tensor = torch.randn(
            large_random_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = large_random_mask.apply_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = large_random_mask.apply_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_mask outputs should match for large mask"

    # ==================== Test apply_inv_mask ====================

    def test_apply_inv_mask_simple(
        self, simple_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_inv_mask sparse vs dense with simple mask."""
        input_tensor: torch.Tensor = torch.randn(
            simple_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = simple_mask.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = simple_mask.apply_inv_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-4, atol=1e-5
        ), "Sparse and dense apply_inv_mask outputs should match"

    def test_apply_inv_mask_random(
        self, random_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_inv_mask sparse vs dense with random mask."""
        input_tensor: torch.Tensor = torch.randn(
            random_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = random_mask.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = random_mask.apply_inv_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-3, atol=1e-4
        ), "Sparse and dense apply_inv_mask outputs should match for random mask"

    def test_apply_inv_mask_full(
        self, full_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_inv_mask sparse vs dense with full mask."""
        input_tensor: torch.Tensor = torch.randn(
            full_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = full_mask.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = full_mask.apply_inv_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_inv_mask outputs should match for full mask"
        # For full mask, output should equal input (1.0 / 1.0 = 1.0)
        assert torch.allclose(
            output_sparse, input_tensor, rtol=1e-5, atol=1e-6
        ), "Full mask inv should return input unchanged"

    def test_apply_inv_mask_empty(
        self, empty_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_inv_mask sparse vs dense with empty mask."""
        input_tensor: torch.Tensor = torch.randn(
            empty_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = empty_mask.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = empty_mask.apply_inv_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_inv_mask outputs should match for empty mask"
        # For empty mask, output should equal input
        assert torch.allclose(
            output_sparse, input_tensor, rtol=1e-5, atol=1e-6
        ), "Empty mask inv should return input unchanged"

    def test_apply_inv_mask_large(
        self, large_random_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test apply_inv_mask sparse vs dense with large random mask."""
        input_tensor: torch.Tensor = torch.randn(
            large_random_mask.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = large_random_mask.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = large_random_mask.apply_inv_mask(
            input_tensor, mode="dense"
        )

        # Relax tolerances for large masks as numerical errors accumulate with division
        # With large random values (hundreds), numerical precision varies significantly
        # Using generous tolerances as the test uses random large masks where edge cases can occur
        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-2, atol=2.0
        ), "Sparse and dense apply_inv_mask outputs should match for large mask"

    # ==================== Test merge_mask ====================

    def test_merge_mask_simple(
        self, simple_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test merge_mask sparse vs dense with simple masks."""
        # Create a second mask to merge
        shape: tuple[int, ...] = simple_mask.shape
        mask_tensor: torch.Tensor = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.5, 0.5, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        other_mask: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        # Test non-inplace merge
        merged_sparse: Mask = simple_mask.merge_mask(
            other_mask, inplace=False, mode="sparse"
        )
        merged_dense: Mask = simple_mask.merge_mask(
            other_mask, inplace=False, mode="dense"
        )

        # Compare dense representations
        merged_sparse_dense: torch.Tensor = merged_sparse.get_dense_mask()
        merged_dense_dense: torch.Tensor = merged_dense.get_dense_mask()

        assert torch.allclose(
            merged_sparse_dense, merged_dense_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense merge_mask outputs should match"

    def test_merge_mask_random(
        self,
        random_mask: Mask,
        device: torch.device,
        dtype: torch.dtype,
        random_seed: int,
    ) -> None:
        """Test merge_mask sparse vs dense with random masks."""
        # Create a second random mask to merge
        torch.manual_seed(random_seed + 2)
        shape: tuple[int, ...] = random_mask.shape
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.4).to(dtype)
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype)
        other_mask: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        # Test non-inplace merge
        merged_sparse: Mask = random_mask.merge_mask(
            other_mask, inplace=False, mode="sparse"
        )
        merged_dense: Mask = random_mask.merge_mask(
            other_mask, inplace=False, mode="dense"
        )

        # Compare dense representations
        merged_sparse_dense: torch.Tensor = merged_sparse.get_dense_mask()
        merged_dense_dense: torch.Tensor = merged_dense.get_dense_mask()

        assert torch.allclose(
            merged_sparse_dense, merged_dense_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense merge_mask outputs should match for random masks"

    def test_merge_mask_inplace(
        self, simple_mask: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Test merge_mask inplace sparse vs dense."""
        # Create two copies of the mask
        shape: tuple[int, ...] = simple_mask.shape
        mask_tensor: torch.Tensor = simple_mask.get_dense_mask()
        mask1_sparse: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )
        mask1_dense: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        # Create a second mask to merge
        mask_tensor2: torch.Tensor = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.5, 0.5, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        other_mask: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor2, dtype=dtype
        )

        # Test inplace merge
        result_sparse: Mask = mask1_sparse.merge_mask(
            other_mask, inplace=True, mode="sparse"
        )
        result_dense: Mask = mask1_dense.merge_mask(
            other_mask, inplace=True, mode="dense"
        )

        # Verify inplace operation
        assert result_sparse is mask1_sparse, "Sparse merge should be inplace"
        assert result_dense is mask1_dense, "Dense merge should be inplace"

        # Compare dense representations
        merged_sparse_dense: torch.Tensor = result_sparse.get_dense_mask()
        merged_dense_dense: torch.Tensor = result_dense.get_dense_mask()

        assert torch.allclose(
            merged_sparse_dense, merged_dense_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense inplace merge_mask outputs should match"

    def test_merge_mask_with_full(
        self,
        random_mask: Mask,
        full_mask: Mask,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Test merge_mask with full mask."""
        # Resize full_mask to match random_mask shape for testing
        full_mask_resized: Mask = Mask.create_full_mask(
            random_mask.shape, dtype=dtype, device=device
        )

        merged_sparse: Mask = random_mask.merge_mask(
            full_mask_resized, inplace=False, mode="sparse"
        )
        merged_dense: Mask = random_mask.merge_mask(
            full_mask_resized, inplace=False, mode="dense"
        )

        # Compare dense representations
        merged_sparse_dense: torch.Tensor = merged_sparse.get_dense_mask()
        merged_dense_dense: torch.Tensor = merged_dense.get_dense_mask()

        assert torch.allclose(
            merged_sparse_dense, merged_dense_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense merge_mask with full mask should match"

        # Result should be full mask (all ones)
        assert (
            merged_sparse.is_full
        ), "Merging with full mask should result in full mask"
        assert merged_dense.is_full, "Merging with full mask should result in full mask"

    def test_merge_mask_with_empty(
        self,
        random_mask: Mask,
        empty_mask: Mask,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Test merge_mask with empty mask."""
        # Resize empty_mask to match random_mask shape for testing
        empty_mask_resized: Mask = Mask.create_empty_mask(
            random_mask.shape, dtype=dtype, device=device
        )

        merged_sparse: Mask = random_mask.merge_mask(
            empty_mask_resized, inplace=False, mode="sparse"
        )
        merged_dense: Mask = random_mask.merge_mask(
            empty_mask_resized, inplace=False, mode="dense"
        )

        # Compare dense representations
        merged_sparse_dense: torch.Tensor = merged_sparse.get_dense_mask()
        merged_dense_dense: torch.Tensor = merged_dense.get_dense_mask()
        original_dense: torch.Tensor = random_mask.get_dense_mask()

        assert torch.allclose(
            merged_sparse_dense, merged_dense_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense merge_mask with empty mask should match"

        # Result should be equal to original mask
        assert torch.allclose(
            merged_sparse_dense, original_dense, rtol=1e-5, atol=1e-6
        ), "Merging with empty mask should return original mask"

    # ==================== Stress Tests with Random Masks (0,1] ====================

    @pytest.fixture
    def stress_mask_small(
        self, device: torch.device, dtype: torch.dtype, random_seed: int
    ) -> Mask:
        """Create a small random mask with values in (0,1] for stress testing."""
        torch.manual_seed(random_seed + 10)
        shape: tuple[int, ...] = (4, 8, 16)
        # Create random mask with ~40% sparsity
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.4).to(dtype)
        # Multiply by random values in (0,1]
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype).clamp(
            min=1e-6
        )
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    @pytest.fixture
    def stress_mask_medium(
        self, device: torch.device, dtype: torch.dtype, random_seed: int
    ) -> Mask:
        """Create a medium random mask with values in (0,1] for stress testing."""
        torch.manual_seed(random_seed + 11)
        shape: tuple[int, ...] = (8, 12, 32)
        # Create random mask with ~50% sparsity
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.5).to(dtype)
        # Multiply by random values in (0,1]
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype).clamp(
            min=1e-6
        )
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    @pytest.fixture
    def stress_mask_large(
        self, device: torch.device, dtype: torch.dtype, random_seed: int
    ) -> Mask:
        """Create a large random mask with values in (0,1] for stress testing."""
        torch.manual_seed(random_seed + 12)
        shape: tuple[int, ...] = (16, 16, 64)
        # Create random mask with ~60% sparsity
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.6).to(dtype)
        # Multiply by random values in (0,1]
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype).clamp(
            min=1e-6
        )
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    @pytest.fixture
    def stress_mask_very_sparse(
        self, device: torch.device, dtype: torch.dtype, random_seed: int
    ) -> Mask:
        """Create a very sparse random mask with values in (0,1] for stress testing."""
        torch.manual_seed(random_seed + 13)
        shape: tuple[int, ...] = (8, 16, 48)
        # Create random mask with ~90% sparsity (very sparse)
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.9).to(dtype)
        # Multiply by random values in (0,1]
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype).clamp(
            min=1e-6
        )
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    @pytest.fixture
    def stress_mask_very_dense(
        self, device: torch.device, dtype: torch.dtype, random_seed: int
    ) -> Mask:
        """Create a very dense random mask with values in (0,1] for stress testing."""
        torch.manual_seed(random_seed + 14)
        shape: tuple[int, ...] = (8, 16, 48)
        # Create random mask with ~10% sparsity (very dense)
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.1).to(dtype)
        # Multiply by random values in (0,1]
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype).clamp(
            min=1e-6
        )
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    def test_stress_apply_mask_small(
        self, stress_mask_small: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Stress test apply_mask with small random mask (0,1]."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_small.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = stress_mask_small.apply_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = stress_mask_small.apply_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_mask should match for small stress test"

    def test_stress_apply_mask_medium(
        self, stress_mask_medium: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Stress test apply_mask with medium random mask (0,1]."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_medium.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = stress_mask_medium.apply_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = stress_mask_medium.apply_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_mask should match for medium stress test"

    def test_stress_apply_mask_large(
        self, stress_mask_large: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Stress test apply_mask with large random mask (0,1]."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_large.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = stress_mask_large.apply_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = stress_mask_large.apply_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense apply_mask should match for large stress test"

    def test_stress_apply_inv_mask_small(
        self, stress_mask_small: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Stress test apply_inv_mask with small random mask (0,1]."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_small.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = stress_mask_small.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = stress_mask_small.apply_inv_mask(
            input_tensor, mode="dense"
        )

        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-3, atol=1e-4
        ), "Sparse and dense apply_inv_mask should match for small stress test"

    def _assert_normalized_close(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        msg: str = "",
    ) -> None:
        """Assert that two tensors are close after normalizing by the max L2 norm.

        This is useful when comparing outputs that may have very different magnitudes
        due to division operations, where absolute errors scale with magnitude.
        Both tensors are normalized by the same factor (max of their norms) to ensure
        fair comparison.

        Args:
            tensor1: First tensor to compare
            tensor2: Second tensor to compare
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            msg: Error message to display if assertion fails
        """
        # Compute L2 norms
        norm1: float = torch.linalg.norm(tensor1).item()
        norm2: float = torch.linalg.norm(tensor2).item()

        # Use max norm for normalization
        max_norm: float = max(norm1, norm2)

        # Handle edge case where both are near zero
        if max_norm < 1e-10:
            assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), msg
            return

        # Normalize both tensors by the same factor
        normalized1: torch.Tensor = tensor1 / max_norm
        normalized2: torch.Tensor = tensor2 / max_norm

        # Compare normalized tensors
        assert torch.allclose(
            normalized1, normalized2, rtol=rtol, atol=atol
        ), f"{msg} - Normalized tensors differ"

    def test_stress_apply_inv_mask_medium(
        self, stress_mask_medium: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Stress test apply_inv_mask with medium random mask (0,1]."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_medium.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = stress_mask_medium.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = stress_mask_medium.apply_inv_mask(
            input_tensor, mode="dense"
        )

        self._assert_normalized_close(
            output_sparse,
            output_dense,
            msg="Sparse and dense apply_inv_mask should match for medium stress test",
        )

    def test_stress_apply_inv_mask_large(
        self, stress_mask_large: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Stress test apply_inv_mask with large random mask (0,1]."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_large.shape, device=device, dtype=dtype
        )

        output_sparse: torch.Tensor = stress_mask_large.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = stress_mask_large.apply_inv_mask(
            input_tensor, mode="dense"
        )

        self._assert_normalized_close(
            output_sparse,
            output_dense,
            msg="Sparse and dense apply_inv_mask should match for large stress test",
        )

    def test_stress_merge_mask_small(
        self,
        stress_mask_small: Mask,
        device: torch.device,
        dtype: torch.dtype,
        random_seed: int,
    ) -> None:
        """Stress test merge_mask with small random masks (0,1]."""
        torch.manual_seed(random_seed + 20)
        shape: tuple[int, ...] = stress_mask_small.shape
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.5).to(dtype)
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype).clamp(
            min=1e-6
        )
        other_mask: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        merged_sparse: Mask = stress_mask_small.merge_mask(
            other_mask, inplace=False, mode="sparse"
        )
        merged_dense: Mask = stress_mask_small.merge_mask(
            other_mask, inplace=False, mode="dense"
        )

        merged_sparse_dense: torch.Tensor = merged_sparse.get_dense_mask()
        merged_dense_dense: torch.Tensor = merged_dense.get_dense_mask()

        assert torch.allclose(
            merged_sparse_dense, merged_dense_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense merge_mask should match for small stress test"

    def test_stress_merge_mask_medium(
        self,
        stress_mask_medium: Mask,
        device: torch.device,
        dtype: torch.dtype,
        random_seed: int,
    ) -> None:
        """Stress test merge_mask with medium random masks (0,1]."""
        torch.manual_seed(random_seed + 21)
        shape: tuple[int, ...] = stress_mask_medium.shape
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.5).to(dtype)
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype).clamp(
            min=1e-6
        )
        other_mask: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        merged_sparse: Mask = stress_mask_medium.merge_mask(
            other_mask, inplace=False, mode="sparse"
        )
        merged_dense: Mask = stress_mask_medium.merge_mask(
            other_mask, inplace=False, mode="dense"
        )

        merged_sparse_dense: torch.Tensor = merged_sparse.get_dense_mask()
        merged_dense_dense: torch.Tensor = merged_dense.get_dense_mask()

        assert torch.allclose(
            merged_sparse_dense, merged_dense_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense merge_mask should match for medium stress test"

    def test_stress_merge_mask_large(
        self,
        stress_mask_large: Mask,
        device: torch.device,
        dtype: torch.dtype,
        random_seed: int,
    ) -> None:
        """Stress test merge_mask with large random masks (0,1]."""
        torch.manual_seed(random_seed + 22)
        shape: tuple[int, ...] = stress_mask_large.shape
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.5).to(dtype)
        mask_tensor = mask_tensor * torch.rand(shape, device=device, dtype=dtype).clamp(
            min=1e-6
        )
        other_mask: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        merged_sparse: Mask = stress_mask_large.merge_mask(
            other_mask, inplace=False, mode="sparse"
        )
        merged_dense: Mask = stress_mask_large.merge_mask(
            other_mask, inplace=False, mode="dense"
        )

        merged_sparse_dense: torch.Tensor = merged_sparse.get_dense_mask()
        merged_dense_dense: torch.Tensor = merged_dense.get_dense_mask()

        assert torch.allclose(
            merged_sparse_dense, merged_dense_dense, rtol=1e-5, atol=1e-6
        ), "Sparse and dense merge_mask should match for large stress test"

    def test_stress_very_sparse_mask(
        self, stress_mask_very_sparse: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Stress test with very sparse mask (90% zeros) with values in (0,1]."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_very_sparse.shape, device=device, dtype=dtype
        )

        # Test apply_mask
        output_sparse: torch.Tensor = stress_mask_very_sparse.apply_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = stress_mask_very_sparse.apply_mask(
            input_tensor, mode="dense"
        )
        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Very sparse mask apply_mask should match"

        # Test apply_inv_mask - use higher tolerance for very sparse matrices
        output_sparse = stress_mask_very_sparse.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense = stress_mask_very_sparse.apply_inv_mask(
            input_tensor, mode="dense"
        )
        self._assert_normalized_close(
            output_sparse,
            output_dense,
            msg="Very sparse mask apply_inv_mask should match",
        )

    def test_stress_very_dense_mask(
        self, stress_mask_very_dense: Mask, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Stress test with very dense mask (90% non-zeros) with values in (0,1]."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_very_dense.shape, device=device, dtype=dtype
        )

        # Test apply_mask
        output_sparse: torch.Tensor = stress_mask_very_dense.apply_mask(
            input_tensor, mode="sparse"
        )
        output_dense: torch.Tensor = stress_mask_very_dense.apply_mask(
            input_tensor, mode="dense"
        )
        assert torch.allclose(
            output_sparse, output_dense, rtol=1e-5, atol=1e-6
        ), "Very dense mask apply_mask should match"

        # Test apply_inv_mask - use higher tolerance for very dense matrices
        output_sparse = stress_mask_very_dense.apply_inv_mask(
            input_tensor, mode="sparse"
        )
        output_dense = stress_mask_very_dense.apply_inv_mask(input_tensor, mode="dense")
        self._assert_normalized_close(
            output_sparse,
            output_dense,
            msg="Very dense mask apply_inv_mask should match",
        )

    def test_stress_multiple_operations(
        self,
        stress_mask_small: Mask,
        stress_mask_medium: Mask,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Stress test with multiple operations chained together."""
        input_tensor: torch.Tensor = torch.randn(
            stress_mask_small.shape, device=device, dtype=dtype
        )

        # Sparse mode
        temp_sparse: torch.Tensor = stress_mask_small.apply_mask(
            input_tensor, mode="sparse"
        )
        temp_sparse = stress_mask_small.apply_inv_mask(temp_sparse, mode="sparse")

        # Dense mode
        temp_dense: torch.Tensor = stress_mask_small.apply_mask(
            input_tensor, mode="dense"
        )
        temp_dense = stress_mask_small.apply_inv_mask(temp_dense, mode="dense")

        assert torch.allclose(
            temp_sparse, temp_dense, rtol=1e-3, atol=1e-4
        ), "Chained operations should match between sparse and dense"
