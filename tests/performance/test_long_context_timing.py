"""Performance tests for long context sparse attention operations.

This module benchmarks sparse vs dense modes for long context scenarios
to determine the most efficient default modes.

Test Configuration:
    - Batch size (B) = 1
    - Number of heads (H) = 32
    - Query length (Q) = 1
    - Key length (K) = 32768

:author: Sparse Attention Hub
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-10-19
:summary: Long context performance benchmarks.
"""

import time
from typing import Dict, Tuple

import pytest
import torch

from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.performance
class TestLongContextTiming:
    """Performance tests for long context attention operations."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Device to run tests on."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self) -> torch.dtype:
        """Default dtype for tests."""
        return torch.float32

    @pytest.fixture
    def long_context_shape(self) -> Tuple[int, int, int, int]:
        """Long context shape: (B=1, H=32, Q=1, K=32768)."""
        return (1, 32, 1, 32768)

    @pytest.fixture
    def sparse_mask_10_percent(
        self,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mask:
        """Create a mask with 10% sparsity (90% of values are non-zero)."""
        torch.manual_seed(42)
        shape: Tuple[int, ...] = long_context_shape
        # 10% sparsity means 90% of values are non-zero
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.1).to(dtype)
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    @pytest.fixture
    def sparse_mask_50_percent(
        self,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mask:
        """Create a mask with 50% sparsity (50% of values are non-zero)."""
        torch.manual_seed(43)
        shape: Tuple[int, ...] = long_context_shape
        # 50% sparsity
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.5).to(dtype)
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    @pytest.fixture
    def sparse_mask_90_percent(
        self,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mask:
        """Create a mask with 90% sparsity (10% of values are non-zero)."""
        torch.manual_seed(44)
        shape: Tuple[int, ...] = long_context_shape
        # 90% sparsity means only 10% of values are non-zero
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.9).to(dtype)
        return Mask.create_mask_from_dense_mask(shape, mask_tensor, dtype=dtype)

    def _prepare_mask_for_mode(self, mask: Mask, mode: str) -> Mask:
        """Prepare mask in the appropriate internal representation for the given mode.

        Args:
            mask: The mask to prepare
            mode: "sparse" or "dense"

        Returns:
            Mask in the appropriate representation
        """
        if mode == "sparse":
            # Ensure mask is in index representation for sparse operations
            if not mask.from_index:
                # Convert to index representation
                indices, ptr, data = mask.get_index_mask()
                return Mask.create_mask_from_indices(
                    mask.shape, indices, ptr, data, dtype=mask.dtype
                )
            return mask
        else:  # dense mode
            # Ensure mask is in dense representation for dense operations
            if not mask.from_dense_mask:
                # Convert to dense representation
                dense_mask = mask.get_dense_mask()
                return Mask.create_mask_from_dense_mask(
                    mask.shape, dense_mask, dtype=mask.dtype
                )
            return mask

    def _benchmark_operation(
        self,
        operation_name: str,
        mask: Mask,
        sparse_fn: callable,
        dense_fn: callable,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark an operation in both sparse and dense modes.

        Args:
            operation_name: Name of the operation being benchmarked
            mask: The mask to use (will be prepared in appropriate representation)
            sparse_fn: Function to execute sparse operation (receives sparse-prepared mask)
            dense_fn: Function to execute dense operation (receives dense-prepared mask)
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations

        Returns:
            Dictionary with timing results and speedup information
        """
        # Prepare masks in appropriate representations
        sparse_mask: Mask = self._prepare_mask_for_mode(mask, "sparse")
        dense_mask: Mask = self._prepare_mask_for_mode(mask, "dense")

        # Warmup for sparse
        for _ in range(warmup_iterations):
            sparse_fn(sparse_mask)

        # Benchmark sparse
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time: float = time.time()
        for _ in range(benchmark_iterations):
            sparse_fn(sparse_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sparse_time: float = (time.time() - start_time) / benchmark_iterations

        # Warmup for dense
        for _ in range(warmup_iterations):
            dense_fn(dense_mask)

        # Benchmark dense
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(benchmark_iterations):
            dense_fn(dense_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dense_time: float = (time.time() - start_time) / benchmark_iterations

        speedup: float = dense_time / sparse_time if sparse_time > 0 else 0.0

        return {
            "operation": operation_name,
            "sparse_time_ms": sparse_time * 1000,
            "dense_time_ms": dense_time * 1000,
            "speedup": speedup,
            "faster_mode": "sparse" if sparse_time < dense_time else "dense",
        }

    def test_apply_mask_timing_10_percent_sparsity(
        self,
        sparse_mask_10_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark apply_mask with 10% sparsity (very dense mask)."""
        input_tensor: torch.Tensor = torch.randn(
            long_context_shape, device=device, dtype=dtype
        )

        results: Dict[str, float] = self._benchmark_operation(
            "apply_mask (10% sparsity)",
            sparse_mask_10_percent,
            lambda mask: mask.apply_mask(input_tensor, mode="sparse"),
            lambda mask: mask.apply_mask(input_tensor, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_apply_mask_timing_50_percent_sparsity(
        self,
        sparse_mask_50_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark apply_mask with 50% sparsity."""
        input_tensor: torch.Tensor = torch.randn(
            long_context_shape, device=device, dtype=dtype
        )

        results: Dict[str, float] = self._benchmark_operation(
            "apply_mask (50% sparsity)",
            sparse_mask_50_percent,
            lambda mask: mask.apply_mask(input_tensor, mode="sparse"),
            lambda mask: mask.apply_mask(input_tensor, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_apply_mask_timing_90_percent_sparsity(
        self,
        sparse_mask_90_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark apply_mask with 90% sparsity (very sparse mask)."""
        input_tensor: torch.Tensor = torch.randn(
            long_context_shape, device=device, dtype=dtype
        )

        results: Dict[str, float] = self._benchmark_operation(
            "apply_mask (90% sparsity)",
            sparse_mask_90_percent,
            lambda mask: mask.apply_mask(input_tensor, mode="sparse"),
            lambda mask: mask.apply_mask(input_tensor, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_apply_inv_mask_timing_10_percent_sparsity(
        self,
        sparse_mask_10_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark apply_inv_mask with 10% sparsity (very dense mask)."""
        input_tensor: torch.Tensor = torch.randn(
            long_context_shape, device=device, dtype=dtype
        )

        results: Dict[str, float] = self._benchmark_operation(
            "apply_inv_mask (10% sparsity)",
            sparse_mask_10_percent,
            lambda mask: mask.apply_inv_mask(input_tensor, mode="sparse"),
            lambda mask: mask.apply_inv_mask(input_tensor, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_apply_inv_mask_timing_50_percent_sparsity(
        self,
        sparse_mask_50_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark apply_inv_mask with 50% sparsity."""
        input_tensor: torch.Tensor = torch.randn(
            long_context_shape, device=device, dtype=dtype
        )

        results: Dict[str, float] = self._benchmark_operation(
            "apply_inv_mask (50% sparsity)",
            sparse_mask_50_percent,
            lambda mask: mask.apply_inv_mask(input_tensor, mode="sparse"),
            lambda mask: mask.apply_inv_mask(input_tensor, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_apply_inv_mask_timing_90_percent_sparsity(
        self,
        sparse_mask_90_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark apply_inv_mask with 90% sparsity (very sparse mask)."""
        input_tensor: torch.Tensor = torch.randn(
            long_context_shape, device=device, dtype=dtype
        )

        results: Dict[str, float] = self._benchmark_operation(
            "apply_inv_mask (90% sparsity)",
            sparse_mask_90_percent,
            lambda mask: mask.apply_inv_mask(input_tensor, mode="sparse"),
            lambda mask: mask.apply_inv_mask(input_tensor, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_merge_mask_timing_10_percent_sparsity(
        self,
        sparse_mask_10_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark merge_mask with 10% sparsity (very dense mask)."""
        torch.manual_seed(45)
        shape: Tuple[int, ...] = long_context_shape
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.15).to(dtype)
        other_mask_base: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        # Prepare other_mask in both representations
        other_mask_sparse: Mask = self._prepare_mask_for_mode(other_mask_base, "sparse")
        other_mask_dense: Mask = self._prepare_mask_for_mode(other_mask_base, "dense")

        results: Dict[str, float] = self._benchmark_operation(
            "merge_mask (10% sparsity)",
            sparse_mask_10_percent,
            lambda mask: mask.merge_mask(
                other_mask_sparse, inplace=False, mode="sparse"
            ),
            lambda mask: mask.merge_mask(other_mask_dense, inplace=False, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_merge_mask_timing_50_percent_sparsity(
        self,
        sparse_mask_50_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark merge_mask with 50% sparsity."""
        torch.manual_seed(46)
        shape: Tuple[int, ...] = long_context_shape
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.5).to(dtype)
        other_mask_base: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        # Prepare other_mask in both representations
        other_mask_sparse: Mask = self._prepare_mask_for_mode(other_mask_base, "sparse")
        other_mask_dense: Mask = self._prepare_mask_for_mode(other_mask_base, "dense")

        results: Dict[str, float] = self._benchmark_operation(
            "merge_mask (50% sparsity)",
            sparse_mask_50_percent,
            lambda mask: mask.merge_mask(
                other_mask_sparse, inplace=False, mode="sparse"
            ),
            lambda mask: mask.merge_mask(other_mask_dense, inplace=False, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_merge_mask_timing_90_percent_sparsity(
        self,
        sparse_mask_90_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Benchmark merge_mask with 90% sparsity (very sparse mask)."""
        torch.manual_seed(47)
        shape: Tuple[int, ...] = long_context_shape
        mask_tensor: torch.Tensor = (torch.rand(shape, device=device) > 0.9).to(dtype)
        other_mask_base: Mask = Mask.create_mask_from_dense_mask(
            shape, mask_tensor, dtype=dtype
        )

        # Prepare other_mask in both representations
        other_mask_sparse: Mask = self._prepare_mask_for_mode(other_mask_base, "sparse")
        other_mask_dense: Mask = self._prepare_mask_for_mode(other_mask_base, "dense")

        results: Dict[str, float] = self._benchmark_operation(
            "merge_mask (90% sparsity)",
            sparse_mask_90_percent,
            lambda mask: mask.merge_mask(
                other_mask_sparse, inplace=False, mode="sparse"
            ),
            lambda mask: mask.merge_mask(other_mask_dense, inplace=False, mode="dense"),
        )

        print(f"\n{'='*70}")
        print(f"Operation: {results['operation']}")
        print(f"Sparse time: {results['sparse_time_ms']:.4f} ms")
        print(f"Dense time:  {results['dense_time_ms']:.4f} ms")
        print(
            f"Speedup:     {results['speedup']:.2f}x ({results['faster_mode']} is faster)"
        )
        print(f"{'='*70}")

    def test_comprehensive_timing_summary(
        self,
        sparse_mask_10_percent: Mask,
        sparse_mask_50_percent: Mask,
        sparse_mask_90_percent: Mask,
        long_context_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Comprehensive timing summary for all operations and sparsity levels."""
        input_tensor: torch.Tensor = torch.randn(
            long_context_shape, device=device, dtype=dtype
        )

        # Test configurations
        masks: Dict[str, Mask] = {
            "10% sparsity": sparse_mask_10_percent,
            "50% sparsity": sparse_mask_50_percent,
            "90% sparsity": sparse_mask_90_percent,
        }

        all_results: list = []

        for sparsity_name, mask in masks.items():
            # Test apply_mask
            results: Dict[str, float] = self._benchmark_operation(
                f"apply_mask ({sparsity_name})",
                mask,
                lambda m: m.apply_mask(input_tensor, mode="sparse"),
                lambda m: m.apply_mask(input_tensor, mode="dense"),
            )
            all_results.append(results)

            # Test apply_inv_mask
            results = self._benchmark_operation(
                f"apply_inv_mask ({sparsity_name})",
                mask,
                lambda m: m.apply_inv_mask(input_tensor, mode="sparse"),
                lambda m: m.apply_inv_mask(input_tensor, mode="dense"),
            )
            all_results.append(results)

        # Print comprehensive summary
        print("\n" + "=" * 100)
        print("COMPREHENSIVE TIMING SUMMARY - LONG CONTEXT (B=1, H=32, Q=1, K=32768)")
        print("=" * 100)
        print(
            f"{'Operation':<40} {'Sparse (ms)':<15} {'Dense (ms)':<15} {'Speedup':<10} {'Winner':<10}"
        )
        print("-" * 100)

        for result in all_results:
            print(
                f"{result['operation']:<40} "
                f"{result['sparse_time_ms']:<15.4f} "
                f"{result['dense_time_ms']:<15.4f} "
                f"{result['speedup']:<10.2f}x "
                f"{result['faster_mode']:<10}"
            )

        print("=" * 100)

        # Recommendations
        print("\nRECOMMENDATIONS FOR DEFAULT MODES:")
        print("-" * 100)

        operation_winners: Dict[str, Dict[str, int]] = {}
        for result in all_results:
            op_base: str = result["operation"].split("(")[0].strip()
            if op_base not in operation_winners:
                operation_winners[op_base] = {"sparse": 0, "dense": 0}
            operation_winners[op_base][result["faster_mode"]] += 1

        for op, counts in operation_winners.items():
            winner: str = "sparse" if counts["sparse"] > counts["dense"] else "dense"
            print(f"{op:<30} -> Default mode should be: {winner.upper()}")
            print(f"{' ' * 30}    (wins {counts[winner]}/3 sparsity levels)")

        print("=" * 100 + "\n")
