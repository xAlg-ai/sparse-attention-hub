"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-10-20
:summary: Performance tests comparing sparse and dense modes of create_from_row_wise_idx.
"""

import time
import pytest
import torch
from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.performance
class TestCreateFromRowWiseIdxPerformance:
    """Performance tests for create_from_row_wise_idx implementations."""

    def test_create_from_row_wise_idx_performance(self):
        """Compare performance of sparse vs dense mode for create_from_row_wise_idx.
        
        Uses long-context attention scenario: batch=1, heads=32, queries=1, keys=32768
        This simulates incremental decoding with attention over long context.
        """
        
        # Setup device
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        print("\n" + "="*80)
        print("Performance Comparison: create_from_row_wise_idx - Sparse vs Dense Mode")
        print("="*80)
        print("\nLong Context Attention Scenario:")
        print("  Batch size (B) = 1")
        print("  Number of heads (H) = 32")
        print("  Query length (Q) = 1")
        print("  Key length (K) = 32768")
        print("  Hidden dim (D) = 128")
        print(f"  Device: {device}")
        print("="*80)
        
        # Test configurations: (shape, k, sparsity_desc)
        # Shape represents (B, H, Q, K) -> flattened to (B*H*Q, K) for mask creation
        # k represents the number of non-zero elements per row (i.e., how many keys to attend to)
        
        B: int = 1
        H: int = 32
        Q: int = 1
        K: int = 32768
        
        # Effective shape for mask: (B*H*Q, K) = (32, 32768)
        mask_shape: tuple = (B * H * Q, K)
        
        test_configs: list = [
            # Very sparse: attending to very few tokens
            (mask_shape, 32, "0.1%", "Very Sparse (32 tokens)"),
            (mask_shape, 64, "0.2%", "Very Sparse (64 tokens)"),
            (mask_shape, 128, "0.4%", "Very Sparse (128 tokens)"),
            
            # Sparse: typical sparse attention patterns
            (mask_shape, 328, "1%", "Sparse (328 tokens)"),
            (mask_shape, 656, "2%", "Sparse (656 tokens)"),
            (mask_shape, 1638, "5%", "Sparse (1638 tokens)"),
            
            # Medium sparse: moderate attention
            (mask_shape, 3276, "10%", "Medium Sparse (3276 tokens)"),
            (mask_shape, 6553, "20%", "Medium Sparse (6553 tokens)"),
            
            # Medium dense: attending to many tokens
            (mask_shape, 16384, "50%", "Medium Dense (16384 tokens)"),
            (mask_shape, 26214, "80%", "Dense (26214 tokens)"),
            
            # Very dense: near-full attention
            (mask_shape, 29491, "90%", "Very Dense (29491 tokens)"),
            (mask_shape, 32440, "99%", "Very Dense (32440 tokens)"),
        ]
        
        # Benchmark parameters
        num_warmup: int = 5
        num_iterations: int = 20
        
        # Results storage
        results: list = []
        
        for shape, k, sparsity, desc in test_configs:
            n: int = shape[-1]
            batch_dims: tuple = shape[:-1]
            
            print(f"\n{desc}")
            print(f"  Shape: {shape}, k={k}, sparsity={sparsity}")
            print("-" * 80)
            
            # Generate test data on device
            row_wise_idx: torch.Tensor = torch.randint(
                0, n, size=batch_dims + (k,), dtype=torch.long, device=device
            )
            data: torch.Tensor = torch.rand(batch_dims + (k,), dtype=torch.float32, device=device)
            
            # Test both mask types
            for mask_type in ["dense", "index"]:
                # Benchmark sparse mode
                times_sparse: list = []
                
                # Warmup
                for _ in range(num_warmup):
                    _ = Mask.create_from_row_wise_idx(
                        shape, row_wise_idx, data, 
                        mask_type=mask_type, 
                        dtype=torch.float32, 
                        mode="sparse"
                    )
                
                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Timing runs
                for _ in range(num_iterations):
                    start: float = time.perf_counter()
                    _ = Mask.create_from_row_wise_idx(
                        shape, row_wise_idx, data, 
                        mask_type=mask_type, 
                        dtype=torch.float32, 
                        mode="sparse"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end: float = time.perf_counter()
                    times_sparse.append((end - start) * 1000)  # Convert to ms
                
                avg_sparse: float = sum(times_sparse) / len(times_sparse)
                
                # Benchmark dense mode
                times_dense: list = []
                
                # Warmup
                for _ in range(num_warmup):
                    _ = Mask.create_from_row_wise_idx(
                        shape, row_wise_idx, data, 
                        mask_type=mask_type, 
                        dtype=torch.float32, 
                        mode="dense"
                    )
                
                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Timing runs
                for _ in range(num_iterations):
                    start: float = time.perf_counter()
                    _ = Mask.create_from_row_wise_idx(
                        shape, row_wise_idx, data, 
                        mask_type=mask_type, 
                        dtype=torch.float32, 
                        mode="dense"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end: float = time.perf_counter()
                    times_dense.append((end - start) * 1000)  # Convert to ms
                
                avg_dense: float = sum(times_dense) / len(times_dense)
                
                # Calculate speedup
                speedup: float = avg_sparse / avg_dense
                winner: str = "Dense" if speedup > 1.0 else "Sparse"
                
                print(f"  mask_type={mask_type:6s} | Sparse: {avg_sparse:8.3f} ms | "
                      f"Dense: {avg_dense:8.3f} ms | Speedup: {speedup:5.2f}x | Winner: {winner}")
                
                # Store results
                results.append({
                    "config": desc,
                    "shape": shape,
                    "k": k,
                    "sparsity": sparsity,
                    "mask_type": mask_type,
                    "sparse_ms": avg_sparse,
                    "dense_ms": avg_dense,
                    "speedup": speedup,
                    "winner": winner
                })
        
        # Summary report
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # Group by sparsity level
        categories: dict = {
            "Very Sparse (0.1-0.4%)": [],
            "Sparse (1-5%)": [],
            "Medium Sparse (10-20%)": [],
            "Dense (50-99%)": []
        }
        
        for result in results:
            config: str = result["config"]
            sparsity_pct: float = float(result["sparsity"].rstrip("%"))
            
            if sparsity_pct <= 0.4:
                categories["Very Sparse (0.1-0.4%)"].append(result)
            elif sparsity_pct <= 5:
                categories["Sparse (1-5%)"].append(result)
            elif sparsity_pct <= 20:
                categories["Medium Sparse (10-20%)"].append(result)
            else:
                categories["Dense (50-99%)"].append(result)
        
        for category_name, category_results in categories.items():
            if not category_results:
                continue
                
            print(f"\n{category_name}:")
            print("-" * 80)
            
            dense_wins: int = sum(1 for r in category_results if r["winner"] == "Dense")
            sparse_wins: int = sum(1 for r in category_results if r["winner"] == "Sparse")
            avg_speedup: float = sum(r["speedup"] for r in category_results) / len(category_results)
            
            # Show detailed results for each config in this category
            for r in category_results:
                winner_symbol: str = "🏆" if r["winner"] == "Dense" else "⚡"
                print(f"  {r['config']:<35s} | Speedup: {r['speedup']:5.2f}x | Winner: {r['winner']} {winner_symbol}")
            
            print(f"\n  Summary:")
            print(f"    Dense wins: {dense_wins}/{len(category_results)} ({100*dense_wins/len(category_results):.0f}%)")
            print(f"    Sparse wins: {sparse_wins}/{len(category_results)} ({100*sparse_wins/len(category_results):.0f}%)")
            print(f"    Average speedup (sparse/dense): {avg_speedup:.2f}x")
            
            if dense_wins > sparse_wins:
                print(f"    🏆 Category Winner: Dense mode")
            else:
                print(f"    ⚡ Category Winner: Sparse mode")
        
        # Overall winner
        print("\n" + "="*80)
        total_dense_wins: int = sum(1 for r in results if r["winner"] == "Dense")
        total_sparse_wins: int = sum(1 for r in results if r["winner"] == "Sparse")
        overall_avg_speedup: float = sum(r["speedup"] for r in results) / len(results)
        
        print(f"Overall Results:")
        print(f"  Dense wins: {total_dense_wins}/{len(results)} ({100*total_dense_wins/len(results):.1f}%)")
        print(f"  Sparse wins: {total_sparse_wins}/{len(results)} ({100*total_sparse_wins/len(results):.1f}%)")
        print(f"  Average speedup (sparse/dense): {overall_avg_speedup:.2f}x")
        
        if total_dense_wins > total_sparse_wins:
            print(f"\n🏆 Overall Winner: Dense mode")
            print(f"   Recommendation: Use mode='dense' as default")
        else:
            print(f"\n🏆 Overall Winner: Sparse mode")
            print(f"   Recommendation: Keep mode='sparse' as default")
        
        print("="*80 + "\n")
        
        return results

    def test_create_from_row_wise_idx_detailed_analysis(self):
        """Detailed performance analysis varying sparsity levels.
        
        Uses long-context attention scenario: batch=1, heads=32, queries=1, keys=32768
        """
        
        # Setup device
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        print("\n" + "="*80)
        print("Detailed Performance Analysis: Sparsity Impact")
        print("="*80)
        
        # Long context attention scenario
        B: int = 1
        H: int = 32
        Q: int = 1
        K: int = 32768
        
        shape: tuple = (B * H * Q, K)  # (32, 32768)
        n: int = shape[-1]
        
        sparsity_levels: list = [
            (32, "0.1%"),
            (64, "0.2%"),
            (164, "0.5%"),
            (328, "1%"),
            (656, "2%"),
            (1638, "5%"),
            (3276, "10%"),
            (6553, "20%"),
            (9830, "30%"),
            (16384, "50%"),
            (22937, "70%"),
            (26214, "80%"),
            (29491, "90%"),
            (31129, "95%"),
            (32440, "99%"),
        ]
        
        num_warmup: int = 5
        num_iterations: int = 20
        
        print(f"\nLong Context Attention Scenario:")
        print(f"  Batch={B}, Heads={H}, Queries={Q}, Keys={K}")
        print(f"  Mask shape: {shape}")
        print(f"  Device: {device}")
        print("\nVarying sparsity from 0.1% to 99%")
        print("-" * 90)
        print(f"{'Sparsity':<10s} | {'k (tokens)':<12s} | {'Sparse (ms)':<12s} | {'Dense (ms)':<12s} | {'Speedup':<8s} | {'Winner':<8s}")
        print("-" * 90)
        
        for k, sparsity_str in sparsity_levels:
            row_wise_idx: torch.Tensor = torch.randint(
                0, n, size=(shape[0], k), dtype=torch.long, device=device
            )
            data: torch.Tensor = torch.rand(shape[0], k, dtype=torch.float32, device=device)
            
            # Sparse mode
            times_sparse: list = []
            for _ in range(num_warmup):
                _ = Mask.create_from_row_wise_idx(
                    shape, row_wise_idx, data, 
                    mask_type="dense", dtype=torch.float32, mode="sparse"
                )
            
            # Synchronize before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            for _ in range(num_iterations):
                start: float = time.perf_counter()
                _ = Mask.create_from_row_wise_idx(
                    shape, row_wise_idx, data, 
                    mask_type="dense", dtype=torch.float32, mode="sparse"
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end: float = time.perf_counter()
                times_sparse.append((end - start) * 1000)
            
            avg_sparse: float = sum(times_sparse) / len(times_sparse)
            
            # Dense mode
            times_dense: list = []
            for _ in range(num_warmup):
                _ = Mask.create_from_row_wise_idx(
                    shape, row_wise_idx, data, 
                    mask_type="dense", dtype=torch.float32, mode="dense"
                )
            
            # Synchronize before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            for _ in range(num_iterations):
                start: float = time.perf_counter()
                _ = Mask.create_from_row_wise_idx(
                    shape, row_wise_idx, data, 
                    mask_type="dense", dtype=torch.float32, mode="dense"
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end: float = time.perf_counter()
                times_dense.append((end - start) * 1000)
            
            avg_dense: float = sum(times_dense) / len(times_dense)
            speedup: float = avg_sparse / avg_dense
            winner: str = "Dense" if speedup > 1.0 else "Sparse"
            
            print(f"{sparsity_str:<10s} | {k:<12d} | {avg_sparse:<12.3f} | {avg_dense:<12.3f} | {speedup:<8.2f} | {winner:<8s}")
        
        print("-" * 90)
        print("\n" + "="*80 + "\n")

