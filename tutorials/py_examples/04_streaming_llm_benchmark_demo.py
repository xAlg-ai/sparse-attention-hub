#!/usr/bin/env python3
"""
Streaming LLM Benchmark Demo

This example demonstrates how to run benchmarks using the sparse-attention-hub framework.
It compares streaming attention (sparse) vs full attention performance on benchmark tasks.

Features:
- StreamingLLM configuration with sink + local attention
- MockBenchmark (default) for quick testing or LongBench for comprehensive evaluation
- Performance comparison between sparse and dense attention
- Comprehensive result analysis and visualization
- Memory usage tracking
- Timing analysis

Usage:
    python 04_streaming_llm_benchmark_demo.py [--model MODEL_NAME] [--benchmark mock|longbench] [--subsets SUBSET1,SUBSET2]
"""

import argparse
import os
import time
import json
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pandas as pd
import matplotlib.pyplot as plt

# Ensure we're in the correct directory and add to Python path
import sys

os.chdir("/data/apdesai/code/sparse-attention-hub")
sys.path.insert(0, "/data/apdesai/code/sparse-attention-hub")

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from benchmark import Benchmark, LongBench, MockBenchmark


class BenchmarkRunner:
    """Comprehensive benchmark runner for streaming attention evaluation."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-mini-instruct",
        device: str = "cuda",
        sink_size: int = 4,
        local_window: int = 64,
        result_dir: str = "./benchmark_results",
        benchmark_type: str = "mock",
    ):
        """Initialize the benchmark runner.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
            sink_size: Number of sink tokens for StreamingLLM
            local_window: Local attention window size
            result_dir: Directory to save results
            benchmark_type: Type of benchmark to use ('mock' or 'longbench')
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sink_size = sink_size
        self.local_window = local_window
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.benchmark_type = benchmark_type

        print(f"🚀 Initializing Benchmark Runner")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Benchmark: {benchmark_type.upper()}")
        print(f"   StreamingLLM Config: Sink({sink_size}) + Local({local_window})")
        print(f"   Results will be saved to: {self.result_dir}")

        # Create StreamingLLM configuration
        self.streaming_config = self._create_streaming_config()

        # Initialize adapters
        self.dense_adapter = None
        self.sparse_adapter = None

    def _create_streaming_config(self) -> ResearchAttentionConfig:
        """Create StreamingLLM attention configuration."""
        sink_config = SinkMaskerConfig(sink_size=self.sink_size)
        local_config = LocalMaskerConfig(window_size=self.local_window)
        return ResearchAttentionConfig(masker_configs=[sink_config, local_config])

    def initialize_adapters(self) -> None:
        """Initialize both dense and sparse attention adapters."""
        print("🔧 Initializing model adapters...")

        # Common model arguments
        common_kwargs = {
            "model_kwargs": {"torch_dtype": torch.bfloat16},
            "device": str(self.device),
        }

        # Dense adapter (no sparse attention)
        print("   ✓ Loading dense attention adapter...")
        self.dense_adapter = ModelAdapterHF(
            model_name=self.model_name,
            sparse_attention_config=None,  # No sparse attention = dense mode
            **common_kwargs,
        )

        # Sparse adapter (StreamingLLM)
        print("   ✓ Loading sparse attention adapter...")
        self.sparse_adapter = ModelAdapterHF(
            model_name=self.model_name,
            sparse_attention_config=self.streaming_config,
            **common_kwargs,
        )

        print("✅ Adapters initialized successfully!")

    def run_benchmark_comparison(
        self,
        benchmark_subsets: Optional[List[str]] = None,
        max_samples: Optional[int] = 10,
    ) -> Dict[str, Dict]:
        """Run benchmark comparison between dense and sparse attention.

        Args:
            benchmark_subsets: List of benchmark subsets to run. For MockBenchmark, this parameter is ignored.
                               For LongBench, if None, uses a small default set.
            max_samples: Maximum number of samples per subset for quick testing (ignored for MockBenchmark)

        Returns:
            Dictionary containing results for both dense and sparse runs
        """
        # Create benchmark instance based on type
        if self.benchmark_type == "mock":
            benchmark = MockBenchmark()
            print(f"🧪 Running MockBenchmark comparison:")
            print(f"   - 5 simple reading comprehension samples")
            print(f"   - 3 different contexts (science, history, geography)")
            print(f"   - Fast execution for testing and learning")
        else:  # longbench
            if benchmark_subsets is None:
                # Use a small subset for demonstration
                benchmark_subsets = ["narrativeqa", "qasper", "samsum"]

            print(f"🧪 Running LongBench comparison on subsets: {benchmark_subsets}")
            benchmark = LongBench(subsets_to_run=benchmark_subsets)

        results = {}

        # Run dense attention benchmark
        print("\n📊 Running Dense Attention Benchmark...")
        if self.dense_adapter is None:
            raise RuntimeError(
                "Dense adapter not initialized. Call initialize_adapters() first."
            )
        results["dense"] = self._run_single_benchmark(
            adapter=self.dense_adapter,
            benchmark=benchmark,
            mode_name="dense",
            max_samples=max_samples,
        )

        # Run sparse attention benchmark
        print("\n⚡ Running Sparse Attention Benchmark...")
        if self.sparse_adapter is None:
            raise RuntimeError(
                "Sparse adapter not initialized. Call initialize_adapters() first."
            )
        results["sparse"] = self._run_single_benchmark(
            adapter=self.sparse_adapter,
            benchmark=benchmark,
            mode_name="sparse",
            max_samples=max_samples,
        )

        return results

    def _run_single_benchmark(
        self,
        adapter: ModelAdapterHF,
        benchmark: Benchmark,
        mode_name: str,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """Run benchmark with a single adapter and track performance metrics.

        Args:
            adapter: The model adapter to use
            benchmark: The benchmark instance
            mode_name: Name for this benchmark run ('dense' or 'sparse')
            max_samples: Maximum samples to process (for quick testing)

        Returns:
            Dictionary containing benchmark results and performance metrics
        """
        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()

        # Create timestamped result directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_result_dir = self.result_dir / f"{mode_name}_{timestamp}"
        mode_result_dir.mkdir(exist_ok=True)

        try:
            # Load and optionally limit dataset size
            print(f"   📋 Loading {mode_name} benchmark data...")
            dataset_df = benchmark._load_datasets()

            if max_samples:
                print(
                    f"   ✂️  Limiting to {max_samples} samples per task for quick testing"
                )
                dataset_df = (
                    dataset_df.groupby("task").head(max_samples).reset_index(drop=True)
                )

            print(f"   📈 Processing {len(dataset_df)} samples...")

            # Run the benchmark
            if mode_name == "sparse":
                # Use sparse attention mode
                with adapter.enable_sparse_mode():
                    metrics = benchmark.run_benchmark(
                        adapter=adapter, result_dir=str(mode_result_dir)
                    )
            else:
                # Use dense attention mode
                metrics = benchmark.run_benchmark(
                    adapter=adapter, result_dir=str(mode_result_dir)
                )

            # Track performance metrics
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Compile results
            performance_metrics = {
                "execution_time_seconds": end_time - start_time,
                "peak_memory_mb": peak / 1024 / 1024,
                "current_memory_mb": current / 1024 / 1024,
                "samples_processed": len(dataset_df),
                "avg_time_per_sample": (end_time - start_time) / len(dataset_df),
            }

            result = {
                "benchmark_metrics": metrics,
                "performance_metrics": performance_metrics,
                "result_dir": str(mode_result_dir),
                "model_config": {
                    "model_name": self.model_name,
                    "mode": mode_name,
                    "streaming_config": (
                        {"sink_size": self.sink_size, "local_window": self.local_window}
                        if mode_name == "sparse"
                        else None
                    ),
                },
            }

            # Save detailed results
            with open(mode_result_dir / "detailed_results.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

            print(f"   ✅ {mode_name.title()} benchmark completed!")
            print(f"      Time: {performance_metrics['execution_time_seconds']:.2f}s")
            print(f"      Peak Memory: {performance_metrics['peak_memory_mb']:.1f}MB")
            print(f"      Avg Score: {metrics.get('average_score', 'N/A')}")

            return result

        except Exception as e:
            print(f"   ❌ Error in {mode_name} benchmark: {str(e)}")
            tracemalloc.stop()
            raise e

    def analyze_results(self, results: Dict[str, Dict]) -> None:
        """Analyze and visualize benchmark comparison results.

        Args:
            results: Results dictionary from run_benchmark_comparison
        """
        print("\n📊 Analyzing Results...")

        # Extract metrics for comparison
        dense_metrics = results["dense"]["performance_metrics"]
        sparse_metrics = results["sparse"]["performance_metrics"]

        dense_benchmark = results["dense"]["benchmark_metrics"]
        sparse_benchmark = results["sparse"]["benchmark_metrics"]

        # Performance comparison
        print("\n📈 Performance Comparison:")
        print("─" * 50)
        print(f"{'Metric':<25} {'Dense':<15} {'Sparse':<15} {'Speedup':<10}")
        print("─" * 50)

        time_speedup = (
            dense_metrics["execution_time_seconds"]
            / sparse_metrics["execution_time_seconds"]
        )
        memory_reduction = (
            (dense_metrics["peak_memory_mb"] - sparse_metrics["peak_memory_mb"])
            / dense_metrics["peak_memory_mb"]
            * 100
        )

        print(
            f"{'Execution Time (s)':<25} {dense_metrics['execution_time_seconds']:<15.2f} {sparse_metrics['execution_time_seconds']:<15.2f} {time_speedup:<10.2f}x"
        )
        print(
            f"{'Peak Memory (MB)':<25} {dense_metrics['peak_memory_mb']:<15.1f} {sparse_metrics['peak_memory_mb']:<15.1f} {memory_reduction:<10.1f}%"
        )
        print(
            f"{'Avg Time/Sample (s)':<25} {dense_metrics['avg_time_per_sample']:<15.3f} {sparse_metrics['avg_time_per_sample']:<15.3f}"
        )

        # Accuracy comparison
        print("\n🎯 Accuracy Comparison:")
        print("─" * 40)
        dense_score = dense_benchmark.get("average_score", 0)
        sparse_score = sparse_benchmark.get("average_score", 0)
        accuracy_retention = (
            (sparse_score / dense_score * 100) if dense_score > 0 else 0
        )

        print(f"Dense Attention Score:    {dense_score:.3f}")
        print(f"Sparse Attention Score:   {sparse_score:.3f}")
        print(f"Accuracy Retention:       {accuracy_retention:.1f}%")

        # Create visualization
        self._create_visualization(results)

        # Summary
        print(f"\n🏆 Summary:")
        print(f"   StreamingLLM achieves {time_speedup:.2f}x speedup")
        print(f"   Reduces memory usage by {memory_reduction:.1f}%")
        print(f"   Retains {accuracy_retention:.1f}% of original accuracy")

    def _create_visualization(self, results: Dict[str, Dict]) -> None:
        """Create visualization comparing dense vs sparse performance."""
        try:
            import matplotlib.pyplot as plt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Dense vs Sparse Attention Benchmark Comparison", fontsize=16)

            # Extract data
            dense_perf = results["dense"]["performance_metrics"]
            sparse_perf = results["sparse"]["performance_metrics"]
            dense_bench = results["dense"]["benchmark_metrics"]
            sparse_bench = results["sparse"]["benchmark_metrics"]

            # Execution time comparison
            times = [
                dense_perf["execution_time_seconds"],
                sparse_perf["execution_time_seconds"],
            ]
            ax1.bar(["Dense", "Sparse"], times, color=["#ff7f0e", "#2ca02c"])
            ax1.set_ylabel("Execution Time (seconds)")
            ax1.set_title("Execution Time Comparison")

            # Memory usage comparison
            memories = [dense_perf["peak_memory_mb"], sparse_perf["peak_memory_mb"]]
            ax2.bar(["Dense", "Sparse"], memories, color=["#ff7f0e", "#2ca02c"])
            ax2.set_ylabel("Peak Memory (MB)")
            ax2.set_title("Memory Usage Comparison")

            # Accuracy comparison
            scores = [
                dense_bench.get("average_score", 0),
                sparse_bench.get("average_score", 0),
            ]
            ax3.bar(["Dense", "Sparse"], scores, color=["#ff7f0e", "#2ca02c"])
            ax3.set_ylabel("Average Score")
            ax3.set_title("Accuracy Comparison")
            ax3.set_ylim(0, 1)

            # Per-sample time comparison
            per_sample_times = [
                dense_perf["avg_time_per_sample"],
                sparse_perf["avg_time_per_sample"],
            ]
            ax4.bar(["Dense", "Sparse"], per_sample_times, color=["#ff7f0e", "#2ca02c"])
            ax4.set_ylabel("Time per Sample (seconds)")
            ax4.set_title("Per-Sample Processing Time")

            plt.tight_layout()

            # Save plot
            plot_path = (
                self.result_dir
                / f"benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"📊 Visualization saved to: {plot_path}")

            # Also try to show if in interactive environment
            try:
                plt.show()
            except:
                pass

        except ImportError:
            print("📊 Matplotlib not available for visualization")
        except Exception as e:
            print(f"📊 Could not create visualization: {str(e)}")


def main():
    """Main function to run the streaming LLM benchmark demo."""
    parser = argparse.ArgumentParser(description="Streaming LLM Benchmark Demo")
    parser.add_argument(
        "--model",
        default="microsoft/Phi-4-mini-instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--benchmark",
        default="mock",
        choices=["mock", "longbench"],
        help="Benchmark type: 'mock' for quick testing (5 samples) or 'longbench' for comprehensive evaluation",
    )
    parser.add_argument(
        "--subsets",
        default="narrativeqa,qasper,samsum",
        help="Comma-separated list of LongBench subsets (ignored for MockBenchmark)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum samples per subset for quick testing (ignored for MockBenchmark)",
    )
    parser.add_argument(
        "--sink-size", type=int, default=4, help="Number of sink tokens"
    )
    parser.add_argument(
        "--local-window", type=int, default=64, help="Local attention window size"
    )
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--result-dir", default="./benchmark_results", help="Directory to save results"
    )

    args = parser.parse_args()

    print("🎯 Streaming LLM Benchmark Demo")
    print("=" * 50)

    # Parse subsets
    subsets = [s.strip() for s in args.subsets.split(",")]

    # Initialize benchmark runner
    runner = BenchmarkRunner(
        model_name=args.model,
        device=args.device,
        sink_size=args.sink_size,
        local_window=args.local_window,
        result_dir=args.result_dir,
        benchmark_type=args.benchmark,
    )

    # Initialize adapters
    runner.initialize_adapters()

    # Run benchmark comparison
    results = runner.run_benchmark_comparison(
        benchmark_subsets=subsets, max_samples=args.max_samples
    )

    # Analyze results
    runner.analyze_results(results)

    print("\n✅ Benchmark demo completed successfully!")
    print(f"📁 Results saved to: {runner.result_dir}")


if __name__ == "__main__":
    main()
