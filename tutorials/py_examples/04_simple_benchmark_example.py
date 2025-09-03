#!/usr/bin/env python3
"""
Simple Benchmark Example

A beginner-friendly example showing how to run a basic benchmark comparison
between dense and sparse attention using the sparse-attention-hub framework.

This example uses the MockBenchmark (5 simple samples) for quick demonstration:
- Easy-to-understand reading comprehension questions
- Short contexts (<250 words each)
- Fast execution for testing and learning

Usage:
    python 04_simple_benchmark_example.py
"""

import os
import time
from pathlib import Path

import torch

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
from benchmark import MockBenchmark


def main():
    """Run a simple benchmark comparison between dense and sparse attention."""

    print("🎯 Simple Benchmark Example")
    print("=" * 40)

    # Configuration
    model_name = "microsoft/Phi-4-mini-instruct"  # Small model for quick testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model: {model_name}")
    print(f"Device: {device}")

    # Create StreamingLLM configuration: Sink (4 tokens) + Local (32 tokens)
    sink_config = SinkMaskerConfig(sink_size=4)
    local_config = LocalMaskerConfig(window_size=32)
    streaming_config = ResearchAttentionConfig(
        masker_configs=[sink_config, local_config]
    )

    print("StreamingLLM Config: Sink(4) + Local(32)")

    # Common model arguments
    model_kwargs = {
        "model_kwargs": {"torch_dtype": torch.bfloat16},
        "device": str(device),
    }

    # Initialize adapters
    print("\n🔧 Loading models...")

    # Sparse adapter (StreamingLLM)
    print("  ✓ Loading sparse attention model...")
    sparse_adapter = ModelAdapterHF(
        model_name=model_name, sparse_attention_config=streaming_config, **model_kwargs
    )

    # Create mock benchmark - fast and simple
    mock_benchmark = MockBenchmark()

    print(f"\n📊 Running MockBenchmark:")
    print(f"  - 5 simple reading comprehension samples")
    print(f"  - 3 different contexts (science, history, geography)")
    print(f"  - Context sharing to test grouping efficiency")

    # Create result directories
    result_dir = Path("./simple_benchmark_results")
    result_dir.mkdir(exist_ok=True)

    sparse_dir = result_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    print("\n🧪 Running Sparse Attention Benchmark...")
    start_time = time.time()

    # Show dataset overview
    dataset_df = mock_benchmark._load_datasets()
    print(
        f"  Processing {len(dataset_df)} samples across {dataset_df['context'].nunique()} contexts..."
    )

    # Run sparse benchmark
    with sparse_adapter.enable_sparse_mode():
        sparse_metrics = mock_benchmark.run_benchmark(
            adapter=sparse_adapter, result_dir=str(sparse_dir)
        )

    sparse_time = time.time() - start_time
    print(f"  ✅ Sparse completed in {sparse_time:.2f}s")
    print(f"     Accuracy: {sparse_metrics.get('accuracy', 'N/A')}")
    print(
        f"     Correct predictions: {sparse_metrics.get('correct_predictions', 'N/A')}/{sparse_metrics.get('total_samples', 'N/A')}"
    )

    print("\n📋 Results Summary:")
    print(f"  • Benchmark: {sparse_metrics['summary']['benchmark']}")
    print(f"  • Task: {sparse_metrics['summary']['task']}")
    print(f"  • Unique contexts: {sparse_metrics['summary']['unique_contexts']}")
    print(f"  • Evaluation method: {sparse_metrics['summary']['evaluation_method']}")

    print(f"\n💾 Results saved to: {sparse_dir}")
    print("   - raw_results.csv: Detailed predictions for each sample")
    print("   - metrics.json: Evaluation metrics and summary")


if __name__ == "__main__":
    main()
