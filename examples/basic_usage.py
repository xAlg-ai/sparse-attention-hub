#!/usr/bin/env python3
"""
Basic usage example for Sparse Attention Hub.

This example demonstrates how to:
1. Create a sparse attention mechanism
2. Set up a model hub for HuggingFace integration
3. Run basic benchmarks
4. Generate visualizations
"""

import torch

from sparse_attention_hub import (
    BenchmarkExecutor,
    Granularity,
    MicroMetricLogger,
    ModelHubHF,
    PipelineHF,
    PlotGenerator,
    SparseAttentionHF,
)
from sparse_attention_hub.benchmark.datasets import LongBench
from sparse_attention_hub.metrics.implementations import SampleVariance
from sparse_attention_hub.sparse_attention.efficient import DoubleSparsity


def main():
    """Main example function."""
    print("Sparse Attention Hub - Basic Usage Example")
    print("=" * 50)

    # 1. Create a sparse attention mechanism
    print("1. Creating sparse attention mechanism...")
    sparse_attention = DoubleSparsity()
    attention_generator = SparseAttentionHF(sparse_attention)
    print("   ✓ DoubleSparsity attention created")

    # 2. Set up model hub (placeholder - would use actual HF model)
    print("\n2. Setting up model hub...")
    model_hub = ModelHubHF(api_token="your_token_here")
    print("   ✓ HuggingFace model hub initialized")

    # 3. Set up metrics logging
    print("\n3. Setting up metrics logging...")
    logger = MicroMetricLogger()
    variance_metric = SampleVariance()
    logger.register_metric(variance_metric)
    logger.enable_metric_logging(variance_metric)
    print("   ✓ Metrics logging configured")

    # 4. Create benchmark executor
    print("\n4. Setting up benchmarks...")
    benchmark_executor = BenchmarkExecutor()
    longbench = LongBench()
    benchmark_executor.register_benchmark(longbench)
    print("   ✓ LongBench registered")

    # 5. Generate sample plots
    print("\n5. Generating sample visualizations...")
    plot_generator = PlotGenerator(storage_path="./example_plots")

    # Generate plots for different granularities
    for granularity in [
        Granularity.PER_TOKEN,
        Granularity.PER_HEAD,
        Granularity.PER_LAYER,
    ]:
        plot_path = plot_generator.generate_plot(granularity)
        print(f"   ✓ Generated {granularity.value} plot: {plot_path}")

    # 6. Log some sample metrics
    print("\n6. Logging sample metrics...")
    sample_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    variance_value = variance_metric.compute(sample_data)
    logger.log("example_location", variance_metric, variance_value)
    print(f"   ✓ Logged sample variance: {variance_value:.4f}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nNext steps:")
    print("- Replace placeholder model with actual HuggingFace model")
    print("- Implement specific attention algorithms")
    print("- Run real benchmarks on your models")
    print("- Explore different sparse attention patterns")


if __name__ == "__main__":
    main()
