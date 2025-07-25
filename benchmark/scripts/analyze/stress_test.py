#!/usr/bin/env python3
"""
Minimalistic benchmark runner for sparse attention evaluation.

This script defines models, sparse attention configurations, and benchmarks,
then runs comprehensive experiments using BenchmarkExecutor.

Usage:
    python benchmark/benchmark.py
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
os.chdir("/home/apd10/code/sparse-attention-hub/")
sys.path.insert(0, "/home/apd10/code/sparse-attention-hub/")

from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig, OracleTopKConfig, OracleTopPMaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig, RandomSamplingMaskerConfig, MagicPigConfig
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# GPU Configuration
GPUS = [0]  # Use all available GPUs
MAX_CONCURRENT_RUNS = 1  # One per GPU

INTENDED_SPARSITY = 0.1

# Model List
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct", 
]

# Sparse Attention Configurations
SPARSE_CONFIGS = [
    ("adaptive_sampling.1", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        OracleTopKConfig(heavy_size=128),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.25, delta=0.25, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.2", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=32),
        LocalMaskerConfig(window_size=32),
        OracleTopKConfig(heavy_size=32),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=512, epsilon=0.25, delta=0.25, init_offset=32, local_offset=32)
    ])),
    ("adaptive_sampling.3", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=32),
        LocalMaskerConfig(window_size=32),
        OracleTopKConfig(heavy_size=32),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=512, epsilon=0.1, delta=0.1, init_offset=32, local_offset=32)
    ])),
    ("adaptive_sampling.4", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=32),
        LocalMaskerConfig(window_size=32),
        OracleTopKConfig(heavy_size=32),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=512, epsilon=0.05, delta=0.05, init_offset=32, local_offset=32)
    ])),
    ("adaptive_sampling.5", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.01),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.2, delta=0.2, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.6", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.01),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.1, delta=0.1, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.7", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.01),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.05, delta=0.05, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.8", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.005),
        LocalMaskerConfig(window_size=0.005),
        OracleTopKConfig(heavy_size=0.01),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.05, delta=0.05, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.9", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.01),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.25, delta=0.25, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.10", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.01),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.30, delta=0.30, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.11", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.01),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.40, delta=0.40, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.12", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.02),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.40, delta=0.40, init_offset=128, local_offset=128)
    ])),
    
    ("adaptive_sampling.13", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.02),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.03, epsilon=0.40, delta=0.40, init_offset=128, local_offset=128)
    ])),

    ("adaptive_sampling.14", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopKConfig(heavy_size=0.03),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.03, epsilon=0.40, delta=0.40, init_offset=128, local_offset=128)
    ])),
    ("adaptive_sampling.15", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.005),
        LocalMaskerConfig(window_size=0.005),
        OracleTopKConfig(heavy_size=0.03),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.03, epsilon=0.40, delta=0.40, init_offset=128, local_offset=128)
    ])),

    # ("adaptive_sampling.2", ResearchAttentionConfig(masker_configs=[
    #     SinkMaskerConfig(sink_size=32),
    #     LocalMaskerConfig(window_size=32),
    #     OracleTopKConfig(heavy_size=32),
    #     AdaptiveSamplingMaskerConfig(base_rate_sampling=512, epsilon=0.1, delta=0.1, init_offset=32, local_offset=32)
    # ])),
    #     ("adaptive_sampling.3", ResearchAttentionConfig(masker_configs=[
    #     SinkMaskerConfig(sink_size=32),
    #     LocalMaskerConfig(window_size=32),
    #     OracleTopKConfig(heavy_size=32),
    #     AdaptiveSamplingMaskerConfig(base_rate_sampling=512, epsilon=0.01, delta=0.01, init_offset=32, local_offset=32)
    # ])),
    # ("adaptive_sampling.4", ResearchAttentionConfig(masker_configs=[
    #     SinkMaskerConfig(sink_size=32),
    #     LocalMaskerConfig(window_size=32),
    #     OracleTopKConfig(heavy_size=32),
    #     AdaptiveSamplingMaskerConfig(base_rate_sampling=512, epsilon=0.1, delta=0.1, init_offset=32, local_offset=32)
    # ])),
    # ("adaptive_sampling.5", ResearchAttentionConfig(masker_configs=[
    #     SinkMaskerConfig(sink_size=32),
    #     LocalMaskerConfig(window_size=32),
    #     OracleTopKConfig(heavy_size=32),
    #     AdaptiveSamplingMaskerConfig(base_rate_sampling=512, epsilon=0.25, delta=0.25, init_offset=32, local_offset=32)
    # ])),
    # ("adaptive_sampling.5", ResearchAttentionConfig(masker_configs=[
    #     SinkMaskerConfig(sink_size=32),
    #     LocalMaskerConfig(window_size=32),
    #     OracleTopKConfig(heavy_size=32),
    #     AdaptiveSamplingMaskerConfig(base_rate_sampling=512, epsilon=0.25, delta=0.25, init_offset=32, local_offset=32)
    # ])),

]

# Benchmark List
# 1. InfiniteBench - using passkey task
infinite_bench_config = BenchmarkConfig(
    benchmark_name="infinite_bench",
    subsets=["passkey"]
)

# 2. Ruler - using 4096 context length
ruler_config = BenchmarkConfig(
    benchmark_name="ruler",
    subsets=["4096"]
)

# 3. Loogle - using shortdep_qa task
loogle_config = BenchmarkConfig(
    benchmark_name="loogle",
    #subsets=["shortdep_qa"],
    #subsets=["longdep_qa"],
    #subsets=["shortdep_cloze"],
    subsets=["longdep_summarization"], 
)

# 4. ZeroScrolls - using gov_report task
zero_scrolls_config = BenchmarkConfig(
    benchmark_name="zero_scrolls",
    subsets=["default"]
)

# 5. LongBenchv2 - using 0shot task
longbenchv2_config = BenchmarkConfig(
    benchmark_name="longbenchv2",
    subsets=["0shot"]
)

# 6. AIME2024 - using single task
aime2024_config = BenchmarkConfig(
    benchmark_name="aime2024",
    subsets=["aime2024"]
)

# 7. AIME2025 - using single task
aime2025_config = BenchmarkConfig(
    benchmark_name="aime2025",
    subsets=["aime2025"]
)

# 8. LongBench (existing) - using narrativeqa task
longbench_config = BenchmarkConfig(
    benchmark_name="longbench",
    subsets=["passage_retrieval_en"]
)

# 9. Mock Benchmark (existing) - using single task
mock_benchmark_config = BenchmarkConfig(
    benchmark_name="mock_benchmark",
    subsets=["reading_comprehension"]
)

# List of all sample configurations
BENCHMARKS = [
    #infinite_bench_config,
    #ruler_config,
    #loogle_config,
    #zero_scrolls_config,
    #longbenchv2_config,
    #aime2024_config,
    #aime2025_config,
    longbench_config,
    #mock_benchmark_config
]


# Adapter Configuration
ADAPTER_CONFIG = AdapterConfig(
    adapter_name="huggingface",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    },
    tokenizer_kwargs={
        "padding_side": "left",
    }
)

# Generation Parameters
GENERATION_KWARGS = {
    "max_new_tokens": 20,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "pad_token_id": None,
}

# Request Parameters
REQUEST_KWARGS = {
    "max_context_length": 32000,
    "max_requests": 5
}

# Execution Settings
RESULT_DIR = "./stress_test_adaptive.this.addmask.addscaling/"
ENABLE_RESUMABILITY = True
TIMEOUT_PER_BENCHMARK = 3600.0  # 1 hour

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Minimalistic Benchmark Suite")
    print("=" * 50)
    
    print(f"🔧 Configuration:")
    print(f"  - GPUs: {GPUS}")
    print(f"  - Models: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"    {i}. {model}")
    print(f"  - Sparse configs: {len(SPARSE_CONFIGS)}")
    for name, config in SPARSE_CONFIGS:
        if config is None:
            print(f"    - {name}: dense (no sparse attention)")
        else:
            sink_size = config.masker_configs[0].sink_size
            window_size = config.masker_configs[1].window_size
            print(f"    - {name}: Sink({sink_size}) + Local({window_size})")
    print(f"  - Benchmarks: {len(BENCHMARKS)}")
    for i, benchmark in enumerate(BENCHMARKS, 1):
        if benchmark.subsets:
            print(f"    {i}. {benchmark.benchmark_name}: {len(benchmark.subsets)} subsets")
        else:
            print(f"    {i}. {benchmark.benchmark_name}: all subsets")
    print(f"  - Max concurrent: {MAX_CONCURRENT_RUNS}")
    print(f"  - Result dir: {RESULT_DIR}")
    print(f"  - Resumability: {'enabled' if ENABLE_RESUMABILITY else 'disabled'}")
    
    # Calculate total combinations
    total_models = len(MODELS)
    total_configs = len(SPARSE_CONFIGS)
    total_benchmarks = sum(len(b.subsets) if b.subsets else 1 for b in BENCHMARKS)
    total_combinations = total_models * total_configs * total_benchmarks
    
    print(f"\n📊 Experiment Matrix: {total_combinations} total combinations")
    print(f"  - Models: {total_models}")
    print(f"  - Sparse configs: {total_configs}")
    print(f"  - Benchmark-subsets: {total_benchmarks}")
    print(f"  - Estimated time: {total_combinations * TIMEOUT_PER_BENCHMARK / 3600:.1f} hours (worst case)")
    
    # Create executor
    print(f"\n🔧 Initializing BenchmarkExecutor...")
    executor = BenchmarkExecutor(
        gpu_ids=GPUS,
        max_concurrent_runs=MAX_CONCURRENT_RUNS,
        base_result_dir=RESULT_DIR,
        enable_resumability=ENABLE_RESUMABILITY,
        required_result_files=["raw_results.csv"],
        timeout_per_benchmark=TIMEOUT_PER_BENCHMARK,
        verbose=True
    )
    
    # Run benchmarks
    print(f"\n🎯 Running Benchmark Matrix...")
    try:
        results = executor.run_benchmark_matrix(
            model_names=MODELS,
            sparse_attention_configs=SPARSE_CONFIGS,
            benchmark_configs=BENCHMARKS,
            adapter_config=ADAPTER_CONFIG,
            generation_kwargs=GENERATION_KWARGS,
            request_kwargs=REQUEST_KWARGS
        )
        
        # Print summary
        print(f"\n✅ Benchmark Execution Completed!")
        print(f"  - Total: {results.progress.total_stubs}")
        print(f"  - Completed: {results.progress.completed_stubs}")
        print(f"  - Failed: {results.progress.failed_stubs}")
        print(f"  - Skipped: {results.progress.skipped_stubs}")
        print(f"  - Results saved to: {RESULT_DIR}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        print(f"  Partial results in: {RESULT_DIR}")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        raise 
