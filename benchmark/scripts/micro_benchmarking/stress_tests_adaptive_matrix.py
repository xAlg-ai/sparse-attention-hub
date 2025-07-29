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
    LocalMaskerConfig, SinkMaskerConfig, OracleTopKConfig, OracleTopPMaskerConfig, HashAttentionTopKMaskerConfig
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

def get_adaptive_config_name(sink_size, window_size, heavy_size, base_rate_sampling, epsilon, delta):
    return f"adaptive_sampling.sink_{sink_size}_window_{window_size}_heavy_{heavy_size}_base_{base_rate_sampling}_epsilon_{epsilon}_delta_{delta}"

def get_adaptive_hat_config_name(sink_size, window_size, heavy_size, base_rate_sampling, epsilon, delta):
    return f"adaptive_sampling_hat.sink_{sink_size}_window_{window_size}_heavy_{heavy_size}_base_{base_rate_sampling}_epsilon_{epsilon}_delta_{delta}"

def get_oracle_top_p_config_name(sink_size, window_size, top_p):
    return f"oracle_top_p_{top_p}.sink_{sink_size}_window_{window_size}"

def get_oracle_top_k_config_name(sink_size, window_size, top_k):
    return f"oracle_top_k_{top_k}.sink_{sink_size}_window_{window_size}"

def get_hashattention_config_name(sink_size, window_size, top_k):
    return f"hashattention.sink_{sink_size}_window_{window_size}_top_k_{top_k}"

def get_random_sampling_config_name(sink_size, window_size, sampling_rate):
    return f"random_sampling.sink_{sink_size}_window_{window_size}_sampling_rate_{sampling_rate}"

# Sparse Attention Configurations
SPARSE_CONFIGS = []
# adaptive sampling + oracle top k + sink + window
for (epsilon, delta) in [(0.01, 0.01), (0.05, 0.05), (0.1, 0.1), (0.25, 0.25), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]:
    for (sink_size, window_size) in [(0.001, 0.001), (0.005, 0.005), (0.01, 0.01), (0.02, 0.02)]:
        for heavy_size in [0.005, 0.01, 0.02, 0.05]:
            for base_rate_sampling in [0.01, 0.05, 0.1]:
                SPARSE_CONFIGS.append((get_adaptive_config_name(sink_size, window_size, heavy_size, base_rate_sampling, epsilon, delta), ResearchAttentionConfig(masker_configs=[
                    SinkMaskerConfig(sink_size=sink_size),
                    LocalMaskerConfig(window_size=window_size),
                    OracleTopKConfig(heavy_size=heavy_size),
                    AdaptiveSamplingMaskerConfig(base_rate_sampling=base_rate_sampling, epsilon=epsilon, delta=delta, init_offset=sink_size, local_offset=window_size)
                ])))

# oracle top p + sink + window
for (sink_size, window_size) in [(0.001, 0.001), (0.005, 0.005), (0.01, 0.01), (0.02, 0.02)]:
    for top_p in [0.5, 0.75, 0.9, 0.95, 0.99]:
        SPARSE_CONFIGS.append((get_oracle_top_p_config_name(sink_size, window_size, top_p), ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=sink_size),
            LocalMaskerConfig(window_size=window_size),
            OracleTopPMaskerConfig(top_p=top_p)
        ])))

# oracle top p + sink + window
for (sink_size, window_size) in [(0.001, 0.001), (0.005, 0.005), (0.01, 0.01), (0.02, 0.02)]:
    for top_k in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        SPARSE_CONFIGS.append((get_oracle_top_k_config_name(sink_size, window_size, top_k), ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=sink_size),
            LocalMaskerConfig(window_size=window_size),
            OracleTopKConfig(heavy_size=top_k)
        ])))

# usa_weight_file = "/home/apd10/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.pt"
weight_file = "/home/apd10/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"

# from sparse_attention_hub.sparse_attention.utils.hashattention_utils import create_hat_weights_file_from_usa
#create_hat_weights_file_from_usa(usa_weight_file, weight_file, num_layers=32, num_heads=32, device="cpu")


# weight_dictionary = convert_usa_weights_to_hash_attention(weight_file, num_layers=32, num_heads=32, device="cpu")

# hashattention + sink + window
for (sink_size, window_size) in [(0.001, 0.001), (0.005, 0.005), (0.01, 0.01), (0.02, 0.02)]:
    for top_k in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.25, 0.3, 0.4, 0.5]:
        SPARSE_CONFIGS.append((get_hashattention_config_name(sink_size, window_size, top_k), ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=sink_size),
            LocalMaskerConfig(window_size=window_size),
            HashAttentionTopKMaskerConfig(heavy_size=top_k, 
                                        hat_bits=32, 
                                        hat_mlp_layers=3, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
        ])))


# adaptive sampling + hat top k + sink + window
for (epsilon, delta) in [(0.01, 0.01), (0.05, 0.05), (0.1, 0.1), (0.25, 0.25), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]:
    for (sink_size, window_size) in [(0.001, 0.001), (0.005, 0.005), (0.01, 0.01), (0.02, 0.02)]:
        for heavy_size in [0.005, 0.01, 0.02, 0.05, 0.1]:
            for base_rate_sampling in [0.01, 0.05, 0.1]:
                SPARSE_CONFIGS.append((get_adaptive_hat_config_name(sink_size, window_size, heavy_size, base_rate_sampling, epsilon, delta), ResearchAttentionConfig(masker_configs=[
                    SinkMaskerConfig(sink_size=sink_size),
                    LocalMaskerConfig(window_size=window_size),
                    HashAttentionTopKMaskerConfig(heavy_size=heavy_size, 
                                        hat_bits=32, 
                                        hat_mlp_layers=3, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
                    AdaptiveSamplingMaskerConfig(base_rate_sampling=base_rate_sampling, epsilon=epsilon, delta=delta, init_offset=sink_size, local_offset=window_size)
                ])))

# random sampling +  sink + window
for (epsilon, delta) in [(0.01, 0.01), (0.05, 0.05), (0.1, 0.1), (0.25, 0.25), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]:
    for (sink_size, window_size) in [(0.001, 0.001), (0.005, 0.005), (0.01, 0.01), (0.02, 0.02)]:
        for sampling_rate in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                SPARSE_CONFIGS.append((get_random_sampling_config_name(sink_size, window_size, sampling_rate), ResearchAttentionConfig(masker_configs=[
                    SinkMaskerConfig(sink_size=sink_size),
                    LocalMaskerConfig(window_size=window_size),
                    RandomSamplingMaskerConfig(sampling_rate=sampling_rate)
                ])))


# 
print("total number of configs: ", len(SPARSE_CONFIGS))

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
    subsets=["longdep_qa"],
    #subsets=["shortdep_cloze"],
    #subsets=["longdep_summarization"], 
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
    loogle_config,
    #zero_scrolls_config,
    #longbenchv2_config,
    #aime2024_config,
    #aime2025_config,
    #longbench_config,
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
    "max_requests": 1
}

# Execution Settings
RESULT_DIR = "./stress_test_adaptive.matrix/"
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
