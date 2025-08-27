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

MACHINE_KEY= "apd10/code"

# Add the project root to the path
os.chdir("/workspace/sparse-attention-hub/")
sys.path.insert(0, "/workspace/sparse-attention-hub/")

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

# HashAttention weights
# usa_weight_file = "/home/apd10/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.pt"
weight_file = "/workspace/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"
# from sparse_attention_hub.sparse_attention.utils.hashattention_utils import create_hat_weights_file_from_usa
#create_hat_weights_file_from_usa(usa_weight_file, weight_file, num_layers=32, num_heads=32, device="cpu")


################################### BENCHMARK CONFIGS #####################################################
#
#
##########################################################################################################

# Sparse Attention Configurations
SPARSE_CONFIGS = []
# dense 
SPARSE_CONFIGS.append(("dense", None))
# ==================== 5 % configs =================

# random sampling 5 %
SPARSE_CONFIGS.append((get_random_sampling_config_name(0.02, 0.02, 0.01), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.02),
    LocalMaskerConfig(window_size=0.02),
    RandomSamplingMaskerConfig(sampling_rate=0.01)
])))
#adaptive sampling with oracle top k 5 %
SPARSE_CONFIGS.append((get_adaptive_config_name(0.001, 0.001, 0.02, 0.01, 0.1, 0.1), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.001),
    LocalMaskerConfig(window_size=0.001),
    OracleTopKConfig(heavy_size=0.02),
    AdaptiveSamplingMaskerConfig(base_rate_sampling=0.01, epsilon=0.1, delta=0.1, init_offset=0.001, local_offset=0.001)
])))
# adative sampling with hat top k 5 %
SPARSE_CONFIGS.append((get_adaptive_hat_config_name(0.01, 0.01, 0.02, 0.01, 0.25, 0.25), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.01),
    LocalMaskerConfig(window_size=0.01),
    HashAttentionTopKMaskerConfig(heavy_size=0.02, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
    AdaptiveSamplingMaskerConfig(base_rate_sampling=0.01, epsilon=0.25, delta=0.25, init_offset=0.001, local_offset=0.001)
])))
# hat top k 5 %
SPARSE_CONFIGS.append((get_hashattention_config_name(0.005, 0.005, 0.04), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.005),
    LocalMaskerConfig(window_size=0.005),
    HashAttentionTopKMaskerConfig(heavy_size=0.04, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
])))

# oracle top p 5 %
SPARSE_CONFIGS.append((get_oracle_top_p_config_name(0.001, 0.001, 0.75), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.001),
    LocalMaskerConfig(window_size=0.001),
    OracleTopPMaskerConfig(top_p=0.75)
])))

# oracle top k 5 %
SPARSE_CONFIGS.append((get_oracle_top_k_config_name(0.005, 0.005, 0.04), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.005),
    LocalMaskerConfig(window_size=0.005),
    OracleTopKConfig(heavy_size=0.04)
])))

# ==================== 10 % configs =================


# random sampling 10 %
SPARSE_CONFIGS.append((get_random_sampling_config_name(0.001, 0.001, 0.1), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.001),
    LocalMaskerConfig(window_size=0.001),
    RandomSamplingMaskerConfig(sampling_rate=0.1)
])))

#adaptive sampling with oracle top k 10 %
SPARSE_CONFIGS.append((get_adaptive_config_name(0.001, 0.001, 0.05, 0.05, 0.25, 0.25), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.001),
    LocalMaskerConfig(window_size=0.001),
    OracleTopKConfig(heavy_size=0.05),
    AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.25, delta=0.25, init_offset=0.001, local_offset=0.001)
])))

# adative sampling with hat top k 10 %
SPARSE_CONFIGS.append((get_adaptive_hat_config_name(0.001, 0.001, 0.05, 0.05, 0.4, 0.4), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.001),
    LocalMaskerConfig(window_size=0.001),
    HashAttentionTopKMaskerConfig(heavy_size=0.05, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
    AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.4, delta=0.4, init_offset=0.001, local_offset=0.001)
])))
# hat top k 10 %
SPARSE_CONFIGS.append((get_hashattention_config_name(0.001, 0.001, 0.09), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.001),
    LocalMaskerConfig(window_size=0.001),
    HashAttentionTopKMaskerConfig(heavy_size=0.09, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
])))

# oracle top p 10 %
SPARSE_CONFIGS.append((get_oracle_top_p_config_name(0.02, 0.02, 0.8), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.02),
    LocalMaskerConfig(window_size=0.02),
    OracleTopPMaskerConfig(top_p=0.8)
])))

# oracle top k 10 %
SPARSE_CONFIGS.append((get_oracle_top_k_config_name(0.001, 0.001, 0.04), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.001),
    LocalMaskerConfig(window_size=0.001),
    OracleTopKConfig(heavy_size=0.1)
])))

# ==================== 20 % configs =================

# random sampling 20 %
SPARSE_CONFIGS.append((get_random_sampling_config_name(0.02, 0.02, 0.2), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.02),
    LocalMaskerConfig(window_size=0.02),
    RandomSamplingMaskerConfig(sampling_rate=0.2)
])))

#adaptive sampling with oracle top k 20 %
SPARSE_CONFIGS.append((get_adaptive_config_name(0.02, 0.02, 0.05, 0.1, 0.3, 0.3), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.02),
    LocalMaskerConfig(window_size=0.02),
    OracleTopKConfig(heavy_size=0.05),
    AdaptiveSamplingMaskerConfig(base_rate_sampling=0.1, epsilon=0.3, delta=0.3, init_offset=0.02, local_offset=0.02)
])))

# adative sampling with hat top k 20 %
SPARSE_CONFIGS.append((get_adaptive_hat_config_name(0.005, 0.005, 0.1, 0.1, 0.25, 0.25), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.005),
    LocalMaskerConfig(window_size=0.005),
    HashAttentionTopKMaskerConfig(heavy_size=0.1, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
    AdaptiveSamplingMaskerConfig(base_rate_sampling=0.1, epsilon=0.25, delta=0.25, init_offset=0.005, local_offset=0.005)
])))
# hat top k 20 %
SPARSE_CONFIGS.append((get_hashattention_config_name(0.005, 0.005, 0.19), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.005),
    LocalMaskerConfig(window_size=0.005),
    HashAttentionTopKMaskerConfig(heavy_size=0.19, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
])))

# oracle top p 20 %
SPARSE_CONFIGS.append((get_oracle_top_p_config_name(0.01, 0.01, 0.95), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.01),
    LocalMaskerConfig(window_size=0.01),
    OracleTopPMaskerConfig(top_p=0.95)
])))

# oracle top k 20 %
SPARSE_CONFIGS.append((get_oracle_top_k_config_name(0.005, 0.005, 0.19), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.005),
    LocalMaskerConfig(window_size=0.005),
    OracleTopKConfig(heavy_size=0.19)
])))

SPARSE_CONFIGS = [
    ("dense", None),
(get_adaptive_hat_config_name(0.01, 0.01, 0.02, 0.01, 0.25, 0.25), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=0.01),
    LocalMaskerConfig(window_size=0.01),
    HashAttentionTopKMaskerConfig(heavy_size=0.02, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
    AdaptiveSamplingMaskerConfig(base_rate_sampling=0.01, epsilon=0.25, delta=0.25, init_offset=0.001, local_offset=0.001)
])),
(get_adaptive_hat_config_name(128, 128, 1./32, 1./32, 0.25, 0.25), ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=128),
    LocalMaskerConfig(window_size=128),
    HashAttentionTopKMaskerConfig(heavy_size=1./32, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
    AdaptiveSamplingMaskerConfig(base_rate_sampling=1./32, epsilon=0.25, delta=0.25, init_offset=128, local_offset=128)
]))
]

SPARSE_CONFIGS = [("dense", None)]
# ==========================================================================

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
loogle_config_1 = BenchmarkConfig(
    benchmark_name="loogle",
    subsets=["longdep_summarization", "longdep_qa"], 
)
loogle_config_2 = BenchmarkConfig(
    benchmark_name="loogle",
    subsets=["shortdep_qa", "shortdep_cloze"], 
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

# 10. Ruler32K - using single task
niah1 = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["niah_single_1"]
)
niah2 = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["niah_single_2"]
)
niah3 = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["niah_single_3"]
)
cwe = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["cwe"]
)
fwe = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["fwe"]
)
vt = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["vt"]
)
qa1 = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["qa_1"]
)
qa2 = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["qa_2"]
)
multikey1 = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["niah_multikey_1"]
)
multikey2 = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["niah_multikey_2"]
)
multikey3 = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["niah_multikey_3"]
)
multikey = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["niah_multiquery"]
)
multivalue = BenchmarkConfig(
    benchmark_name="ruler16K",
    subsets=["niah_multivalue"]
)

# List of all sample configurations
BENCHMARKS = [
    #infinite_bench_config,
    #ruler_config,
    #loogle_config_1,
    #loogle_config_2,
    #zero_scrolls_config,
    #longbenchv2_config,
    #aime2024_config,
    #aime2025_config,
    #longbench_config,
    #mock_benchmark_config,
    #niah1, niah2, niah3, cwe, fwe, vt, qa1, qa2, multikey1, multikey2, multikey3, multikey, multivalue
    cwe
]


# Adapter Configuration
ADAPTER_CONFIG = AdapterConfig(
    adapter_name="huggingface",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        #"attn_implementation": "flash_attention_2",
    },
    tokenizer_kwargs={
        "padding_side": "left",
    }
)

# Generation Parameters
GENERATION_KWARGS = {
    "max_new_tokens": 120,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "pad_token_id": None,
}

# Request Parameters
REQUEST_KWARGS = {
    "max_context_length": 32000,
    "max_requests": 10,
}

# Execution Settings
RESULT_DIR = "./full_benchmark.matrix/"
ENABLE_RESUMABILITY = False
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
