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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# GPU Configuration
GPUS = [4, 5]  # Use all available GPUs
MAX_CONCURRENT_RUNS = 2  # One per GPU

# Model List
MODELS = [
    "microsoft/Phi-4-mini-instruct", 
    "meta-llama/Llama-3.2-1B-Instruct",  
]

# Sparse Attention Configurations
SPARSE_CONFIGS = [
    # Dense baseline (no sparse attention)
    ("dense", None),
    
    # StreamingLLM configurations
    ("streaming_conservative", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=4),
        LocalMaskerConfig(window_size=16)
    ])),
    
    ("streaming_balanced", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=4),
        LocalMaskerConfig(window_size=32)
    ])),
    
    ("streaming_aggressive", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=4),
        LocalMaskerConfig(window_size=64)
    ])),
    
    ("streaming_very_aggressive", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=4),
        LocalMaskerConfig(window_size=128)
    ])),
]

# Benchmark List
BENCHMARKS = [
    BenchmarkConfig("longbench", [
        "narrativeqa",  # Story understanding
        "triviaqa",     # Factual QA
    ]),
]

# Adapter Configuration
ADAPTER_CONFIG = AdapterConfig(
    adapter_name="huggingface",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    },
    tokenizer_kwargs={
        "padding_side": "left",
    }
)

# Generation Parameters
GENERATION_KWARGS = {
    "max_new_tokens": 50,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "pad_token_id": None,
}

# Request Parameters
REQUEST_KWARGS = {
    "max_context_length": 256,
    "max_requests": 2,  # Limit for testing
}

# Execution Settings
RESULT_DIR = "./benchmark_results"
ENABLE_RESUMABILITY = False
TIMEOUT_PER_BENCHMARK = 3600.0  # 1 hour

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Minimalistic Benchmark Suite")
    print("=" * 50)
    
    # Validate GPU availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU only")
        GPUS = []
        MAX_CONCURRENT_RUNS = 1
    else:
        available_gpus = list(range(torch.cuda.device_count()))
        if GPUS:
            GPUS = [gpu for gpu in GPUS if gpu in available_gpus]
            if not GPUS:
                print(f"⚠️  No specified GPUs available, using all: {available_gpus}")
                GPUS = available_gpus
        else:
            GPUS = available_gpus
    
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
