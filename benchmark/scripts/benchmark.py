#!/usr/bin/env python3
"""
Minimalistic benchmark runner for sparse attention evaluation.

This script defines models, sparse attention configurations, and benchmarks,
then runs comprehensive experiments using BenchmarkExecutor.

Usage:
    python benchmark/benchmark.py
"""

import sys
import torch
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

# Add the project root to the path (go up two directories from scripts/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    PQCacheConfig
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# GPU Configuration
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]  # Use all 8 GPUs
MAX_CONCURRENT_RUNS = 8  # One per GPU

# Model List
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct"
]

# Sparse Attention Configurations
SPARSE_CONFIGS = [
    # Dense baseline (no sparse attention)
    ("dense", None),
    
    # PQCache configuration (matching original implementation exactly)
    # Original: compress_ratio=0.1, recent_ratio=0.5, sink_size=32
    # For 16K sequence: (16384-32) * 0.1 * 0.5 = 817 tokens each for heavy and recent
    ("pqcache", ResearchAttentionConfig(masker_configs=[
        PQCacheConfig(
            heavy_size=1635,     # Total compressed tokens: (16384-32) * 0.1 ≈ 1635
            pq_sub_dim=64,       # Subvector dimension (128 head_dim / 2 subvectors)
            pq_bits=6,           # Number of bits per subvector (64 centroids)
            kmeans_iters=25,     # K-means iterations (matching original)
            sink_size=32         # Number of sink tokens (matching original)
        ),
        # Note: PQCache internally handles recent tokens with recent_ratio=0.5
        # No need for separate LocalMaskerConfig
    ])),
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
    subsets=["shortdep_qa"]
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

# 8. LongBench - using hotpotqa task
longbench_config = BenchmarkConfig(
    benchmark_name="longbench",
    subsets=["hotpotqa"]
)

# 9. Mock Benchmark (existing) - using single task
mock_benchmark_config = BenchmarkConfig(
    benchmark_name="mock_benchmark",
    subsets=["reading_comprehension"]
)

# List of all sample configurations
BENCHMARKS = [
    # Comment out other benchmarks for now
    # infinite_bench_config,
    # ruler_config,
    # loogle_config,
    # zero_scrolls_config,
    # longbenchv2_config,
    # aime2024_config,
    # aime2025_config,
    longbench_config,  # Only run LongBench with HotpotQA
    # mock_benchmark_config
]


# Adapter Configuration
ADAPTER_CONFIG = AdapterConfig(
    adapter_name="huggingface",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
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
    "max_context_length": 32768,  # Increased for LongBench HotpotQA
    "max_requests": 100,  # Run 100 queries
}

# Execution Settings
RESULT_DIR = "./benchmark_results"
ENABLE_RESUMABILITY = True
TIMEOUT_PER_BENCHMARK = 3600.0  # 1 hour

# Query Parallelization Mode
# If True: Run one config at a time but distribute queries across GPUs
# If False: Run different configs in parallel (default behavior)
QUERY_PARALLEL_MODE = True

# ============================================================================
# QUERY PARALLEL EXECUTION FUNCTIONS
# ============================================================================

# Import Ray for distributed GPU execution
try:
    import ray
except ImportError:
    print("Ray not installed. To use query-parallel mode, install with: pip install ray")
    QUERY_PARALLEL_MODE = False

@ray.remote(num_gpus=1)
def run_query_subset(args: Tuple[str, Any, BenchmarkConfig, Dict, int, int, int]) -> Tuple[int, pd.DataFrame]:
    """Run a subset of queries on a specific GPU."""
    model_name, sparse_config, benchmark_config, adapter_config, gpu_id, start_idx, end_idx = args
    
    # Import here to avoid issues with multiprocessing
    import os
    import pandas as pd
    import torch
    from sparse_attention_hub.adapters import ModelAdapterHF
    from benchmark.benchmark_registry import create_benchmark_instance
    
    # Ray automatically assigns GPUs, so we'll use cuda:0
    device = "cuda:0"
    
    print(f"[Ray Worker] Starting queries {start_idx}-{end_idx}")
    
    # Verify GPU selection
    if torch.cuda.is_available():
        actual_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(actual_device)
        device_props = torch.cuda.get_device_properties(actual_device)
        print(f"[Worker for queries {start_idx}-{end_idx}] Using device {actual_device}: {device_name}, Memory: {device_props.total_memory / 1e9:.1f} GB")
    
    # Create adapter with specific GPU
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_config,
        model_kwargs={**adapter_config.model_kwargs},
        generate_kwargs=GENERATION_KWARGS,
        device=device
    )
    
    # Get benchmark instance
    benchmark = create_benchmark_instance(
        benchmark_name=benchmark_config.benchmark_name,
        subsets=benchmark_config.subsets
    )
    
    # Create temporary result dir for this GPU
    temp_result_dir = Path(f"{RESULT_DIR}_temp_gpu{gpu_id}")
    temp_result_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full dataset and slice it
    print(f"[Ray Worker] Loading dataset...")
    full_dataset = benchmark._load_datasets()
    subset_dataset = full_dataset.iloc[start_idx:end_idx].copy()
    
    # Process the subset directly
    try:
        print(f"[Ray Worker] Processing {len(subset_dataset)} requests...")
        results_df = benchmark._process_all_requests(
            adapter,
            subset_dataset,
            GENERATION_KWARGS,
            {"max_requests": end_idx - start_idx}
        )
        
        # Save the subset results
        results_df.to_csv(temp_result_dir / "raw_results.csv", index=False)
        print(f"[Ray Worker] Completed {len(results_df)} queries")
        
    except Exception as e:
        print(f"[Ray Worker] Error: {e}")
        import traceback
        traceback.print_exc()
        results_df = pd.DataFrame()
    
    return gpu_id, results_df

def run_query_parallel_benchmark(model_name: str, sparse_config_name: str, sparse_config: Any, 
                                 benchmark_config: BenchmarkConfig):
    """Run a single benchmark configuration with queries distributed across GPUs."""
    print(f"\n{'='*80}")
    print(f"Running Query-Parallel Benchmark with Ray:")
    print(f"  Model: {model_name}")
    print(f"  Config: {sparse_config_name}")
    print(f"  Benchmark: {benchmark_config.benchmark_name}")
    print(f"  GPUs: {GPUS}")
    print(f"{'='*80}\n")
    
    # Initialize Ray with GPU resources
    if not ray.is_initialized():
        ray.init(num_gpus=len(GPUS))
    
    # Calculate query distribution
    total_queries = REQUEST_KWARGS["max_requests"]
    num_gpus = len(GPUS)
    queries_per_gpu = total_queries // num_gpus
    remainder = total_queries % num_gpus
    
    # Create work assignments
    work_items = []
    start_idx = 0
    for i, gpu_id in enumerate(GPUS):
        # Distribute remainder queries to first GPUs
        end_idx = start_idx + queries_per_gpu + (1 if i < remainder else 0)
        work_items.append((
            model_name, sparse_config, benchmark_config, 
            ADAPTER_CONFIG, gpu_id, start_idx, end_idx
        ))
        start_idx = end_idx
    
    # Submit tasks to Ray
    futures = [run_query_subset.remote(work_item) for work_item in work_items]
    
    # Get results
    results = ray.get(futures)
    
    # Filter out empty results and aggregate
    valid_results = [r[1] for r in results if not r[1].empty]
    
    if not valid_results:
        print(f"❌ No valid results obtained from any GPU")
        return pd.DataFrame()
    
    # Concatenate all results
    all_results = pd.concat(valid_results, ignore_index=True)
    
    # Save aggregated results
    model_clean = model_name.replace("/", "_")
    final_result_dir = Path(RESULT_DIR) / model_clean / sparse_config_name / f"{benchmark_config.benchmark_name}_{'_'.join(benchmark_config.subsets)}"
    final_result_dir.mkdir(parents=True, exist_ok=True)
    
    all_results.to_csv(final_result_dir / "raw_results.csv", index=False)
    
    # Compute metrics if benchmark has post_run_evaluate
    from benchmark.benchmark_registry import create_benchmark_instance
    benchmark = create_benchmark_instance(
        benchmark_name=benchmark_config.benchmark_name,
        subsets=benchmark_config.subsets
    )
    
    if hasattr(benchmark, 'post_run_evaluate'):
        try:
            # post_run_evaluate expects a dataframe, not a path
            metrics = benchmark.post_run_evaluate(all_results)
            
            # Save metrics
            metrics_path = final_result_dir / "metrics.json"
            import json
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            print(f"\n📊 Metrics computed and saved to: {metrics_path}")
        except Exception as e:
            print(f"⚠️  Failed to compute metrics: {e}")
    
    # Clean up temporary directories
    for gpu_id in GPUS:
        temp_dir = Path(f"{RESULT_DIR}_temp_gpu{gpu_id}")
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
    
    print(f"✅ Completed! Results saved to: {final_result_dir}")
    
    return all_results

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
    
    if QUERY_PARALLEL_MODE:
        # Query-parallel mode: Run one config at a time, distribute queries across GPUs
        print(f"\n🔄 Running in QUERY-PARALLEL mode")
        print(f"   Distributing {REQUEST_KWARGS['max_requests']} queries across {len(GPUS)} GPUs")
        print(f"   Queries per GPU: ~{REQUEST_KWARGS['max_requests'] // len(GPUS)}")
        
        all_results = {}
        
        # Iterate through each configuration sequentially
        for model in MODELS:
            for config_name, sparse_config in SPARSE_CONFIGS:
                for benchmark_config in BENCHMARKS:
                    # Check if already completed (resumability)
                    model_clean = model.replace("/", "_")
                    result_path = Path(RESULT_DIR) / model_clean / config_name / f"{benchmark_config.benchmark_name}_{'_'.join(benchmark_config.subsets)}"
                    
                    if ENABLE_RESUMABILITY and (result_path / "raw_results.csv").exists():
                        print(f"\n⏭️  Skipping {model} - {config_name} - {benchmark_config.benchmark_name} (already completed)")
                        continue
                    
                    # Run this configuration with queries distributed across GPUs
                    try:
                        results = run_query_parallel_benchmark(
                            model, config_name, sparse_config, benchmark_config
                        )
                        all_results[(model, config_name, benchmark_config.benchmark_name)] = results
                        print(f"\n✅ Completed {model} - {config_name} - {benchmark_config.benchmark_name}")
                    except Exception as e:
                        print(f"\n❌ Failed {model} - {config_name} - {benchmark_config.benchmark_name}: {str(e)}")
                        import traceback
                        traceback.print_exc()
        
        print(f"\n✅ Query-parallel benchmark execution completed!")
        print(f"  - Results saved to: {RESULT_DIR}")
        
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
    
    else:
        # Default mode: Run different configs in parallel
        print(f"\n🔄 Running in DEFAULT mode (config-parallel)")
        
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
