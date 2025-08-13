#!/usr/bin/env python3
"""
Full End-to-End Benchmark Execution and Optimizer for Sparse Attention Methods.

This script performs a robust, two-stage process for each combination of
model, benchmark, and sparse attention configuration:
1.  **Search**: It uses Ray Tune to run a hyperparameter search with lightweight
    settings to quickly discover the optimal parameters.
2.  **Validate**: It takes the single best configuration found during the search
    and runs a final, thorough benchmark with it to get a definitive score.

## Usage Examples

### Basic Usage
```bash
# Run full benchmark suite with all sparse attention configs
python benchmark/raytune/run_full_benchmark.py

# Run in debug mode (quick test with minimal configs)
python benchmark/raytune/run_full_benchmark.py --debug

# Run only dense baseline (no sparse attention)
python benchmark/raytune/run_full_benchmark.py --dense-only

# Print all available configurations without running
python benchmark/raytune/run_full_benchmark.py --print-configs
```

### Advanced Usage
```bash
# Custom search parameters for faster exploration
python benchmark/raytune/run_full_benchmark.py \
    --search-timeout 600 \
    --search-max-new-tokens 10 \
    --search-max-context-length 4096 \
    --num-samples 20

# Custom validation parameters for thorough evaluation
python benchmark/raytune/run_full_benchmark.py \
    --validation-timeout 7200 \
    --validation-max-new-tokens 200 \
    --validation-max-context-length 64000 \
    --validation-max-requests 50

# Run with custom result directory suffix
python benchmark/raytune/run_full_benchmark.py --result-dir-suffix "_experiment_v1"
```

## Command-Line Arguments

### General Options
- `--debug`: Run quick test configuration with minimal settings
- `--num-samples`: Number of Ray Tune samples per optimization (default: 50)
- `--dense-only`: Run only dense configuration without sparse attention
- `--result-dir-suffix`: Suffix to add to result directory names
- `--print-configs`: Print all sparse configurations and exit

### Search Phase Parameters (for finding optimal configs)
- `--search-timeout`: Timeout for each search trial in seconds (default: 1800)
- `--search-max-new-tokens`: Max new tokens for search trials (default: 50)
- `--search-max-context-length`: Max context length for search trials (default: 16384)
- `--search-max-requests`: Max requests for search trials (default: 15)

### Validation Phase Parameters (for final evaluation)
- `--validation-timeout`: Timeout for final validation in seconds (default: 3600)
- `--validation-max-new-tokens`: Max new tokens for validation (default: 100)
- `--validation-max-context-length`: Max context length for validation (default: 32000)
- `--validation-max-requests`: Max requests for validation (default: 25)

## Sparse Attention Configurations

The script evaluates 19 different sparse attention configurations across 3 sparsity levels:

### 5% Sparsity
- Random Sampling (2% sink + 2% window + 1% sampling)
- Adaptive Sampling with Oracle Top-K
- Adaptive Sampling with HashAttention Top-K
- HashAttention Top-K
- Oracle Top-P (75%)
- Oracle Top-K

### 10% Sparsity
- Random Sampling (0.1% sink + 0.1% window + 10% sampling)
- Adaptive Sampling with Oracle Top-K
- Adaptive Sampling with HashAttention Top-K
- HashAttention Top-K
- Oracle Top-P (80%)
- Oracle Top-K

### 20% Sparsity
- Random Sampling (2% sink + 2% window + 20% sampling)
- Adaptive Sampling with Oracle Top-K
- Adaptive Sampling with HashAttention Top-K
- HashAttention Top-K
- Oracle Top-P (95%)
- Oracle Top-K

## Benchmarks

The script runs the following benchmark tasks:
- **InfiniteBench**: passkey task for extreme long context
- **Ruler**: 4096 context length evaluation
- **Loogle**: longdep_summarization, longdep_qa, shortdep_qa, shortdep_cloze
- **ZeroScrolls**: default configuration
- **LongBenchv2**: 0-shot evaluation
- **AIME2024/2025**: Mathematical reasoning tasks
- **LongBench**: passage_retrieval_en
- **Mock Benchmark**: reading_comprehension (for testing)

## Output Structure

Results are saved in two directories:
- `./search_results/`: Ray Tune optimization results
- `./validation_results/`: Final validation results for best configurations

Each run produces:
- Raw benchmark results (CSV)
- Micro metrics (JSONL) with attention errors and density
- Final summary (JSON) with all scores and best configurations

## Notes

- Requires GPU(s) with CUDA support
- HashAttention weights file should be available at the specified path
- Ray Tune must be installed: `pip install "ray[tune]" hyperopt`
- The script automatically handles resumability for interrupted runs

To add new models, benchmarks, or masker presets, modify the `get_run_configurations` function.
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{current_dir}:{root_path}"

# --- Core Imports ---
import torch
from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import AdapterConfig, BenchmarkConfig, BenchmarkResult
from optimizer_factory import create_optimizer

# --- Masker Config Imports ---
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
    OracleTopKConfig,
    OracleTopPMaskerConfig,
    HashAttentionTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
    RandomSamplingMaskerConfig,
    MagicPigConfig,
)

# --- Ray Tune Imports ---
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.stopper import TrialPlateauStopper
except ImportError:
    print("Error: Ray Tune is required. Install with: pip install \"ray[tune]\" hyperopt")
    sys.exit(1)


class ComprehensiveBenchmarkRunner:
    """Runs a benchmark for a model and sparse attention config, returning a score."""

    def __init__(self, config: dict, verbose: bool = False):
        self.config = config
        self.executor = BenchmarkExecutor(
            gpu_ids=config["gpu_ids"],
            max_concurrent_runs=config["max_concurrent_runs"],
            base_result_dir=config["result_dir"],
            enable_resumability=True,
            required_result_files=["raw_results.csv"],
            timeout_per_benchmark=config["timeout_per_benchmark"],
            verbose=verbose,
        )
        self.adapter_config = AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.generation_kwargs = {"max_new_tokens": config["max_new_tokens"], "do_sample": False}
        self.request_kwargs = {
            "max_context_length": config["max_context_length"],
            "max_requests": config["max_requests"],
        }
        self.results_cache = {}

    def _extract_micro_metrics(self, result_dir: Path) -> dict:
        import math
        micro_metrics_file = result_dir / "micro_metrics.jsonl"
        if not micro_metrics_file.exists():
            raise FileNotFoundError(f"micro_metrics.jsonl not found in {result_dir}")

        errors, densities = [], []
        with open(micro_metrics_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    metric, value = entry.get("metric"), entry.get("value")
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        if metric == "research_attention_output_error": errors.append(float(value))
                        elif metric == "research_attention_density": densities.append(float(value))
                except (json.JSONDecodeError, ValueError, TypeError): continue
        return {"attention_error": sum(errors) / len(errors) if errors else 1.0, "density": sum(densities) / len(densities) if densities else 1.0}

    def __call__(self, attention_config, task_name: str, model_name: str) -> float:
        config_key = f"{model_name}_{task_name}_{hash(str(attention_config))}"
        if config_key in self.results_cache: return self.results_cache[config_key]

        try:
            if "/" in task_name:
                benchmark_name, subset_name = task_name.split("/", 1)
            else:
                benchmark_name, subset_name = task_name, None

            benchmark_config = BenchmarkConfig(
                benchmark_name=benchmark_name, 
                subsets=[subset_name] if subset_name else None
            )
            
            results = self.executor.run_benchmark_matrix(
                model_names=[model_name],
                sparse_attention_configs=[("optimized", attention_config)],
                benchmark_configs=[benchmark_config],
                adapter_config=self.adapter_config,
                generation_kwargs=self.generation_kwargs,
                request_kwargs=self.request_kwargs,
            )

            if results.progress.completed_stubs > 0 and hasattr(results, "individual_results"):
                completed = [r for r in results.individual_results if isinstance(r, BenchmarkResult)]
                if completed:
                    result_dir = Path(completed[0].stub.result_dir)
                    metrics = self._extract_micro_metrics(result_dir)
                    error, density = metrics["attention_error"], metrics["density"]
                    score = error + 0.1 * density + (5.0 if density > 0.5 else 0.0)
                    self.results_cache[config_key] = score
                    return score
        except Exception as e:
            print(f"    ✗ Error in benchmark runner: {e}")
            traceback.print_exc()

        print(f"    Warning: Could not compute a valid score for {model_name} on {task_name}. Returning penalty.")
        self.results_cache[config_key] = 5.0
        return 5.0

# Helper functions for generating configuration names
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

def get_run_configurations(args: argparse.Namespace) -> dict:
    """Defines the complete configuration for the optimization run."""
    num_gpus = torch.cuda.device_count()

    # Get the HashAttention weights file path
    machine_key = "ubuntu"
    weight_file = f"/home/{machine_key}/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"
    
    # If weight file doesn't exist, try a fallback path
    if not os.path.exists(weight_file):
        weight_file = "./hat_weights.pkl"  # fallback to local file
        print(f"Warning: HashAttention weights not found at expected path, using {weight_file}")

    # Generate all sparse configurations from the provided script
    sparse_configs = []
    
    # Dense baseline
    sparse_configs.append(("dense", None))
    
    # ==================== 5% sparsity configs =================
    # Random sampling 5%
    sparse_configs.append((get_random_sampling_config_name(0.02, 0.02, 0.01), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.02),
        LocalMaskerConfig(window_size=0.02),
        RandomSamplingMaskerConfig(sampling_rate=0.01)
    ])))
    
    # Adaptive sampling with oracle top k 5%
    sparse_configs.append((get_adaptive_config_name(0.001, 0.001, 0.02, 0.01, 0.1, 0.1), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.02),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.01, epsilon=0.1, delta=0.1, init_offset=0.001, local_offset=0.001)
    ])))
    
    # Adaptive sampling with HAT top k 5%
    sparse_configs.append((get_adaptive_hat_config_name(0.01, 0.01, 0.02, 0.01, 0.25, 0.25), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        HashAttentionTopKMaskerConfig(heavy_size=0.02, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.01, epsilon=0.25, delta=0.25, init_offset=0.001, local_offset=0.001)
    ])))
    
    # HAT top k 5%
    sparse_configs.append((get_hashattention_config_name(0.005, 0.005, 0.04), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.005),
        LocalMaskerConfig(window_size=0.005),
        HashAttentionTopKMaskerConfig(heavy_size=0.04, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
    ])))
    
    # Oracle top p 5%
    sparse_configs.append((get_oracle_top_p_config_name(0.001, 0.001, 0.75), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopPMaskerConfig(top_p=0.75)
    ])))
    
    # Oracle top k 5%
    sparse_configs.append((get_oracle_top_k_config_name(0.005, 0.005, 0.04), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.005),
        LocalMaskerConfig(window_size=0.005),
        OracleTopKConfig(heavy_size=0.04)
    ])))
    
    # ==================== 10% sparsity configs =================
    # Random sampling 10%
    sparse_configs.append((get_random_sampling_config_name(0.001, 0.001, 0.1), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        RandomSamplingMaskerConfig(sampling_rate=0.1)
    ])))
    
    # Adaptive sampling with oracle top k 10%
    sparse_configs.append((get_adaptive_config_name(0.001, 0.001, 0.05, 0.05, 0.25, 0.25), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.05),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.25, delta=0.25, init_offset=0.001, local_offset=0.001)
    ])))
    
    # Adaptive sampling with HAT top k 10%
    sparse_configs.append((get_adaptive_hat_config_name(0.001, 0.001, 0.05, 0.05, 0.4, 0.4), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        HashAttentionTopKMaskerConfig(heavy_size=0.05, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.4, delta=0.4, init_offset=0.001, local_offset=0.001)
    ])))
    
    # HAT top k 10%
    sparse_configs.append((get_hashattention_config_name(0.001, 0.001, 0.09), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        HashAttentionTopKMaskerConfig(heavy_size=0.09, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
    ])))
    
    # Oracle top p 10%
    sparse_configs.append((get_oracle_top_p_config_name(0.02, 0.02, 0.8), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.02),
        LocalMaskerConfig(window_size=0.02),
        OracleTopPMaskerConfig(top_p=0.8)
    ])))
    
    # Oracle top k 10%
    sparse_configs.append((get_oracle_top_k_config_name(0.001, 0.001, 0.1), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.1)
    ])))
    
    # ==================== 20% sparsity configs =================
    # Random sampling 20%
    sparse_configs.append((get_random_sampling_config_name(0.02, 0.02, 0.2), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.02),
        LocalMaskerConfig(window_size=0.02),
        RandomSamplingMaskerConfig(sampling_rate=0.2)
    ])))
    
    # Adaptive sampling with oracle top k 20%
    sparse_configs.append((get_adaptive_config_name(0.02, 0.02, 0.05, 0.1, 0.3, 0.3), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.02),
        LocalMaskerConfig(window_size=0.02),
        OracleTopKConfig(heavy_size=0.05),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.1, epsilon=0.3, delta=0.3, init_offset=0.02, local_offset=0.02)
    ])))
    
    # Adaptive sampling with HAT top k 20%
    sparse_configs.append((get_adaptive_hat_config_name(0.005, 0.005, 0.1, 0.1, 0.25, 0.25), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.005),
        LocalMaskerConfig(window_size=0.005),
        HashAttentionTopKMaskerConfig(heavy_size=0.1, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.1, epsilon=0.25, delta=0.25, init_offset=0.005, local_offset=0.005)
    ])))
    
    # HAT top k 20%
    sparse_configs.append((get_hashattention_config_name(0.005, 0.005, 0.19), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.005),
        LocalMaskerConfig(window_size=0.005),
        HashAttentionTopKMaskerConfig(heavy_size=0.19, hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128, hat_mlp_activation="silu", hat_weight_file=weight_file, hat_weights=None),
    ])))
    
    # Oracle top p 20%
    sparse_configs.append((get_oracle_top_p_config_name(0.01, 0.01, 0.95), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.01),
        LocalMaskerConfig(window_size=0.01),
        OracleTopPMaskerConfig(top_p=0.95)
    ])))
    
    # Oracle top k 20%
    sparse_configs.append((get_oracle_top_k_config_name(0.005, 0.005, 0.19), ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.005),
        LocalMaskerConfig(window_size=0.005),
        OracleTopKConfig(heavy_size=0.19)
    ])))
    
    # For Ray Tune optimization, we'll create a smaller subset for the search
    # and then use all configs for final validation
    if args.dense_only:
        # Only run dense configuration
        selected_sparse_configs = [("dense", None)]
    elif args.debug:
        # In debug mode, just test a few configs
        selected_sparse_configs = sparse_configs[:3]
    else:
        # In production, we might want to optimize across all configs or a subset
        # For now, let's use the full set
        selected_sparse_configs = sparse_configs
    
    # Convert configs to optimizer-compatible format
    # The optimizer expects classes, not instances
    masker_config_presets = {}      # For optimizer (classes)
    sparse_attention_configs = {}   # For validation (full configs)
    
    for name, config in selected_sparse_configs:
        if config is not None:
            # Extract the class from each config instance for the optimizer
            masker_classes = []
            for masker_config in config.masker_configs:
                masker_classes.append(type(masker_config))
            masker_config_presets[name] = masker_classes
            sparse_attention_configs[name] = config  # Store full config for validation
        else:
            masker_config_presets[name] = None
            sparse_attention_configs[name] = None
    
    test_suites = {"default": list(masker_config_presets.keys()), "debug": list(masker_config_presets.keys())[:3]}

    # --- Decouple Search and Validation Parameters ---
    if args.debug:
        # Use smaller, faster settings for the search phase in debug mode
        search_params = {
            "timeout_per_benchmark": 300, "max_new_tokens": 10,
            "max_context_length": 4096, "max_requests": 2,
        }
        # Use slightly more thorough settings for debug validation
        validation_params = {
            "timeout_per_benchmark": 600, "max_new_tokens": 30,
            "max_context_length": 16384, "max_requests": 5,
        }
        base_config = {
            "models": ["meta-llama/Llama-3.1-8B-Instruct"], 
            "benchmarks": [
                "loogle/shortdep_qa",  # Quick benchmark for debug
            ],
            "masker_presets": {p: masker_config_presets[p] for p in test_suites["debug"]},
            "num_samples": 8,
        }
    else:
        # For production, use specific flags for each stage
        search_params = {
            "timeout_per_benchmark": args.search_timeout, "max_new_tokens": args.search_max_new_tokens,
            "max_context_length": args.search_max_context_length, "max_requests": args.search_max_requests,
        }
        validation_params = {
            "timeout_per_benchmark": args.validation_timeout, "max_new_tokens": args.validation_max_new_tokens,
            "max_context_length": args.validation_max_context_length, "max_requests": args.validation_max_requests,
        }
        base_config = {
            "models": ["meta-llama/Llama-3.1-8B-Instruct"],
            "benchmarks": [
                # InfiniteBench
                "infinite_bench/passkey",
                # Ruler
                "ruler/4096",
                # Loogle
                "loogle/longdep_summarization",
                "loogle/longdep_qa",
                "loogle/shortdep_qa",
                "loogle/shortdep_cloze",
                # ZeroScrolls
                "zero_scrolls/default",
                # LongBenchv2
                "longbenchv2/0shot",
                # AIME benchmarks
                "aime2024/aime2024",
                "aime2025/aime2025",
                # LongBench
                "longbench/passage_retrieval_en",
                # Mock benchmark for testing
                "mock_benchmark/reading_comprehension",
            ],
            "masker_presets": {p: masker_config_presets[p] for p in test_suites["default"]},
            "num_samples": args.num_samples,
        }

    # Combine into a final, structured configuration
    return {
        **base_config,
        "search_params": search_params,
        "validation_params": validation_params,
        "gpu_ids": list(range(num_gpus)),
        "max_concurrent_runs": num_gpus,
        "result_dir": f"./search_results{args.result_dir_suffix}", # Base directory for the search phase
        "detailed_result_dir": f"./validation_results{args.result_dir_suffix}", # Base directory for the validation phase
        "sparse_configs": selected_sparse_configs,  # Store the full list for reference
        "sparse_attention_configs": sparse_attention_configs,  # Store full config objects for validation
    }

def get_ray_tune_components(config: dict) -> dict:
    scheduler = ASHAScheduler(time_attr="training_iteration", max_t=20, grace_period=5, reduction_factor=2)
    search_alg = HyperOptSearch(metric="combined_score", mode="min", n_initial_points=max(1, config["num_samples"] // 4))
    stopper = TrialPlateauStopper(metric="combined_score", std=0.005, num_results=5, grace_period=8, mode="min")
    return {"scheduler": scheduler, "search_alg": search_alg, "stop": stopper}

def create_optimization_objective(config: dict, model_name: str, task_name: str, optimizer):
    """Creates the objective function that Ray Tune will execute for each trial."""
    def objective(trial_config: dict):
        # The worker always uses the lighter search parameters for speed
        worker_config = {**config, **config["search_params"]}
        worker_config["gpu_ids"] = [0]
        worker_config["max_concurrent_runs"] = 1
        
        benchmark_runner = ComprehensiveBenchmarkRunner(worker_config)
        attention_config = optimizer.create_config_from_params(trial_config)
        score = benchmark_runner(attention_config, task_name, model_name)
        return {"combined_score": score}
    return objective

def run_optimization_and_validation(model_name: str, benchmark_task: str, preset_name: str, masker_configs: list, config: dict, full_sparse_config=None) -> dict:
    """Runs the two-stage Search-then-Validate process for one combination."""
    print(f"\n--- Running: {model_name} | {benchmark_task} | {preset_name} ---")
    
    # Handle dense configuration (no masker configs)
    if masker_configs is None or preset_name == "dense":
        print("  Running dense configuration (no optimization needed)...")
        validation_config = {**config, **config["validation_params"]}
        validation_config["result_dir"] = os.path.join(config["detailed_result_dir"], preset_name)
        
        validator = ComprehensiveBenchmarkRunner(validation_config, verbose=True)
        start_time = time.time()
        print(f"    Running validation benchmark: {model_name} on {benchmark_task}...")
        final_score = validator(full_sparse_config, benchmark_task, model_name)  # Use full config
        runtime = time.time() - start_time
        print(f"    Validation benchmark completed in {runtime:.1f}s")
        print(f"     ✓ Final validation score: {final_score:.4f}")
        
        return {
            "best_search_score": final_score,
            "final_validation_score": final_score,
            "best_config": None,
            "best_params": {},
            "num_trials": 1,
        }
    
    # Stage 1: Search using the lighter 'search_params'
    print("  1. Searching for optimal configuration...")
    try:
        optimizer = create_optimizer(masker_configs)
        objective = create_optimization_objective(config, model_name, benchmark_task, optimizer)
        tune_components = get_ray_tune_components(config)
        sanitized_task_name = benchmark_task.replace('/', '_')
        
        analysis = tune.run(
            objective, config=optimizer.create_search_space(benchmark_task),
            num_samples=config["num_samples"], metric="combined_score", mode="min",
            resources_per_trial={"CPU": 1, "GPU": 1.0},
            name=f"opt_{model_name.split('/')[-1]}_{sanitized_task_name}_{preset_name}",
            storage_path=config["storage_path"], verbose=1, resume=False,
            max_concurrent_trials=config["max_concurrent_runs"], **tune_components
        )
        best_trial = analysis.get_best_trial("combined_score", "min", "last")
        best_config_obj = optimizer.create_config_from_params(best_trial.config)
        best_search_score = best_trial.last_result['combined_score']
        print(f"     ✓ Best search score: {best_search_score:.4f}")
    except Exception as e:
        print(f"     ✗ Search stage failed: {e}"); traceback.print_exc()
        return {"error": f"Search failed: {e}"}

    # Stage 2: Validate using the more thorough 'validation_params'
    print("  2. Validating the best configuration...")
    try:
        # Create a new config for validation by merging base and validation params
        validation_config = {**config, **config["validation_params"]}
        validation_config["result_dir"] = os.path.join(config["detailed_result_dir"], preset_name)
        
        validator = ComprehensiveBenchmarkRunner(validation_config, verbose=True)
        start_time = time.time()
        print(f"    Running validation benchmark: {model_name} on {benchmark_task}...")
        final_score = validator(best_config_obj, benchmark_task, model_name)
        runtime = time.time() - start_time
        print(f"    Validation benchmark completed in {runtime:.1f}s")
        print(f"     ✓ Final validation score: {final_score:.4f}")
    except Exception as e:
        print(f"     ✗ Validation stage failed: {e}"); traceback.print_exc()
        return {"error": f"Validation failed: {e}"}

    return {
        "best_search_score": best_search_score,
        "final_validation_score": final_score,
        "best_config": best_config_obj,
        "best_params": best_trial.config,
        "num_trials": len(analysis.trials),
    }

def run_optimization_matrix(config: dict) -> tuple[dict, str]:
    print("Starting Full Benchmark Optimization and Validation Matrix"); print("=" * 80)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_path = os.path.abspath(f"./ray_results_{timestamp}")
    config["storage_path"] = storage_path
    print(f"Ray Tune results will be saved to: {storage_path}")

    all_results = {}
    for model_name in config["models"]:
        all_results[model_name] = {}
        print(f"\nModel: {model_name}"); print("-" * 60)
        for benchmark_task in config["benchmarks"]:
            all_results[model_name][benchmark_task] = {}
            for preset_name, masker_configs in config["masker_presets"].items():
                full_sparse_config = config.get("sparse_attention_configs", {}).get(preset_name)
                combo_result = run_optimization_and_validation(model_name, benchmark_task, preset_name, masker_configs, config, full_sparse_config)
                all_results[model_name][benchmark_task][preset_name] = combo_result
    return all_results, storage_path

def print_summary(results: dict):
    print("\n" + "=" * 80); print("--- FINAL BENCHMARK SUMMARY ---"); print("=" * 80)
    best_overall_score, best_overall_config = float("inf"), {}
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}"); print("-" * 70)
        for benchmark_task, task_results in model_results.items():
            print(f"\n  Benchmark: {benchmark_task}")
            for masker_preset, result in task_results.items():
                if "error" in result:
                    print(f"    {masker_preset:25s}: FAILED ({result['error']})"); continue
                score = result.get("final_validation_score", float("inf"))
                search_score = result.get("best_search_score", float("inf"))
                print(f"    {masker_preset:25s}: {score:.4f} (Search score: {search_score:.4f})")
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_config = {"model": model_name, "benchmark": benchmark_task, "masker": masker_preset, "score": score, "params": result.get("best_params")}
    print("\n" + "--- Best Overall Configuration ---")
    if best_overall_config:
        for key, value in best_overall_config.items(): print(f"  {key.capitalize():<12}: {value}")
    else: print("  No successful runs completed.")
    print("-" * 32)

def define_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full benchmark optimization and validation runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General arguments
    parser.add_argument("--debug", action="store_true", help="Run a quick test configuration, ignoring other flags.")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of Ray Tune samples per optimization search.")
    parser.add_argument("--dense-only", action="store_true", help="Run only dense configuration without sparse attention.")
    parser.add_argument("--result-dir-suffix", type=str, default="", help="Suffix to add to result directory names.")
    parser.add_argument("--print-configs", action="store_true", help="Print all sparse configurations and exit.")

    # Search-specific arguments
    search_group = parser.add_argument_group('Search Parameters (for finding the best config)')
    search_group.add_argument("--search-timeout", type=int, default=1800, help="Timeout for each search trial.")
    search_group.add_argument("--search-max-new-tokens", type=int, default=50, help="Max new tokens for search trials.")
    search_group.add_argument("--search-max-context-length", type=int, default=16384, help="Max context length for search trials.")
    search_group.add_argument("--search-max-requests", type=int, default=15, help="Max requests for search trials.")

    # Validation-specific arguments
    validation_group = parser.add_argument_group('Validation Parameters (for the final run with the best config)')
    validation_group.add_argument("--validation-timeout", type=int, default=3600, help="Timeout for the final validation run.")
    validation_group.add_argument("--validation-max-new-tokens", type=int, default=100, help="Max new tokens for the final validation run.")
    validation_group.add_argument("--validation-max-context-length", type=int, default=32000, help="Max context length for the final validation run.")
    validation_group.add_argument("--validation-max-requests", type=int, default=25, help="Max requests for the final validation run.")
    
    return parser.parse_args()

def main():
    args = define_cli_args()
    config = get_run_configurations(args)
    
    # Print configurations if requested
    if args.print_configs:
        print("\n" + "=" * 80)
        print("SPARSE ATTENTION CONFIGURATIONS")
        print("=" * 80)
        for i, (name, cfg) in enumerate(config.get("sparse_configs", [])):
            print(f"\n{i+1}. {name}")
            if cfg is not None:
                print("   Maskers:")
                for masker in cfg.masker_configs:
                    print(f"     - {masker.__class__.__name__}")
            else:
                print("   Dense (no sparse attention)")
        print("\n" + "=" * 80)
        print(f"Total configurations: {len(config.get('sparse_configs', []))}")
        print("=" * 80)
        return
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, runtime_env={"working_dir": str(root_path)})

    mode = "Quick Test" if args.debug else "Full Production"
    print(f"Starting {mode} Optimization & Validation...")
    print(f"Ray Version: {ray.__version__}, GPUs Available: {torch.cuda.device_count()}")
    
    # Print execution summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Models ({len(config['models'])}):") 
    for model in config['models']:
        print(f"  - {model}")
    print(f"\nBenchmarks ({len(config['benchmarks'])}):") 
    for benchmark in config['benchmarks']:
        print(f"  - {benchmark}")
    print(f"\nSparse Configurations ({len(config['masker_presets'])}):") 
    for i, preset in enumerate(list(config['masker_presets'].keys())[:5]):
        print(f"  - {preset}")
    if len(config['masker_presets']) > 5:
        print(f"  ... and {len(config['masker_presets']) - 5} more")
    
    total_combinations = len(config['models']) * len(config['benchmarks']) * len(config['masker_presets'])
    print(f"\nTotal combinations to run: {total_combinations}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    try:
        results, storage_path = run_optimization_matrix(config)
        print_summary(results)
        print(f"\nDetailed validation results saved to: {config['detailed_result_dir']}")
        print(f"View optimization progress with: tensorboard --logdir {storage_path}")
        
        results_file = Path(storage_path) / "final_summary.json"
        def json_serializer(obj): return str(obj)
            
        print(f"Saving summary to: {results_file}")
        # Create directory if it doesn't exist
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f: json.dump(results, f, indent=2, default=json_serializer)
        print("Summary saved successfully.")
    except KeyboardInterrupt:
        print("\nWarning: Optimization interrupted by user.")
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {e}"); traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        print(f"\nTotal script time: {total_time / 3600:.2f} hours ({total_time:.0f} seconds)")
        ray.shutdown()
        print("Script finished.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()