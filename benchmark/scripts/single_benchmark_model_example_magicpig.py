#!/usr/bin/env python3
"""
Simple Benchmark Example (Refactored)

A beginner-friendly example showing how to run a basic benchmark comparison
between dense and sparse attention using the sparse-attention-hub framework.

This version programmatically generates configurations to easily sweep over
hyperparameters for MagicPig attention.
"""

import os
import time
from pathlib import Path
import psutil
import gc
import torch
import sys

# Set project root and add to Python path
project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import SinkMaskerConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import MagicPigConfig
from benchmark import Loogle
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

def get_system_memory_usage():
    """Get current system memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def clear_memory():
    """Clear GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def run_single_config(model_name, device, config_name, sparse_attention_config,
                      benchmark, base_result_dir, request_kwargs):
    """Run benchmark for a single configuration and return metrics with timing/memory info."""
    print(f"\n{'='*60}\nRunning configuration: {config_name}\n{'='*60}")
    clear_memory()

    initial_gpu_mem, initial_sys_mem = get_gpu_memory_usage(), get_system_memory_usage()
    start_time = time.time()

    print("  ✓ Loading model...")
    model_load_start = time.time()
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        generate_kwargs={"max_new_tokens": 32},
        device=device
    )
    model_load_time = time.time() - model_load_start
    after_model_gpu_mem, after_model_sys_mem = get_gpu_memory_usage(), get_system_memory_usage()

    print(f"  ✓ Model loaded in {model_load_time:.2f}s")
    print(f"  ✓ GPU memory: {after_model_gpu_mem:.1f} MB (+{after_model_gpu_mem - initial_gpu_mem:.1f} MB)")
    print(f"  ✓ System memory: {after_model_sys_mem:.1f} MB (+{after_model_sys_mem - initial_sys_mem:.1f} MB)")

    result_dir = base_result_dir / config_name
    result_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(
        log_path=result_dir,
        enabled_metrics=["research_attention_density", "research_attention_output_error"]
    )

    print("  ✓ Running benchmark...")
    benchmark_start = time.time()
    metrics = benchmark.run_benchmark(adapter, result_dir, request_kwargs=request_kwargs)
    benchmark_time = time.time() - benchmark_start
    metric_logger.flush()

    final_gpu_mem, final_sys_mem = get_gpu_memory_usage(), get_system_memory_usage()
    metrics.update({
        'timing': {'model_load_time': model_load_time, 'benchmark_time': benchmark_time, 'total_time': time.time() - start_time},
        'memory': {'gpu_usage_mb': final_gpu_mem - initial_gpu_mem, 'sys_usage_mb': final_sys_mem - initial_sys_mem}
    })

    print(f"  ✓ Benchmark completed in {benchmark_time:.2f}s (total: {metrics['timing']['total_time']:.2f}s)")
    print(f"  ✓ Final GPU memory: {final_gpu_mem:.1f} MB")

    del adapter
    clear_memory()
    return metrics

def generate_magicpig_configs(param_grid, sink_size=8):
    """
    Generates a list of benchmark configurations from a grid of parameters.

    Args:
        param_grid (list of dict): A list where each dict contains keys 'l', 'k', and 'center'.
        sink_size (int): The sink size for the SinkMaskerConfig.

    Returns:
        list of dict: A list of configurations ready for the benchmark loop.
    """
    configs = []
    for params in param_grid:
        l, k, packing = params['l'], params['k'], params['packing']
        name = f"sink{sink_size}_L{l}_K{k}_packing_{packing}"
        config_obj = ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=sink_size),
            MagicPigConfig(lsh_l=l, lsh_k=k, packing=packing)
        ])
        configs.append({"name": name, "config": config_obj})
    return configs

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    benchmark = Loogle(["shortdep_qa"])
    base_result_dir = Path("./test_magicpig_packing")
    base_result_dir.mkdir(exist_ok=True)
    request_kwargs = {"max_requests": 2, "max_context_length": 16000}

    # ## DEFINE YOUR PARAMETER SWEEP HERE ##
    # Simply add or remove dictionaries to test different combinations.
    param_grid = [
        {'l': 32, 'k': 4, "packing": "int64"},
        {'l': 32, 'k': 4, "packing": "float32"},
        {'l': 96, 'k': 4, "packing": "int64"},
        {'l': 96, 'k': 4, "packing": "float32"},
        {'l': 32, 'k': 8, "packing": "int64"},
        {'l': 32, 'k': 8, "packing": "float32"},
        {'l': 96, 'k': 8, "packing": "int64"},
        {'l': 96, 'k': 8, "packing": "float32"},
    ]

    # Generate configurations dynamically from the parameter grid
    configs = generate_magicpig_configs(param_grid, sink_size=8)
    all_results = {}

    print(f"Starting benchmark for {len(configs)} configurations...")
    print(f"Model: {model_name}, Device: {device}")

    for i, config_info in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Starting configuration: {config_info['name']}")
        try:
            metrics = run_single_config(
                model_name, device, config_info['name'], config_info['config'],
                benchmark, base_result_dir, request_kwargs
            )
            all_results[config_info['name']] = metrics
        except Exception as e:
            print(f"ERROR in configuration {config_info['name']}: {e}")
            all_results[config_info['name']] = {"error": str(e)}
            clear_memory()

    # --- FINAL SUMMARY ---
    print(f"\n{'='*80}\nFINAL SUMMARY\n{'='*80}")
    summary_lines = []
    for config_name, result in all_results.items():
        print(f"\n{config_name}:")
        summary_lines.append(f"{config_name}:\n")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            summary_lines.append(f"  ERROR: {result['error']}\n")
        else:
            timing = result.get('timing', {})
            memory = result.get('memory', {})
            if timing:
                print(f"  Total time: {timing.get('total_time', 0):.2f}s")
                summary_lines.append(f"  Total time: {timing.get('total_time', 0):.2f}s\n")
            if memory:
                print(f"  GPU usage: {memory.get('gpu_usage_mb', 0):.1f} MB")
                summary_lines.append(f"  GPU usage: {memory.get('gpu_usage_mb', 0):.1f} MB\n")
        summary_lines.append("\n")

    summary_file = base_result_dir / "summary_results.txt"
    with open(summary_file, 'w') as f:
        f.writelines(summary_lines)
    print(f"\nResults summary saved to: {summary_file}")
    print("Detailed metrics saved in respective subdirectories.")

    return all_results

if __name__ == "__main__":
    main()