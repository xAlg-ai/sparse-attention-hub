#!/usr/bin/env python3
"""
Simple demo of the hyperparameter optimization system.

This script demonstrates how to run a quick optimization + benchmark job
with minimal setup.
"""

import logging
import os
import sys
from pathlib import Path

# Set project root and add to Python path (like magic_pig_experiments)
project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels from benchmark/optimizer/scripts/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from benchmark.optimizer.hyperparameter_optimization import OptimizationConfig
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_demo():
    """Run a simple demo of the optimized benchmark system."""
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION + BENCHMARK DEMO")
    print("=" * 80)
    
    # Configuration
    model_names = ["meta-llama/Llama-3.1-8B-Instruct"]
    
    # Create sparse attention configs
    magic_pig_config = ResearchAttentionConfig(
        masker_configs=[MagicPigConfig(
            lsh_l=8,
            lsh_k=32,
            center=True,
            packing="int64",
            seed=42
        )]
    )
    
    sparse_configs = [
        ("dense", None),
        ("magic_pig", magic_pig_config)
    ]
    
    # Create benchmark configs
    benchmark_configs = [
        BenchmarkConfig(
            benchmark_name="loogle",
            subsets=["shortdep_qa", "longdep_qa"]  # Both loogle tasks for testing
        )
    ]
    
    # Create adapter config with quick iteration settings
    adapter_config = AdapterConfig(
        model_kwargs={"torch_dtype": "auto"},
        tokenizer_kwargs={}
    )
    
    result_dir = "./demo_results"
    
    print("Configuration:")
    print(f"  Models: {model_names}")
    print(f"  Sparse configs: {[name for name, _ in sparse_configs]}")
    print(f"  Benchmarks: {[bc.benchmark_name for bc in benchmark_configs]}")
    print(f"  Result directory: {result_dir}")
    print("  Optimization samples: 3 (very quick demo)")
    print("  Requests per trial: 2 (ultra-fast iteration)")
    print("  Requests per benchmark: 2 (multi-task testing)")
    print()
    
    print("Running optimized benchmarks...")
    print("This will:")
    print("1. Optimize Magic Pig hyperparameters using Ray Tune (2 requests per trial)")
    print("2. Run benchmarks with both dense and optimized Magic Pig (2 requests each)")
    print("3. Test both shortdep_qa and longdep_qa tasks (separate optimization per task)")
    print("4. Save results and optimization summary")
    print()
    
    # Example 1: Global optimization (single best config for all tasks)
    print("=" * 80)
    print("EXAMPLE 1: Global Optimization (single best config)")
    print("=" * 80)
    
    # Create custom optimization config for global optimization
    global_optimization_config = OptimizationConfig(
        enabled=True,
        num_samples=3,  # Fewer samples for quick demo 
        max_concurrent=1,  # Single concurrent trial
        optimization_metric="combined_score",
        optimization_mode="min",
        cache_dir=f"{result_dir}/hyperparameter_cache_global",
        quick_eval_requests=2,  # Only 2 requests per optimization trial
        use_per_task_config=False  # Global optimization
    )
    
    try:
        # Use the optimized executor with global config
        from benchmark.optimizer.optimized_executor import create_optimized_benchmark_executor
        
        print("Running GLOBAL optimization...")
        executor = create_optimized_benchmark_executor(
            gpu_ids=[0],
            max_concurrent_runs=1,
            base_result_dir=f"{result_dir}/global",
            optimization_config=global_optimization_config,
            enable_optimization=True
        )
        
        executor.run_benchmark_matrix(
            model_names=model_names,
            sparse_attention_configs=sparse_configs,
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config,
            request_kwargs={"max_requests": 2}
        )
        
        print("✅ Global optimization completed!")
        
    except Exception as e:
        print(f"❌ Global optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Per-task optimization (best config for each task)
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Per-Task Optimization (best config per task)")
    print("=" * 80)
    
    # Create custom optimization config for per-task optimization
    per_task_optimization_config = OptimizationConfig(
        enabled=True,
        num_samples=3,  # Fewer samples for quick demo 
        max_concurrent=1,  # Single concurrent trial
        optimization_metric="combined_score",
        optimization_mode="min",
        cache_dir=f"{result_dir}/hyperparameter_cache_per_task",
        quick_eval_requests=2,  # Only 2 requests per optimization trial
        use_per_task_config=True  # Per-task optimization
    )
    
    try:
        print("Running PER-TASK optimization...")
        executor = create_optimized_benchmark_executor(
            gpu_ids=[0],
            max_concurrent_runs=1,
            base_result_dir=f"{result_dir}/per_task",
            optimization_config=per_task_optimization_config,
            enable_optimization=True
        )
        
        executor.run_benchmark_matrix(
            model_names=model_names,
            sparse_attention_configs=sparse_configs,
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config,
            request_kwargs={"max_requests": 2}
        )
        
        print("✅ Per-task optimization completed!")
        
    except Exception as e:
        print(f"❌ Per-task optimization failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Summary section (runs regardless of success/failure)
    print("\n" + "=" * 80)
    print("BOTH OPTIMIZATION MODES COMPLETED!")
    print("=" * 80)
    
    # Show results for both modes
    result_path = Path(result_dir)
    if result_path.exists():
        print(f"\nResults saved to: {result_path.resolve()}")
        
        # Show global optimization results
        global_path = result_path / "global"
        if global_path.exists():
            print(f"\nGlobal optimization results: {global_path}")
            
        # Show per-task optimization results  
        per_task_path = result_path / "per_task"
        if per_task_path.exists():
            print(f"Per-task optimization results: {per_task_path}")
        
    print("\nNext steps:")
    print("- Compare global vs per-task optimization results")
    print("- Use the get_cached_config.py script to retrieve optimized configs")
    print("- Try: python benchmark/scripts/get_cached_config.py --config-type sparse_attention --show-all")
    print("- Or: python benchmark/scripts/get_cached_config.py --config-type sparse_attention --create-optimized")
    
    return 0


if __name__ == "__main__":
    setup_logging()
    try:
        result = run_demo()
        sys.exit(result)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
