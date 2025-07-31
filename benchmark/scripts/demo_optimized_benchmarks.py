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
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels from benchmark/scripts/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from benchmark.hyperparameter_optimization import OptimizationConfig
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
    
    # Create custom optimization config for very quick iteration
    optimization_config = OptimizationConfig(
        enabled=True,
        num_samples=3,  # Fewer samples for quick demo 
        max_concurrent=1,  # Single concurrent trial
        optimization_metric="combined_score",
        optimization_mode="min",
        cache_dir=f"{result_dir}/hyperparameter_cache",
        quick_eval_requests=2  # Only 2 requests per optimization trial
    )
    
    try:
        # Use the new direct approach with custom optimization config
        from benchmark.optimized_executor import create_optimized_benchmark_executor
        
        executor = create_optimized_benchmark_executor(
            gpu_ids=[0],
            max_concurrent_runs=1,
            base_result_dir=result_dir,
            optimization_config=optimization_config,
            enable_optimization=True
        )
        
        executor.run_benchmark_matrix(
            model_names=model_names,
            sparse_attention_configs=sparse_configs,
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config,
            request_kwargs={"max_requests": 2}  # Back to 2 requests for faster multi-task testing
        )
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Show results
        result_path = Path(result_dir)
        if result_path.exists():
            print(f"\nResults saved to: {result_path.resolve()}")
            
            # Show optimization summary if available
            opt_summary = result_path / "optimization_summary.txt"
            if opt_summary.exists():
                print("\nOptimization Summary:")
                print("-" * 40)
                with open(opt_summary, 'r') as f:
                    print(f.read())
            
        print("\nNext steps:")
        print("- Check the results directory for detailed benchmark outputs")
        print("- Optimization results are cached for future runs")
        print("- Try different models or benchmark subsets")
        print("- Increase optimization_samples for better results")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(run_demo())
