#!/usr/bin/env python3
"""
CLI script for running optimized benchmarks with declarative hyperparameter tuning.

This script demonstrates the complete workflow using the new declarative search space system:
1. Automatic search space discovery from config classes
2. Hyperparameter optimization for sparse attention configurations
3. Benchmark execution using optimized configurations
4. Result analysis and comparison

Usage:
    python run_optimized_benchmarks.py --model microsoft/DialoGPT-medium --benchmark loogle --config magic_pig
    python run_optimized_benchmarks.py --help
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Set project root and add to Python path
project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels from benchmark/optimizer/scripts/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark.optimizer.optimized_executor import run_optimized_benchmarks
from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from benchmark.optimizer.generic_config_optimizer import auto_register_config

# Sparse attention config imports
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('optimized_benchmarks.log')
        ]
    )


def auto_register_sparse_configs() -> None:
    """Auto-register all sparse attention configs for declarative optimization."""
    print("🔧 Auto-registering sparse attention configs for declarative optimization...")
    
    # Register MagicPigConfig for automatic search space discovery
    try:
        auto_register_config(MagicPigConfig, "magic_pig")
        print("✅ Registered MagicPigConfig as 'magic_pig'")
    except Exception as e:
        print(f"⚠️  MagicPigConfig registration failed: {e}")
    
    # Register other configs as needed
    try:
        auto_register_config(LocalMaskerConfig, "local_masker")
        print("✅ Registered LocalMaskerConfig as 'local_masker'")
    except Exception as e:
        print(f"⚠️  LocalMaskerConfig registration failed: {e}")
    
    try:
        auto_register_config(SinkMaskerConfig, "sink_masker")
        print("✅ Registered SinkMaskerConfig as 'sink_masker'")
    except Exception as e:
        print(f"⚠️  SinkMaskerConfig registration failed: {e}")


def create_sparse_configs(config_names: List[str]) -> List[tuple]:
    """Create sparse attention configurations from names using declarative system."""
    configs = []
    
    # Always include dense baseline if not explicitly included
    if "dense" not in config_names:
        configs.append(("dense", None))
    
    for config_name in config_names:
        if config_name == "dense":
            configs.append(("dense", None))
        elif config_name == "magic_pig":
            # Magic Pig with declarative optimization - None triggers auto-optimization
            configs.append(("magic_pig", None))
        elif config_name == "streaming_conservative":
            # StreamingLLM conservative configuration (fixed config)
            streaming_config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=4),
                LocalMaskerConfig(window_size=16)
            ])
            configs.append(("streaming_conservative", streaming_config))
        elif config_name == "local_only":
            # Local attention only with declarative optimization
            configs.append(("local_only", None))
        elif config_name == "sink_only":
            # Sink attention only with declarative optimization
            configs.append(("sink_only", None))
        else:
            raise ValueError(f"Unknown sparse config: {config_name}. Supported configs: dense, magic_pig, streaming_conservative, local_only, sink_only")
    
    return configs


def create_benchmark_configs(benchmark_names: List[str], subsets: Optional[List[str]] = None) -> List[BenchmarkConfig]:
    """Create benchmark configurations from names."""
    configs = []
    
    for benchmark_name in benchmark_names:
        if benchmark_name == "loogle":
            # Use specific subsets if provided, otherwise use defaults
            if subsets:
                benchmark_subsets = subsets
            else:
                benchmark_subsets = ["shortdep_qa", "longdep_qa"]
            
            configs.append(BenchmarkConfig(
                benchmark_name="loogle",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "infinite_bench":
            # InfiniteBench - using passkey task
            benchmark_subsets = subsets if subsets else ["passkey"]
            configs.append(BenchmarkConfig(
                benchmark_name="infinite_bench",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "ruler":
            # Ruler - using 4096 context length
            benchmark_subsets = subsets if subsets else ["4096"]
            configs.append(BenchmarkConfig(
                benchmark_name="ruler",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "zero_scrolls":
            # ZeroScrolls - using gov_report task
            benchmark_subsets = subsets if subsets else ["default"]
            configs.append(BenchmarkConfig(
                benchmark_name="zero_scrolls",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "longbenchv2":
            # LongBenchv2 - using 0shot task
            benchmark_subsets = subsets if subsets else ["0shot"]
            configs.append(BenchmarkConfig(
                benchmark_name="longbenchv2",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "aime2024":
            # AIME2024 - using single task
            benchmark_subsets = subsets if subsets else ["aime2024"]
            configs.append(BenchmarkConfig(
                benchmark_name="aime2024",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "aime2025":
            # AIME2025 - using single task
            benchmark_subsets = subsets if subsets else ["aime2025"]
            configs.append(BenchmarkConfig(
                benchmark_name="aime2025",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "longbench":
            # LongBench - using narrativeqa task
            benchmark_subsets = subsets if subsets else ["narrativeqa"]
            configs.append(BenchmarkConfig(
                benchmark_name="longbench",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "mock_benchmark":
            # Mock Benchmark - using reading_comprehension task
            benchmark_subsets = subsets if subsets else ["reading_comprehension"]
            configs.append(BenchmarkConfig(
                benchmark_name="mock_benchmark",
                subsets=benchmark_subsets
            ))
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Supported benchmarks: loogle, infinite_bench, ruler, zero_scrolls, longbenchv2, aime2024, aime2025, longbench, mock_benchmark")
    
    return configs


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run optimized benchmarks with declarative hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with Magic Pig optimization
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --benchmark loogle --config magic_pig --samples 5

  # Full evaluation with multiple configs
  python run_optimized_benchmarks.py --model microsoft/Phi-4-mini-instruct --benchmark infinite_bench,ruler --config dense,magic_pig --samples 20

  # Per-task optimization with custom subsets
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.2-1B-Instruct --benchmark loogle --subsets shortdep_qa --config magic_pig --samples 10
        """
    )
    
    # Model and configuration
    parser.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--benchmark", required=True, help="Benchmark name(s) (comma-separated)")
    parser.add_argument("--config", required=True, help="Sparse attention config(s) (comma-separated)")
    parser.add_argument("--subsets", help="Benchmark subsets (comma-separated, optional)")
    
    # Optimization settings
    parser.add_argument("--samples", type=int, default=10, help="Number of optimization samples (default: 10)")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent optimization trials (default: 2)")
    parser.add_argument("--quick-eval", type=int, default=5, help="Quick evaluation requests per trial (default: 5)")
    parser.add_argument("--per-task", action="store_true", default=True, help="Enable per-task optimization (default: True)")
    
    # Execution settings
    parser.add_argument("--gpu-ids", default="0", help="GPU IDs to use (comma-separated, default: 0)")
    parser.add_argument("--max-concurrent-runs", type=int, default=1, help="Max concurrent benchmark runs (default: 1)")
    parser.add_argument("--result-dir", default="./optimized_benchmark_results", help="Result directory (default: ./optimized_benchmark_results)")
    parser.add_argument("--max-requests", type=int, default=10, help="Max requests per benchmark (default: 10)")
    parser.add_argument("--max-context-length", type=int, default=1024, help="Max context length (default: 1024)")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens for generation (default: 50)")
    
    # Logging and debugging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-optimization", action="store_true", help="Disable hyperparameter optimization")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Parse arguments
    model_names = [args.model]
    benchmark_names = args.benchmark.split(",")
    config_names = args.config.split(",")
    subsets = args.subsets.split(",") if args.subsets else None
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    
    # Auto-register configs for declarative optimization
    auto_register_sparse_configs()
    
    # Create configurations
    sparse_configs = create_sparse_configs(config_names)
    benchmark_configs = create_benchmark_configs(benchmark_names, subsets)
    
    # Create adapter config
    adapter_config = AdapterConfig(
        model_kwargs={"torch_dtype": "auto"},
        tokenizer_kwargs={}
    )
    
    # Print configuration summary
    print("🚀 DECLARATIVE OPTIMIZED BENCHMARK RUNNER")
    print("=" * 60)
    print(f"Model: {model_names[0]}")
    print(f"Benchmarks: {[bc.benchmark_name for bc in benchmark_configs]}")
    print(f"Sparse configs: {[name for name, _ in sparse_configs]}")
    print(f"Optimization samples: {args.samples}")
    print(f"Per-task optimization: {'enabled' if args.per_task else 'disabled'}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Result directory: {args.result_dir}")
    print()
    
    # Show declarative optimization info
    print("🔧 Declarative Optimization Features:")
    print("  ✅ Automatic search space discovery")
    print("  ✅ Auto-composition of composite configs")
    print("  ✅ Per-task hyperparameter optimization")
    print("  ✅ Result caching and resumability")
    print()
    
    try:
        # Run optimized benchmarks
        run_optimized_benchmarks(
            model_names=model_names,
            sparse_attention_configs=sparse_configs,
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config,
            result_dir=args.result_dir,
            gpu_ids=gpu_ids,
            max_concurrent_runs=args.max_concurrent_runs,
            enable_optimization=not args.no_optimization,
            optimization_samples=args.samples,
            max_concurrent=args.max_concurrent,
            quick_eval_requests=args.quick_eval,
            use_per_task_config=args.per_task,
            request_kwargs={
                "max_requests": args.max_requests,
                "max_context_length": args.max_context_length
            },
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0
            }
        )
        
        print(f"\n✅ Optimization and benchmarking completed successfully!")
        print(f"   Results saved to: {args.result_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        print(f"   Partial results in: {args.result_dir}")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
