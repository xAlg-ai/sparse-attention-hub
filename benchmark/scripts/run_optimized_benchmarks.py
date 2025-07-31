#!/usr/bin/env python3
"""
CLI script for running optimized benchmarks with hyperparameter tuning.

This script demonstrates the complete workflow of:
1. Hyperparameter optimization for sparse attention configurations
2. Benchmark execution using optimized configurations
3. Result analysis and comparison

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

# Set project root and add to Python path (like magic_pig_experiments)
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels from benchmark/scripts/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark.optimized_executor import run_optimized_benchmarks
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

# Sparse attention config imports
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
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


def create_sparse_configs(config_names: List[str]) -> List[tuple]:
    """Create sparse attention configurations from names."""
    configs = []
    
    # Always include dense baseline
    if "dense" not in config_names:
        configs.append(("dense", None))
    
    for config_name in config_names:
        if config_name == "dense":
            configs.append(("dense", None))
        elif config_name == "magic_pig":
            # Base Magic Pig config - will be optimized
            magic_pig_config = ResearchAttentionConfig(
                masker_configs=[MagicPigConfig(
                    lsh_l=8,
                    lsh_k=32,
                    center=True,
                    packing="int64",
                    seed=42
                )]
            )
            configs.append(("magic_pig", magic_pig_config))
        else:
            raise ValueError(f"Unknown sparse config: {config_name}")
    
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
        else:
            # For other benchmarks, use provided subsets or None
            configs.append(BenchmarkConfig(
                benchmark_name=benchmark_name,
                subsets=subsets
            ))
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run optimized benchmarks with hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Magic Pig optimization
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --benchmark loogle --config magic_pig
  
  # Multiple models and configs
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --config dense magic_pig --benchmark loogle
  
  # Specific benchmark subsets
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --config magic_pig --benchmark loogle --subset shortdep_qa
  
  # Disable optimization (just run benchmarks)
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --config magic_pig --benchmark loogle --no-optimization
  
  # Custom optimization settings
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --config magic_pig --benchmark loogle --optimization-samples 50 --max-concurrent 8
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model", 
        nargs="+", 
        required=True,
        help="Model name(s) to benchmark (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    )
    
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark name(s) to run (e.g., loogle)"
    )
    
    parser.add_argument(
        "--config",
        nargs="+",
        default=["dense", "magic_pig"],
        help="Sparse attention config(s) to test (default: dense magic_pig)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--subset",
        nargs="*",
        help="Benchmark subset(s) to run (default: all available)"
    )
    
    parser.add_argument(
        "--result-dir",
        default="./optimized_benchmark_results",
        help="Directory to store results (default: ./optimized_benchmark_results)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Maximum concurrent benchmark runs (default: 2)"
    )
    
    parser.add_argument(
        "--gpu-pool",
        nargs="*",
        type=int,
        default=[0],
        help="GPU IDs to use (default: [0])"
    )
    
    # Optimization settings
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Disable hyperparameter optimization"
    )
    
    parser.add_argument(
        "--optimization-samples",
        type=int,
        default=20,
        help="Number of optimization samples per config (default: 20)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent optimization trials (default: 4)"
    )
    
    parser.add_argument(
        "--optimization-cache-dir",
        default=None,
        help="Directory to cache optimization results (default: result_dir/hyperparameter_cache)"
    )
    
    # Other options
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip existing benchmark results (default: True)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="PyTorch dtype for model loading (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting optimized benchmark execution")
    logger.info(f"Models: {args.model}")
    logger.info(f"Benchmarks: {args.benchmark}")
    logger.info(f"Sparse configs: {args.config}")
    logger.info(f"Optimization enabled: {not args.no_optimization}")
    
    try:
        # Create configurations
        sparse_configs = create_sparse_configs(args.config)
        benchmark_configs = create_benchmark_configs(args.benchmark, args.subset)
        
        # Create adapter config
        adapter_config = AdapterConfig(
            model_kwargs={"torch_dtype": args.torch_dtype},
            tokenizer_kwargs={}
        )
        
        # Set up cache directory
        cache_dir = args.optimization_cache_dir
        if cache_dir is None:
            cache_dir = f"{args.result_dir}/hyperparameter_cache"
        
        logger.info(f"Result directory: {args.result_dir}")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Run optimized benchmarks
        run_optimized_benchmarks(
            model_names=args.model,
            sparse_attention_configs=sparse_configs,
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config,
            result_dir=args.result_dir,
            gpu_ids=args.gpu_pool,
            max_concurrent_runs=args.num_workers,
            enable_optimization=not args.no_optimization,
            optimization_samples=args.optimization_samples
        )
        
        logger.info("Benchmark execution completed successfully")
        
        # Print summary
        result_path = Path(args.result_dir)
        if result_path.exists():
            logger.info(f"Results saved to: {result_path.resolve()}")
            
            # Look for optimization summary
            opt_summary = result_path / "optimization_summary.txt"
            if opt_summary.exists():
                logger.info(f"Optimization summary: {opt_summary.resolve()}")
                print("\n" + "="*60)
                print("OPTIMIZATION SUMMARY")
                print("="*60)
                with open(opt_summary, 'r') as f:
                    print(f.read())
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
