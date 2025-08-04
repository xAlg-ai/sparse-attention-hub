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
project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels from benchmark/optimizer/scripts/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark.optimizer.optimized_executor import run_optimized_benchmarks
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

# Sparse attention config imports
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)


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
    
    # Always include dense baseline if not explicitly included
    if "dense" not in config_names:
        configs.append(("dense", None))
    
    for config_name in config_names:
        if config_name == "dense":
            configs.append(("dense", None))
        elif config_name == "magic_pig":
            # Base Magic Pig config - parameters will be optimized by MagicPigOptimizer
            # We pass None to indicate this config should be optimized
            configs.append(("magic_pig", None))
        elif config_name == "streaming_conservative":
            # StreamingLLM conservative configuration
            streaming_config = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=4),
                LocalMaskerConfig(window_size=16)
            ])
            configs.append(("streaming_conservative", streaming_config))
        else:
            raise ValueError(f"Unknown sparse config: {config_name}. Supported configs: dense, magic_pig, streaming_conservative")
    
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
            # LongBench (existing) - using narrativeqa task
            benchmark_subsets = subsets if subsets else ["narrativeqa"]
            configs.append(BenchmarkConfig(
                benchmark_name="longbench",
                subsets=benchmark_subsets
            ))
        elif benchmark_name == "mock_benchmark":
            # Mock Benchmark (existing) - using single task
            benchmark_subsets = subsets if subsets else ["reading_comprehension"]
            configs.append(BenchmarkConfig(
                benchmark_name="mock_benchmark",
                subsets=benchmark_subsets
            ))
        else:
            # For other benchmarks, use provided subsets or None (all available)
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
  
  # Test streaming conservative configuration (no optimization needed)
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --benchmark loogle --config streaming_conservative --no-optimization
  
  # Multiple models and configs
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --config dense magic_pig streaming_conservative --benchmark loogle
  
  # Multiple benchmarks
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --config magic_pig --benchmark loogle infinite_bench ruler
  
  # Specific benchmark subsets
  python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --config magic_pig --benchmark loogle --subset shortdep_qa
  
  # Full benchmark suite with streaming conservative
  python run_optimized_benchmarks.py --model microsoft/Phi-4-mini-instruct --config dense streaming_conservative --benchmark loogle infinite_bench ruler zero_scrolls longbenchv2 --no-optimization
  
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
        help="Benchmark name(s) to run. Options: loogle, infinite_bench, ruler, zero_scrolls, longbenchv2, aime2024, aime2025, longbench, mock_benchmark"
    )
    
    parser.add_argument(
        "--config",
        nargs="+",
        default=["dense", "magic_pig"],
        help="Sparse attention config(s) to test. Options: dense, magic_pig, streaming_conservative (default: dense magic_pig)"
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
    
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Maximum number of requests per benchmark subtask (default: no limit)"
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
        
        # Set up request kwargs if max_requests is specified
        request_kwargs = {}
        if args.max_requests is not None:
            request_kwargs["max_requests"] = args.max_requests
            logger.info(f"Limiting requests per subtask to: {args.max_requests}")
        
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
            optimization_samples=args.optimization_samples,
            request_kwargs=request_kwargs if request_kwargs else None
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
