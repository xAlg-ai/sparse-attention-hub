#!/usr/bin/env python3
"""
Script to retrieve and display cached optimization results.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Set project root and add to Python path (like magic_pig_experiments)
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels from benchmark/scripts/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark.hyperparameter_optimization import HyperparameterOptimizer, OptimizationConfig


def main():
    parser = argparse.ArgumentParser(description="Retrieve cached sparse attention optimization results")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--config-type", required=True, help="Configuration type (e.g., 'sparse_attention')")
    parser.add_argument("--task", help="Specific task to get config for (optional)")
    parser.add_argument("--cache-dir", default="./hyperparameter_cache", help="Cache directory path")
    parser.add_argument("--show-all", action="store_true", help="Show all cached configs for this type")
    parser.add_argument("--create-optimized", action="store_true", help="Create OptimizedSparseConfig from cache")
    parser.add_argument("--per-task", action="store_true", default=True, help="Enable per-task config usage (default: True)")
    
    args = parser.parse_args()
    
    # Initialize optimizer with configuration
    optimization_config = OptimizationConfig(
        enabled=True,
        cache_dir=args.cache_dir
    )
    
    optimizer = HyperparameterOptimizer(optimization_config)
    
    if args.show_all:
        # Show all cached configs for this type
        all_configs = optimizer.get_all_cached_configs_for_type(args.model, args.config_type)
        
        if not all_configs:
            print(f"No cached configurations found for {args.model}/{args.config_type}")
            # List available cache files
            cache_files = list(optimizer.cache_dir.glob("*.json"))
            if cache_files:
                print(f"Available cache files in {optimizer.cache_dir}:")
                for cache_file in sorted(cache_files):
                    print(f"  - {cache_file.name}")
            else:
                print(f"No cache files found in {optimizer.cache_dir}")
            return
            
        print(f"Found {len(all_configs)} cached configurations for {args.config_type}:")
        print("=" * 80)
        
        for task_key, cached_result in all_configs.items():
            print(f"\nTask: {task_key}")
            print(f"Combined Score: {cached_result['best_metrics']['combined_score']:.4f}")
            print(f"Best Parameters: {json.dumps(cached_result['best_params'], indent=2)}")
            print("-" * 40)
            
    elif args.create_optimized:
        # Create OptimizedSparseConfig from cache
        optimized_config = optimizer.create_optimized_sparse_config_from_cache(
            args.model, args.config_type, args.per_task
        )
        
        if optimized_config:
            print("Created OptimizedSparseConfig:")
            print("=" * 80)
            print(f"Config Type: {optimized_config.config_type}")
            print(f"Is Optimized: {optimized_config.is_optimized}")
            print(f"Global Best Parameters: {json.dumps(optimized_config.optimized_params, indent=2)}")
            print(f"Per-task configs available: {len(optimized_config.per_task_configs)}")
            
            if optimized_config.per_task_configs:
                print("\nPer-task configurations:")
                for task_key, task_params in optimized_config.per_task_configs.items():
                    print(f"  {task_key}: {json.dumps(task_params, indent=4)}")
        else:
            print(f"No cached configurations found for {args.model}/{args.config_type}")
            # List available cache files
            cache_files = list(optimizer.cache_dir.glob("*.json"))
            if cache_files:
                print(f"Available cache files in {optimizer.cache_dir}:")
                for cache_file in sorted(cache_files):
                    print(f"  - {cache_file.name}")
            else:
                print(f"No cache files found in {optimizer.cache_dir}")
            
    else:
        # Get specific cached config by building the cache key
        if args.task:
            # Task format should be like "loogle_shortdep_qa"
            if "_" in args.task:
                benchmark_name, subset = args.task.rsplit("_", 1)  
            else:
                benchmark_name = args.task
                subset = None
        else:
            print("Please specify --task, --show-all, or --create-optimized")
            return
            
        cached_result = optimizer.get_cached_best_config(args.model, args.config_type, benchmark_name, subset)
        
        if cached_result:
            cache_key = optimizer.get_cache_key(args.model, args.config_type, benchmark_name, subset)
            print(f"Cached result for {cache_key}:")
            print("=" * 80)
            print(f"Best Parameters: {json.dumps(cached_result['best_params'], indent=2)}")
            print(f"Best Metrics: {json.dumps(cached_result['best_metrics'], indent=2)}")
            print(f"Optimization Metadata: {json.dumps(cached_result['optimization_metadata'], indent=2)}")
        else:
            print(f"No cached result found for task: {args.task}")
            # List available cache files  
            cache_files = list(optimizer.cache_dir.glob("*.json"))
            if cache_files:
                print(f"Available cache files in {optimizer.cache_dir}:")
                for cache_file in sorted(cache_files):
                    print(f"  - {cache_file.name}")
            else:
                print(f"No cache files found in {optimizer.cache_dir}")


if __name__ == "__main__":
    main()
