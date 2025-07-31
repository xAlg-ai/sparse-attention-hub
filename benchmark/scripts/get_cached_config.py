#!/usr/bin/env python3
"""
Utility script to get cached optimized configurations.

This script allows you to retrieve previously optimized hyperparameter configurations
without running the full optimization process.
"""

import os
import sys
from pathlib import Path

# Set project root and add to Python path (like magic_pig_experiments)
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels from benchmark/scripts/
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark.hyperparameter_optimization import HyperparameterOptimizer, OptimizationConfig


def main():
    """Main function to demonstrate cached config retrieval."""
    
    # Configuration
    cache_dir = "./demo_results/hyperparameter_cache"
    
    # Create optimizer with cache directory
    optimization_config = OptimizationConfig(
        enabled=True,
        cache_dir=cache_dir
    )
    
    optimizer = HyperparameterOptimizer(optimization_config)
    
    # Example queries
    queries = [
        {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "config_type": "magic_pig", 
            "benchmark_name": "loogle",
            "subset": "shortdep_qa"
        },
        {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "config_type": "magic_pig",
            "benchmark_name": "loogle", 
            "subset": "longdep_qa"
        }
    ]
    
    print("=" * 80)
    print("CACHED HYPERPARAMETER CONFIGURATION RETRIEVAL")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}")
    print()
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query['model_name']} + {query['config_type']} + {query['benchmark_name']}/{query['subset']}")
        print("-" * 60)
        
        # Get cached config
        cached_result = optimizer.get_cached_best_config(**query)
        
        if cached_result:
            print("✅ Found cached optimization result:")
            print(f"   Best parameters: {cached_result['best_params']}")
            print(f"   Best metrics: {cached_result['best_metrics']}")
            print(f"   Optimization metadata: {cached_result['optimization_metadata']}")
            
            # Try to create the actual config object
            optimized_config = optimizer.create_optimized_config_from_cache(**query)
            if optimized_config:
                print("✅ Successfully created optimized config object")
            else:
                print("❌ Failed to create config object from cached params")
                
        else:
            print("❌ No cached optimization found")
            cache_key = optimizer.get_cache_key(query['model_name'], query['config_type'], query['benchmark_name'], query['subset'])
            print(f"   Cache key would be: {cache_key}")
            
        print()
    
    # Show available cache files
    cache_path = Path(cache_dir)
    if cache_path.exists():
        cache_files = list(cache_path.glob("*.json"))
        print(f"Available cache files ({len(cache_files)}):")
        for cache_file in sorted(cache_files):
            print(f"  - {cache_file.name}")
    else:
        print("Cache directory does not exist")


if __name__ == "__main__":
    main()
