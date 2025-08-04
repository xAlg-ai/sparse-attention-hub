#!/usr/bin/env python3
"""
Full Pipeline Demo of the Declarative Search Space System

This script demonstrates the complete optimization pipeline using the new
declarative search space system. It shows how config classes define their
own search spaces and are automatically composed without manual setup.
"""

import logging
import os
import sys
from pathlib import Path

# Set project root and add to Python path
project_root = Path(__file__).resolve().parents[3]
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from benchmark.optimizer.hyperparameter_optimization import OptimizationConfig
from benchmark.optimizer.optimized_executor import run_optimized_benchmarks
from benchmark.optimizer.generic_config_optimizer import (
    auto_create_composite_optimizer, 
    auto_register_config,
    create_optimizer_for_config
)


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demo_search_space_discovery():
    """Demo 1: Show how search spaces are automatically discovered."""
    print("=" * 80)
    print("DEMO 1: AUTOMATIC SEARCH SPACE DISCOVERY")
    print("=" * 80)
    
    from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import (
        LocalMaskerConfig, SinkMaskerConfig
    )
    
    # Test individual config search spaces
    configs_to_test = [
        (MagicPigConfig, "MagicPigConfig"),
        (LocalMaskerConfig, "LocalMaskerConfig"), 
        (SinkMaskerConfig, "SinkMaskerConfig")
    ]
    
    for config_class, config_name in configs_to_test:
        print(f"\n--- {config_name} Default Search Space ---")
        
        # Get default search space directly from config class
        if hasattr(config_class, 'get_default_search_space'):
            search_space = config_class.get_default_search_space()
            print(f"✅ {config_name} defines its own search space:")
            for param, space in search_space.items():
                # Show readable values for Ray Tune objects
                if hasattr(space, '_spec') and 'categories' in space._spec:
                    print(f"  {param}: tune.choice({space._spec['categories']})")
                elif hasattr(space, 'low') and hasattr(space, 'high'):
                    print(f"  {param}: tune.uniform({space.low}, {space.high})")
                else:
                    print(f"  {param}: {space}")
        else:
            print(f"⚠️  {config_name} has no default search space defined")
        
        # Test optimizer creation
        optimizer = create_optimizer_for_config(
            config_class=config_class,
            config_name=config_name.lower()
        )
        
        # Generate search space using auto-discovery
        auto_search_space = optimizer.create_search_space()
        print(f"✅ Auto-discovered search space for {config_name}:")
        for param, space in auto_search_space.items():
            # Show readable values for Ray Tune objects
            if hasattr(space, '_spec') and 'categories' in space._spec:
                print(f"  {param}: tune.choice({space._spec['categories']})")
            elif hasattr(space, 'low') and hasattr(space, 'high'):
                print(f"  {param}: tune.uniform({space.low}, {space.high})")
            else:
                print(f"  {param}: {space}")


def demo_composite_optimization():
    """Demo 2: Show composite optimization with auto-discovery."""
    print("\n" + "=" * 80)
    print("DEMO 2: COMPOSITE OPTIMIZATION WITH AUTO-DISCOVERY")
    print("=" * 80)
    
    from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import (
        LocalMaskerConfig, SinkMaskerConfig
    )
    
    print("\n--- Creating Composite Optimizer ---")
    composite_optimizer = auto_create_composite_optimizer(
        masker_configs=[SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig],
        config_name="magic_pig_composite"
    )
    
    # Generate combined search space
    search_space = composite_optimizer.create_search_space()
    print(f"✅ Auto-discovered composite search space:")
    for param, space in search_space.items():
        # Show readable values for Ray Tune objects
        if hasattr(space, '_spec') and 'categories' in space._spec:
            print(f"  {param}: tune.choice({space._spec['categories']})")
        elif hasattr(space, 'low') and hasattr(space, 'high'):
            print(f"  {param}: tune.uniform({space.low}, {space.high})")
        else:
            print(f"  {param}: {space}")
    
    # Test composite config creation
    if search_space:
        test_params = {}
        for param, space in search_space.items():
            if hasattr(space, '_spec') and 'categories' in space._spec:
                test_params[param] = space._spec['categories'][0]
            elif hasattr(space, 'low') and hasattr(space, 'high'):
                test_params[param] = (space.low + space.high) / 2
            else:
                test_params[param] = space
        
        print(f"\n✅ Test parameters: {test_params}")
        
        try:
            test_config = composite_optimizer.create_config_from_params(test_params)
            print(f"✅ Successfully created composite config: {test_config}")
            print(f"   Config type: {type(test_config)}")
            if hasattr(test_config, 'masker_configs'):
                print(f"   Number of maskers: {len(test_config.masker_configs)}")
                for i, masker in enumerate(test_config.masker_configs):
                    print(f"     Masker {i}: {type(masker).__name__} - {masker}")
        except Exception as e:
            print(f"❌ Failed to create composite config: {e}")


def demo_auto_registration():
    """Demo 3: Show automatic config registration."""
    print("\n" + "=" * 80)
    print("DEMO 3: AUTOMATIC CONFIG REGISTRATION")
    print("=" * 80)
    
    from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig
    
    print("\n--- Auto-Registering MagicPigConfig ---")
    try:
        registered_name = auto_register_config(MagicPigConfig)
        print(f"✅ Successfully registered {MagicPigConfig.__name__} as '{registered_name}'")
        
        # Test that it's now available in the optimizer registry
        from benchmark.optimizer.hyperparameter_optimization import get_sparse_optimizer, list_available_optimizers
        
        available_optimizers = list_available_optimizers()
        print(f"Available optimizers: {available_optimizers}")
        
        if registered_name in available_optimizers:
            optimizer = get_sparse_optimizer(registered_name)
            search_space = optimizer.get_search_space()
            print(f"✅ Successfully retrieved registered optimizer")
            print(f"   Search space keys: {list(search_space.keys())}")
        else:
            print(f"⚠️  Registered optimizer '{registered_name}' not found in registry")
            
    except Exception as e:
        print(f"❌ Auto-registration failed: {e}")
        import traceback
        traceback.print_exc()


def demo_full_optimization_pipeline():
    """Demo 4: Run the complete optimization pipeline."""
    print("\n" + "=" * 80)
    print("DEMO 4: FULL OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    print("\nThis demo runs the complete optimization pipeline:")
    print("1. Auto-discovers search spaces from config classes")
    print("2. Runs hyperparameter optimization with Ray Tune (per task)")
    print("3. Executes benchmarks with optimized configurations")
    print("4. Saves results and optimization summary")
    print("5. Compares dense vs optimized sparse attention")
    print()
    
    # Configuration for quick demo
    model_names = ["meta-llama/Llama-3.1-8B-Instruct"]
    
    # Create benchmark configs
    benchmark_configs = [
        BenchmarkConfig(
            benchmark_name="loogle",
            subsets=["shortdep_qa", "longdep_qa"]  # Both loogle tasks
        )
    ]
    
    # Create adapter config
    adapter_config = AdapterConfig(
        model_kwargs={"torch_dtype": "auto"},
        tokenizer_kwargs={}
    )
    
    result_dir = "./demo_full_pipeline_results"
    
    print("Configuration:")
    print(f"  Models: {model_names}")
    print(f"  Sparse configs: ['dense', 'magic_pig']")
    print(f"  Benchmarks: {[bc.benchmark_name for bc in benchmark_configs]}")
    print(f"  Subsets: {[subset for bc in benchmark_configs for subset in bc.subsets]}")
    print(f"  Result directory: {result_dir}")
    print(f"  Optimization samples: 2 (quick demo)")
    print(f"  Requests per trial: 1 (ultra-fast)")
    print()
    
    try:
        # Run the full pipeline
        print("🚀 Starting full optimization pipeline...")
        
        run_optimized_benchmarks(
            model_names=model_names,
            sparse_attention_configs=[
                ("dense", None),  # Dense attention (no optimization needed)
                ("magic_pig", None)  # Magic Pig with optimization
            ],
            benchmark_configs=benchmark_configs,
            adapter_config=adapter_config,
            result_dir=result_dir,
            gpu_ids=[0],
            max_concurrent_runs=1,
            enable_optimization=True,
            optimization_samples=2,  # Very small for demo
            request_kwargs={"max_requests": 1, "max_context_length": 256}  # Ultra-fast
        )
        
        print(f"\n✅ Full pipeline completed successfully!")
        print(f"   Results saved to: {result_dir}")
        
        # Check for optimization summary
        summary_file = Path(result_dir) / "optimization_summary.txt"
        if summary_file.exists():
            print(f"   Optimization summary: {summary_file}")
        
        # Check for cache files
        cache_dir = Path(result_dir) / "hyperparameter_cache"
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json"))
            print(f"   Cache files: {len(cache_files)} optimization results cached")
        
    except Exception as e:
        print(f"\n❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def demo_usage_examples():
    """Demo 5: Show usage examples for the new system."""
    print("\n" + "=" * 80)
    print("DEMO 5: USAGE EXAMPLES")
    print("=" * 80)
    
    print("\n--- Example 1: Simple Single Config ---")
    print("""
# Before (Manual):
optimizer = create_optimizer_for_config(
    MagicPigConfig, "magic_pig",
    overrides={
        "lsh_l": tune.choice([4, 6, 8, 10, 12]),
        "lsh_k": tune.choice([2, 4, 6, 8]),
        # ... manual specification of every parameter
    }
)

# After (Declarative):
optimizer = create_optimizer_for_config(
    MagicPigConfig, "magic_pig"
    # Automatically uses MagicPigConfig.get_default_search_space()
)
""")
    
    print("\n--- Example 2: Composite Config ---")
    print("""
# Before (Manual):
composite_optimizer = create_composite_optimizer(
    [SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig], 
    "magic_pig",
    overrides={
        "sinkmasker_sink_size": tune.choice([16, 32, 64, 96, 128]),
        "localmasker_window_size": tune.choice([16, 32, 64, 96, 128]),
        "magicpig_lsh_l": tune.choice([4, 6, 8, 10, 12]),
        # ... manual specification for every parameter of every masker
    }
)

# After (Declarative):
composite_optimizer = auto_create_composite_optimizer(
    [SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig], 
    "magic_pig"
    # Automatically combines all three default search spaces
)
""")
    
    print("\n--- Example 3: Auto-Registration ---")
    print("""
# Register any config class with minimal setup:
auto_register_config(MagicPigConfig)  # Uses default name
auto_register_config(MagicPigConfig, "magic_pig")  # Custom name

# Now "magic_pig" is available in the optimizer registry
optimizer = get_sparse_optimizer("magic_pig")
""")
    
    print("\n--- Example 4: Full Pipeline ---")
    print("""
# Complete optimization + benchmarking pipeline:
run_optimized_benchmarks(
    model_names=["meta-llama/Llama-3.1-8B-Instruct"],
    sparse_attention_configs=[("magic_pig", None)],  # None = optimize
    benchmark_configs=[BenchmarkConfig("loogle", ["shortdep_qa"])],
    adapter_config=AdapterConfig(),
    result_dir="./results",
    enable_optimization=True,
    optimization_samples=20
)
""")


def main():
    """Run all demos."""
    print("🚀 DECLARATIVE SEARCH SPACE SYSTEM - FULL PIPELINE DEMO")
    print("This demo shows the complete system from search space discovery")
    print("to full optimization pipeline execution.")
    
    setup_logging()
    
    try:
        # Run all demos
        demo_search_space_discovery()
        demo_composite_optimization()
        demo_auto_registration()
        demo_usage_examples()
        
        # Ask user if they want to run the full pipeline
        print("\n" + "=" * 80)
        print("FULL PIPELINE DEMO")
        print("=" * 80)
        print("The next demo will run the complete optimization pipeline.")
        print("This includes:")
        print("  - Hyperparameter optimization with Ray Tune")
        print("  - Benchmark execution with optimized configs")
        print("  - Result caching and analysis")
        print()
        
        response = input("Do you want to run the full pipeline demo? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            demo_full_optimization_pipeline()
        else:
            print("Skipping full pipeline demo.")
        
        print("\n" + "=" * 80)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Benefits of the New System:")
        print("✅ No manual search space specification required")
        print("✅ Automatic discovery of default search spaces")
        print("✅ Automatic composition of composite configs")
        print("✅ Declarative and extensible")
        print("✅ Backward compatible with manual overrides")
        print("✅ Full pipeline integration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()