#!/usr/bin/env python3
"""
Multi-GPU Ray Tune optimization example for MagicPig

This script demonstrates how to run Ray Tune optimization across multiple GPUs
for faster hyperparameter search.
"""

import sys
import os
from pathlib import Path

# Set project root and add to Python path
project_root = Path(__file__).resolve().parents[3]
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Add the magic pig experiments directory to the path for local imports
magic_pig_dir = Path(__file__).parent
sys.path.insert(0, str(magic_pig_dir))

try:
    from raytune_magicpig_optimizer import MagicPigOptimizer
    from colorama import Fore, Style
    import colorama
    try:
        import torch
    except ImportError:
        torch = None
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install ray[tune] hyperopt pynvml colorama psutil")
    sys.exit(1)


def run_multi_gpu_optimization():
    """Run optimization across multiple GPUs"""
    
    colorama.init(autoreset=True)
    
    print(f"{Style.BRIGHT}{Fore.CYAN}🚀 Multi-GPU MagicPig Optimization")
    print("=" * 60)
    
    # Check available GPUs
    if torch is None or not torch.cuda.is_available():
        print(f"{Fore.RED}❌ CUDA/PyTorch not available. This script requires GPUs.")
        print(f"{Fore.YELLOW}⚠️  Defaulting to single GPU configuration...")
        max_concurrent = 1
        gpu_per_trial = 1.0
        num_gpus = 1
    else:
        num_gpus = torch.cuda.device_count()
        print(f"{Fore.GREEN}✓ Detected {num_gpus} GPUs")
    
    if num_gpus < 2:
        print(f"{Fore.YELLOW}⚠️  Only {num_gpus} GPU detected. This demo is designed for multiple GPUs.")
        print(f"{Fore.YELLOW}   Continuing with single GPU configuration...")
        max_concurrent = 1
        gpu_per_trial = 1.0
    else:
        # Configure for multi-GPU usage
        max_concurrent = min(num_gpus, 8)  # Use up to 8 GPUs or available GPUs
        gpu_per_trial = 1.0  # 1 GPU per trial
        print(f"{Fore.CYAN}📊 Configuration:")
        print(f"   - Max concurrent trials: {max_concurrent}")
        print(f"   - GPU per trial: {gpu_per_trial}")
        print("   - CPU per trial: 4.0")
    
    try:
        # Create optimizer
        optimizer = MagicPigOptimizer(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            benchmark_tasks=["shortdep_qa"],
            max_requests=2,  # Small for demo
            results_base_dir="./multigpu_raytune_asha"  # Custom results directory
        )
        
        print(f"\n{Fore.YELLOW}🔍 Starting multi-GPU optimization...")
        print(f"{Fore.YELLOW}This will run {max_concurrent} trials in parallel")
        
        # Run optimization with multi-GPU configuration
        results = optimizer.run_optimization(
            num_samples=20,  # More samples to benefit from parallelization
            max_concurrent=max_concurrent,
            cpu_per_trial=4.0,  # More CPUs per trial for better performance
            gpu_per_trial=gpu_per_trial,
            use_asha=True,  # Enable ASHA scheduler for early stopping
            metric="combined_score",
            mode="min"
        )
        
        # Analyze results
        print(f"\n{Fore.GREEN}📊 Analyzing results...")
        best_result, top_configs = optimizer.analyze_results(results, top_k=5)
        
        print(f"\n{Style.BRIGHT}{Fore.GREEN}🏆 Multi-GPU Optimization Completed!")
        print(f"{Fore.CYAN}Best configuration found:")
        print(f"  - Combined Score: {best_result.metrics['combined_score']:.4f}")
        print(f"  - Attention Error: {best_result.metrics['attention_error']:.4f}")
        print(f"  - Density: {best_result.metrics['density']:.4f}")
        print(f"  - GPU Runtime: {best_result.metrics['gpu_runtime_s']:.2f}s")
        
        print(f"\n{Fore.MAGENTA}💡 Performance Tips for Multi-GPU:")
        print("   1. Increase num_samples (50-200) to fully utilize parallel GPUs")
        print("   2. Use ASHA scheduler for early stopping of poor trials")
        print("   3. Monitor GPU utilization: nvidia-smi")
        print("   4. Adjust max_concurrent based on available GPU memory")
        
        return results
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_production_multi_gpu():
    """Example production configuration for multi-GPU optimization"""
    
    print(f"\n{Style.BRIGHT}{Fore.MAGENTA}🔧 Production Multi-GPU Configuration Example")
    print("=" * 60)
    
    production_config = {
        "num_samples": 100,
        "max_concurrent": 8,
        "cpu_per_trial": 8.0,
        "gpu_per_trial": 1.0,
        "use_asha": True
    }
    
    print(f"{Fore.CYAN}Recommended production settings:")
    for key, value in production_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n{Fore.YELLOW}To run production optimization, modify the config above and run:")
    print(f"{Fore.WHITE}optimizer.run_optimization(**production_config)")


if __name__ == "__main__":
    print(f"{Style.BRIGHT}{Fore.BLUE}Multi-GPU Ray Tune Optimization Demo")
    print(f"{Fore.BLUE}{'='*60}")
    
    # Run the multi-GPU optimization demo
    results = run_multi_gpu_optimization()
    
    if results:
        # Show production configuration example
        run_production_multi_gpu()
        
        print(f"\n{Style.BRIGHT}{Fore.GREEN}✅ Demo completed successfully!")
        print(f"{Fore.CYAN}Check the results in: ./custom_results/raytune_detailed_results/")
    else:
        print(f"\n{Fore.RED}❌ Demo failed. Check the error messages above.")
