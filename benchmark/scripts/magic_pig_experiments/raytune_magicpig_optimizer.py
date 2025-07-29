#!/usr/bin/env python3
"""
Ray Tune Hyperparameter Optimization for MagicPig Sparse Attention

This script uses Ray Tune to find optimal MagicPig configurations that minimize
attention error while maintaining low sparsity/density. Extends the modular
AttentionOptimizerBase class for easy integration with other attention mechanisms.

Usage:
    python raytune_magicpig_optimizer.py
"""

import os
import sys
import time
import gc
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings

import torch
import numpy as np

# Try to import required libraries
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search import ConcurrencyLimiter
    import pynvml
    import colorama
    from colorama import Fore, Style
    import psutil
except ImportError as e:
    print(f"Error: Required library missing. Please install: {e.name}")
    print("Run: pip install ray[tune] hyperopt pynvml colorama psutil")
    sys.exit(1)

# Suppress Ray warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set project root and add to Python path
project_root = Path(__file__).resolve().parents[3]  # Fixed: was 2, should be 3 to get to repo root
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import SinkMaskerConfig, LocalMaskerConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import MagicPigConfig
from benchmark import Loogle
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

# Import the base optimizer
from raytune_attention_optimizer import AttentionOptimizerBase


class MagicPigOptimizer(AttentionOptimizerBase):
    """Modular optimizer for MagicPig configurations using Ray Tune"""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 benchmark_tasks: List[str] = None,
                 max_requests: int = 2,
                 max_context_length: int = 16000,
                 device: Optional[torch.device] = None,
                 results_base_dir: Optional[str] = None):
        
        super().__init__(
            model_name=model_name,
            benchmark_class=Loogle,
            benchmark_tasks=benchmark_tasks,
            max_requests=max_requests,
            max_context_length=max_context_length,
            device=device,
            attention_type="magicpig",
            results_base_dir=results_base_dir
        )

    def create_attention_config(self, params: Dict[str, Any]) -> Tuple[str, ResearchAttentionConfig]:
        """Create MagicPig configuration from hyperparameters"""
        # Extract parameters
        lsh_l = int(params['l'])
        lsh_k = int(params['k'])
        packing = params['packing']
        center = params['center']
        sink_size = int(params['sink_size'])
        window_size = int(params['window_size'])
        
        # Create configuration name based on actual parameter values
        # Sort parameters for consistent naming
        param_parts = []
        for key in sorted(params.keys()):
            value = params[key]
            # Shorten common parameter names for cleaner config names
            key_short = {
                'sink_size': 'sink',
                'window_size': 'win', 
                'packing': 'pack',
                'center': 'ctr',
                'l': 'L',
                'k': 'K'
            }.get(key, key)
            
            # Format boolean values
            if isinstance(value, bool):
                value = 'T' if value else 'F'
            
            param_parts.append(f"{key_short}{value}")
        
        # Create unique config name from all parameters
        config_name = "_".join(param_parts)
        
        # Create sparse attention configuration
        config = ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=sink_size),
            LocalMaskerConfig(window_size=window_size),
            MagicPigConfig(lsh_l=lsh_l, lsh_k=lsh_k, packing=packing, center=center)
        ])
        
        return config_name, config

    def get_search_space(self) -> Dict[str, Any]:
        """Define the hyperparameter search space for MagicPig"""
        return {
            # MagicPig specific parameters
            'l': tune.choice([16, 32, 64, 96, 128]),           # LSH hash functions
            'k': tune.choice([4, 8, 16, 32]),                   # LSH hash bits
            'packing': tune.choice(['int64']),       # Packing method
            'center': tune.choice([True]),               # Centering
            
            # Fixed masker parameters
            'sink_size': tune.choice([64, 128, 256]),           # Sink tokens
            'window_size': tune.choice([64, 128, 256, 512]),    # Local window
        }

    def get_hyperopt_search_space(self) -> Dict[str, Any]:
        """Define search space for HyperOpt (more efficient for continuous spaces)"""
        try:
            from hyperopt import hp
            return {
                'l': hp.choice('l', [16, 32, 64, 96, 128]),
                'k': hp.choice('k', [4, 8, 16, 32]),
                'packing': hp.choice('packing', ['int64']),
                'center': hp.choice('center', [True]),
                'sink_size': hp.choice('sink_size', [64, 128, 256]),
                'window_size': hp.choice('window_size', [64, 128, 256, 512]),
            }
        except ImportError:
            return None


def main():
    """Main execution function"""
    
    # Initialize colorama for colored output
    colorama.init(autoreset=True)
    
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    BENCHMARK_TASKS = ["shortdep_qa"]
    MAX_REQUESTS = 2  # Increased for better evaluation
    MAX_CONTEXT_LENGTH = 16000
    
    # Multi-GPU optimization settings
    NUM_SAMPLES = 50           # Number of configurations to try
    MAX_CONCURRENT = 8         # Use all 8 GPUs
    CPU_PER_TRIAL = 4.0        # More CPU cores per trial
    GPU_PER_TRIAL = 1.0        # 1 full GPU per trial
    results_base_dir = "./multigpu_magicpig_results_constrained"  # Custom results directory
    
    print(f"{Style.BRIGHT}{Fore.BLUE}🚀 Multi-GPU MagicPig Ray Tune Optimization")
    print(f"{'='*60}")
    print(f"{Fore.CYAN}Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Tasks: {BENCHMARK_TASKS}")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Max Concurrent: {MAX_CONCURRENT}")
    print(f"  GPU per Trial: {GPU_PER_TRIAL}")
    print(f"  CPU per Trial: {CPU_PER_TRIAL}")
    print()
    
    try:
        # Create optimizer
        optimizer = MagicPigOptimizer(
            model_name=MODEL_NAME,
            benchmark_tasks=BENCHMARK_TASKS,
            max_requests=MAX_REQUESTS,
            max_context_length=MAX_CONTEXT_LENGTH,
            results_base_dir=results_base_dir
        )
        
        print(f"{Fore.YELLOW}🔍 Starting multi-GPU optimization...")
        print(f"{Fore.YELLOW}This will use {MAX_CONCURRENT} GPUs in parallel")
        
        # Run optimization with multi-GPU settings
        results = optimizer.run_optimization(
            num_samples=NUM_SAMPLES,
            max_concurrent=MAX_CONCURRENT,
            cpu_per_trial=CPU_PER_TRIAL,
            gpu_per_trial=GPU_PER_TRIAL,
            use_asha=True,
            metric="combined_score",
            mode="min"
        )
        
        # Analyze results
        print(f"\n{Fore.GREEN}📊 Analyzing results...")
        best_result, top_configs = optimizer.analyze_results(results, top_k=10)
        
        print(f"\n{Style.BRIGHT}{Fore.GREEN}🏆 Multi-GPU Optimization completed successfully!")
        print(f"{Fore.CYAN}Best configuration found:")
        print(f"  Combined Score: {best_result.metrics['combined_score']:.4f}")
        print(f"  Attention Error: {best_result.metrics['attention_error']:.4f}")
        print(f"  Density: {best_result.metrics['density']:.4f}")
        print(f"  GPU Runtime: {best_result.metrics['gpu_runtime_s']:.2f}s")
        print(f"\n{Fore.GREEN}📁 Results saved to: {results_base_dir}/raytune_detailed_results/")
        print("Use the best configuration in your sparse attention experiments.")
        
    except Exception as e:
        print(f"\n{Style.BRIGHT}{Fore.RED}❌ Optimization failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Clean up Ray
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
