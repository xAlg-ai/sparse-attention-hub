#!/usr/bin/env python3
"""Simple benchmark-integrated optimizer with minimal code."""

import sys
import os
import logging
from typing import Dict, Any

# Add paths for imports
sys.path.append('/scratch/krishna/inference/longcontext/sparse-attention-hub/benchmark/optimizer')
sys.path.append('/scratch/krishna/inference/longcontext/sparse-attention-hub')

try:
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    print("Ray not available - using mock for testing")
    class MockTune:
        @staticmethod
        def choice(options): return options[0]
        @staticmethod 
        def run(*args, **kwargs): return type('MockAnalysis', (), {'get_best_trial': lambda *a: type('MockTrial', (), {'config': {}})()})()
    tune = MockTune()
    RAY_AVAILABLE = False

from generic_config_optimizer import create_composite_optimizer

# Import masker configs
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import MagicPigConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import LocalMaskerConfig, SinkMaskerConfig


class SimpleBenchmarkOptimizer:
    """Minimal wrapper to integrate optimizer with benchmarks."""
    
    def __init__(self, masker_configs, config_name, benchmark_runner=None):
        """Initialize with masker configs and optional benchmark runner.
        
        Args:
            masker_configs: List of masker config classes
            config_name: Name for the configuration
            benchmark_runner: Function that takes (config, task_name) and returns score
        """
        self.optimizer = create_composite_optimizer(masker_configs, config_name)
        self.benchmark_runner = benchmark_runner or self._mock_benchmark
        self.logger = logging.getLogger(__name__)
    
    def _mock_benchmark(self, config, task_name):
        """Mock benchmark that returns random score - replace with real benchmark."""
        import random
        score = random.random()
        self.logger.info(f"Mock benchmark: {config} -> {score:.3f}")
        return score
    
    def optimize_for_task(self, task_name: str, num_samples: int = 10):
        """Run optimization for a task with minimal setup."""
        print(f"🚀 Optimizing {self.optimizer.config_type_name} for {task_name} ({num_samples} samples)")
        
        if not RAY_AVAILABLE:
            print("⚠️  Ray not available - using mock optimization")
            search_space = self.optimizer.create_search_space(task_name)
            # Use first choice for each parameter
            best_params = {}
            for param, space in search_space.items():
                if hasattr(space, '__iter__') and not isinstance(space, str):
                    best_params[param] = list(space)[0] if space else None
                else:
                    best_params[param] = space
            return self.optimizer.create_config_from_params(best_params)
        
        # Store references for the objective function
        optimizer_ref = self.optimizer
        benchmark_runner_ref = self.benchmark_runner
        
        # Simple objective function that avoids serialization issues
        def objective(config):
            attention_config = optimizer_ref.create_config_from_params(config)
            score = benchmark_runner_ref(attention_config, task_name)
            return {"score": score}
        
        # Initialize Ray without distributed workers to avoid import issues
        import ray
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
        
        analysis = tune.run(
            objective,
            config=self.optimizer.create_search_space(task_name),
            num_samples=num_samples,
            resources_per_trial={"cpu": 1, "gpu": 0.25},
            name=f"optimize_{self.optimizer.config_type_name}_{task_name}",
            storage_path=os.path.abspath("./ray_results"),
            verbose=1
        )
        
        best_trial = analysis.get_best_trial("score", "max", "last")
        best_config = self.optimizer.create_config_from_params(best_trial.config)
        
        print(f"✅ Best config found: {best_config}")
        return best_config


def create_optimizer(masker_types="magic_pig_local_sink", benchmark_runner=None):
    """Factory function to create optimizers with minimal setup.
    
    Args:
        masker_types: Preset combination or list of masker classes
        benchmark_runner: Optional benchmark function
    """
    # Preset combinations for easy use
    presets = {
        "local_sink": [SinkMaskerConfig, LocalMaskerConfig], 
        "sink_local_magic_pig": [SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig], 
    }
    
    if isinstance(masker_types, str):
        if masker_types not in presets:
            raise ValueError(f"Unknown preset '{masker_types}'. Available: {list(presets.keys())}")
        masker_configs = presets[masker_types]
        config_name = masker_types
    else:
        masker_configs = masker_types
        config_name = "_".join([c.__name__.lower().replace('config', '') for c in masker_configs])
    
    return SimpleBenchmarkOptimizer(masker_configs, config_name, benchmark_runner)


def main():
    """Demo usage."""
    
    # Method 1: Simple preset
    optimizer = create_optimizer("magic_pig_local_sink")
    best_config = optimizer.optimize_for_task("longbench_qasper", num_samples=5)
    print(f"\n📊 Best config: {best_config}")
    
    # Method 2: Custom maskers
    custom_optimizer = create_optimizer([MagicPigConfig, LocalMaskerConfig])
    custom_config = custom_optimizer.optimize_for_task("passkey_task", num_samples=3)
    print(f"\n📊 Custom config: {custom_config}")
    
    # Method 3: With custom benchmark (when available)
    def my_benchmark(config, task_name):
        # Your benchmark logic here
        return 0.85  # Mock score
    
    bench_optimizer = create_optimizer("magic_pig", my_benchmark)
    bench_config = bench_optimizer.optimize_for_task("my_task", num_samples=2) 
    print(f"\n📊 Benchmark config: {bench_config}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
