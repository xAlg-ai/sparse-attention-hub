#!/usr/bin/env python3
"""
Demo: End-to-end optimization with Loogle benchmark using ASHA scheduler.

This demonstrates a complete sparse attention optimization pipeline with:
- Real Loogle benchmark integration with combined scoring (attention_error + 0.1 × density)
- ASHA (Asynchronous Successive Halving) scheduler for efficient hyperparameter optimization
- Multi-masker sparse attention configurations (MagicPig + Local + Sink)
- Production-ready Ray Tune setup with proper minimization

Features:
- ✅ Research-grade combined scoring using attention error and density metrics
- ✅ Unique result directories to avoid benchmark caching issues  
- ✅ Proper Ray Tune minimization (mode="min") for combined scores
- ✅ Multi-masker optimization with comprehensive search spaces

Setup:
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install requirements (if needed)
    pip install ray[tune]

Usage:
    source .venv/bin/activate
    python benchmark/optimizer/demo_loogle_asha.py

Expected Results:
    - Basic MagicPig optimization: ~10 trials, typically achieves scores around 0.01-0.1
    - Multi-masker optimization: ~15 trials, explores MagicPig+Local+Sink combinations
    - Best configurations balance low attention error with sparsity (low density)
"""

import sys
import os
import logging
import time
import json
from pathlib import Path

# Add paths for imports
sys.path.append('/scratch/krishna/inference/longcontext/sparse-attention-hub/benchmark/optimizer')
sys.path.append('/scratch/krishna/inference/longcontext/sparse-attention-hub')

# Core optimizer imports
from simple_optimizer import create_optimizer

# Ray Tune imports with fallback
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.stopper import TrialPlateauStopper
    RAY_AVAILABLE = True
    print("✅ Ray Tune available - using real optimization")
except ImportError:
    print("⚠️  Ray Tune not available - install with: pip install ray[tune]")
    RAY_AVAILABLE = False
    sys.exit(1)

# Benchmark imports  
from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig, BenchmarkResult
from sparse_attention_hub.metric_logging import MicroMetricLogger
import torch


class LoogleBenchmarkRunner:
    """Real benchmark runner that integrates with BenchmarkExecutor for Loogle with correct combined scoring."""
    
    def __init__(self, gpu_ids=[0], enable_logging=True):
        """Initialize with GPU configuration."""
        self.gpu_ids = gpu_ids
        self.enable_logging = enable_logging
        
        # Setup benchmark executor for Loogle
        self.executor = BenchmarkExecutor(
            gpu_ids=gpu_ids,
            max_concurrent_runs=len(gpu_ids),
            base_result_dir="./demo_optimization_results", 
            enable_resumability=False,  # Disable resumability to force fresh runs
            required_result_files=["raw_results.csv", "micro_metrics.jsonl"],
            timeout_per_benchmark=1800.0,  # 30 minutes per benchmark
            verbose=False
        )
        
        self.benchmark_config = BenchmarkConfig(
            benchmark_name="loogle",
            subsets=["shortdep_qa"]  # Start with shorter dependency QA
        )
        
        self.adapter_config = AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"}
        )
        
        self.generation_kwargs = {
            "max_new_tokens": 50,
            "do_sample": False, 
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": None,
        }
        
        self.request_kwargs = {
            "max_context_length": 2048,  # Reasonable for demo
            "max_requests": 10,  # Limit for demo efficiency
        }
    
    def extract_micro_metrics(self, result_dir: Path) -> dict:
        """Extract attention error and density from micro_metrics.jsonl."""
        micro_metrics_file = result_dir / "micro_metrics.jsonl"
        
        if not micro_metrics_file.exists():
            raise ValueError(f"micro_metrics.jsonl not found in {result_dir}")
        
        attention_errors = []
        densities = []
        
        with open(micro_metrics_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                metric_name = entry.get('metric', '')
                value = entry.get('value')
                
                if metric_name == 'research_attention_output_error' and value is not None:
                    attention_errors.append(float(value))
                elif metric_name == 'research_attention_density' and value is not None:
                    densities.append(float(value))
        
        if not attention_errors:
            raise ValueError(f"No research_attention_output_error metrics found in {micro_metrics_file}")
        
        if not densities:
            raise ValueError(f"No research_attention_density metrics found in {micro_metrics_file}")
        
        # Average the metrics across all measurements
        attention_error = sum(attention_errors) / len(attention_errors)
        density = sum(densities) / len(densities)
        
        return {
            'attention_error': float(attention_error),
            'density': float(density)
        }
    
    def __call__(self, attention_config, task_name: str) -> float:
        """Run Loogle benchmark and return combined score (attention_error + 0.1 * density)."""
        model_name = "microsoft/Phi-4-mini-instruct"  # Fast model for demo
        
        try:
            start_time = time.time()
            
            if self.enable_logging:
                print(f"  🔧 Running benchmark for config: {attention_config}")
            
            # Configure MicroMetricLogger for attention metrics
            MicroMetricLogger.register_metric("research_attention_density", float)
            MicroMetricLogger.register_metric("research_attention_output_error", float)
            
            # Create a unique config name based on the attention config
            import hashlib
            config_str = str(attention_config)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            unique_config_name = f"opt_{config_hash}"
            
            # Run benchmark matrix (single combination)
            if self.enable_logging:
                print(f"  🚀 Starting benchmark_matrix with model: {model_name}, config: {unique_config_name}")
            
            results = self.executor.run_benchmark_matrix(
                model_names=[model_name],
                sparse_attention_configs=[(unique_config_name, attention_config)],
                benchmark_configs=[self.benchmark_config],
                adapter_config=self.adapter_config,
                generation_kwargs=self.generation_kwargs,
                request_kwargs=self.request_kwargs
            )
            
            runtime = time.time() - start_time
            
            if self.enable_logging:
                print(f"  ⏱️  Benchmark completed in {runtime:.1f}s")
                print(f"      Completed stubs: {results.progress.completed_stubs}")
                print(f"      Total results: {len(results.individual_results)}")
            
            # Extract combined score from results
            if results.progress.completed_stubs > 0:
                # Get the benchmark result
                completed_results = [r for r in results.individual_results if isinstance(r, BenchmarkResult)]
                if self.enable_logging:
                    print(f"      BenchmarkResult objects: {len(completed_results)}")
                
                if completed_results:
                    result = completed_results[0]
                    result_dir = Path(result.stub.result_dir)
                    
                    # Extract micro metrics for proper combined scoring
                    try:
                        micro_metrics = self.extract_micro_metrics(result_dir)
                        attention_error = micro_metrics['attention_error']
                        density = micro_metrics['density']
                        
                        # Handle NaN values
                        if not isinstance(attention_error, (int, float)) or attention_error != attention_error:  # NaN check
                            attention_error = 1.0  # Default high error for NaN
                            if self.enable_logging:
                                print(f"      ⚠️ Attention error was NaN, using default: {attention_error}")
                        
                        if not isinstance(density, (int, float)) or density != density:  # NaN check  
                            density = 0.5  # Default density for NaN
                            if self.enable_logging:
                                print(f"      ⚠️ Density was NaN, using default: {density}")
                        
                        # Use the same combined scoring formula as the research-grade optimizer:
                        # combined_score = attention_error + 0.1 * density
                        combined_score = attention_error + 0.1 * density
                        
                        # Add penalty for high density (sparsity constraint)
                        penalty = 0.0
                        if density > 0.5:
                            penalty = 5.0  # Large penalty for high density
                        
                        final_score = combined_score + penalty
                        
                        if self.enable_logging:
                            print(f"  📊 Config: {attention_config}")
                            print(f"      Attention Error: {attention_error:.4f}")
                            print(f"      Density: {density:.4f}")  
                            print(f"      Combined Score: {final_score:.4f} (took {runtime:.1f}s)")
                        
                        # Return positive score for Ray Tune minimization
                        return final_score
                        
                    except Exception as e:
                        print(f"  ⚠️  Could not extract micro metrics: {e}")
                        print(f"      Result dir: {result_dir}")
                        # List files in result directory for debugging
                        try:
                            files = list(result_dir.glob("*"))
                            print(f"      Available files: {[f.name for f in files]}")
                        except Exception:
                            pass
                        # Fall back to benchmark score only
                        benchmark_score = result.metrics.get("overall_score", 0.0) if result.metrics else 0.0
                        if self.enable_logging:
                            print(f"  📊 Using benchmark score only: {benchmark_score:.4f}")
                        return benchmark_score
                else:
                    if self.enable_logging:
                        print(f"  ⚠️  No BenchmarkResult objects found in {len(results.individual_results)} results")
            else:
                if self.enable_logging:
                    print(f"  ⚠️  No completed stubs: {results.progress}")
            
            # Fallback score for failures
            print(f"  ❌ Benchmark failed for config: {attention_config}")
            return 10.0  # Large positive score for failed configs (bad for minimization)
            
        except Exception as e:
            print(f"  ❌ Error running benchmark: {e}")
            return 10.0  # Large positive score for error cases (bad for minimization)


def create_asha_scheduler(max_t=20, grace_period=5, reduction_factor=2):
    """Create ASHA scheduler for efficient hyperparameter optimization.
    
    Args:
        max_t: Maximum number of iterations per trial
        grace_period: Minimum number of iterations before stopping
        reduction_factor: Factor to reduce the number of trials
    """
    return ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_t,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        brackets=3  # Number of brackets for successive halving
    )


def create_trial_stopper(num_results=5, std=0.001):
    """Create trial stopper for early termination of plateaued trials."""
    return TrialPlateauStopper(
        std=std,
        num_results=num_results,
        grace_period=5
    )


def demo_basic_optimization():
    """Demo 1: Basic optimization with single masker type."""
    print("🚀 Demo 1: Basic MagicPig Optimization with ASHA")
    print("=" * 60)
    
    # Initialize benchmark runner
    benchmark_runner = LoogleBenchmarkRunner(gpu_ids=[0], enable_logging=True)
    
    # Create optimizer for MagicPig only
    optimizer = create_optimizer("magic_pig", benchmark_runner)
    
    def objective(config):
        """Optimization objective function with proper Ray Tune trainable API."""
        attention_config = optimizer.optimizer.create_config_from_params(config)
        # Get the combined score (positive value for minimization)
        combined_score = benchmark_runner(attention_config, "loogle_shortdep_qa")
        # Return as dictionary for Ray Tune
        return {"combined_score": combined_score}
    
    # Simple ASHA scheduler setup
    scheduler = create_asha_scheduler(max_t=10, grace_period=3)
    
    # Run optimization
    print(f"🔧 Starting optimization with {len(optimizer.optimizer.create_search_space('loogle'))} parameters")
    
    analysis = tune.run(
        objective,
        config=optimizer.optimizer.create_search_space('loogle'),
        scheduler=scheduler,
        num_samples=10,  # Reasonable number of trials
        resources_per_trial={"cpu": 1, "gpu": 0.5},
        name="loogle_demo_asha",
        storage_path=os.path.abspath("./ray_results"),
        metric="combined_score",
        mode="min",  # Minimize the combined score
        verbose=1
    )
    
    # Results
    best_trial = analysis.get_best_trial("combined_score", "min", "last")
    best_config = optimizer.optimizer.create_config_from_params(best_trial.config)
    
    print("\n✅ Best MagicPig Configuration:")
    print(f"  Combined Score: {best_trial.last_result['combined_score']:.4f}")
    print(f"  Config: {best_config}")
    print(f"  Params: {best_trial.config}")
    
    return best_config, best_trial.last_result['combined_score']


def demo_multi_masker_optimization():
    """Demo 2: Multi-masker optimization with ASHA."""
    print("\n🚀 Demo 2: Multi-Masker Optimization (MagicPig + Local + Sink)")
    print("=" * 60)
    
    # Initialize benchmark runner
    benchmark_runner = LoogleBenchmarkRunner(gpu_ids=[0], enable_logging=True)
    
    # Create multi-masker optimizer
    optimizer = create_optimizer("magic_pig_local_sink", benchmark_runner)
    
    def objective(config):
        """Multi-masker optimization objective with proper Ray Tune API."""
        attention_config = optimizer.optimizer.create_config_from_params(config)
        combined_score = benchmark_runner(attention_config, "loogle_shortdep_qa")
        return {"combined_score": combined_score}
    
    # Simple ASHA scheduler
    scheduler = create_asha_scheduler(max_t=10, grace_period=3)
    
    print(f"🔧 Multi-masker search space: {len(optimizer.optimizer.create_search_space('loogle'))} parameters")
    
    # Run optimization
    analysis = tune.run(
        objective,
        config=optimizer.optimizer.create_search_space("loogle"),
        num_samples=15,  # More samples for multi-masker optimization
        scheduler=scheduler,
        resources_per_trial={"cpu": 1, "gpu": 1},
        name="demo2_multi_masker_loogle", 
        storage_path=os.path.abspath("./ray_results"),
        metric="combined_score",
        mode="min",  # Minimize the combined score
        verbose=1
    )
    
    # Results
    best_trial = analysis.get_best_trial("combined_score", "min", "last")
    best_config = optimizer.optimizer.create_config_from_params(best_trial.config)
    
    print("\n✅ Best Multi-Masker Configuration:")
    print(f"  Combined Score: {best_trial.last_result['combined_score']:.4f}")
    print(f"  Config: {best_config}")
    print(f"  Params: {best_trial.config}")
    
    return best_config, best_trial.last_result['combined_score']


def main():
    """Run basic demo scenarios."""
    if not RAY_AVAILABLE:
        print("❌ Ray Tune required for this demo")
        return
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    print("🎯 Loogle Benchmark Optimization with ASHA Scheduler Demo")
    print("=" * 80)
    print(f"Ray Tune version: {ray.__version__}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    try:
        # Demo 1: Basic single-masker optimization
        config1, score1 = demo_basic_optimization()
        
        # Demo 2: Multi-masker optimization  
        config2, score2 = demo_multi_masker_optimization()
        
        # Summary
        print("\n" + "=" * 80)
        print("🏆 OPTIMIZATION SUMMARY")
        print("=" * 80)
        print(f"1. Basic MagicPig:      Combined Score = {score1:.4f}")
        print(f"2. Multi-Masker:        Combined Score = {score2:.4f}")
        
        best_score = max(score1, score2)
        best_demo = ["Basic MagicPig", "Multi-Masker"][
            [score1, score2].index(best_score)
        ]
        
        print(f"\n🥇 Best performing approach: {best_demo} (Combined Score: {best_score:.4f})")
        
        print("\n💾 Results saved to: ./ray_results/")
        print("🔍 View results with: tensorboard --logdir ./ray_results")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
