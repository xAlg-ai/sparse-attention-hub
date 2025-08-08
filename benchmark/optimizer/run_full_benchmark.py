#!/usr/bin/env python3
"""
Full Benchmark Runner: Comprehensive optimization across multiple benchmarks and models.

This script runs the complete optimization pipeline across:
- Multiple models (Phi-4, Llama-3.1, etc.)  
- Multiple benchmarks (Loogle, InfiniteBench, Ruler)
- Multiple masker configurations (sink_local_magic_pig, local_sink)
- Production Ray Tune setup with ASHA scheduling

Setup:
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    pip install ray[tune] hyperopt

Usage:
    source venv/bin/activate
    python benchmark/optimizer/run_full_benchmark.py [--debug]

Arguments:
    --debug: Run a smaller test with limited configurations

Requirements:
    pip install ray[tune] hyperopt
"""

import sys
import os
import argparse
import logging
import time
import json
from datetime import datetime
from pathlib import Path

# Add paths for imports and set PYTHONPATH for Ray workers
optimizer_path = '/scratch/krishna/inference/longcontext/sparse-attention-hub/benchmark/optimizer'
root_path = '/scratch/krishna/inference/longcontext/sparse-attention-hub'

sys.path.append(optimizer_path)
sys.path.append(root_path)

# Set PYTHONPATH environment variable for Ray workers
current_pythonpath = os.environ.get('PYTHONPATH', '')
paths_to_add = [optimizer_path, root_path]
new_paths = [p for p in paths_to_add if p not in current_pythonpath]
if new_paths:
    if current_pythonpath:
        os.environ['PYTHONPATH'] = current_pythonpath + ':' + ':'.join(new_paths)
    else:
        os.environ['PYTHONPATH'] = ':'.join(new_paths)

# Core imports
from simple_optimizer import create_optimizer

# Ray Tune imports 
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.stopper import TrialPlateauStopper
    from ray.tune.search.hyperopt import HyperOptSearch
    RAY_AVAILABLE = True
except ImportError:
    print("❌ Ray Tune required - install with: pip install ray[tune] hyperopt")
    sys.exit(1)

# Benchmark execution imports
from benchmark.executor import BenchmarkExecutor 
from benchmark.executor_config import BenchmarkConfig, AdapterConfig, BenchmarkResult
import torch


class ComprehensiveBenchmarkRunner:
    """Comprehensive benchmark runner for multiple models and benchmarks."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        
        # Setup benchmark executor
        self.executor = BenchmarkExecutor(
            gpu_ids=config['gpu_ids'],
            max_concurrent_runs=config['max_concurrent_runs'],
            base_result_dir=config['result_dir'],
            enable_resumability=True,
            required_result_files=["raw_results.csv"],
            timeout_per_benchmark=config['timeout_per_benchmark'],
            verbose=False
        )
        
        # Adapter configuration
        self.adapter_config = AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"}
        )
        
        # Generation parameters
        self.generation_kwargs = {
            "max_new_tokens": config['max_new_tokens'],
            "do_sample": False,
            "temperature": 1.0, 
            "top_p": 1.0,
            "pad_token_id": None,
        }
        
        # Request parameters
        self.request_kwargs = {
            "max_context_length": config['max_context_length'],
            "max_requests": config['max_requests'],
        }
        
        # Results cache for efficiency
        self.results_cache = {}
    
    def extract_micro_metrics(self, result_dir: Path) -> dict:
        """Extract attention error and density from micro_metrics.jsonl."""
        import math
        
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
                    try:
                        val = float(value)
                        if not math.isnan(val) and math.isfinite(val):
                            attention_errors.append(val)
                    except (ValueError, TypeError):
                        continue
                elif metric_name == 'research_attention_density' and value is not None:
                    try:
                        val = float(value)
                        if not math.isnan(val) and math.isfinite(val):
                            densities.append(val)
                    except (ValueError, TypeError):
                        continue
        
        if not attention_errors:
            print("⚠️ No valid attention error metrics found, using default penalty")
            attention_error = 1.0  # High error as penalty
        else:
            attention_error = sum(attention_errors) / len(attention_errors)
        
        if not densities:
            print("⚠️ No valid density metrics found, using default penalty")
            density = 1.0  # High density as penalty
        else:
            density = sum(densities) / len(densities)
        
        return {
            'attention_error': attention_error,
            'density': density
        }
    
    def __call__(self, attention_config, task_name: str, model_name: str) -> float:
        """Run benchmark and return combined score (lower is better)."""
        
        # Create cache key
        config_key = f"{model_name}_{task_name}_{hash(str(attention_config))}"
        if config_key in self.results_cache:
            return self.results_cache[config_key]
        
        try:
            # Try to find existing result directory first (for resumed/existing results)
            result_base_dir = Path(self.config['result_dir'])
            model_path_name = model_name.replace('/', '_').replace('-', '-')
            result_dir = result_base_dir / model_path_name / "optimized" / task_name
            
            if result_dir.exists() and (result_dir / "micro_metrics.jsonl").exists():
                print(f"    📂 Found existing results for {model_name} on {task_name}, extracting metrics...")
                try:
                    # Extract micro metrics from the result directory
                    micro_metrics = self.extract_micro_metrics(result_dir)
                    
                    # Get attention error (higher = worse)
                    attention_error = micro_metrics['attention_error']
                    
                    # Get density (higher = less sparse)  
                    density = micro_metrics['density']
                    
                    # Handle NaN attention error - convert to penalty
                    import math
                    if math.isnan(attention_error):
                        print("    ⚠️ NaN attention error detected, using penalty value")
                        attention_error = 1.0  # High penalty for NaN
                    
                    # Calculate combined score (research-grade formula)
                    combined_score = attention_error + 0.1 * density
                    
                    # Add penalty for high density (sparsity constraint)
                    penalty = 0.0
                    if density > 0.5:
                        penalty = 5.0  # Large penalty for high density
                    
                    final_score = combined_score + penalty
                    
                    # Cache and return (lower score is better)
                    self.results_cache[config_key] = final_score
                    print(f"    ✓ {model_name} on {task_name}: {final_score:.4f} (error={attention_error:.4f}, density={density:.4f}) [from existing]")
                    return final_score
                    
                except Exception as e:
                    print(f"    ⚠️ Failed to extract micro metrics from existing results: {e}")
            
            # Parse task_name to get benchmark and subset
            if "_" in task_name:
                benchmark_name, subset_name = task_name.split("_", 1)
            else:
                benchmark_name = task_name
                subset_name = "default"
            
            benchmark_config = BenchmarkConfig(
                benchmark_name=benchmark_name,
                subsets=[subset_name] if subset_name != "default" else None
            )
            
            start_time = time.time()
            
            # Run benchmark
            results = self.executor.run_benchmark_matrix(
                model_names=[model_name],
                sparse_attention_configs=[("optimized", attention_config)],
                benchmark_configs=[benchmark_config],
                adapter_config=self.adapter_config,
                generation_kwargs=self.generation_kwargs,
                request_kwargs=self.request_kwargs
            )
            
            runtime = time.time() - start_time
            
            # Extract and calculate combined score from new results
            if results.progress.completed_stubs > 0:
                # Use individual_results instead of results
                if hasattr(results, 'individual_results'):
                    all_results = results.individual_results
                else:
                    all_results = []
                
                completed_results = [r for r in all_results if isinstance(r, BenchmarkResult)]
                if completed_results:
                    result = completed_results[0]
                    
                    # Get the result directory from the benchmark stub
                    result_dir = Path(result.stub.result_dir)
                    
                    try:
                        # Extract micro metrics from the result directory
                        micro_metrics = self.extract_micro_metrics(result_dir)
                        
                        # Get attention error (higher = worse)
                        attention_error = micro_metrics['attention_error']
                        
                        # Get density (higher = less sparse)  
                        density = micro_metrics['density']
                        
                        # Handle NaN attention error - convert to penalty
                        import math
                        if math.isnan(attention_error):
                            print("    ⚠️ NaN attention error detected, using penalty value")
                            attention_error = 1.0  # High penalty for NaN
                        
                        # Calculate combined score (research-grade formula)
                        combined_score = attention_error + 0.1 * density
                        
                        # Add penalty for high density (sparsity constraint)
                        penalty = 0.0
                        if density > 0.5:
                            penalty = 5.0  # Large penalty for high density
                        
                        final_score = combined_score + penalty
                        
                        # Cache and return (lower score is better)
                        self.results_cache[config_key] = final_score
                        print(f"    ✓ {model_name} on {task_name}: {final_score:.4f} (error={attention_error:.4f}, density={density:.4f}, {runtime:.1f}s)")
                        return final_score
                        
                    except Exception as e:
                        print(f"    ⚠️ Failed to extract micro metrics from new results: {e}")
            
            # Fallback - try one more time to extract from expected result directory
            if result_dir.exists() and (result_dir / "micro_metrics.jsonl").exists():
                print(f"    📂 Attempting final extraction from result directory...")
                try:
                    micro_metrics = self.extract_micro_metrics(result_dir)
                    attention_error = micro_metrics['attention_error']
                    density = micro_metrics['density']
                    
                    import math
                    if math.isnan(attention_error):
                        attention_error = 1.0
                    
                    combined_score = attention_error + 0.1 * density
                    penalty = 5.0 if density > 0.5 else 0.0
                    final_score = combined_score + penalty
                    
                    self.results_cache[config_key] = final_score
                    print(f"    ✓ {model_name} on {task_name}: {final_score:.4f} (error={attention_error:.4f}, density={density:.4f}) [fallback]")
                    return final_score
                except Exception as e:
                    print(f"    ⚠️ Fallback extraction failed: {e}")
            
            # Final fallback for failed benchmarks - return high penalty score
            print(f"    ❌ {model_name} on {task_name}: Failed to extract any metrics")
            penalty_score = 10.0  # High penalty for failed runs
            self.results_cache[config_key] = penalty_score
            return penalty_score
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            penalty_score = 10.0  # High penalty for errors
            return penalty_score


def get_benchmark_configurations(debug=False):
    """Get benchmark configurations for optimization."""
    
    if debug:
        # Quick test configuration
        return {
            'models': ["meta-llama/Llama-3.1-8B-Instruct"],
            'benchmarks': [
                "loogle_shortdep_qa"
            ],
            'masker_presets': ["sink_local_magic_pig"],  # Use composite masker: magic_pig -> local -> sink
            'gpu_ids': [0],
            'max_concurrent_runs': 1,
            'result_dir': "./quick_optimization_results",
            'timeout_per_benchmark': 600,  # 10 minutes
            'max_new_tokens': 30,
            'max_context_length': 1024,
            'max_requests': 5,
            'num_samples': 10
        }
    
    # Full production configuration
    return {
        'models': [
            "meta-llama/Llama-3.2-8B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct"
        ],
        'benchmarks': [
            # Core benchmarks - focus on loogle
            "loogle_shortdep_qa", 
            "loogle_longdep_qa"
        ],
        'masker_presets': [
            "local_sink",
            "sink_local_magic_pig"  # Use composite masker: sink -> local -> magic_pig 
        ],
        'gpu_ids': [0, 1, 2, 3],  # Use all available GPUs
        'max_concurrent_runs': 4,
        'result_dir': "./full_optimization_results", 
        'timeout_per_benchmark': 1800,  # 30 minutes
        'max_new_tokens': 50,
        'max_context_length': 2048,
        'max_requests': 20,
        'num_samples': 50
    }


def run_detailed_benchmark_execution(attention_config, task_name: str, model_name: str, masker_preset: str):
    """Run detailed benchmark execution with specific configuration to get full performance metrics."""
    
    # Configuration for detailed benchmark run
    config = {
        'gpu_ids': [0],
        'max_concurrent_runs': 1,
        'result_dir': f"./detailed_results_{masker_preset}",
        'timeout_per_benchmark': 1800,  # 30 minutes
        'max_new_tokens': 50,
        'max_context_length': 2048,
        'max_requests': 20,
    }
    
    # Initialize benchmark executor with verbose logging
    executor = BenchmarkExecutor(
        gpu_ids=config['gpu_ids'],
        max_concurrent_runs=config['max_concurrent_runs'],
        base_result_dir=config['result_dir'],
        enable_resumability=True,
        required_result_files=["raw_results.csv"],
        timeout_per_benchmark=config['timeout_per_benchmark'],
        verbose=True  # Enable verbose logging for detailed output
    )
    
    # Setup adapter configuration
    adapter_config = AdapterConfig(
        adapter_name="huggingface",
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        tokenizer_kwargs={"padding_side": "left"}
    )
    
    # Generation parameters
    generation_kwargs = {
        "max_new_tokens": config['max_new_tokens'],
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "pad_token_id": None,
    }
    
    # Request parameters
    request_kwargs = {
        "max_context_length": config['max_context_length'],
        "max_requests": config['max_requests'],
    }
    
    # Parse benchmark task
    benchmark_name, subset_name = task_name.split("_", 1)
    benchmark_config = BenchmarkConfig(
        benchmark_name=benchmark_name,
        subsets=[subset_name]
    )
    
    try:
        # Run the detailed benchmark
        results = executor.run_benchmark_matrix(
            model_names=[model_name],
            sparse_attention_configs=[(f"best_{masker_preset}", attention_config)],
            benchmark_configs=[benchmark_config],
            adapter_config=adapter_config,
            generation_kwargs=generation_kwargs,
            request_kwargs=request_kwargs
        )
        
        # Get result directory
        model_sanitized = model_name.replace('/', '_').replace('-', '-')
        result_dir = Path(config['result_dir']) / model_sanitized / f"best_{masker_preset}" / task_name
        
        return {
            'progress': {
                'total': results.progress.total_stubs,
                'completed': results.progress.completed_stubs,
                'failed': results.progress.failed_stubs,
                'skipped': results.progress.skipped_stubs
            },
            'result_dir': str(result_dir),
            'files_generated': [str(f) for f in result_dir.iterdir()] if result_dir.exists() else []
        }
        
    except Exception as e:
        print(f"      ❌ Detailed benchmark execution failed: {e}")
        return {
            'error': str(e),
            'progress': {'total': 0, 'completed': 0, 'failed': 1, 'skipped': 0}
        }


def create_optimization_objective(benchmark_runner, model_name, task_name, optimizer):
    """Create optimization objective for a specific model-task combination."""
    
    def objective(config):
        """Objective function for Ray Tune."""
        attention_config = optimizer.optimizer.create_config_from_params(config)
        combined_score = benchmark_runner(attention_config, task_name, model_name)
        
        # Return the combined score for Ray Tune (lower is better)
        return {"combined_score": combined_score}
    
    return objective


def run_optimization_matrix(config):
    """Run optimization across the full model×benchmark×masker matrix."""
    
    print("🚀 Starting Full Benchmark Optimization Matrix")
    print("=" * 80)
    print(f"Models: {len(config['models'])}")
    print(f"Benchmarks: {len(config['benchmarks'])}")
    print(f"Masker presets: {len(config['masker_presets'])}")
    print(f"GPU IDs: {config['gpu_ids']}")
    print(f"Samples per combination: {config['num_samples']}")
    
    total_combinations = len(config['models']) * len(config['benchmarks']) * len(config['masker_presets'])
    print(f"Total optimization combinations: {total_combinations}")
    
    # Create unique storage path for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_storage_path = os.path.abspath(f"./ray_results_{timestamp}")
    print(f"Ray Tune results will be saved to: {unique_storage_path}")
    
    # Initialize benchmark runner
    benchmark_runner = ComprehensiveBenchmarkRunner(config)
    
    # Store all results
    optimization_results = {}
    
    # Run optimization for each combination
    for model_name in config['models']:
        model_results = {}
        
        print(f"\n📱 Model: {model_name}")
        print("-" * 60)
        
        for benchmark_task in config['benchmarks']:
            task_results = {}
            
            print(f"\n  📊 Benchmark: {benchmark_task}")
            
            for masker_preset in config['masker_presets']:
                print(f"\n    🔧 Masker: {masker_preset}")
                
                try:
                    # Create optimizer for this masker preset
                    optimizer = create_optimizer(masker_preset, benchmark_runner)
                    
                    # Create objective function
                    objective = create_optimization_objective(
                        benchmark_runner, model_name, benchmark_task, optimizer
                    )
                    
                    # ASHA scheduler configuration
                    scheduler = ASHAScheduler(
                        time_attr="training_iteration",
                        max_t=20,
                        grace_period=5,
                        reduction_factor=2,
                        brackets=3
                    )
                    
                    # Trial stopper for early termination
                    stopper = TrialPlateauStopper(
                        metric="combined_score",
                        std=0.005,
                        num_results=5,
                        grace_period=8,
                        mode="min"  # Minimize combined score
                    )
                    
                    # HyperOpt search algorithm for better exploration
                    search_alg = HyperOptSearch(
                        metric="combined_score",
                        mode="min",  # Minimize combined score
                        n_initial_points=config['num_samples'] // 4
                    )
                    
                    # Run optimization
                    analysis = tune.run(
                        objective,
                        config=optimizer.optimizer.create_search_space(benchmark_task),
                        num_samples=config['num_samples'],
                        scheduler=scheduler,
                        stop=stopper,
                        search_alg=search_alg,
                        metric="combined_score",  # Specify the metric to optimize
                        mode="min",               # Minimize the combined score
                        resources_per_trial={"cpu": 1, "gpu": 0.25},
                        name=f"opt_{model_name.split('/')[-1]}_{benchmark_task}_{masker_preset}",
                        storage_path=unique_storage_path,
                        verbose=1,
                        resume=False  # Don't resume to avoid state file issues
                    )
                    
                    # Extract best result
                    best_trial = analysis.get_best_trial("combined_score", "min", "last")
                    best_config = optimizer.optimizer.create_config_from_params(best_trial.config)
                    best_score = best_trial.last_result['combined_score']
                    
                    print(f"      ✅ Best combined score: {best_score:.4f} (lower is better)")
                    
                    # Run final benchmark with best configuration for validation
                    print(f"      🔬 Running final validation with best config...")
                    final_score = benchmark_runner(best_config, benchmark_task, model_name)
                    print(f"      📊 Final validation score: {final_score:.4f}")
                    
                    # Run detailed benchmark execution with best configuration to get full performance metrics
                    print(f"      🚀 Running detailed benchmark execution with best config...")
                    detailed_results = run_detailed_benchmark_execution(
                        best_config, benchmark_task, model_name, masker_preset
                    )
                    print(f"      📈 Detailed benchmark execution completed")
                    
                    # Store results
                    task_results[masker_preset] = {
                        'best_score': best_score,
                        'final_validation_score': final_score,
                        'best_config': best_config,
                        'best_params': best_trial.config,
                        'num_trials': len(analysis.trials),
                        'detailed_results': detailed_results
                    }
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    task_results[masker_preset] = {
                        'best_score': 0.0,
                        'error': str(e)
                    }
            
            model_results[benchmark_task] = task_results
        
        optimization_results[model_name] = model_results
    
    return optimization_results, unique_storage_path


def extract_benchmark_metrics(result_dir: Path) -> str:
    """Extract key benchmark performance metrics from result directory."""
    import math
    
    if not result_dir.exists():
        return ""
    
    metrics_parts = []
    
    # Check for raw results CSV to get accuracy/performance metrics
    raw_results_file = result_dir / "raw_results.csv"
    if raw_results_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(raw_results_file)
            if 'score' in df.columns:
                avg_score = df['score'].mean()
                metrics_parts.append(f"avg_score={avg_score:.3f}")
            if 'accuracy' in df.columns:
                avg_accuracy = df['accuracy'].mean()
                metrics_parts.append(f"accuracy={avg_accuracy:.3f}")
        except Exception:
            # Fallback: try to read CSV manually
            try:
                with open(raw_results_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has header + data
                        header = lines[0].strip().split(',')
                        if 'score' in header:
                            score_idx = header.index('score')
                            scores = []
                            for line in lines[1:]:
                                try:
                                    score = float(line.split(',')[score_idx])
                                    scores.append(score)
                                except:
                                    continue
                            if scores:
                                avg_score = sum(scores) / len(scores)
                                metrics_parts.append(f"avg_score={avg_score:.3f}")
            except Exception:
                pass
    
    # Check for metrics.json file
    metrics_file = result_dir / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                if isinstance(metrics_data, dict):
                    for key, value in metrics_data.items():
                        if isinstance(value, (int, float)) and key not in ['attention_error', 'density']:
                            metrics_parts.append(f"{key}={value:.3f}")
        except Exception:
            pass
    
    # Check micro_metrics.jsonl for research metrics summary
    micro_metrics_file = result_dir / "micro_metrics.jsonl"
    if micro_metrics_file.exists():
        try:
            attention_errors = []
            densities = []
            with open(micro_metrics_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get('metric') == 'research_attention_output_error':
                        val = entry.get('value')
                        if val is not None and not math.isnan(float(val)):
                            attention_errors.append(float(val))
                    elif entry.get('metric') == 'research_attention_density':
                        val = entry.get('value')
                        if val is not None and not math.isnan(float(val)):
                            densities.append(float(val))
            
            if attention_errors:
                avg_error = sum(attention_errors) / len(attention_errors)
                metrics_parts.append(f"attn_error={avg_error:.4f}")
            if densities:
                avg_density = sum(densities) / len(densities)
                metrics_parts.append(f"density={avg_density:.4f}")
        except Exception:
            pass
    
    return " ".join(metrics_parts) if metrics_parts else "no_metrics"


def print_optimization_summary(results, storage_path="./ray_results"):
    """Print comprehensive optimization summary."""
    
    print("\n" + "=" * 80)
    print("🏆 FULL OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    # Find overall best configuration (lowest combined_score is best)
    best_overall_score = float('inf')
    best_overall_config = None
    
    for model_name, model_results in results.items():
        print(f"\n📱 {model_name}")
        print("-" * 70)
        
        for benchmark_task, task_results in model_results.items():
            print(f"\n  📊 {benchmark_task}")
            
            task_best_score = float('inf')
            task_best_masker = None
            
            for masker_preset, result in task_results.items():
                score = result.get('best_score', 10.0)  # Default high penalty
                final_score = result.get('final_validation_score', 10.0)
                
                # Show benchmark metrics if available
                detailed_results = result.get('detailed_results', {})
                if detailed_results and 'result_dir' in detailed_results:
                    result_dir = Path(detailed_results['result_dir'])
                    benchmark_metrics = extract_benchmark_metrics(result_dir)
                    if benchmark_metrics:
                        print(f"    {masker_preset:20s}: {score:.4f} (validation: {final_score:.4f}) | Benchmark: {benchmark_metrics}")
                    else:
                        print(f"    {masker_preset:20s}: {score:.4f} (validation: {final_score:.4f})")
                else:
                    print(f"    {masker_preset:20s}: {score:.4f} (validation: {final_score:.4f})")
                
                if score < task_best_score:  # Lower is better
                    task_best_score = score
                    task_best_masker = masker_preset
                
                if score < best_overall_score:  # Lower is better
                    best_overall_score = score
                    best_overall_config = {
                        'model': model_name,
                        'benchmark': benchmark_task,
                        'masker': masker_preset,
                        'score': score,
                        'benchmark_metrics': benchmark_metrics if 'benchmark_metrics' in locals() else None
                    }
            
            if task_best_masker:
                print(f"    → Best: {task_best_masker} ({task_best_score:.4f})")
        
        # Model summary
        if model_results:
            all_scores = []
            for task_results in model_results.values():
                task_scores = [r.get('best_score', 10.0) for r in task_results.values()]  # Default high penalty
                all_scores.extend(task_scores)
            
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 10.0
            print(f"\n  📈 Model average: {avg_score:.4f} (lower is better)")
    
    # Overall summary
    print("\n🥇 BEST OVERALL CONFIGURATION (lowest combined score):")
    if best_overall_config:
        print(f"  Model: {best_overall_config['model']}")
        print(f"  Benchmark: {best_overall_config['benchmark']}")
        print(f"  Masker: {best_overall_config['masker']}")
        print(f"  Combined Score: {best_overall_config['score']:.4f} (lower is better)")
        if best_overall_config.get('benchmark_metrics'):
            print(f"  Benchmark Metrics: {best_overall_config['benchmark_metrics']}")
    
    print(f"\n💾 Detailed results saved to: {storage_path}/")
    print(f"🔍 View with: tensorboard --logdir {storage_path}")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description="Full benchmark optimization runner")
    parser.add_argument("--debug", action="store_true", 
                       help="Run quick test with limited configurations")
    args = parser.parse_args()
    
    if not RAY_AVAILABLE:
        print("❌ Ray Tune required for this benchmark")
        return
    
    # Get configuration
    config = get_benchmark_configurations(debug=args.debug)
    
    # Initialize Ray with proper environment
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True, 
            log_to_driver=False,
            runtime_env={
                "env_vars": {"PYTHONPATH": os.environ.get('PYTHONPATH', '')},
                "working_dir": "/scratch/krishna/inference/longcontext/sparse-attention-hub"
            }
        )
    
    mode = "Quick Test" if args.debug else "Full Production"
    print(f"🎯 {mode} Optimization Starting...")
    print(f"Ray Tune version: {ray.__version__}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    try:
        # Run optimization matrix
        start_time = time.time()
        results, storage_path = run_optimization_matrix(config)
        total_time = time.time() - start_time
        
        # Print summary
        print_optimization_summary(results, storage_path)
        
        print(f"\n⏱️  Total optimization time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        
        # Save results to file
        import json
        results_file = Path(config['result_dir']) / "optimization_summary.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for model, model_data in results.items():
            serializable_results[model] = {}
            for benchmark, benchmark_data in model_data.items():
                serializable_results[model][benchmark] = {}
                for masker, masker_data in benchmark_data.items():
                    clean_data = masker_data.copy()
                    # Remove non-serializable config objects
                    if 'best_config' in clean_data:
                        clean_data['best_config'] = str(clean_data['best_config'])
                    serializable_results[model][benchmark][masker] = clean_data
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📄 Summary saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
