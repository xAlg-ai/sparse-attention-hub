#!/usr/bin/env python3
"""
Two-Phase Benchmark System for Sparse Attention Methods.

Phase 1: Hyperparameter search to find optimal configs for each (model, task, masker) combination
Phase 2: Parallel benchmark execution using the discovered optimal configs

Usage:
    # Run both phases (default)
    python benchmark/raytune/run_two_phase_benchmark.py
    
    # Run only Phase 1 (config search)
    python benchmark/raytune/run_two_phase_benchmark.py --phase 1
    
    # Run only Phase 2 (benchmark execution)  
    python benchmark/raytune/run_two_phase_benchmark.py --phase 2
    
    # Debug mode (minimal configs, fast execution)
    python benchmark/raytune/run_two_phase_benchmark.py --debug
    
    # Force re-search in Phase 1
    python benchmark/raytune/run_two_phase_benchmark.py --phase 1 --force-search
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import pickle

# Path setup
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{current_dir}:{root_path}"

import torch
import pandas as pd
from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import AdapterConfig, BenchmarkConfig, BenchmarkResult
from optimizer_factory import create_optimizer

# Import all masker configs
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
    OracleTopKConfig,
    OracleTopPMaskerConfig,
    HashAttentionTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
    RandomSamplingMaskerConfig,
    MagicPigConfig,
)

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
except ImportError:
    print("Error: Ray Tune required. Install with: pip install 'ray[tune]' hyperopt")
    sys.exit(1)


# Note: Configuration names are based on the masker classes used, not parameter values
# Parameter values come from Ray Tune search, not from these initial configs


@dataclass
class OptimalConfig:
    """Stores optimal configuration found in Phase 1."""
    model: str
    task: str
    masker_name: str
    sparse_config: Optional[ResearchAttentionConfig]
    masker_classes: Optional[List] = field(default=None)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    search_time: float = 0.0
    num_trials: int = 0


def create_sparsity_objective(target_density: float, penalty_weight: float = 10.0):
    """Create an objective function that targets a specific sparsity level.
    
    Args:
        target_density: Target density level (e.g., 0.05 for 5% density)
        penalty_weight: Weight for penalty when density exceeds target
        
    Returns:
        Objective function that can be used for optimization
    """
    def objective(error: float, density: float) -> float:
        # Base objective: heavily weight error, lightly weight density
        base_score = 0.99 * error + 0.01 * density
        
        # Add penalty if density exceeds target
        penalty = penalty_weight * max(0, density - target_density)
        
        return base_score + penalty
    
    objective.__name__ = f"objective_sparsity_{int(target_density * 100)}_percent"
    return objective


# Pre-defined objective functions for common sparsity levels
OBJECTIVE_FUNCTIONS = {
    "sparsity_5": create_sparsity_objective(0.05),
    "sparsity_10": create_sparsity_objective(0.10),
    "sparsity_15": create_sparsity_objective(0.15),
    "sparsity_20": create_sparsity_objective(0.20),
    "sparsity_25": create_sparsity_objective(0.25),
    "default": lambda error, density: error + 0.1 * density + (5.0 if density > 0.5 else 0.0),
}


class Phase1BenchmarkRunner:
    """Handles individual benchmark runs during config search."""
    
    def __init__(self, config: dict):
        self.config = config
        self.executor = BenchmarkExecutor(
            gpu_ids=[0],  # Single GPU per trial  
            max_concurrent_runs=1,
            base_result_dir=config["search_result_dir"],
            enable_resumability=False,
            required_result_files=["raw_results.csv"],
            timeout_per_benchmark=config["search_timeout"],
            verbose=False,
        )
        self.adapter_config = AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.generation_kwargs = {
            "max_new_tokens": config["search_max_new_tokens"],
            "do_sample": False
        }
        self.request_kwargs = {
            "max_context_length": config["search_max_context_length"],
            "max_requests": config["search_max_requests"],
        }
        
        # Get objective function
        self.objective_name = config.get("objective_function", "default")
        self.objective_function = OBJECTIVE_FUNCTIONS.get(self.objective_name, OBJECTIVE_FUNCTIONS["default"])
        logging.info(f"Using objective function: {self.objective_name}")

    def __call__(self, attention_config, task_name: str, model_name: str) -> Tuple[float, float, float]:
        """Run benchmark and return (score, density, error) tuple."""
        try:
            benchmark_name, subset_name = task_name.split("/", 1) if "/" in task_name else (task_name, None)
            benchmark_config = BenchmarkConfig(
                benchmark_name=benchmark_name, 
                subsets=[subset_name] if subset_name else None
            )
            
            results = self.executor.run_benchmark_matrix(
                model_names=[model_name],
                sparse_attention_configs=[("search", attention_config)],
                benchmark_configs=[benchmark_config],
                adapter_config=self.adapter_config,
                generation_kwargs=self.generation_kwargs,
                request_kwargs=self.request_kwargs,
            )

            # Extract score from results
            if results.progress.completed_stubs > 0 and hasattr(results, "individual_results"):
                completed = [r for r in results.individual_results if isinstance(r, BenchmarkResult)]
                if completed:
                    result_dir = Path(completed[0].stub.result_dir)
                    metrics = self._extract_micro_metrics(result_dir)
                    error, density = metrics["attention_error"], metrics["density"]
                    
                    # For dense configuration (density=1.0, error=0.0), use a simple score
                    if density == 1.0 and error == 0.0:
                        # Dense baseline: use benchmark accuracy metrics instead of sparse metrics
                        score = 0.1  # Small baseline score for dense
                    else:
                        # Use the selected objective function
                        score = self.objective_function(error, density)
                        # Also print to stdout so the test script can detect it
                        print(f"Objective: {self.objective_name}, Error: {error:.4f}, Density: {density:.4f}, Score: {score:.4f}")
                        logging.info(f"Objective: {self.objective_name}, Error: {error:.4f}, Density: {density:.4f}, Score: {score:.4f}")
                    
                    return score, density, error
                    
        except Exception as e:
            logging.error(f"Benchmark failed: {e}")
            
        return 5.0, 1.0, 1.0  # Penalty score, worst-case density, and worst-case error
    
    def _extract_micro_metrics(self, result_dir: Path) -> dict:
        """Extract attention error and density from micro metrics."""
        micro_metrics_file = result_dir / "micro_metrics.jsonl"
        if not micro_metrics_file.exists():
            # For dense configuration, micro_metrics.jsonl won't exist since no sparse attention is used
            # Return default values: 0 error (perfect) and 1.0 density (fully dense)
            logging.info(f"micro_metrics.jsonl not found in {result_dir}, using dense defaults")
            return {"attention_error": 0.0, "density": 1.0}
            
        errors, densities = [], []
        with open(micro_metrics_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    metric, value = entry.get("metric"), entry.get("value")
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        if metric == "research_attention_output_error": 
                            errors.append(float(value))
                        elif metric == "research_attention_density": 
                            densities.append(float(value))
                except (json.JSONDecodeError, ValueError, TypeError): 
                    continue
                    
        return {
            "attention_error": sum(errors) / len(errors) if errors else 1.0, 
            "density": sum(densities) / len(densities) if densities else 1.0
        }


class ConfigSearchManager:
    """Manages Phase 1: Hyperparameter search for optimal configs."""
    
    def __init__(self, base_config: dict):
        self.config = base_config
        # Add timestamp to the results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(base_config["optimal_configs_dir"])
        self.results_dir = base_dir / f"run_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp
        print(f"Saving optimal configs to: {self.results_dir}")
        
    def search_optimal_config(
        self, 
        model: str, 
        task: str, 
        masker_name: str, 
        masker_classes: Optional[List],
        full_sparse_config: Optional[ResearchAttentionConfig] = None
    ) -> OptimalConfig:
        """Search for optimal hyperparameters for a single combination."""
        
        config_file = self.results_dir / f"{model}_{task}_{masker_name}.json".replace("/", "_")
        
        # Check if already exists
        if config_file.exists() and not self.config.get("force_search", False):
            print(f"  → Loading existing config")
            return self._load_config(config_file)
        
        # Handle dense config (no optimization needed)
        if masker_classes is None:
            optimal = OptimalConfig(
                model=model,
                task=task,
                masker_name=masker_name,
                sparse_config=None,
                masker_classes=None,
                hyperparams={},
                score=0.0,
                search_time=0.0,
                num_trials=1
            )
            self._save_config(optimal, config_file)
            return optimal
        
        # Run hyperparameter search
        print(f"  → Running hyperparameter search...")
        start_time = time.time()
        
        try:
            # Create optimizer with template config for fixed parameters
            optimizer = create_optimizer(masker_classes, full_sparse_config)
            
            # Show what we're searching
            search_space = optimizer.create_search_space(task)
            print(f"  → Search space parameters:")
            for param, space_obj in search_space.items():
                # Extract actual values from Ray Tune objects
                if hasattr(space_obj, 'categories'):
                    values = space_obj.categories
                    print(f"     - {param}: {values}")
                else:
                    print(f"     - {param}: {space_obj}")
            
            # Create objective function
            def objective(trial_config):
                runner = Phase1BenchmarkRunner(self.config)
                attention_config = optimizer.create_config_from_params(trial_config)
                score, density, error = runner(attention_config, task, model)
                return {"combined_score": score, "density": density, "error": error}
            
            # Get Ray Tune components
            search_space = optimizer.create_search_space(task)
            scheduler = ASHAScheduler(
                time_attr="training_iteration",
                max_t=20,
                grace_period=5,
                reduction_factor=2
            )
            search_alg = HyperOptSearch(
                metric="combined_score",
                mode="min",
                n_initial_points=max(1, self.config["num_samples"] // 4)
            )
            
            # Run Ray Tune
            sanitized_name = f"{model}_{task}_{masker_name}".replace("/", "_")
            analysis = tune.run(
                objective,
                config=search_space,
                num_samples=self.config["num_samples"],
                metric="combined_score",
                mode="min",
                scheduler=scheduler,
                #search_alg=search_alg,
                resources_per_trial={"CPU": 1, "GPU": 1.0},
                storage_path=os.path.abspath(self.config["ray_results_dir"]),
                name=sanitized_name,
                verbose=1,  # Show Ray Tune progress
                stop={"training_iteration": 1},  # One evaluation per config
            )
            
            # Get best config
            best_trial = analysis.get_best_trial("combined_score", "min", "last")
            best_config = optimizer.create_config_from_params(best_trial.config)
            
            # Save detailed trial information for post-analysis
            trials_info = []
            for trial in analysis.trials:
                trial_info = {
                    "trial_id": trial.trial_id,
                    "config": trial.config,
                    "score": trial.last_result.get("combined_score", float('inf')) if trial.last_result else float('inf'),
                    "status": trial.status,
                    "start_time": trial.start_time.isoformat() if hasattr(trial, 'start_time') and trial.start_time else None,
                    "metric_history": trial.metric_analysis.get("combined_score", {}) if hasattr(trial, 'metric_analysis') else {}
                }
                trials_info.append(trial_info)
            
            # Save trial details to separate file
            trials_file = self.results_dir / f"{model}_{task}_{masker_name}_trials.json".replace("/", "_")
            with open(trials_file, "w") as f:
                json.dump({
                    "model": model,
                    "task": task,
                    "masker_name": masker_name,
                    "objective_function": self.config.get("objective_function", "default"),
                    "best_trial_id": best_trial.trial_id,
                    "trials": trials_info,
                    "analysis_dataframe_path": str(self.results_dir / f"{model}_{task}_{masker_name}_analysis.csv".replace("/", "_"))
                }, f, indent=2)
            
            # Save Ray analysis dataframe for detailed analysis
            df = analysis.dataframe()
            df.to_csv(self.results_dir / f"{model}_{task}_{masker_name}_analysis.csv".replace("/", "_"), index=False)
            
            optimal = OptimalConfig(
                model=model,
                task=task,
                masker_name=masker_name,
                sparse_config=best_config,
                masker_classes=masker_classes,
                hyperparams=best_trial.config,
                score=best_trial.last_result["combined_score"],
                search_time=time.time() - start_time,
                num_trials=len(analysis.trials)
            )
            
            self._save_config(optimal, config_file)
            return optimal
            
        except Exception as e:
            print(f"  ✗ Search failed: {e}")
            traceback.print_exc()
            # Return failure config
            optimal = OptimalConfig(
                model=model,
                task=task,
                masker_name=masker_name,
                sparse_config=full_sparse_config,  # Use the full config passed in
                masker_classes=masker_classes,
                hyperparams={},
                score=5.0,
                search_time=time.time() - start_time,
                num_trials=0
            )
            self._save_config(optimal, config_file)
            return optimal
    
    def _save_config(self, config: OptimalConfig, filepath: Path):
        """Save configuration to JSON."""
        data = asdict(config)
        
        # Convert sparse config to serializable format
        if config.sparse_config:
            data["sparse_config"] = self._serialize_sparse_config(config.sparse_config)
        
        # Convert masker classes to strings
        if config.masker_classes:
            data["masker_classes"] = [cls.__name__ for cls in config.masker_classes]
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_config(self, filepath: Path) -> OptimalConfig:
        """Load configuration from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Reconstruct sparse config if present
        if data.get("sparse_config"):
            data["sparse_config"] = self._deserialize_sparse_config(data["sparse_config"])
        
        # Reconstruct masker classes from strings
        if data.get("masker_classes"):
            # Map class names to actual classes
            class_map = {
                "LocalMaskerConfig": LocalMaskerConfig,
                "SinkMaskerConfig": SinkMaskerConfig,
                "OracleTopKConfig": OracleTopKConfig,
                "OracleTopPMaskerConfig": OracleTopPMaskerConfig,
                "HashAttentionTopKMaskerConfig": HashAttentionTopKMaskerConfig,
                "AdaptiveSamplingMaskerConfig": AdaptiveSamplingMaskerConfig,
                "RandomSamplingMaskerConfig": RandomSamplingMaskerConfig,
                "MagicPigConfig": MagicPigConfig,
            }
            data["masker_classes"] = [class_map[name] for name in data["masker_classes"]]
        
        return OptimalConfig(**data)
    
    def _serialize_sparse_config(self, config: ResearchAttentionConfig) -> dict:
        """Convert ResearchAttentionConfig to JSON-serializable format."""
        if config is None:
            return None
            
        # Serialize each masker config
        masker_configs = []
        for masker in config.masker_configs:
            masker_dict = {
                "type": type(masker).__name__,
                "params": {}
            }
            # Add all attributes
            for attr in dir(masker):
                if not attr.startswith("_") and hasattr(masker, attr):
                    value = getattr(masker, attr)
                    if isinstance(value, (int, float, str, bool, type(None))):
                        masker_dict["params"][attr] = value
            masker_configs.append(masker_dict)
        
        return {
            "type": "ResearchAttentionConfig",
            "masker_configs": masker_configs
        }
    
    def _deserialize_sparse_config(self, data: dict) -> ResearchAttentionConfig:
        """Reconstruct ResearchAttentionConfig from JSON data."""
        if data is None:
            return None
            
        if data.get("type") != "ResearchAttentionConfig":
            return None
            
        # Map config types to classes
        config_map = {
            "LocalMaskerConfig": LocalMaskerConfig,
            "SinkMaskerConfig": SinkMaskerConfig,
            "OracleTopKConfig": OracleTopKConfig,
            "OracleTopPMaskerConfig": OracleTopPMaskerConfig,
            "HashAttentionTopKMaskerConfig": HashAttentionTopKMaskerConfig,
            "AdaptiveSamplingMaskerConfig": AdaptiveSamplingMaskerConfig,
            "RandomSamplingMaskerConfig": RandomSamplingMaskerConfig,
            "MagicPigConfig": MagicPigConfig,
        }
        
        # Reconstruct masker configs
        masker_configs = []
        for masker_data in data.get("masker_configs", []):
            config_class = config_map.get(masker_data["type"])
            if config_class:
                # Create instance with parameters
                params = masker_data.get("params", {})
                masker_configs.append(config_class(**params))
        
        return ResearchAttentionConfig(masker_configs=masker_configs)


def run_phase_one(config: dict) -> Dict[str, OptimalConfig]:
    """Phase 1: Find optimal configurations for all combinations."""
    print("\n" + "="*80)
    print("PHASE 1: HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Models: {len(config['models'])}")
    print(f"Tasks: {len(config['tasks'])}")
    print(f"Optimal Configs: {len(config['optimal_configs'])}")
    print(f"To Optimize Configs: {len(config['to_optimize_configs'])}")
    print(f"Total Combinations to optimize: {len(config['models']) * len(config['tasks']) * len(config['to_optimize_configs'])}")
    print(f"Samples per search: {config['num_samples']}")
    print(f"Objective Function: {config['objective_function']}")
    
    # Display objective function details
    if config['objective_function'].startswith('sparsity_'):
        target = int(config['objective_function'].split('_')[1])
        print(f"  → Targeting {target}% density (0.{target:02d} fraction)")
        print(f"  → Formula: 0.99 * error + 0.01 * density + penalty for exceeding target")
    
    print("\nSearch Configuration:")
    print(f"  → Max new tokens: {config['search_max_new_tokens']}")
    print(f"  → Max context length: {config['search_max_context_length']}")
    print(f"  → Max requests per trial: {config['search_max_requests']}")
    print(f"  → Timeout per trial: {config['search_timeout']}s")
    
    print("\nNote: For each sparse config, Ray Tune will search different hyperparameter")
    print("values (e.g., window_size, sink_size, sampling_rate) to find the best combination.")
    print("="*80)
    
    manager = ConfigSearchManager(config)
    optimal_configs = {}
    
    total = len(config["models"]) * len(config["tasks"]) * len(config["to_optimize_configs"]) + len(config["models"]) * len(config["tasks"]) * len(config["optimal_configs"])
    current = 0
    
    for model in config["models"]:
        print(f"\nModel: {model}")
        print("-" * 60)
        
        for task in config["tasks"]:
            for masker_name, (masker_classes, full_config) in config["to_optimize_configs_map"].items():
                current += 1
                key = f"{model}_{task}_{masker_name}".replace("/", "_")
                
                print(f"\n[{current}/{total}] Task: {task} | Config: {masker_name}")
                optimal = manager.search_optimal_config(
                    model, task, masker_name, masker_classes, full_config
                )
                optimal_configs[key] = optimal
            
            for masker_name, (masker_classes, full_config) in config["optimal_configs_map"].items():
                current += 1
                key = f"{model}_{task}_{masker_name}".replace("/", "_")
                
                optimal = OptimalConfig(
                    model=model,
                    task=task,
                    masker_name=masker_name,
                    sparse_config=full_config,
                    masker_classes=masker_classes,
                    hyperparams={},
                    score=0.0,
                    search_time=0.0,
                    num_trials=0
                )
                manager._save_config(optimal, Path(manager.results_dir) / f"{key}.json")
                optimal_configs[key] = optimal
    
    print(f"\n{'='*80}")
    print(f"Phase 1 complete. Found {len(optimal_configs)} optimal configurations.")
    print(f"Configs saved to: {manager.results_dir}")
    print(f"Run identifier: {manager.timestamp}")
    print(f"\nTo use these configs in Phase 2:")
    print(f"  python {sys.argv[0]} --phase 2  # Uses most recent configs")
    print(f"  python {sys.argv[0]} --phase 2 --config-run run_{manager.timestamp}  # Uses this specific run")
    print(f"{'='*80}")
    
    return optimal_configs


def run_phase_two(config: dict, optimal_configs: Dict[str, OptimalConfig]) -> dict:
    """Phase 2: Run benchmarks with optimal configurations."""
    print("\n" + "="*80)
    print("PHASE 2: BENCHMARK EXECUTION")
    print("="*80)
    
    # Build unique sparse configs from optimal configs
    unique_sparse_configs = []
    seen = set()
    config_usage = {}  # Track which (model, task) use each config
    
    for key, opt_config in optimal_configs.items():
        config_str = str(opt_config.sparse_config) if opt_config.sparse_config else "None"
        if config_str not in seen:
            seen.add(config_str)
            unique_sparse_configs.append((
                opt_config.masker_name,
                opt_config.sparse_config
            ))
            config_usage[config_str] = []
        config_usage[config_str].append((opt_config.model, opt_config.task))
    
    print(f"Unique sparse configurations: {len(unique_sparse_configs)}")
    print(f"Models: {len(config['models'])}")
    print(f"Tasks: {len(config['tasks'])}")
    print(f"Total benchmark runs: {len(config['models']) * len(config['tasks']) * len(unique_sparse_configs)}")
    print(f"GPUs available: {len(config['gpu_ids'])}")
    print("="*80)
    
    # Create executor
    executor = BenchmarkExecutor(
        gpu_ids=config["gpu_ids"],
        max_concurrent_runs=len(config["gpu_ids"]),
        base_result_dir=config["benchmark_results_dir"],
        enable_resumability=True,
        required_result_files=["raw_results.csv"],
        timeout_per_benchmark=config["benchmark_timeout"],
        verbose=True
    )
    
    # Create benchmark configs
    benchmark_configs = []
    for task in config["tasks"]:
        if "/" in task:
            name, subset = task.split("/", 1)
            benchmark_configs.append(BenchmarkConfig(
                benchmark_name=name,
                subsets=[subset]
            ))
        else:
            benchmark_configs.append(BenchmarkConfig(
                benchmark_name=task,
                subsets=None
            ))
    
    # Run benchmarks
    print("\nStarting benchmark execution...")
    results = executor.run_benchmark_matrix(
        model_names=config["models"],
        sparse_attention_configs=unique_sparse_configs,
        benchmark_configs=benchmark_configs,
        adapter_config=AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"}
        ),
        generation_kwargs={
            "max_new_tokens": config["benchmark_max_new_tokens"],
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": None,
        },
        request_kwargs={
            "max_context_length": config["benchmark_max_context_length"],
            "max_requests": config["benchmark_max_requests"]
        }
    )
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "objective_function": config["objective_function"],
        "config_run_used": config.get("config_run_dir", "unknown"),
        "phase1_optimal_configs": {
            k: {
                "model": v.model,
                "task": v.task,
                "masker_name": v.masker_name,
                "score": v.score,
                "hyperparams": v.hyperparams,
                "search_time": v.search_time,
                "num_trials": v.num_trials
            } for k, v in optimal_configs.items()
        },
        "phase2_results": {
            "total": results.progress.total_stubs,
            "completed": results.progress.completed_stubs,
            "failed": results.progress.failed_stubs,
            "skipped": results.progress.skipped_stubs,
        },
        "configuration": {
            "models": config["models"],
            "tasks": config["tasks"],
            "num_sparse_configs": len(unique_sparse_configs),
            "objective_function": config["objective_function"],
            "benchmark_timeout": config["benchmark_timeout"],
            "max_new_tokens": config["benchmark_max_new_tokens"],
            "max_context_length": config["benchmark_max_context_length"],
        }
    }
    
    summary_file = Path(config["benchmark_results_dir"]) / "benchmark_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Phase 2 complete.")
    print(f"Results saved to: {config['benchmark_results_dir']}")
    print(f"Summary saved to: {summary_file}")
    print(f"Completed: {results.progress.completed_stubs}/{results.progress.total_stubs}")
    print(f"Failed: {results.progress.failed_stubs}")
    print(f"{'='*80}")
    
    return summary


def get_masker_list_name(masker_classes: List) -> str:
    """Generate a name based on the masker classes being used."""
    if not masker_classes:
        return "dense"
    
    # Extract just the key part of each masker name
    parts = []
    for cls in masker_classes:
        name = cls.__name__.replace("MaskerConfig", "").replace("Config", "")
        # Convert camelCase to lowercase
        name = ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
        parts.append(name)
    
    return "_".join(parts)


def get_all_sparse_configs(weight_file: str = None) -> List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]:
    """Get all sparse attention configurations.
    Returns list of (name, full_config, masker_classes) tuples.
    
    Note: The configs returned here are only used to determine which masker classes
    to use. The actual parameter values will be determined by Ray Tune search.
    """
    assert weight_file is not None, "Weight file is required for HashAttention Masker"
    optimal_configs = []
    to_optimize_configs = []
    

    ############################## optimal configs ##############################
    # 1. Dense baseline
    optimal_configs.append(("dense", None, None))
    
    # 2. Oracle top k (already included above with adaptive, but also standalone)
    classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopKConfig]
    name = get_masker_list_name(classes)
    config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        OracleTopKConfig(heavy_size=0.095)
    ])
    optimal_configs.append((name, config, classes))

    #3. HashAttention top k
    classes = [SinkMaskerConfig, LocalMaskerConfig, HashAttentionTopKMaskerConfig]
    name = get_masker_list_name(classes)
    config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        HashAttentionTopKMaskerConfig(
            heavy_size=0.095,
            hat_bits=32,
            hat_mlp_layers=3,
            hat_mlp_hidden_size=128,
            hat_mlp_activation="silu",
            hat_weight_file=weight_file
        ),
    ])
    optimal_configs.append((name, config, classes))
    
    # 4. Random sampling with sink and local
    # classes = [SinkMaskerConfig, LocalMaskerConfig, RandomSamplingMaskerConfig]
    # name = get_masker_list_name(classes)
    # config = ResearchAttentionConfig(masker_configs=[
    #     SinkMaskerConfig(sink_size=128),  # Middle value from search space [4, 8, 16, 32, 64, 128]
    #     LocalMaskerConfig(window_size=128),  # Middle value from search space [32, 64, 128, 256]
    #     RandomSamplingMaskerConfig(sampling_rate=0.095)  # Middle value from search space [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    # ])
    # optimal_configs.append((name, config, classes))
    
    ############################## to optimize configs ##############################


    # 1. Adaptive sampling with oracle top k
    classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopKConfig, AdaptiveSamplingMaskerConfig]
    name = get_masker_list_name(classes)
    config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        OracleTopKConfig(heavy_size=0.10),  # Middle value from search space
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,  # Middle value
            epsilon=0.25,  # Middle value
            delta=0.25,  # Middle value
            init_offset=128,  # Middle value
            local_offset=128  # Middle value
        )
    ])
    to_optimize_configs.append((name, config, classes))

    # 2. Adaptive sampling with oracle top p

    classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopPMaskerConfig, AdaptiveSamplingMaskerConfig]
    name = get_masker_list_name(classes)
    config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        OracleTopPMaskerConfig(top_p=0.10),  # Middle value from search space
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,  # Middle value
            epsilon=0.25,  # Middle value
            delta=0.25,  # Middle value
            init_offset=128,  # Middle value
            local_offset=128  # Middle value
        )
    ])
    to_optimize_configs.append((name, config, classes))
    
    # 3. Adaptive sampling with HAT top k
    classes = [SinkMaskerConfig, LocalMaskerConfig, HashAttentionTopKMaskerConfig, AdaptiveSamplingMaskerConfig]
    name = get_masker_list_name(classes)
    config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        HashAttentionTopKMaskerConfig(
            heavy_size=0.05,  # Required parameter
            hat_bits=32,  # Required parameter
            hat_mlp_layers=3,  # Required parameter
            hat_mlp_hidden_size=128,  # Required parameter
            hat_mlp_activation="silu",  # Required parameter
            hat_weight_file=weight_file  # Weight file is required
        ),
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,
            epsilon=0.25,
            delta=0.25,
            init_offset=128,
            local_offset=128
        )
    ])
    to_optimize_configs.append((name, config, classes))
    
    
    # 4. Oracle top p
    classes = [SinkMaskerConfig, LocalMaskerConfig, OracleTopPMaskerConfig]
    name = get_masker_list_name(classes)
    config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        OracleTopPMaskerConfig(top_p=0.7)  # Default middle value from search space
    ])
    to_optimize_configs.append((name, config, classes))
    

    # 5. MagicPig config
    classes = [SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig]
    name = get_masker_list_name(classes)
    config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        MagicPigConfig(
            lsh_l=8,  # Default value from search space
            lsh_k=8   # Default value from search space
        )
    ])
    to_optimize_configs.append((name, config, classes))
    
    return optimal_configs, to_optimize_configs


def get_run_configuration(args: argparse.Namespace) -> dict:
    """Build complete configuration from command-line arguments."""
    num_gpus = torch.cuda.device_count()
    
    # Get HashAttention weights file
    weight_file = f"/workspace/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"
    if not os.path.exists(weight_file):
        weight_file = "./hat_weights.pkl"
        print(f"Warning: HashAttention weights not found, using {weight_file}")
    
    # Get all sparse configs
    optimal_configs, to_optimize_configs = get_all_sparse_configs(weight_file)
    
    # Filter configs based on debug mode
    if args.debug:
        sparse_configs = to_optimize_configs[:3]  # Just first 3 for debug
        models = ["meta-llama/Llama-3.1-8B-Instruct"]
        tasks = ["loogle/shortdep_qa"]
        num_samples = 8
    else:
        models = ["meta-llama/Llama-3.1-8B-Instruct"]
        tasks = [
                # "infinite_bench/passkey",
                # "ruler/4096",
                #"loogle/longdep_summarization",
                #"loogle/longdep_qa",
                #"loogle/shortdep_qa",
                #"loogle/shortdep_cloze",
                # "zero_scrolls/default",
                # "longbenchv2/0shot",
                # "aime2024/aime2024",
                # "aime2025/aime2025",
                # "longbench/passage_retrieval_en",
                # "mock_benchmark/reading_comprehension",
                "ruler32k/qa_1",
                "ruler32k/qa_2",
                "ruler32k/fwe",
                "ruler32k/niah_multikey_2",
                "ruler32k/niah_multikey_3",
                "ruler32k/niah_multivalue",
        ]
        num_samples = args.num_samples
    
    # Build config maps
    optimal_configs_map = {}
    to_optimize_configs_map = {}
    for name, full_config, classes in optimal_configs:
        optimal_configs_map[name] = (classes, full_config)
    for name, full_config, classes in to_optimize_configs:
        to_optimize_configs_map[name] = (classes, full_config)
    
    return {
        "models": models,
        "tasks": tasks,
        "optimal_configs": optimal_configs,
        "to_optimize_configs": to_optimize_configs,
        "optimal_configs_map": optimal_configs_map,
        "to_optimize_configs_map": to_optimize_configs_map,
        "gpu_ids": list(range(num_gpus)),
        "num_samples": num_samples,
        "objective_function": args.objective,
        
        # Directories
        "optimal_configs_dir": args.optimal_configs_dir,
        "benchmark_results_dir": args.benchmark_results_dir,
        "ray_results_dir": args.ray_results_dir,
        "search_result_dir": os.path.join(args.ray_results_dir, "search_runs"),
        
        # Phase 1 params
        "search_timeout": args.search_timeout,
        "search_max_new_tokens": args.search_max_new_tokens,
        "search_max_context_length": args.search_max_context_length,
        "search_max_requests": args.search_max_requests,
        "force_search": args.force_search,
        
        # Phase 2 params
        "benchmark_timeout": args.benchmark_timeout,
        "benchmark_max_new_tokens": args.benchmark_max_new_tokens,
        "benchmark_max_context_length": args.benchmark_max_context_length,
        "benchmark_max_requests": args.benchmark_max_requests,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase benchmark system for sparse attention methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Phase control
    parser.add_argument("--phase", type=int, choices=[1, 2], 
                       help="Run specific phase only (1=search, 2=benchmark)", required=True)
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode with minimal configs")
    parser.add_argument("--force-search", action="store_true",
                       help="Force re-run of Phase 1 even if configs exist")
    
    # Objective function selection
    parser.add_argument("--objective", type=str, default="default",
                       choices=list(OBJECTIVE_FUNCTIONS.keys()),
                       help="Objective function to use for optimization", required=True)
    
    # Config run selection for Phase 2
    parser.add_argument("--config-run", type=str,
                       help="Specific config run directory to use for Phase 2 (e.g., 'run_20240315_143022')")
    
    # Directories
    parser.add_argument("--optimal-configs-dir", default="./optimal_configs",
                       help="Directory for storing optimal configurations")
    parser.add_argument("--benchmark-results-dir", default="./benchmark_results",
                       help="Directory for benchmark results")
    parser.add_argument("--ray-results-dir", default="./ray_results",
                       help="Directory for Ray Tune results")
    
    # Phase 1 arguments
    phase1_group = parser.add_argument_group('Phase 1 - Config Search')
    phase1_group.add_argument("--num-samples", type=int, default=1,
                             help="Number of samples per hyperparameter search", required=True)
    phase1_group.add_argument("--search-timeout", type=int, default=900,
                             help="Timeout per search trial (seconds)")
    phase1_group.add_argument("--search-max-new-tokens", type=int, default=5,
                             help="Max new tokens for search trials", required=True)
    phase1_group.add_argument("--search-max-context-length", type=int, default=100000,
                             help="Max context length for search trials", required=True)
    phase1_group.add_argument("--search-max-requests", type=int, default=5,
                             help="Max requests per search trial", required=True)
    
    # Phase 2 arguments
    phase2_group = parser.add_argument_group('Phase 2 - Benchmark Execution')
    phase2_group.add_argument("--benchmark-timeout", type=int, default=3600,
                             help="Timeout per benchmark (seconds)")
    phase2_group.add_argument("--benchmark-max-new-tokens", type=int, default=100,
                             help="Max new tokens for benchmarks")
    phase2_group.add_argument("--benchmark-max-context-length", type=int, default=32000,
                             help="Max context length for benchmarks")
    phase2_group.add_argument("--benchmark-max-requests", type=int, default=25,
                             help="Max requests per benchmark")
    
    args = parser.parse_args()
    
    # Build configuration
    config = get_run_configuration(args)
    
    print("Two-Phase Benchmark System")
    print(f"Ray Version: {ray.__version__}, GPUs Available: {torch.cuda.device_count()}")
    print(f"Mode: {'Debug' if args.debug else 'Production'}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, 
                runtime_env={"working_dir": str(root_path)})
    
    start_time = time.time()
    
    try:
        # Phase 1: Config Search
        if args.phase is None or args.phase == 1:
            optimal_configs = run_phase_one(config)
            # If running both phases, store the config directory
            if args.phase is None and optimal_configs:
                # Get the manager's results directory from any config
                first_key = next(iter(optimal_configs))
                manager = ConfigSearchManager(config)
                config["config_run_dir"] = str(manager.results_dir)
        else:
            # Load existing configs for Phase 2
            print("\nLoading existing optimal configurations...")
            base_config_dir = Path(args.optimal_configs_dir)
            
            # Find the most recent run directory or use specified one
            if args.config_run:
                config_dir = base_config_dir / args.config_run
                if not config_dir.exists():
                    print(f"Error: Specified config run {config_dir} does not exist.")
                    return
            else:
                # Find the most recent run_* directory
                run_dirs = sorted([d for d in base_config_dir.glob("run_*") if d.is_dir()])
                if not run_dirs:
                    # Fallback to base directory for backward compatibility
                    config_dir = base_config_dir
                    if not any(config_dir.glob("*.json")):
                        print(f"Error: No optimal configs found. Run Phase 1 first.")
                        return
                else:
                    config_dir = run_dirs[-1]  # Most recent
                    print(f"Using most recent config run: {config_dir.name}")
            
            # Create a dummy manager just for loading
            manager = ConfigSearchManager(config)
            manager.results_dir = config_dir  # Override the directory
            
            optimal_configs = {}
            for config_file in config_dir.glob("*.json"):
                if config_file.name.endswith("_trials.json"):
                    continue
                try:
                    opt_config = manager._load_config(config_file)
                    key = config_file.stem
                    optimal_configs[key] = opt_config
                except Exception as e:
                    print(f"Warning: Failed to load {config_file}: {e}")
                    
            print(f"Loaded {len(optimal_configs)} configurations from {config_dir}")
            # Store which config run was used
            config["config_run_dir"] = str(config_dir)
        
        # Phase 2: Benchmark Execution
        if args.phase is None or args.phase == 2:
            if not optimal_configs:
                print("\nError: No optimal configurations found. Run Phase 1 first.")
                return
                
            results = run_phase_two(config, optimal_configs)
            
            # Print final summary
            print("\n" + "="*80)
            print("FINAL SUMMARY")
            print("="*80)
            if args.phase is None:
                print(f"Phase 1: Found {len(optimal_configs)} optimal configurations")
            if results:
                print(f"Phase 2: Completed {results['phase2_results']['completed']}/{results['phase2_results']['total']} benchmarks")
                print(f"         Failed: {results['phase2_results']['failed']}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time / 3600:.2f} hours ({total_time:.0f} seconds)")
        ray.shutdown()
        print("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
