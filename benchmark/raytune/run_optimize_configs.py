#!/usr/bin/env python3
"""
Hyperparameter search for optimal sparse attention configurations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{current_dir}:{root_path}"

import torch
import ray

from config_builders.utility import OBJECTIVE_FUNCTIONS, OptimalConfig
from config_builders.factory import build_all_configs

# Import all masker configs
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig

# Import search manager
from search_manager import ConfigSearchManager

# Import run configuration
from OPTIMIZATION_EXPERIMENT import (
    MODEL_CONFIGS,
    DEFAULT_MODEL,
    RUN_TASKS,
    OBJECTIVE,
    NUM_SAMPLES,
    SEARCH_MAX_NEW_TOKENS,
    SEARCH_MAX_CONTEXT_LENGTH,
    SEARCH_MAX_REQUESTS,
    FORCE_SEARCH,
    OPTIMAL_CONFIGS_DIR,
    RAY_RESULTS_DIR,
    SEARCH_TIMEOUT,
    ACTORS_PER_GPU,
    MEMORY_OBJECTIVE,
    BUILDER_NAMES,
)

def get_all_sparse_configs(weight_file: str = None, objective: str = "default", memory_objective: str = None, builder_names: List[str] = None) -> List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]:
    """Get all sparse attention configurations.
    Returns list of (name, full_config, masker_classes) tuples.
    
    Note: The configs returned here are only used to determine which masker classes
    to use. The actual parameter values will be determined by Ray Tune search.
    
    Args:
        weight_file: Path to weight file (required)
        objective: Objective function name (e.g., "sparsity_5")
        memory_objective: Memory objective parameter for configs that need it
        builder_names: List of builder names to use
        
    Returns:
        Tuple of (optimal_configs, to_optimize_configs)
    """
    assert weight_file is not None, "Weight file is required for HashAttention Masker"
    
    # Use factory to build all configs
    optimal_configs, to_optimize_configs = build_all_configs(
        weight_file=weight_file,
        objective=objective,
        builder_names=builder_names or BUILDER_NAMES,
        memory_objective=memory_objective
    )
    
    return optimal_configs, to_optimize_configs


def get_run_configuration() -> dict:
    """Build complete configuration from RUN_CONFIG.py."""
    num_gpus: int = torch.cuda.device_count()
    
    # Get model configuration  
    model_config: Dict[str, str] = MODEL_CONFIGS[DEFAULT_MODEL]
    weight_file: str = model_config["weight_file"]
    model_name: str = model_config["model_name"]

    if not os.path.exists(weight_file):
        weight_file = "./hat_weights.pkl"
        print(f"Warning: HashAttention weights not found, using {weight_file}")
    
    # Get all sparse configs
    optimal_configs, to_optimize_configs = get_all_sparse_configs(
        weight_file, 
        objective=OBJECTIVE, 
        memory_objective=MEMORY_OBJECTIVE,
        builder_names=BUILDER_NAMES
    )
    
    # Set models, tasks, and num_samples
    models: List[str] = [model_name]
    tasks: List[str] = RUN_TASKS
    num_samples: int = NUM_SAMPLES
    
    # Build config maps
    optimal_configs_map: Dict[str, tuple] = {}
    to_optimize_configs_map: Dict[str, tuple] = {}
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
        "objective_function": OBJECTIVE,
        
        # Directories
        "optimal_configs_dir": OPTIMAL_CONFIGS_DIR,
        "ray_results_dir": RAY_RESULTS_DIR,
        "search_result_dir": os.path.join(RAY_RESULTS_DIR, "search_runs"),
        
        # Search params
        "search_timeout": SEARCH_TIMEOUT,
        "search_max_new_tokens": SEARCH_MAX_NEW_TOKENS,
        "search_max_context_length": SEARCH_MAX_CONTEXT_LENGTH,
        "search_max_requests": SEARCH_MAX_REQUESTS,
        "force_search": FORCE_SEARCH,
    }


def run_search(config: Dict[str, Any], actors_per_gpu: int = 1) -> Dict[str, OptimalConfig]:
    """Find optimal configurations for all combinations.
    
    This function orchestrates the search process across all model/task/config
    combinations, using ConfigSearchManager to handle individual searches.
    
    Args:
        config: Dictionary containing search configuration with keys:
            - models: List of model names
            - tasks: List of task names
            - optimal_configs: List of optimal configs (don't need search)
            - to_optimize_configs: List of configs to optimize
            - optimal_configs_map: Map of optimal configs
            - to_optimize_configs_map: Map of configs to optimize
            - num_samples: Number of samples per search
            - objective_function: Objective function name
            - search_max_new_tokens: Max new tokens for search
            - search_max_context_length: Max context length
            - search_max_requests: Max requests per trial
            - search_timeout: Timeout per trial
        actors_per_gpu: Number of actors per GPU for resource allocation
        
    Returns:
        Dictionary mapping config keys to OptimalConfig objects
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH")
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
        target: int = int(config['objective_function'].split('_')[1])
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
    
    manager: ConfigSearchManager = ConfigSearchManager(config)
    optimal_configs: Dict[str, OptimalConfig] = {}
    
    total: int = len(config["models"]) * len(config["tasks"]) * len(config["to_optimize_configs"]) + len(config["models"]) * len(config["tasks"]) * len(config["optimal_configs"])
    current: int = 0
    
    for model in config["models"]:
        print(f"\nModel: {model}")
        print("-" * 60)
        
        for task in config["tasks"]:
            for masker_name, (masker_classes, full_config) in config["to_optimize_configs_map"].items():
                current += 1
                key: str = f"{model}_{task}_{masker_name}".replace("/", "_")
                
                print(f"\n[{current}/{total}] Task: {task} | Config: {masker_name}")
                optimal: OptimalConfig = manager.search_optimal_config(
                    model, task, masker_name, masker_classes, full_config, actors_per_gpu
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
    print(f"Search complete. Found {len(optimal_configs)} optimal configurations.")
    print(f"Configs saved to: {manager.results_dir}")
    print(f"Run identifier: {manager.timestamp}")
    print(f"{'='*80}")
    
    return optimal_configs

def main() -> None:
    """Hyperparameter search for sparse attention methods.
    
    All configuration is loaded from RUN_CONFIG.py. Modify that file to change
    search parameters instead of passing command-line arguments.
    """
    # Validate objective function
    if OBJECTIVE not in OBJECTIVE_FUNCTIONS:
        raise ValueError(f"Invalid objective function '{OBJECTIVE}'. Choose from: {list(OBJECTIVE_FUNCTIONS.keys())}")
    
    config: Dict[str, Any] = get_run_configuration()
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, 
                runtime_env={"working_dir": str(root_path)})
    
    optimal_configs: Dict[str, OptimalConfig] = run_search(config, ACTORS_PER_GPU)
    ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
