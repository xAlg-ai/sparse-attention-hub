#!/usr/bin/env python3
"""
Full End-to-End Benchmark Execution and Optimizer.

This script performs a robust, two-stage process for each combination of
model, benchmark, and sparse attention configuration:
1.  **Search**: It uses Ray Tune to run a hyperparameter search with lightweight
    settings to quickly discover the optimal parameters.
2.  **Validate**: It takes the single best configuration found during the search
    and runs a final, thorough benchmark with it to get a definitive score.

The entire process is designed to be modular and easy to extend. To add new
models, benchmarks, or masker presets, see the `get_run_configurations` function.
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{current_dir}:{root_path}"

# --- Core Imports ---
import torch
from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import AdapterConfig, BenchmarkConfig, BenchmarkResult
from optimizer_factory import create_optimizer

# --- Masker Config Imports ---
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.magic_pig import (
    MagicPigConfig,
)

# --- Ray Tune Imports ---
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.stopper import TrialPlateauStopper
except ImportError:
    print("Error: Ray Tune is required. Install with: pip install \"ray[tune]\" hyperopt")
    sys.exit(1)


class ComprehensiveBenchmarkRunner:
    """Runs a benchmark for a model and sparse attention config, returning a score."""

    def __init__(self, config: dict, verbose: bool = False):
        self.config = config
        self.executor = BenchmarkExecutor(
            gpu_ids=config["gpu_ids"],
            max_concurrent_runs=config["max_concurrent_runs"],
            base_result_dir=config["result_dir"],
            enable_resumability=True,
            required_result_files=["raw_results.csv"],
            timeout_per_benchmark=config["timeout_per_benchmark"],
            verbose=verbose,
        )
        self.adapter_config = AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.generation_kwargs = {"max_new_tokens": config["max_new_tokens"], "do_sample": False}
        self.request_kwargs = {
            "max_context_length": config["max_context_length"],
            "max_requests": config["max_requests"],
        }
        self.results_cache = {}

    def _extract_micro_metrics(self, result_dir: Path) -> dict:
        import math
        micro_metrics_file = result_dir / "micro_metrics.jsonl"
        if not micro_metrics_file.exists():
            raise FileNotFoundError(f"micro_metrics.jsonl not found in {result_dir}")

        errors, densities = [], []
        with open(micro_metrics_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    metric, value = entry.get("metric"), entry.get("value")
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        if metric == "research_attention_output_error": errors.append(float(value))
                        elif metric == "research_attention_density": densities.append(float(value))
                except (json.JSONDecodeError, ValueError, TypeError): continue
        return {"attention_error": sum(errors) / len(errors) if errors else 1.0, "density": sum(densities) / len(densities) if densities else 1.0}

    def __call__(self, attention_config, task_name: str, model_name: str) -> float:
        config_key = f"{model_name}_{task_name}_{hash(str(attention_config))}"
        if config_key in self.results_cache: return self.results_cache[config_key]

        try:
            if "/" in task_name:
                benchmark_name, subset_name = task_name.split("/", 1)
            else:
                benchmark_name, subset_name = task_name, None

            benchmark_config = BenchmarkConfig(
                benchmark_name=benchmark_name, 
                subsets=[subset_name] if subset_name else None
            )
            
            results = self.executor.run_benchmark_matrix(
                model_names=[model_name],
                sparse_attention_configs=[("optimized", attention_config)],
                benchmark_configs=[benchmark_config],
                adapter_config=self.adapter_config,
                generation_kwargs=self.generation_kwargs,
                request_kwargs=self.request_kwargs,
            )

            if results.progress.completed_stubs > 0 and hasattr(results, "individual_results"):
                completed = [r for r in results.individual_results if isinstance(r, BenchmarkResult)]
                if completed:
                    result_dir = Path(completed[0].stub.result_dir)
                    metrics = self._extract_micro_metrics(result_dir)
                    error, density = metrics["attention_error"], metrics["density"]
                    score = error + 0.1 * density + (5.0 if density > 0.5 else 0.0)
                    self.results_cache[config_key] = score
                    return score
        except Exception as e:
            print(f"    ✗ Error in benchmark runner: {e}")
            traceback.print_exc()

        print(f"    Warning: Could not compute a valid score for {model_name} on {task_name}. Returning penalty.")
        self.results_cache[config_key] = 5.0
        return 5.0

def get_run_configurations(args: argparse.Namespace) -> dict:
    """Defines the complete configuration for the optimization run."""
    num_gpus = torch.cuda.device_count()

    masker_config_presets = {
        "local_sink": [SinkMaskerConfig, LocalMaskerConfig],
        "sink_local_magic_pig": [SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig],
    }
    test_suites = {"default": ["local_sink", "sink_local_magic_pig"], "debug": ["sink_local_magic_pig"]}

    # --- Decouple Search and Validation Parameters ---
    if args.debug:
        # Use smaller, faster settings for the search phase in debug mode
        search_params = {
            "timeout_per_benchmark": 300, "max_new_tokens": 10,
            "max_context_length": 4096, "max_requests": 2,
        }
        # Use slightly more thorough settings for debug validation
        validation_params = {
            "timeout_per_benchmark": 600, "max_new_tokens": 30,
            "max_context_length": 16384, "max_requests": 5,
        }
        base_config = {
            "models": ["meta-llama/Llama-3.1-8B-Instruct"], "benchmarks": ["loogle/shortdep_qa"],
            "masker_presets": {p: masker_config_presets[p] for p in test_suites["debug"]},
            "num_samples": 8,
        }
    else:
        # For production, use specific flags for each stage
        search_params = {
            "timeout_per_benchmark": args.search_timeout, "max_new_tokens": args.search_max_new_tokens,
            "max_context_length": args.search_max_context_length, "max_requests": args.search_max_requests,
        }
        validation_params = {
            "timeout_per_benchmark": args.validation_timeout, "max_new_tokens": args.validation_max_new_tokens,
            "max_context_length": args.validation_max_context_length, "max_requests": args.validation_max_requests,
        }
        base_config = {
            "models": ["meta-llama/Llama-3.2-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
            "benchmarks": ["loogle/shortdep_qa", "loogle/longdep_qa"],
            "masker_presets": {p: masker_config_presets[p] for p in test_suites["default"]},
            "num_samples": args.num_samples,
        }

    # Combine into a final, structured configuration
    return {
        **base_config,
        "search_params": search_params,
        "validation_params": validation_params,
        "gpu_ids": list(range(num_gpus)),
        "max_concurrent_runs": num_gpus,
        "result_dir": "./search_results", # Base directory for the search phase
        "detailed_result_dir": "./validation_results", # Base directory for the validation phase
    }

def get_ray_tune_components(config: dict) -> dict:
    scheduler = ASHAScheduler(time_attr="training_iteration", max_t=20, grace_period=5, reduction_factor=2)
    search_alg = HyperOptSearch(metric="combined_score", mode="min", n_initial_points=max(1, config["num_samples"] // 4))
    stopper = TrialPlateauStopper(metric="combined_score", std=0.005, num_results=5, grace_period=8, mode="min")
    return {"scheduler": scheduler, "search_alg": search_alg, "stop": stopper}

def create_optimization_objective(config: dict, model_name: str, task_name: str, optimizer):
    """Creates the objective function that Ray Tune will execute for each trial."""
    def objective(trial_config: dict):
        # The worker always uses the lighter search parameters for speed
        worker_config = {**config, **config["search_params"]}
        worker_config["gpu_ids"] = [0]
        worker_config["max_concurrent_runs"] = 1
        
        benchmark_runner = ComprehensiveBenchmarkRunner(worker_config)
        attention_config = optimizer.create_config_from_params(trial_config)
        score = benchmark_runner(attention_config, task_name, model_name)
        return {"combined_score": score}
    return objective

def run_optimization_and_validation(model_name: str, benchmark_task: str, preset_name: str, masker_configs: list, config: dict) -> dict:
    """Runs the two-stage Search-then-Validate process for one combination."""
    print(f"\n--- Running: {model_name} | {benchmark_task} | {preset_name} ---")
    
    # Stage 1: Search using the lighter 'search_params'
    print("  1. Searching for optimal configuration...")
    try:
        optimizer = create_optimizer(masker_configs)
        objective = create_optimization_objective(config, model_name, benchmark_task, optimizer)
        tune_components = get_ray_tune_components(config)
        sanitized_task_name = benchmark_task.replace('/', '_')
        
        analysis = tune.run(
            objective, config=optimizer.create_search_space(benchmark_task),
            num_samples=config["num_samples"], metric="combined_score", mode="min",
            resources_per_trial={"CPU": 1, "GPU": 1.0},
            name=f"opt_{model_name.split('/')[-1]}_{sanitized_task_name}_{preset_name}",
            storage_path=config["storage_path"], verbose=1, resume=False,
            max_concurrent_trials=config["max_concurrent_runs"], **tune_components
        )
        best_trial = analysis.get_best_trial("combined_score", "min", "last")
        best_config_obj = optimizer.create_config_from_params(best_trial.config)
        best_search_score = best_trial.last_result['combined_score']
        print(f"     ✓ Best search score: {best_search_score:.4f}")
    except Exception as e:
        print(f"     ✗ Search stage failed: {e}"); traceback.print_exc()
        return {"error": f"Search failed: {e}"}

    # Stage 2: Validate using the more thorough 'validation_params'
    print("  2. Validating the best configuration...")
    try:
        # Create a new config for validation by merging base and validation params
        validation_config = {**config, **config["validation_params"]}
        validation_config["result_dir"] = os.path.join(config["detailed_result_dir"], preset_name)
        
        validator = ComprehensiveBenchmarkRunner(validation_config, verbose=True)
        start_time = time.time()
        print(f"    Running validation benchmark: {model_name} on {benchmark_task}...")
        final_score = validator(best_config_obj, benchmark_task, model_name)
        runtime = time.time() - start_time
        print(f"    Validation benchmark completed in {runtime:.1f}s")
        print(f"     ✓ Final validation score: {final_score:.4f}")
    except Exception as e:
        print(f"     ✗ Validation stage failed: {e}"); traceback.print_exc()
        return {"error": f"Validation failed: {e}"}

    return {
        "best_search_score": best_search_score,
        "final_validation_score": final_score,
        "best_config": best_config_obj,
        "best_params": best_trial.config,
        "num_trials": len(analysis.trials),
    }

def run_optimization_matrix(config: dict) -> tuple[dict, str]:
    print("Starting Full Benchmark Optimization and Validation Matrix"); print("=" * 80)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_path = os.path.abspath(f"./ray_results_{timestamp}")
    config["storage_path"] = storage_path
    print(f"Ray Tune results will be saved to: {storage_path}")

    all_results = {}
    for model_name in config["models"]:
        all_results[model_name] = {}
        print(f"\nModel: {model_name}"); print("-" * 60)
        for benchmark_task in config["benchmarks"]:
            all_results[model_name][benchmark_task] = {}
            for preset_name, masker_configs in config["masker_presets"].items():
                combo_result = run_optimization_and_validation(model_name, benchmark_task, preset_name, masker_configs, config)
                all_results[model_name][benchmark_task][preset_name] = combo_result
    return all_results, storage_path

def print_summary(results: dict):
    print("\n" + "=" * 80); print("--- FINAL BENCHMARK SUMMARY ---"); print("=" * 80)
    best_overall_score, best_overall_config = float("inf"), {}
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}"); print("-" * 70)
        for benchmark_task, task_results in model_results.items():
            print(f"\n  Benchmark: {benchmark_task}")
            for masker_preset, result in task_results.items():
                if "error" in result:
                    print(f"    {masker_preset:25s}: FAILED ({result['error']})"); continue
                score = result.get("final_validation_score", float("inf"))
                search_score = result.get("best_search_score", float("inf"))
                print(f"    {masker_preset:25s}: {score:.4f} (Search score: {search_score:.4f})")
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_config = {"model": model_name, "benchmark": benchmark_task, "masker": masker_preset, "score": score, "params": result.get("best_params")}
    print("\n" + "--- Best Overall Configuration ---")
    if best_overall_config:
        for key, value in best_overall_config.items(): print(f"  {key.capitalize():<12}: {value}")
    else: print("  No successful runs completed.")
    print("-" * 32)

def define_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full benchmark optimization and validation runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General arguments
    parser.add_argument("--debug", action="store_true", help="Run a quick test configuration, ignoring other flags.")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of Ray Tune samples per optimization search.")

    # Search-specific arguments
    search_group = parser.add_argument_group('Search Parameters (for finding the best config)')
    search_group.add_argument("--search-timeout", type=int, default=900, help="Timeout for each search trial.")
    search_group.add_argument("--search-max-new-tokens", type=int, default=20, help="Max new tokens for search trials.")
    search_group.add_argument("--search-max-context-length", type=int, default=8192, help="Max context length for search trials.")
    search_group.add_argument("--search-max-requests", type=int, default=10, help="Max requests for search trials.")

    # Validation-specific arguments
    validation_group = parser.add_argument_group('Validation Parameters (for the final run with the best config)')
    validation_group.add_argument("--validation-timeout", type=int, default=1800, help="Timeout for the final validation run.")
    validation_group.add_argument("--validation-max-new-tokens", type=int, default=50, help="Max new tokens for the final validation run.")
    validation_group.add_argument("--validation-max-context-length", type=int, default=16384, help="Max context length for the final validation run.")
    validation_group.add_argument("--validation-max-requests", type=int, default=20, help="Max requests for the final validation run.")
    
    return parser.parse_args()

def main():
    args = define_cli_args()
    config = get_run_configurations(args)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, runtime_env={"working_dir": str(root_path)})

    mode = "Quick Test" if args.debug else "Full Production"
    print(f"Starting {mode} Optimization & Validation..."); print(f"Ray Version: {ray.__version__}, GPUs Available: {torch.cuda.device_count()}")
    start_time = time.time()
    try:
        results, storage_path = run_optimization_matrix(config)
        print_summary(results)
        print(f"\nDetailed validation results saved to: {config['detailed_result_dir']}")
        print(f"View optimization progress with: tensorboard --logdir {storage_path}")
        
        results_file = Path(storage_path) / "final_summary.json"
        def json_serializer(obj): return str(obj)
            
        print(f"Saving summary to: {results_file}")
        with open(results_file, "w") as f: json.dump(results, f, indent=2, default=json_serializer)
        print("Summary saved successfully.")
    except KeyboardInterrupt:
        print("\nWarning: Optimization interrupted by user.")
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {e}"); traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        print(f"\nTotal script time: {total_time / 3600:.2f} hours ({total_time:.0f} seconds)")
        ray.shutdown()
        print("Script finished.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()