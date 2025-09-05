#!/usr/bin/env python3
"""
Ray-based parallel benchmark runner with efficient resource management.

This implementation uses Ray for:
- Distributed execution with automatic resource management
- Efficient model caching through Ray actors
- Built-in fault tolerance and progress tracking
- Optimal task scheduling to minimize model loading

Usage:
    python benchmark/raytune/run_ray_benchmarks.py --config-run run_20250818_203531
    python benchmark/raytune/run_ray_benchmarks.py --config-run run_20250818_203531 --resume
"""

import argparse
import json
import logging
import os
import sys
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import traceback

# Path setup
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])

import ray
from ray.util.queue import Queue as RayQueue
from ray.util.actor_pool import ActorPool

from benchmark.executor_config import AdapterConfig
from benchmark.benchmark_registry import create_benchmark_instance
from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

# Import all masker configs
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import *
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import *


@dataclass
class BenchmarkTask:
    """Single benchmark task to execute."""
    task_id: str
    model_name: str
    task_name: str
    masker_name: str
    sparse_config: Optional[Dict]  # JSON-serializable config
    result_dir: str
    generation_kwargs: Dict[str, Any]
    request_kwargs: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Result from a benchmark execution."""
    task_id: str
    success: bool
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    gpu_id: Optional[int] = None
    model_load_time: float = 0.0


@ray.remote(num_gpus=1)
class GPUBenchmarkActor:
    """Ray actor that runs benchmarks on a specific GPU with fresh model initialization for each task."""
    
    def __init__(self, actor_id: int, adapter_config: Dict):
        self.actor_id = actor_id
        self.adapter_config = adapter_config
        
        # Ray sets CUDA_VISIBLE_DEVICES for us, so GPU 0 is always the correct device
        self.gpu_id = 0  # Always use device 0 in the actor's visible GPU space
        torch.cuda.set_device(self.gpu_id)
        
        # Get actual GPU info for logging
        gpu_name = torch.cuda.get_device_name(self.gpu_id)
        logging.info(f"Actor {actor_id} initialized on GPU {gpu_name}")
    
    def _reconstruct_sparse_config(self, config_data: Optional[Dict]) -> Optional[ResearchAttentionConfig]:
        """Reconstruct ResearchAttentionConfig from JSON data."""
        if not config_data or not config_data.get("masker_configs"):
            return None
        
        config_class_map = {
            "LocalMaskerConfig": LocalMaskerConfig,
            "SinkMaskerConfig": SinkMaskerConfig,
            "OracleTopKConfig": OracleTopKConfig,
            "OracleTopPMaskerConfig": OracleTopPMaskerConfig,
            "HashAttentionTopKMaskerConfig": HashAttentionTopKMaskerConfig,
            "AdaptiveSamplingMaskerConfig": AdaptiveSamplingMaskerConfig,
            "RandomSamplingMaskerConfig": RandomSamplingMaskerConfig,
            "MagicPigConfig": MagicPigConfig,
        }
        
        masker_configs = []
        for masker_data in config_data["masker_configs"]:
            config_class = config_class_map.get(masker_data["type"])
            if config_class:
                try:
                    params = masker_data.get("params", {})
                    masker_configs.append(config_class(**params))
                except Exception as e:
                    logging.warning(f"Failed to create {masker_data['type']}: {e}")
        
        return ResearchAttentionConfig(masker_configs=masker_configs) if masker_configs else None
    
    def _create_fresh_model(self, model_name: str, sparse_config: Optional[Dict], 
                          masker_name: str, task_name: str) -> Tuple[ModelAdapterHF, float]:
        """Create a fresh model from scratch for each task.
        
        This ensures no state leakage between tasks with different sparse configs.
        Returns (model, load_time).
        """
        logging.info(f"Actor {self.actor_id}: Creating fresh model for {task_name} with {masker_name}")
        
        # Clear any GPU cache before loading
        torch.cuda.empty_cache()
        
        start_time = time.time()
        
        # Reconstruct sparse config
        sparse_attention_config = self._reconstruct_sparse_config(sparse_config)
        
        # Create completely fresh model instance
        adapter = ModelAdapterHF(
            model_name=model_name,
            sparse_attention_config=sparse_attention_config,
            model_kwargs=self.adapter_config["model_kwargs"],
            tokenizer_kwargs=self.adapter_config["tokenizer_kwargs"]
        )
        
        load_time = time.time() - start_time
        logging.info(f"Actor {self.actor_id}: Model created in {load_time:.1f}s")
        
        return adapter, load_time
    
    def run_benchmark(self, task: BenchmarkTask) -> BenchmarkResult:
        """Execute a single benchmark task."""
        total_start = time.time()
        
        adapter = None
        try:
            # Create fresh model for this task
            adapter, model_load_time = self._create_fresh_model(
                task.model_name, task.sparse_config, task.masker_name, task.task_name
            )
            
            # Parse benchmark info
            benchmark_name, subset = (task.task_name.split("/", 1) 
                                    if "/" in task.task_name 
                                    else (task.task_name, None))
            
            # Create benchmark instance
            benchmark = create_benchmark_instance(
                benchmark_name=benchmark_name,
                subsets=[subset] if subset else None
            )
            
            # Setup result directory
            Path(task.result_dir).mkdir(parents=True, exist_ok=True)
            
            # Check if already completed
            metrics_file = Path(task.result_dir) / "metrics.json"
            if metrics_file.exists():
                logging.info(f"Actor {self.actor_id}: Skipping completed {task.task_id}")
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return BenchmarkResult(
                    task_id=task.task_id,
                    success=True,
                    metrics=metrics,
                    execution_time=0.0,
                    gpu_id=None,
                    model_load_time=0.0
                )
            
            # Setup micro metrics
            metric_logger = MicroMetricLogger()
            metric_logger.configure_logging(
                log_path=task.result_dir,
                enabled_metrics=["research_attention_density", "research_attention_output_error"],
                max_records=5000,
                sampling_factor=0.1
            )
            
            # Run benchmark
            benchmark_start = time.time()
            logging.info(f"Actor {self.actor_id}: Running {task.task_id}")
            
            metrics = benchmark.run_benchmark(
                adapter=adapter,
                result_dir=task.result_dir,
                generation_kwargs=task.generation_kwargs,
                request_kwargs=task.request_kwargs
            )
            
            metric_logger.flush()
            
            execution_time = time.time() - total_start
            
            return BenchmarkResult(
                task_id=task.task_id,
                success=True,
                metrics=metrics,
                execution_time=execution_time,
                gpu_id=None,
                model_load_time=model_load_time
            )
            
        except Exception as e:
            logging.error(f"Actor {self.actor_id}: Task {task.task_id} failed: {e}")
            traceback.print_exc()
            
            return BenchmarkResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - total_start,
                gpu_id=None
            )
        
        finally:
            # Always clean up the model to ensure no state leakage
            if adapter is not None:
                logging.info(f"Actor {self.actor_id}: Cleaning up model for {task.task_id}")
                try:
                    del adapter
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.warning(f"Actor {self.actor_id}: Cleanup error: {e}")
    
    def get_stats(self) -> Dict:
        """Return actor statistics."""
        return {
            "actor_id": self.actor_id,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "status": "active"
        }
    
    def cleanup(self):
        """Clean up resources."""
        logging.info(f"Actor {self.actor_id}: Final cleanup")
        torch.cuda.empty_cache()


def prepare_tasks(tasks: List[BenchmarkTask]) -> List[BenchmarkTask]:
    """Prepare tasks for execution.
    
    Since each task has unique optimized parameters from Phase 1,
    every task requires fresh model initialization.
    """
    return tasks


def serialize_sparse_config(config: Optional[ResearchAttentionConfig]) -> Optional[Dict]:
    """Convert ResearchAttentionConfig to JSON-serializable format."""
    if config is None:
        return None
    
    masker_configs = []
    for masker in config.masker_configs:
        masker_dict = {
            "type": type(masker).__name__,
            "params": {}
        }
        # Extract all public attributes
        for attr in dir(masker):
            if not attr.startswith("_"):
                value = getattr(masker, attr)
                if isinstance(value, (int, float, str, bool, type(None))):
                    masker_dict["params"][attr] = value
        masker_configs.append(masker_dict)
    
    return {
        "type": "ResearchAttentionConfig",
        "masker_configs": masker_configs
    }


def load_optimal_configs(config_dir: Path) -> List[BenchmarkTask]:
    """Load optimal configurations and create benchmark tasks."""
    tasks = []
    
    for config_file in config_dir.glob("*.json"):
        if config_file.name.endswith(("_trials.json", "_analysis.csv")):
            continue
        
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
            
            task_id = f"{data['model']}_{data['task']}_{data['masker_name']}".replace("/", "_")
            
            task = BenchmarkTask(
                task_id=task_id,
                model_name=data["model"],
                task_name=data["task"],
                masker_name=data["masker_name"],
                sparse_config=data.get("sparse_config"),
                result_dir="",  # Will be set later
                generation_kwargs={},  # Will be set later
                request_kwargs={}  # Will be set later
            )
            tasks.append(task)
            
        except Exception as e:
            logging.warning(f"Failed to load {config_file}: {e}")
    
    return tasks


@ray.remote
def progress_reporter(total_tasks: int, result_queue: RayQueue) -> None:
    """Ray task that reports progress from result queue."""
    completed = 0
    failed = 0
    start_time = time.time()
    total_model_load_time = 0.0
    
    while completed + failed < total_tasks:
        try:
            result = result_queue.get(timeout=10)
            
            if result.success:
                completed += 1
                total_model_load_time += result.model_load_time
                
                print(f"[{completed + failed}/{total_tasks}] ✓ {result.task_id} "
                      f"({result.execution_time:.1f}s, model load: {result.model_load_time:.1f}s)")
            else:
                failed += 1
                print(f"[{completed + failed}/{total_tasks}] ✗ {result.task_id} - {result.error}")
            
            # Print progress stats every 10 tasks
            if (completed + failed) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (completed + failed) / elapsed
                eta = (total_tasks - completed - failed) / rate if rate > 0 else 0
                avg_load_time = total_model_load_time / max(1, completed)
                print(f"\n--- Progress: {completed + failed}/{total_tasks} "
                      f"({rate:.2f} tasks/s, ETA: {eta/60:.1f} min) ---")
                print(f"--- Avg model load time: {avg_load_time:.1f}s ---\n")
                
        except Exception:
            continue
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed: {completed}, Failed: {failed}")
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"Total model load time: {total_model_load_time/60:.1f} minutes")
    print(f"Throughput: {completed/total_time*3600:.1f} tasks/hour")


def main():
    parser = argparse.ArgumentParser(description="Ray-based parallel benchmark runner")
    parser.add_argument("--config-run", type=str, required=True,
                       help="Config run directory name")
    parser.add_argument("--optimal-configs-dir", default="./optimal_configs")
    parser.add_argument("--benchmark-results-dir", default="./benchmark_vt_full_10pct", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1000, required=True)
    parser.add_argument("--max-context-length", type=int, default=100000, required=True)
    parser.add_argument("--max-requests", type=int, default=1000, required=True)
    parser.add_argument("--num-actors", type=int, default=None,
                       help="Number of Ray actors (default: number of GPUs)")
    parser.add_argument("--actors-per-gpu", type=int, default=None,
                       help="Number of actors per GPU for better utilization (overrides --num-actors)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing results")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running benchmarks")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode - run only 2-4 benchmarks to test functionality")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print(f"\n{'='*80}")
    print(f"RAY BENCHMARK RUNNER")
    print(f"{'='*80}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Get GPU info
    num_gpus = int(ray.available_resources().get("GPU", 0))
    if num_gpus == 0:
        print("Error: No GPUs available")
        sys.exit(1)
    
    # Determine number of actors
    if args.actors_per_gpu:
        num_actors = num_gpus * args.actors_per_gpu
        print(f"Creating {args.actors_per_gpu} actors per GPU for maximum utilization")
    elif args.num_actors:
        num_actors = args.num_actors
    else:
        # Default to number of GPUs
        num_actors = num_gpus
        # In debug mode, still use all GPUs unless specified
        if args.debug:
            print(f"Debug mode: using all {num_actors} GPUs for maximum utilization")
        
    print(f"Ray cluster: {ray.available_resources()}")
    print(f"Using {num_actors} actors on {num_gpus} GPUs")
    
    # Load configurations
    config_dir = Path(args.optimal_configs_dir) / args.config_run
    if not config_dir.exists():
        print(f"Error: Config directory {config_dir} not found")
        sys.exit(1)
    
    print(f"\nLoading configurations from {config_dir}...")
    tasks = load_optimal_configs(config_dir)
    print(f"Loaded {len(tasks)} configurations")
    
    # Debug mode adjustments
    if args.debug:
        print("\n⚠️  DEBUG MODE ENABLED ⚠️")
        print("  - Will run only a subset of benchmarks")
        print("  - Using reduced parameters for faster testing")
        
        # Filter tasks for debug mode - take diverse samples
        debug_tasks = []
        # Get one dense config
        dense_tasks = [t for t in tasks if t.masker_name == "dense"]
        if dense_tasks:
            debug_tasks.append(dense_tasks[0])
        
        # Get 2-3 sparse configs with different maskers
        sparse_tasks = [t for t in tasks if t.masker_name != "dense"]
        seen_maskers = set()
        for task in sparse_tasks:
            if task.masker_name not in seen_maskers and len(debug_tasks) < 4:
                debug_tasks.append(task)
                seen_maskers.add(task.masker_name)
        
        tasks = debug_tasks
        print(f"  - Selected {len(tasks)} tasks for debug run:")
        for task in tasks:
            print(f"    * {task.model_name} / {task.masker_name} / {task.task_name}")
        
        # Override parameters for faster execution
        generation_kwargs = {
            "max_new_tokens": 20,  # Much smaller for debug
            "do_sample": False,
        }
        
        request_kwargs = {
            "max_context_length": 4096,  # Smaller context
            "max_requests": 2,  # Just 2 requests per benchmark
        }
        
        print(f"\n  Debug parameters:")
        print(f"    - max_new_tokens: 20 (vs {args.max_new_tokens})")
        print(f"    - max_context_length: 4096 (vs {args.max_context_length})")
        print(f"    - max_requests: 2 (vs {args.max_requests})")
        
    else:
        # Normal mode - use full parameters
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        }
        
        request_kwargs = {
            "max_context_length": args.max_context_length,
            "max_requests": args.max_requests
        }
    
    # Update tasks with full configuration
    for task in tasks:
        task.result_dir = os.path.join(
            args.benchmark_results_dir,
            task.model_name.replace("/", "_"),
            task.masker_name,
            task.task_name.replace("/", "_")
        )
        task.generation_kwargs = generation_kwargs
        task.request_kwargs = request_kwargs
    
    # Prepare tasks
    print("\nPreparing tasks...")
    tasks = prepare_tasks(tasks)
    
    # Dry run mode - show what would be executed
    if args.dry_run:
        print(f"\n{'='*80}")
        if args.debug:
            print("DRY RUN MODE (DEBUG) - No benchmarks will be executed")
        else:
            print("DRY RUN MODE - No benchmarks will be executed")
        print(f"{'='*80}")
        
        # Group tasks by model and masker for analysis
        task_groups = defaultdict(list)
        for task in tasks:
            key = (task.model_name, task.masker_name)
            task_groups[key].append(task)
        
        print(f"\nTask Summary:")
        print(f"  Total tasks: {len(tasks)}")
        print(f"  Unique model/masker combinations: {len(task_groups)}")
        print(f"  Actors to be created: {num_actors}")
        
        # Check existing results
        completed_count = 0
        for task in tasks:
            metrics_file = Path(task.result_dir) / "metrics.json"
            if metrics_file.exists() and not args.resume:
                completed_count += 1
        
        if completed_count > 0:
            print(f"  Already completed: {completed_count} (would be skipped)")
            print(f"  To be executed: {len(tasks) - completed_count}")
        
        print(f"\nTask Groups (optimized order):")
        print("-" * 80)
        
        for i, ((model, masker), group_tasks) in enumerate(task_groups.items()):
            print(f"\n{i+1}. {model} + {masker}")
            print(f"   Tasks ({len(group_tasks)}):")
            for task in group_tasks[:3]:  # Show first 3
                status = "✓" if (Path(task.result_dir) / "metrics.json").exists() else "○"
                print(f"     {status} {task.task_name}")
            if len(group_tasks) > 3:
                print(f"     ... and {len(group_tasks) - 3} more")
        
        # Estimate resource usage
        print(f"\nResource Estimates:")
        print("-" * 80)
        model_sizes = {
            "Llama-3.1-8B": 16,  # GB in bfloat16
            "Phi-4-mini": 7,
            # Add more model estimates
        }
        
        est_model_size = 16  # Default estimate
        for model_key in model_sizes:
            if model_key in tasks[0].model_name:
                est_model_size = model_sizes[model_key]
                break
        
        print(f"  Estimated model size: ~{est_model_size} GB per model")
        print(f"  Total unique model configurations: {len(tasks)}")
        print(f"  GPU memory required per actor: ~{est_model_size} GB")
        
        # Execution plan
        print(f"\nExecution Plan:")
        print("-" * 80)
        print(f"  1. Initialize Ray with {num_actors} GPU actors")
        print(f"  2. Each actor processes tasks independently")
        print(f"  3. Fresh model initialization for each task:")
        print(f"     - Each task has unique optimized parameters from Phase 1")
        print(f"     - Total model loads: {len(tasks)} (one per task)")
        
        # Show example of different configs
        if len(tasks) >= 2:
            print(f"\nExample configurations showing parameter differences:")
            
            # Find tasks with same masker but different parameters
            masker_groups = defaultdict(list)
            for task in tasks:
                masker_groups[task.masker_name].append(task)
            
            # Show first group with multiple tasks
            for masker_name, group_tasks in masker_groups.items():
                if len(group_tasks) >= 2 and masker_name != "dense":
                    for i, task in enumerate(group_tasks[:2]):
                        print(f"\n  {task.masker_name} for {task.task_name}:")
                        if task.sparse_config and task.sparse_config.get("masker_configs"):
                            for masker in task.sparse_config["masker_configs"][:2]:
                                params = masker.get("params", {})
                                param_str = ", ".join([f"{k}={v}" for k, v in sorted(params.items())[:3]])
                                print(f"    - {masker['type']}: {param_str}...")
                    break
        
        print(f"\nGeneration Configuration:")
        print(f"  max_new_tokens: {args.max_new_tokens}")
        print(f"  max_context_length: {args.max_context_length}")
        print(f"  max_requests: {args.max_requests}")
        
        print(f"\nResults will be saved to:")
        print(f"  {args.benchmark_results_dir}/")
        print(f"    └── <model_name>/")
        print(f"        └── <masker_name>/")
        print(f"            └── <task_name>/")
        print(f"                ├── raw_results.csv")
        print(f"                ├── metrics.json")
        print(f"                └── micro_metrics.jsonl")
        
        print(f"\n{'='*80}")
        print("Dry run complete. Remove --dry-run to execute benchmarks.")
        print(f"{'='*80}")
        return
    
    # Create adapter config
    adapter_config = {
        "adapter_name": "huggingface",
        "model_kwargs": {"torch_dtype": torch.bfloat16},
        "tokenizer_kwargs": {"padding_side": "left"}
    }
    
    # Create Ray actors
    print(f"\nCreating {num_actors} Ray actors...")
    actors = []
    
    # Calculate GPU resources per actor
    if args.actors_per_gpu and args.actors_per_gpu > 1:
        # When multiple actors per GPU, each gets a fraction
        gpu_per_actor = 1.0 / args.actors_per_gpu
        print(f"Each actor will use {gpu_per_actor:.2f} GPU resources")
        
        # Create actors with fractional GPU resources
        for i in range(num_actors):
            # Have to use options to set fractional GPU
            actor = GPUBenchmarkActor.options(num_gpus=gpu_per_actor).remote(i, adapter_config)
            actors.append(actor)
    else:
        # Standard: one actor per GPU
        for i in range(num_actors):
            actor = GPUBenchmarkActor.remote(i, adapter_config)
            actors.append(actor)
    
    # Create result queue and progress reporter
    result_queue = RayQueue(maxsize=len(tasks))
    progress_task = progress_reporter.remote(len(tasks), result_queue)
    
    # Create actor pool for load balancing
    pool = ActorPool(actors)
    
    # Submit all tasks
    print(f"\nSubmitting {len(tasks)} tasks...")
    print("-" * 80)
    
    start_time = time.time()
    
    # Submit tasks to actor pool
    # ActorPool.submit expects (fn, value) where fn(actor, value) is called
    for task in tasks:
        pool.submit(lambda actor, task: actor.run_benchmark.remote(task), task)
    
    # Collect results
    while pool.has_next():
        result = pool.get_next()
        result_queue.put(result)
    
    # Wait for progress reporter
    ray.get(progress_task)
    
    # Get actor statistics
    print("\nActor statistics:")
    for actor in actors:
        stats = ray.get(actor.get_stats.remote())
        print(f"  Actor {stats['actor_id']} ({stats['gpu_name']}): {stats['status']}")
    
    # Cleanup
    print("\nCleaning up...")
    for actor in actors:
        ray.get(actor.cleanup.remote())
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {args.benchmark_results_dir}")
    print(f"{'='*80}")
    
    ray.shutdown()


if __name__ == "__main__":
    main()
