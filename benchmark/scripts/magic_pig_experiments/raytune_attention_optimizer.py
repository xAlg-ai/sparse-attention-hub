#!/usr/bin/env python3
"""
Modular Ray Tune Optimizer Base Classes

This module provides extensible base classes for optimizing different sparse attention
mechanisms using Ray Tune. The design allows easy extension to new attention types
while maintaining consistent optimization workflows.

Usage:
    from raytune_attention_optimizer import AttentionOptimizerBase
    
    class MyCustomOptimizer(AttentionOptimizerBase):
        def get_search_space(self):
            # Define your search space
            pass
            
        def create_attention_config(self, params):
            # Create your attention config
            pass
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
from abc import ABC, abstractmethod

import torch

# Try to import required libraries
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune import CLIReporter
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

# Import project modules after path setup
try:
    from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
    from sparse_attention_hub.adapters import ModelAdapterHF
    from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
except ImportError as e:
    print(f"Error importing sparse attention modules: {e}")
    print("Make sure you're running from the correct directory and the sparse attention hub is installed")
    sys.exit(1)


class SystemMonitor:
    """Enhanced system monitoring for Ray Tune trials"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.is_cuda = device.type == 'cuda'
        self.gpu_handle = None
        
        if self.is_cuda:
            try:
                pynvml.nvmlInit()
                device_index = torch.cuda.current_device()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
            except Exception as e:
                print(f"Warning: GPU monitoring initialization failed: {e}")
                self.is_cuda = False

    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        stats = {'cpu_memory_mb': self.process.memory_info().rss / 1024**2}
        
        if self.is_cuda and self.gpu_handle:
            try:
                driver_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                stats.update({
                    'gpu_allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
                    'gpu_total_used_mb': driver_info.used / 1024**2,
                    'gpu_total_free_mb': driver_info.free / 1024**2
                })
            except Exception:
                pass
        
        return stats

    def start_capture(self) -> None:
        self.start_time = time.perf_counter()
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_event.record()

    def stop_capture(self) -> Dict[str, float]:
        end_time = time.perf_counter()
        metrics = {"wall_time_s": end_time - self.start_time}
        
        if self.is_cuda:
            try:
                self.end_event.record()
                torch.cuda.synchronize(self.device)
                metrics.update({
                    "gpu_runtime_ms": self.start_event.elapsed_time(self.end_event),
                    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2
                })
            except Exception:
                pass
        
        return metrics

    def cleanup(self):
        if self.is_cuda:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class AttentionOptimizerBase(ABC):
    """
    Abstract base class for sparse attention optimization using Ray Tune
    
    This class provides the common framework for optimizing any sparse attention
    mechanism. Subclasses need to implement the search space definition and
    configuration creation methods.
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 benchmark_class=None,
                 benchmark_tasks: List[str] = None,
                 max_requests: int = 2,
                 max_context_length: int = 16000,
                 device: Optional[torch.device] = None,
                 attention_type: str = "base",
                 results_base_dir: Optional[str] = None):
        
        self.model_name = model_name
        self.benchmark_tasks = benchmark_tasks or ["shortdep_qa"]
        self.max_requests = max_requests
        self.max_context_length = max_context_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention_type = attention_type
        
        # Set results base directory (default to current working directory)
        if results_base_dir is None:
            self.results_base_dir = Path.cwd()
        else:
            self.results_base_dir = Path(results_base_dir).resolve()
        
        # Initialize benchmark - must be provided by subclass
        if benchmark_class is None:
            # Default to Loogle if available
            try:
                from benchmark import Loogle
                self.benchmark = Loogle(self.benchmark_tasks)
            except ImportError:
                raise ValueError("Must provide benchmark_class or have Loogle available")
        else:
            self.benchmark = benchmark_class(self.benchmark_tasks)
        
        colorama.init(autoreset=True)
        
    def generate_experiment_name(self) -> str:
        """
        Generate a unique experiment name based on the search space configuration
        This prevents running the same search space multiple times
        
        Returns:
            Unique experiment name based on search space hash
        """
        import hashlib
        
        search_space = self.get_search_space()
        
        # Create a deterministic string representation of the search space
        search_space_str = str(sorted(search_space.items()))
        
        # Generate a short hash of the search space
        search_hash = hashlib.md5(search_space_str.encode()).hexdigest()[:8]
        
        # Include key search space characteristics in the name
        space_summary = []
        for param_name, param_spec in search_space.items():
            if hasattr(param_spec, '_spec'):
                spec = param_spec._spec
                if spec['type'] == 'choice':
                    space_summary.append(f"{param_name}x{len(spec['categories'])}")
                elif spec['type'] in ['uniform', 'loguniform']:
                    space_summary.append(f"{param_name}_{spec['type']}")
                elif spec['type'] == 'quniform':
                    count = int((spec['upper'] - spec['lower']) / spec.get('q', 1)) + 1
                    space_summary.append(f"{param_name}x{count}")
        
        # Create experiment name: attention_type_summary_hash
        summary_str = "_".join(space_summary[:3])  # Limit to first 3 params to keep name reasonable
        experiment_name = f"{self.attention_type}_{summary_str}_{search_hash}"
        
        return experiment_name
        
    def calculate_search_space_size(self) -> int:
        """
        Calculate the total size of the search space
        
        Returns:
            Total number of possible configurations in the search space
        """
        search_space = self.get_search_space()
        total_size = 1
        
        for param_name, param_spec in search_space.items():
            if hasattr(param_spec, '_spec'):
                spec = param_spec._spec
                if spec['type'] == 'choice':
                    # For choice parameters, count the number of choices
                    total_size *= len(spec['categories'])
                elif spec['type'] in ['uniform', 'loguniform', 'quniform']:
                    # For continuous parameters, this is theoretically infinite
                    # but we'll estimate based on practical resolution
                    if spec['type'] == 'quniform':
                        # For quantized uniform, we can calculate exact count
                        q = spec.get('q', 1)
                        count = int((spec['upper'] - spec['lower']) / q) + 1
                    else:
                        # For continuous uniform, estimate with reasonable resolution
                        count = 100  # Assume ~100 practical values
                    total_size *= count
                elif spec['type'] == 'randint':
                    # For random integers
                    total_size *= (spec['upper'] - spec['lower'])
            else:
                # Fallback: try to infer from the parameter object
                if hasattr(param_spec, 'categories'):
                    total_size *= len(param_spec.categories)
                else:
                    # Conservative estimate for unknown parameter types
                    total_size *= 10
        
        return total_size

    def print_search_space_info(self):
        """Print detailed information about the search space"""
        search_space = self.get_search_space()
        total_size = self.calculate_search_space_size()
        
        print(f"\n{Style.BRIGHT}{Fore.CYAN}🔍 Search Space Configuration:")
        print(f"{'-'*50}")
        
        for param_name, param_spec in search_space.items():
            if hasattr(param_spec, '_spec'):
                spec = param_spec._spec
                if spec['type'] == 'choice':
                    choices = spec['categories']
                    print(f"  {Fore.YELLOW}{param_name}: {Fore.WHITE}choice({len(choices)} options) - {choices}")
                elif spec['type'] == 'uniform':
                    print(f"  {Fore.YELLOW}{param_name}: {Fore.WHITE}uniform({spec['lower']}, {spec['upper']})")
                elif spec['type'] == 'loguniform':
                    print(f"  {Fore.YELLOW}{param_name}: {Fore.WHITE}loguniform({spec['lower']}, {spec['upper']})")
                elif spec['type'] == 'quniform':
                    print(f"  {Fore.YELLOW}{param_name}: {Fore.WHITE}quniform({spec['lower']}, {spec['upper']}, q={spec.get('q', 1)})")
                elif spec['type'] == 'randint':
                    print(f"  {Fore.YELLOW}{param_name}: {Fore.WHITE}randint({spec['lower']}, {spec['upper']})")
                else:
                    print(f"  {Fore.YELLOW}{param_name}: {Fore.WHITE}{spec['type']}")
            else:
                print(f"  {Fore.YELLOW}{param_name}: {Fore.WHITE}{type(param_spec).__name__}")
        
        print(f"\n{Style.BRIGHT}{Fore.MAGENTA}📊 Total Search Space Size: {Fore.WHITE}{total_size:,} configurations")
        if total_size > 1000000:
            print(f"  {Fore.YELLOW}⚠️  Large search space - consider using more samples for thorough exploration")
        elif total_size < 50:
            print(f"  {Fore.GREEN}✓ Small search space - can be exhaustively explored")
        else:
            print(f"  {Fore.CYAN}ℹ️  Moderate search space - good for optimization")

    @abstractmethod
    def get_search_space(self) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for Ray Tune
        
        Returns:
            Dictionary defining the search space using tune.choice, tune.uniform, etc.
        """
        pass

    @abstractmethod
    def create_attention_config(self, params: Dict[str, Any]) -> Tuple[str, ResearchAttentionConfig]:
        """
        Create attention configuration from hyperparameters
        
        Args:
            params: Dictionary of hyperparameters from Ray Tune
            
        Returns:
            Tuple of (config_name, ResearchAttentionConfig)
        """
        pass

    def get_hyperopt_search_space(self) -> Optional[Dict[str, Any]]:
        """
        Optional: Define search space for HyperOpt (more efficient for continuous spaces)
        If not implemented, will use Ray Tune's default search space
        
        Returns:
            Dictionary with hyperopt search space definitions or None
        """
        return None

    def clear_memory(self):
        """Clean up GPU and system memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def extract_metrics(self, benchmark_results: Dict[str, Any], 
                       result_dir: Path) -> Dict[str, float]:
        """
        Extract optimization metrics from MicroMetricLogger output files
        
        Args:
            benchmark_results: Results from benchmark.run_benchmark() (for reference)
            result_dir: Directory where metric logs are stored
            
        Returns:
            Dictionary with metrics for optimization
        """
        attention_errors = []
        densities = []
        
        # Read metrics from the single MicroMetricLogger file
        micro_metrics_file = result_dir / "micro_metrics.jsonl"
        if micro_metrics_file.exists():
            try:
                with open(micro_metrics_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            metric_name = data.get('metric', '')
                            value = data.get('value')
                            
                            if metric_name == 'research_attention_output_error' and value is not None:
                                attention_errors.append(float(value))
                            elif metric_name == 'research_attention_density' and value is not None:
                                densities.append(float(value))
            except Exception as e:
                print(f"Warning: Could not read micro metrics: {e}")
        
        # Raise errors if required metrics are not found
        if not attention_errors:
            available_files = list(result_dir.glob("*.jsonl"))
            raise ValueError(f"No research_attention_output_error metrics found in {micro_metrics_file}. "
                           f"Available metric files: {available_files}")
        
        if not densities:
            available_files = list(result_dir.glob("*.jsonl"))
            raise ValueError(f"No research_attention_density metrics found in {micro_metrics_file}. "
                           f"Available metric files: {available_files}")
        
        # Calculate means
        attention_error = sum(attention_errors) / len(attention_errors)
        density = sum(densities) / len(densities)
        
        return {
            'attention_error': float(attention_error),
            'density': float(density)
        }

    def evaluate_config(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single configuration and return metrics for Ray Tune
        
        Args:
            config: Hyperparameter configuration from Ray Tune
            
        Returns:
            Dictionary with metrics including:
            - attention_error: Primary metric to minimize
            - density: Secondary metric to minimize (sparsity measure)
            - gpu_runtime_s: Performance metric
            - memory_mb: Resource usage metric
            - combined_score: Multi-objective score for optimization
        """
        
        # Clean up before starting
        self.clear_memory()
        
        monitor = SystemMonitor(self.device)
        
        try:
            # Create configuration
            config_name, sparse_config = self.create_attention_config(config)
            
            print(f"\n{Fore.BLUE}Evaluating {self.attention_type}: {config_name}")
            
            # Load model with sparse attention
            monitor.start_capture()
            
            adapter = ModelAdapterHF(
                model_name=self.model_name,
                sparse_attention_config=sparse_config,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2"
                },
                generate_kwargs={"max_new_tokens": 32},
                device=self.device
            )
            
            # Model loaded, stop timing
            monitor.stop_capture()
            
            # Set up metric logging - use a persistent directory structure
            # Create a unique directory that won't be deleted by Ray workers
            # Use the configurable results base directory
            results_base = self.results_base_dir / "raytune_detailed_results" / f"{self.attention_type}_optimization"
            results_base.mkdir(parents=True, exist_ok=True)
            result_dir = results_base / f"{self.attention_type}_{config_name}"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            metric_logger = MicroMetricLogger()
            metric_logger.configure_logging(
                log_path=result_dir,
                enabled_metrics=["research_attention_density", "research_attention_output_error"]
            )
            
            # Run benchmark
            request_kwargs = {
                "max_requests": self.max_requests,
                "max_context_length": self.max_context_length
            }
            
            monitor.start_capture()
            benchmark_results = self.benchmark.run_benchmark(
                adapter, result_dir, request_kwargs=request_kwargs
            )
            eval_metrics = monitor.stop_capture()
            
            metric_logger.flush()
            
            # Extract key metrics
            extracted_metrics = self.extract_metrics(benchmark_results, result_dir)
            
            attention_error = extracted_metrics['attention_error']
            density = extracted_metrics['density']

            # 1. Calculate the base score
            base_score = attention_error + 0.1 * density
            
            # 2. Define a large penalty
            penalty = 0.0

            if density > 0.5:
                penalty = 5.0  # Large penalty for high density

            # Prepare final metrics for Ray Tune
            final_metrics = {
                'attention_error': attention_error,
                'density': density,
                'gpu_runtime_s': eval_metrics.get('gpu_runtime_ms', 0) / 1000.0,
                'memory_mb': eval_metrics.get('peak_gpu_memory_mb', 0),
                'wall_time_s': eval_metrics.get('wall_time_s', 0),
                # Multi-objective score: weighted combination of error and density
                'combined_score': base_score + penalty
            }
            
            print(f"  {Fore.GREEN}✓ Attention Error: {attention_error:.4f}")
            print(f"  {Fore.GREEN}✓ Density: {density:.4f}")
            print(f"  {Fore.GREEN}✓ Combined Score: {final_metrics['combined_score']:.4f}")

            ray.train.report(final_metrics)
            
            # Clean up
            del adapter
            self.clear_memory()
            monitor.cleanup()
            
            return final_metrics
            
        except Exception as e:
            print(f"  {Fore.RED}✗ Error: {str(e)}")
            print(f"  {Fore.RED}Traceback: {traceback.format_exc()}")
            
            # Clean up on error
            self.clear_memory()
            monitor.cleanup()
            
            # Return poor metrics to avoid this configuration
            return {
                'attention_error': 10.0,  # Very high error
                'density': 1.0,           # Maximum density
                'gpu_runtime_s': 999.0,   # High runtime penalty
                'memory_mb': 999999.0,    # High memory penalty
                'wall_time_s': 999.0,
                'combined_score': 11.0    # Very poor combined score
            }

    def run_optimization(self, 
                        num_samples: int = 50,
                        max_concurrent: int = 8,
                        cpu_per_trial: float = 4.0,
                        gpu_per_trial: float = 1.0,
                        use_asha: bool = True,
                        metric: str = "combined_score",
                        mode: str = "min",
                        results_dir: str = None) -> ray.tune.ExperimentAnalysis:
        """
        Run Ray Tune optimization
        
        Args:
            num_samples: Number of configurations to try
            max_concurrent: Maximum concurrent trials
            cpu_per_trial: CPU cores per trial
            gpu_per_trial: GPU fraction per trial
            use_asha: Whether to use ASHA scheduler for early stopping
            metric: Metric to optimize
            mode: Optimization mode ("min" or "max")
            results_dir: Directory to save results
            
        Returns:
            ExperimentAnalysis object with results
        """
        
        if results_dir is None:
            results_dir = f"./raytune_{self.attention_type}_results"
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_gpus=8,  # Specify total number of GPUs available
                num_cpus=None,  # Let Ray auto-detect CPUs
                log_to_driver=False  # Reduce log verbosity
            )
        
        print(f"\n{Style.BRIGHT}{Fore.MAGENTA}{'='*60}")
        print(f"Starting Ray Tune Optimization for {self.attention_type.title()}")
        print(f"{'='*60}")
        print(f"Model: {Fore.CYAN}{self.model_name}")
        print(f"Benchmark Tasks: {Fore.CYAN}{self.benchmark_tasks}")
        print(f"Number of Samples: {Fore.CYAN}{num_samples}")
        print(f"Max Concurrent Trials: {Fore.CYAN}{max_concurrent}")
        print(f"Device: {Fore.CYAN}{self.device}")
        print(f"Optimization Metric: {Fore.CYAN}{metric} ({mode})")
        print(f"Results Directory: {Fore.CYAN}{results_dir}")
        print(f"Detailed Results Base: {Fore.CYAN}{self.results_base_dir / 'raytune_detailed_results'}")
        
        # Generate unique experiment name based on search space
        experiment_name = self.generate_experiment_name()
        
        # Print search space information
        self.print_search_space_info()
        print(f"\n{Style.BRIGHT}{Fore.CYAN}🔬 Experiment Name: {Fore.WHITE}{experiment_name}")
        print(f"  {Fore.YELLOW}ℹ️  This name is generated from your search space configuration")
        print(f"  {Fore.YELLOW}ℹ️  Same search space = same experiment name (avoids duplicates)")
        
        # Set up search algorithm
        hyperopt_space = self.get_hyperopt_search_space()
        search_alg = None
        
        if hyperopt_space:
            search_alg = HyperOptSearch(
                hyperopt_space,
                metric=metric,
                mode=mode
            )
            # Limit concurrency
            search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)

        if search_alg is None:
            print(f"{Fore.YELLOW}⚠️  No search algorithm found. Using default grid search.")

        # Set up scheduler for early stopping
        scheduler = None
        if use_asha:
            scheduler = ASHAScheduler(
                metric=metric,
                mode=mode,
                max_t=1,  # Since we only have one evaluation per trial
                grace_period=1
            )
        
        reporter = CLIReporter(
            metric_columns=[
                "combined_score", 
                "attention_error", 
                "density", 
                "gpu_runtime_s",
                "wall_time_s",
                "memory_mb"
            ],
            max_progress_rows=15,
            metric="combined_score",
            mode="min"
        )
        
        # Configure the tuner
        tuner = tune.Tuner(
            tune.with_resources(
                self.evaluate_config,
                resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial}
            ),
            param_space=self.get_search_space() if not search_alg else {},
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent,
            ),
            run_config=ray.air.RunConfig(
                name=experiment_name,  # Use the generated experiment name
                storage_path=str(Path(results_dir).resolve()),
                verbose=1,
                progress_reporter=reporter,
            )
        )
        
        # Run optimization
        print(f"\n{Fore.GREEN}Starting optimization...")
        results = tuner.fit()
        
        print(f"\n{Style.BRIGHT}{Fore.GREEN}Optimization completed!")
        
        return results

    def analyze_results(self, results: ray.tune.ExperimentAnalysis, 
                       top_k: int = 10, metric: str = "combined_score", 
                       mode: str = "min"):
        """Analyze and display optimization results"""
        
        print(f"\n{Style.BRIGHT}{Fore.MAGENTA}{'='*60}")
        print("OPTIMIZATION RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        # Get best results
        best_result = results.get_best_result(metric=metric, mode=mode)
        
        print(f"\n{Style.BRIGHT}{Fore.GREEN}🏆 BEST CONFIGURATION ({self.attention_type.upper()}):")
        print(f"Combined Score: {Fore.CYAN}{best_result.metrics['combined_score']:.4f}")
        print(f"Attention Error: {Fore.CYAN}{best_result.metrics['attention_error']:.4f}")
        print(f"Density: {Fore.CYAN}{best_result.metrics['density']:.4f}")
        print(f"GPU Runtime: {Fore.CYAN}{best_result.metrics['gpu_runtime_s']:.2f}s")
        print(f"Memory Usage: {Fore.CYAN}{best_result.metrics['memory_mb']:.1f}MB")
        
        print(f"\n{Style.BRIGHT}Best Hyperparameters:")
        for param, value in best_result.config.items():
            print(f"  {param}: {Fore.CYAN}{value}")
        
        # Get top-k results
        df = results.get_dataframe()
        df_sorted = df.sort_values(metric, ascending=(mode == "min")).head(top_k)
        
        print(f"\n{Style.BRIGHT}{Fore.YELLOW}📊 TOP {top_k} CONFIGURATIONS:")
        print(f"{'Rank':<4} {'Combined':<10} {'Error':<10} {'Density':<10} {'Runtime':<10}")
        print("-" * 60)
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"{i:<4} {row['combined_score']:<10.4f} {row['attention_error']:<10.4f} "
                  f"{row['density']:<10.4f} {row['gpu_runtime_s']:<10.2f}")
        
        # Save detailed results
        results_file = Path(f"./raytune_{self.attention_type}_detailed_results.json")
        detailed_results = {
            'attention_type': self.attention_type,
            'best_config': dict(best_result.config),
            'best_metrics': dict(best_result.metrics),
            'top_configs': df_sorted.to_dict('records')
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n{Fore.GREEN}📁 Detailed results saved to: {results_file}")
        
        return best_result, df_sorted
