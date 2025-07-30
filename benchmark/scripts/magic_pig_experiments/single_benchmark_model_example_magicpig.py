#!/usr/bin/env python3
"""
Advanced Benchmark Example

An improved script for benchmarking dense and sparse attention that provides
more accurate GPU/CPU memory utilization and runtime metrics.

Key Improvements:
- A dedicated `SystemMonitor` class to encapsulate monitoring logic.
- Uses `pynvml` for comprehensive GPU memory stats (like `nvidia-smi`).
- Uses `torch.cuda.Event` for precise GPU-side runtime measurement.
- Measures and reports peak memory usage during benchmark execution.
- Adds color to console output for improved readability using `colorama`.
- Saves all results to a consolidated JSON file for easy analysis.
"""
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import psutil
import torch

# Try to import necessary libraries and provide helpful error messages.
try:
    import colorama
    import pynvml
    from colorama import Fore, Style
except ImportError as e:
    print(
        f"Error: A required library is missing. Please install it.\nMissing module: {e.name}"
    )
    sys.exit(1)

# Set project root and add to Python path
project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from benchmark import Loogle
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    MagicPigConfig,
)


# (SystemMonitor class and other helpers are unchanged)
class SystemMonitor:
    def __init__(self, device: torch.device):
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.is_cuda = device.type == "cuda"
        self.gpu_handle = None
        if self.is_cuda:
            pynvml.nvmlInit()
            device_index = torch.cuda.current_device()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def get_cpu_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024**2

    def get_gpu_memory_mb(self) -> dict:
        if not self.is_cuda:
            return {"torch_allocated": 0, "driver_total_used": 0}
        driver_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        return {
            "torch_allocated": torch.cuda.memory_allocated(self.device) / 1024**2,
            "driver_total_used": driver_info.used / 1024**2,
        }

    def start_capture(self) -> None:
        self.start_time = time.perf_counter()
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_event.record()

    def stop_capture(self) -> dict:
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize(self.device)
            gpu_runtime_ms = self.start_event.elapsed_time(self.end_event)
            peak_torch_allocated_mb = (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
        else:
            gpu_runtime_ms, peak_torch_allocated_mb = 0, 0
        return {
            "wall_time_s": time.perf_counter() - self.start_time,
            "gpu_runtime_ms": gpu_runtime_ms,
            "peak_torch_gpu_mem_mb": peak_torch_allocated_mb,
        }

    def shutdown(self):
        if self.is_cuda:
            pynvml.nvmlShutdown()


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_single_config(
    model_name,
    device,
    config_name,
    sparse_attention_config,
    benchmark,
    base_result_dir,
    request_kwargs,
):
    header = f"{Style.BRIGHT}{Fore.MAGENTA}"
    success = f"{Fore.GREEN}✓{Style.RESET_ALL}"
    print(f"\n{header}{'='*60}\nRunning configuration: {config_name}\n{'='*60}")
    clear_memory()
    monitor = SystemMonitor(device)

    print(f"  {success} Loading model...")
    initial_mem_cpu, initial_mem_gpu = (
        monitor.get_cpu_memory_mb(),
        monitor.get_gpu_memory_mb(),
    )
    monitor.start_capture()
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        generate_kwargs={"max_new_tokens": 32},
        device=device,
    )
    load_metrics = monitor.stop_capture()
    after_load_mem_cpu, after_load_mem_gpu = (
        monitor.get_cpu_memory_mb(),
        monitor.get_gpu_memory_mb(),
    )
    print(f"  {success} Model loaded in {Fore.CYAN}{load_metrics['wall_time_s']:.2f}s")
    print(
        f"  {success} GPU memory (Driver): {Fore.CYAN}{after_load_mem_gpu['driver_total_used']:.1f} MB{Style.RESET_ALL} (+{after_load_mem_gpu['driver_total_used'] - initial_mem_gpu['driver_total_used']:.1f} MB)"
    )
    print(
        f"  {success} System memory: {Fore.CYAN}{after_load_mem_cpu:.1f} MB{Style.RESET_ALL} (+{after_load_mem_cpu - initial_mem_cpu:.1f} MB)"
    )

    result_dir = base_result_dir / config_name
    result_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(
        log_path=result_dir,
        enabled_metrics=[
            "research_attention_density",
            "research_attention_output_error",
        ],
    )
    print(f"  {success} Running benchmark...")
    monitor.start_capture()
    benchmark_results = benchmark.run_benchmark(
        adapter, result_dir, request_kwargs=request_kwargs
    )
    benchmark_perf_metrics = monitor.stop_capture()
    metric_logger.flush()
    print(f"  {success} Benchmark completed.")
    print(
        f"    - {Fore.YELLOW}Wall Time:{Style.RESET_ALL} {Fore.CYAN}{benchmark_perf_metrics['wall_time_s']:.2f}s"
    )
    print(
        f"    - {Fore.YELLOW}Actual GPU Runtime:{Style.RESET_ALL} {Fore.CYAN}{benchmark_perf_metrics['gpu_runtime_ms'] / 1000:.2f}s"
    )
    print(
        f"    - {Fore.YELLOW}Peak GPU Memory (PyTorch Tensors):{Style.RESET_ALL} {Fore.CYAN}{benchmark_perf_metrics['peak_torch_gpu_mem_mb']:.1f} MB"
    )

    final_mem_gpu = monitor.get_gpu_memory_mb()
    full_results = {
        "benchmark_results": benchmark_results,
        "performance": {
            "timing": {
                "model_load_wall_time_s": load_metrics["wall_time_s"],
                "benchmark_wall_time_s": benchmark_perf_metrics["wall_time_s"],
                "benchmark_gpu_runtime_ms": benchmark_perf_metrics["gpu_runtime_ms"],
            },
            "memory": {
                "model_load_gpu_mem_mb": after_load_mem_gpu["driver_total_used"]
                - initial_mem_gpu["driver_total_used"],
                "model_load_cpu_mem_mb": after_load_mem_cpu - initial_mem_cpu,
                "peak_benchmark_torch_gpu_mem_mb": benchmark_perf_metrics[
                    "peak_torch_gpu_mem_mb"
                ],
                "final_total_driver_gpu_mem_mb": final_mem_gpu["driver_total_used"],
            },
        },
    }
    monitor.shutdown()
    del adapter
    clear_memory()
    return full_results


def generate_magicpig_configs(param_grid, sink_size=128, window_size=128):
    configs = []
    for params in param_grid:
        l, k, packing, center = (
            params["l"],
            params["k"],
            params["packing"],
            params["center"],
        )
        name = f"sink{sink_size}_L{l}_K{k}_pack_{packing}_center_{center}"
        cfg = ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=sink_size),
                LocalMaskerConfig(window_size=window_size),
                MagicPigConfig(lsh_l=l, lsh_k=k, packing=packing, center=center),
            ]
        )
        configs.append({"name": name, "config": cfg})
    return configs


def main():
    # Initialize colorama. autoreset=True ensures styles are reset after each print.
    colorama.init(autoreset=True)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark = Loogle(["shortdep_qa"])
    base_result_dir = Path("./test_magicpig_results")
    base_result_dir.mkdir(exist_ok=True)
    request_kwargs = {"max_requests": 2, "max_context_length": 16000}

    param_grid = [
        {"l": 32, "k": 8, "packing": "int64", "center": True},
        {"l": 32, "k": 8, "packing": "float32", "center": True},
        {"l": 96, "k": 8, "packing": "int64", "center": True},
        {"l": 96, "k": 8, "packing": "float32", "center": True},
        {"l": 32, "k": 8, "packing": "int64", "center": False},
        {"l": 32, "k": 8, "packing": "float32", "center": False},
        {"l": 96, "k": 8, "packing": "int64", "center": False},
        {"l": 96, "k": 8, "packing": "float32", "center": False},
    ]

    configs = generate_magicpig_configs(param_grid)
    all_results = {}

    print(f"{Style.BRIGHT}Starting benchmark for {len(configs)} configurations...")
    print(
        f"Model: {Fore.CYAN}{model_name}{Style.RESET_ALL}, Device: {Fore.CYAN}{device}"
    )

    for i, cfg_info in enumerate(configs, 1):
        print(
            f"\n{Style.BRIGHT}{Fore.BLUE}[{i}/{len(configs)}] Starting configuration: {cfg_info['name']}"
        )
        try:
            metrics = run_single_config(
                model_name,
                device,
                cfg_info["name"],
                cfg_info["config"],
                benchmark,
                base_result_dir,
                request_kwargs,
            )
            all_results[cfg_info["name"]] = metrics
        except Exception as e:
            error_msg = f"ERROR in configuration {cfg_info['name']}: {e}"
            print(f"{Fore.RED}{error_msg}")
            all_results[cfg_info["name"]] = {"error": str(e)}
            clear_memory()

    # --- FINAL SUMMARY ---
    header = f"{Style.BRIGHT}{Fore.MAGENTA}"
    print(f"\n{header}{'='*80}\nFINAL SUMMARY\n{'='*80}")

    summary_lines = []
    for cfg_name, result in all_results.items():
        line = f"{Style.BRIGHT}--- {cfg_name} ---\n"
        if "error" in result:
            line += f"  {Fore.RED}ERROR: {result['error']}\n"
        else:
            perf = result.get("performance", {})
            bench_res = result.get("benchmark_results", {})
            timing = perf.get("timing", {})
            memory = perf.get("memory", {})
            gpu_runtime_s = timing.get("benchmark_gpu_runtime_ms", 0) / 1000
            peak_gpu_mem_mb = memory.get("peak_benchmark_torch_gpu_mem_mb", 0)
            cpu_mem_mb = memory.get("model_load_cpu_mem_mb", 0)

            line += f"  {Fore.YELLOW}GPU Runtime:{Style.RESET_ALL} {Fore.CYAN}{gpu_runtime_s:.2f}s\n"
            line += f"  {Fore.YELLOW}Peak GPU Memory (Tensors):{Style.RESET_ALL} {Fore.CYAN}{peak_gpu_mem_mb:.1f} MB\n"
            line += f"  {Fore.YELLOW}CPU Memory (Model Load):{Style.RESET_ALL} {Fore.CYAN}{cpu_mem_mb:.1f} MB\n"

            # --- THIS IS THE CORRECTED PART ---
            if bench_res:
                # The benchmark result is a simple {metric_name: value} dictionary.
                # We do not need a nested loop.
                for metric_name, metric_value in bench_res.items():
                    if isinstance(metric_value, (int, float)):
                        line += f"  {Fore.YELLOW}{metric_name}:{Style.RESET_ALL} {Fore.CYAN}{metric_value:.4f}\n"

        print(line, end="")
        summary_lines.append(line)

    text_summary_file = base_result_dir / "summary_results.txt"
    with open(text_summary_file, "w") as f:
        plain_text_summary = [
            re.sub(r"\033\[[0-9;]*m", "", line) for line in summary_lines
        ]
        f.writelines(plain_text_summary)
    print(f"\n{Fore.GREEN}Text summary saved to: {text_summary_file}")

    json_summary_file = base_result_dir / "benchmark_results.json"
    with open(json_summary_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"{Fore.GREEN}Full results JSON saved to: {json_summary_file}")


if __name__ == "__main__":
    main()
