# Benchmark Runner Scripts

This directory contains scripts for running benchmarks with optimal configurations from Phase 1.

## Scripts

### 1. run_ray_benchmarks.py
The main benchmark runner using Ray for efficient parallel execution.

<<<<<<< HEAD
1. **Phase 1**: Hyperparameter search to find optimal configs for each (model, task, masker) combination
2. **Phase 2**: Parallel benchmark execution using the discovered optimal configs


## Quick Start

```bash 
## install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

## clone repo, to branch feature/raytune
git clone https://github.com/xAlg-ai/sparse-attention-hub
git checkout feature/raytune

## build env
uv sync
### need to install flash-attn after init to avoid torch dependencies
uv add flash-attn --no-build-isolation
source .venv/bin/activate

## login required for huggingface
export HF_TOKEN=...
huggingface-cli login --token $HF_TOKEN


## run benchmark in debug mode
python benchmark/raytune/run_full_benchmark.py --debug
```

Expected output:

```
.
├── optimal_configs/                    # Phase 1 outputs
│   └── run_20240315_143022/           # Timestamped run directory
│       ├── meta-llama_Llama-3.1-8B-Instruct_loogle_shortdep_qa_sink_local_5pct.json          # Best config
│       ├── meta-llama_Llama-3.1-8B-Instruct_loogle_shortdep_qa_sink_local_5pct_trials.json   # Trial details
│       ├── meta-llama_Llama-3.1-8B-Instruct_loogle_shortdep_qa_sink_local_5pct_analysis.csv  # Ray analysis
│       └── ... (3 files per model-task-masker combination)
│
├── ray_results/                        # Ray Tune working directory
│   └── search_runs/                    # Hyperparameter search experiments
│       └── ... (Ray Tune experiment artifacts)
│
└── benchmark_results/                  # Phase 2 outputs
    ├── benchmark_summary.json          # Overall benchmark summary
    └── meta-llama_Llama-3.1-8B-Instruct/     # Sanitized model name
        ├── dense/                             # Dense baseline config
        │   └── loogle_shortdep_qa/            # Benchmark_subset
        │       └── raw_results.csv
        ├── sink_local_oracle_top_k_adaptive_sampling/  # Sparse config name
        │   └── loogle_shortdep_qa/
        │       ├── raw_results.csv
        │       └── micro_metrics.jsonl        # Sparse attention metrics
        └── sink_local_random_sampling/        # Another sparse config
            └── loogle_shortdep_qa/
                ├── raw_results.csv
                └── micro_metrics.jsonl
```






## Basic Usage
=======
**Features:**
- Stateful Ray actors managing GPU resources
- Fresh model initialization for each task (required due to unique optimized parameters)
- Real-time progress tracking with ETA
- Dry run mode to preview execution plan
- Debug mode for testing with reduced parameters
- Automatic GPU resource management
- Resume capability (skips completed benchmarks)
>>>>>>> 6d836e5 (add parallel benchmark executor with ray, visualization scripts)

**Usage:**
```bash
# Basic usage (uses all available GPUs)
python benchmark/raytune/run_ray_benchmarks.py --config-run run_20250818_203531

# Dry run to see what will be executed
python benchmark/raytune/run_ray_benchmarks.py --config-run run_20250818_203531 --dry-run

# Debug mode - run 2-4 tasks with reduced parameters for testing
python benchmark/raytune/run_ray_benchmarks.py --config-run run_20250818_203531 --debug

# Single GPU execution
python benchmark/raytune/run_ray_benchmarks.py --config-run run_20250818_203531 --num-actors 1

# Maximum utilization with multiple actors per GPU (e.g., 2 actors per GPU)
python benchmark/raytune/run_ray_benchmarks.py --config-run run_20250818_203531 --actors-per-gpu 2

# Resume from previous run
python benchmark/raytune/run_ray_benchmarks.py --config-run run_20250818_203531 --resume

# Custom parameters
python benchmark/raytune/run_ray_benchmarks.py \
    --config-run run_20250818_203531 \
    --max-new-tokens 200 \
    --max-context-length 64000 \
    --max-requests 50 \
    --benchmark-results-dir ./my_results
```

### 2. list_benchmark_tasks.py
Utility to list and inspect benchmark tasks from optimal configurations.

**Usage:**
```bash
# List all tasks in table format
python benchmark/raytune/list_benchmark_tasks.py --config-run run_20250818_203531

# Group by model
python benchmark/raytune/list_benchmark_tasks.py --config-run run_20250818_203531 --group-by model

# Export to CSV
python benchmark/raytune/list_benchmark_tasks.py --config-run run_20250818_203531 --format csv > tasks.csv

# Filter tasks
python benchmark/raytune/list_benchmark_tasks.py \
    --config-run run_20250818_203531 \
    --filter-task loogle \
    --filter-masker adaptive

# Simple format for scripting
python benchmark/raytune/list_benchmark_tasks.py --config-run run_20250818_203531 --format simple
```

## Performance Tips

1. **Model Loading**: Each task requires fresh model initialization due to unique optimized parameters from Phase 1. Model loading time is tracked and reported.

2. **Actor Count**: 
   - Default: 1 actor per GPU for maximum parallelism
   - Debug mode: Limited to 2 actors for faster testing
   - Custom: Use `--num-actors` to control parallelism

3. **Debug Mode**: Use `--debug` for quick testing:
   - Runs only 2-4 diverse tasks
   - Reduces max_new_tokens to 20
   - Limits context length to 4096
   - Processes only 2 requests per benchmark

4. **Resume**: Completed benchmarks are automatically skipped based on the presence of `metrics.json`.

## Output Structure

Results are saved in the following structure:
```
benchmark_results_ray/
├── meta-llama_Llama-3.1-8B-Instruct/
│   ├── dense/
│   │   ├── loogle_longdep_qa/
│   │   │   ├── raw_results.csv
│   │   │   ├── metrics.json
│   │   │   └── micro_metrics.jsonl
│   │   └── ...
│   ├── sink_local_random_sampling/
│   │   └── ...
│   └── ...
└── ...
```

## Monitoring Progress

The Ray runner provides real-time progress updates:
- Current task completion status with execution time
- Model loading time for each task
- Average model load time statistics
- Estimated time remaining (ETA)
- Tasks per second throughput
- Total execution and model loading time summary