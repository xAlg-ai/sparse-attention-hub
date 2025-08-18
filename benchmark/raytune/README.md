# Two-Phase Benchmark System

Automated sparse attention optimization with distinct search and execution phases.

## Architecture

`run_full_benchmark.py` implements a two-phase workflow:

1. **Phase 1**: Hyperparameter search to find optimal configs for each (model, task, masker) combination
2. **Phase 2**: Parallel benchmark execution using the discovered optimal configs


## Quick Start

```bash 
## install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

## clone repo, to branch feature/raytune
git clone https://github.com/xAlg-ai/sparse-attention-hub
git checkout feature/raytune

## build env
uv sync
### need to install flash-attn after init to avoid torch dependencies
uv add flash-attn --no-build-isolation


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

```bash
# Run both phases (default)
python benchmark/raytune/run_full_benchmark.py

# Run only Phase 1 (config search)
python benchmark/raytune/run_full_benchmark.py --phase 1

# Run only Phase 2 (benchmark execution)  
python benchmark/raytune/run_full_benchmark.py --phase 2

# Debug mode (minimal configs, fast execution)
python benchmark/raytune/run_full_benchmark.py --debug

# Force re-search in Phase 1
python benchmark/raytune/run_full_benchmark.py --phase 1 --force-search
```

## Configuration

Models and tasks are configured in `get_run_configuration()`:

```python
models = ["meta-llama/Llama-3.1-8B-Instruct"]
tasks = [
    "infinite_bench/passkey",
    "ruler/4096", 
    "loogle/longdep_qa",
    "zero_scrolls/default",
    # ... more tasks
]
```

Sparse configs are generated in `get_all_sparse_configs()` with sparsity levels: 5%, 10%, 25%, 50%.

## How It Works

### Phase 1: Hyperparameter Search
- Uses Ray Tune to search optimal parameters for each masker configuration
- Runs with lightweight settings (small context, few tokens) for speed
- Saves best configs to `phase1_results/{model}_{task}_{masker}.json`

### Phase 2: Benchmark Execution  
- Loads optimal configs from Phase 1
- Runs full benchmarks with production settings
- Saves results to `phase2_results/`

## Adding New Components

### New Tasks
Add to `tasks` list in `get_run_configuration()`:
```python
tasks = [
    "infinite_bench/passkey",
    "your_benchmark/subset",  # Add here
]
```

### New Models
Add to `models` list in `get_run_configuration()`:
```python
models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "your-org/your-model",  # Add here
]
```

### New Masker Configs
Add to `get_all_sparse_configs()` following the pattern:
```python
# Example: Add custom masker at 5% sparsity
configs.append((
    "custom_masker_5pct",
    ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.02),
        YourMaskerConfig(param=0.03)
    ]),
    [SinkMaskerConfig, YourMaskerConfig]  # Classes for Ray Tune
))
```

## Phase Control

### Phase 1 Parameters
```bash
--phase 1                        # Run search only
--force-search                   # Ignore existing results
--num-samples 50                 # Trials per config
--search-timeout 900             # 15min timeout per trial
--search-max-new-tokens 20       # Search with minimal tokens
--search-max-context-length 8192 # Search with small context
--search-max-requests 10         # Max requests per trial
```

### Phase 2 Parameters
```bash
--phase 2                        # Run benchmark only
--config-run run_20240315_143022 # Use specific config run
--max-concurrent-benchmarks 4    # Parallel benchmarks
--benchmark-timeout 7200         # 2hr timeout
--max-new-tokens 200             # Full generation
--max-context-length 64000       # Full context
```

### Combined Execution
```bash
# Run both phases with custom settings
python run_full_benchmark.py \
    --num-samples 100 \
    --search-max-context-length 16384 \
    --max-new-tokens 500
```

## Output Structure

```
.
├── optimal_configs/                    # Phase 1: Hyperparameter search results
│   └── run_YYYYMMDD_HHMMSS/           # Timestamped configuration run
│       ├── {model}_{task}_{masker}.json          # Best hyperparameters
│       ├── {model}_{task}_{masker}_trials.json   # All trial details
│       └── {model}_{task}_{masker}_analysis.csv  # Ray Tune analysis
│
├── ray_results/                        # Ray Tune experiment artifacts
│   └── search_runs/                    # Intermediate search data
│       └── search_{model}_{task}_{masker}/
│           └── ... (checkpoints, logs, etc.)
│
└── benchmark_results/                  # Phase 2: Full benchmark results
    ├── benchmark_summary.json          # Aggregated results summary
    └── {model}/                        # Model-specific results (e.g., meta-llama_Llama-3.1-8B-Instruct)
        └── {sparse_config}/            # Sparse config name (e.g., dense, sink_local_5pct)
            └── {benchmark}_{subset}/   # Combined benchmark-subset directory
                ├── raw_results.csv           # Evaluation metrics
                └── micro_metrics.jsonl       # Sparse attention details (only for sparse configs)
```

### File Descriptions

**Phase 1 Files:**
- **Best hyperparameters** (`{model}_{task}_{masker}.json`): Optimal settings discovered for each combination
- **Trial details** (`*_trials.json`): Complete record of all hyperparameter search trials
- **Ray analysis** (`*_analysis.csv`): Detailed metrics for all trials (loss, accuracy, runtime, etc.)

**Phase 2 Files:**
- **Raw results** (`raw_results.csv`): Final evaluation metrics (accuracy, perplexity, etc.)
- **Micro metrics** (`micro_metrics.jsonl`): Per-sample sparse attention statistics (only for sparse configs)
- **Benchmark summary** (`benchmark_summary.json`): Consolidated results across all model/task/masker combinations

**Notes:**
- Model names containing "/" are replaced with "_" in filenames (e.g., `meta-llama/Llama-3.1-8B` becomes `meta-llama_Llama-3.1-8B`)
- Sparse config names reflect the masker types and parameters (e.g., `sink_local_oracle_top_k_adaptive_sampling` combines multiple masker strategies)
- The `dense` configuration serves as a baseline with no sparse attention applied

## Key Files
- `run_full_benchmark.py`: Main two-phase system implementation
- `optimizer_factory.py`: Ray Tune search space generation
- `analyze_trials.py`: Utility to analyze Phase 1 results