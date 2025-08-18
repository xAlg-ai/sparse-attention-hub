# Two-Phase Benchmark System

Automated sparse attention optimization with distinct search and execution phases.

## Architecture

`run_full_benchmark.py` implements a two-phase workflow:

1. **Phase 1**: Hyperparameter search to find optimal configs for each (model, task, masker) combination
2. **Phase 2**: Parallel benchmark execution using the discovered optimal configs

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
phase1_results/
├── llama-3.1-8b-instruct_loogle-shortdep-qa_sink_local_5pct.json
├── llama-3.1-8b-instruct_loogle-shortdep-qa_adaptive_oracle_5pct.json
└── ... (best configs for each combination)

phase2_results/
├── benchmark_results_2024-01-15_10-30-00.csv
└── ... (full benchmark results)
```

## Key Files
- `run_full_benchmark.py`: Main two-phase system implementation
- `optimizer_factory.py`: Ray Tune search space generation
- `analyze_trials.py`: Utility to analyze Phase 1 results