# Sparse Attention Configuration Optimizer

This framework automates the process of finding the optimal hyperparameters for sparse attention configurations and validates their performance across a matrix of models and benchmarks.

It uses a robust **"Search then Validate"** workflow:
1.  **Search**: Uses Ray Tune to run a fast search with lightweight benchmark settings to find the best configuration.
2.  **Validate**: Takes the single best configuration and runs a final, thorough benchmark to get a definitive performance score.

---
## How It Works: The "Search then Validate" Workflow

The entire process is orchestrated by the `execute_full_benchmark.py` script. For each combination of model, benchmark, and masker preset, it performs two main stages:

1.  **Search Stage** 
    * The script constructs a hyperparameter **search space** from the specified masker configurations.
    * It launches a **Ray Tune** optimization job, which runs multiple trials in parallel.
    * Each trial runs a benchmark with a unique set of hyperparameters, using **lightweight settings** (e.g., fewer tokens, smaller context) to get a score quickly.
    * Ray Tune identifies the trial that produced the **best score**.

2.  **Validation Stage** 
    * The script takes the single **best configuration** discovered in the search stage.
    * It then runs the benchmark one final time using **thorough, high-quality settings** (e.g., longer context, more generated tokens).
    * This produces a definitive **final validation score** and detailed logs for the winning configuration.

---
## How to Extend the Framework

The framework is designed for easy extension by modifying a single function in `execute_full_benchmark.py`: `get_run_configurations()`.

### Adding a New Model or Benchmark
Simply add the string ID to the corresponding list within the function. Use the `benchmark_name/subset_name` format for clarity.

```python
# In get_run_configurations() in execute_full_benchmark.py

# Add a new model
"models": [
    "meta-llama/Llama-3.2-8B-Instruct",
    "mistralai/Mistral-7B-v0.1"  # <-- ADD NEW MODEL HERE
],

# Add a new benchmark
"benchmarks": [
    "loogle/shortdep_qa",
    "new_benchmark/new_subset"  # <-- ADD NEW BENCHMARK HERE
],
```

### Adding a New Masker Preset
Import your masker's ...Config class at the top of the file.

Add a new entry to the masker_config_presets dictionary. The key is the preset's name, and the value is a list of the masker config classes to combine.

```python

# In get_run_configurations() in execute_full_benchmark.py

# 1. Import your custom masker config
from my_cool_maskers.cool_masker import CoolMaskerConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers import LocalMaskerConfig

# 2. Add a new preset
masker_config_presets = {
    "local_sink": [SinkMaskerConfig, LocalMaskerConfig],
    "sink_local_magic_pig": [SinkMaskerConfig, LocalMaskerConfig, MagicPigConfig],
    "new_preset": [SinkMaskerConfig, NewMaskerConfig], # <-- ADD PRESET HERE
}
```

### Key Files
- execute_full_benchmark.py: Main entry point. This is the file you run and the primary file you'll edit to change the test matrix.
- optimizer_factory.py: The core engine that builds search spaces from your config classes. You should not need to edit this file unless changing the fundamental optimization logic.


### Command-Line Arguments
The script uses decoupled parameters for the search and validation stages.
--debug: Runs a small, fast test.   
--num-samples: Controls how many configurations Ray Tune will try during the search.   
--search-*: A group of flags (--search-timeout, --search-max-new-tokens, etc.) to control the lightweight search trials.    
--validation-*: A group of flags (--validation-timeout, etc.) to control the final, thorough benchmark run.    
Run with --help to see all available options and their default values.   