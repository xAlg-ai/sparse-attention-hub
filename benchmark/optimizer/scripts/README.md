# Benchmark Optimizer Scripts

This directory contains scripts for running optimized benchmarks using the declarative hyperparameter optimization system.

## Scripts

### `run_optimized_benchmarks.py`
Main CLI script for running optimized benchmarks with declarative hyperparameter tuning.

**Features:**
- Automatic search space discovery from config classes
- Per-task hyperparameter optimization
- Result caching and resumability
- Support for multiple sparse attention configurations

**Usage:**
```bash
# Quick demo with Magic Pig optimization
python run_optimized_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --benchmark loogle --config magic_pig --samples 5

# Full evaluation with multiple configs
python run_optimized_benchmarks.py --model microsoft/Phi-4-mini-instruct --benchmark infinite_bench,ruler --config dense,magic_pig --samples 20

# Per-task optimization with custom subsets
python run_optimized_benchmarks.py --model meta-llama/Llama-3.2-1B-Instruct --benchmark loogle --subsets shortdep_qa --config magic_pig --samples 10
```

**Supported Configurations:**
- `dense`: Dense attention (no optimization)
- `magic_pig`: Magic Pig with declarative optimization
- `streaming_conservative`: StreamingLLM conservative configuration
- `local_only`: Local attention only with declarative optimization
- `sink_only`: Sink attention only with declarative optimization

**Supported Benchmarks:**
- `loogle`: Short and long dependency QA tasks
- `infinite_bench`: Passkey task
- `ruler`: 4096 context length task
- `zero_scrolls`: Government report task
- `longbenchv2`: 0-shot task
- `aime2024`: AIME 2024 task
- `aime2025`: AIME 2025 task
- `longbench`: Narrative QA task
- `mock_benchmark`: Reading comprehension task

### `demo_declarative_optimization.py`
Comprehensive demo script showing the complete declarative optimization pipeline.

**Features:**
- Search space discovery demonstration
- Composite optimization examples
- Auto-registration examples
- Full pipeline demonstration
- Usage examples

**Usage:**
```bash
python demo_declarative_optimization.py
```

## Declarative Optimization System

The new declarative system provides several benefits:

1. **Automatic Search Space Discovery**: Config classes define their own search spaces
2. **Auto-Composition**: Composite configs are automatically created from individual configs
3. **Per-Task Optimization**: Each benchmark task gets its own optimized configuration
4. **Result Caching**: Optimization results are cached for reuse
5. **Backward Compatibility**: Still supports manual overrides when needed

## Migration from Old System

The old scripts (`demo_optimized_benchmarks.py` and `get_cached_config.py`) have been removed as they used the manual optimization system. The new declarative system provides the same functionality with much cleaner code and better maintainability.

## Key Improvements

- **No manual search space specification required**
- **Automatic discovery of default search spaces**
- **Automatic composition of composite configs**
- **Declarative and extensible**
- **Backward compatible with manual overrides**
- **Full pipeline integration** 