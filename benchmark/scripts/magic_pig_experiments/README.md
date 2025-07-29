# Ray Tune Integration for Sparse Attention Optimization

This directory contains a modular Ray Tune integration for optimizing sparse attention configurations, starting with MagicPig. The system is designed to find the best configurations that minimize attention error while maintaining optimal sparsity/density.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate the environment
source /scratch/krishna/.venv/bin/activate

# Install Ray Tune dependencies (if not already installed)
pip install ray[tune] hyperopt pynvml colorama psutil
```

### 2. Single GPU Demo

```bash
cd /scratch/krishna/inference/longcontext/sparse-attention-hub/benchmark/scripts/magic_pig_experiments
python demo_raytune_optimization.py
```

### 3. Multi-GPU Demo (8 GPUs)

```bash
cd /scratch/krishna/inference/longcontext/sparse-attention-hub/benchmark/scripts/magic_pig_experiments
python demo_raytune_optimization_multi_gpu.py
```

## 📁 File Structure

```
magic_pig_experiments/
├── raytune_attention_optimizer.py      # Base optimizer class
├── raytune_magicpig_optimizer.py       # MagicPig-specific optimizer
├── demo_raytune_optimization.py        # Single GPU demo
├── demo_raytune_optimization_multi_gpu.py  # Multi-GPU demo
└── README.md                           # This file
```

## 🔧 Key Features

### ✅ Modular Design
- **Base Optimizer**: `AttentionOptimizer` class for extensibility
- **MagicPig Optimizer**: Specific implementation for MagicPig configs
- **Easy Extension**: Add new optimizers by inheriting from the base class

### ✅ Multi-GPU Support
- Configurable concurrent trials across multiple GPUs
- Automatic GPU detection and allocation
- Resource management (CPU/GPU per trial)

### ✅ Robust Configuration Management
- **Dynamic Config Names**: Generated from parameter values to avoid duplicate runs
- **Hash-based Experiment Names**: Unique experiment tracking per search space
- **Configurable Results Directory**: Persistent results storage

### ✅ Dashboard Integration
- Results compatible with `dashboard.py`
- Proper metric extraction from `micro_metrics.jsonl`
- Persistent results for visualization

### ✅ Advanced Ray Tune Features
- **ASHA Scheduler**: Early stopping for poor configurations
- **HyperOpt**: Bayesian optimization for efficient search
- **Concurrency Limiting**: Prevent resource overload
- **Search Space Size Reporting**: Know the optimization scale

## 📊 Usage Examples

### Basic Optimization

```python
from raytune_magicpig_optimizer import MagicPigOptimizer

# Create optimizer
optimizer = MagicPigOptimizer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    benchmark_tasks=["shortdep_qa"],
    max_requests=10,
    results_base_dir="./my_results"
)

# Run optimization
results = optimizer.run_optimization(
    num_samples=50,
    max_concurrent=4,
    cpu_per_trial=4.0,
    gpu_per_trial=1.0,
    use_asha=True
)

# Analyze results
best_result, top_configs = optimizer.analyze_results(results, top_k=5)
```

### Multi-GPU Optimization

```python
# Configure for 8 GPUs
results = optimizer.run_optimization(
    num_samples=100,
    max_concurrent=8,      # 8 concurrent trials
    cpu_per_trial=8.0,     # More CPUs per trial
    gpu_per_trial=1.0,     # 1 GPU per trial
    use_asha=True,
    metric="combined_score",
    mode="min"
)
```

### Custom Search Space

```python
# Override search space in your optimizer subclass
def get_search_space(self):
    return {
        "sink_size": tune.choice([16, 32, 64, 128]),
        "local_size": tune.choice([128, 256, 512, 1024]),
        "sink_density": tune.uniform(0.1, 0.9),
        "local_density": tune.uniform(0.1, 0.9),
        # Add your parameters here
    }
```

## 🎯 Optimization Metrics

The optimizer tracks multiple metrics to find the best configurations:

- **`attention_error`**: L2 error between sparse and full attention
- **`density`**: Sparsity level (lower is better for efficiency)
- **`gpu_runtime_s`**: GPU execution time
- **`memory_mb`**: Peak GPU memory usage
- **`combined_score`**: Weighted combination optimizing both error and density

### Combined Score Formula

```python
combined_score = attention_error + 0.1 * density
```

This balances minimizing attention error while maintaining sparsity.

### Live Progress Table

The Ray Tune progress table now displays all key metrics including the combined score:

```
│ Trial name            status         iter     total time (s)     attention_error     density     combined_score     gpu_runtime_s     memory_mb │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ _trainable_1cced3ec   TERMINATED        1           31.0649           0.0969149      0.324396        0.1294        26.2844           38614.9 │
│ _trainable_ad1d6180   TERMINATED        1           40.5874           0.327407       0.0282808       0.3302        35.9407           80211.8 │
```

This allows you to monitor the multi-objective optimization in real-time.

## 📈 Multi-GPU Performance Tips

1. **Scale num_samples**: Use 50-200 samples to fully utilize parallel GPUs
2. **Use ASHA scheduler**: Enable early stopping with `use_asha=True`
3. **Monitor resources**: Watch GPU utilization with `nvidia-smi`
4. **Adjust concurrency**: Set `max_concurrent` based on available GPU memory
5. **Increase CPU allocation**: More CPUs per trial for data processing

## 🔍 Results Analysis

### Best Configuration
```python
best_result, top_configs = optimizer.analyze_results(results, top_k=5)
print(f"Best combined score: {best_result.metrics['combined_score']:.4f}")
print(f"Attention error: {best_result.metrics['attention_error']:.4f}")
print(f"Density: {best_result.metrics['density']:.4f}")
```

### Dashboard Visualization
```bash
# View results in dashboard
cd /scratch/krishna/inference/longcontext/sparse-attention-hub
python dashboard.py --results_dir ./custom_results/raytune_detailed_results
```

## 🛠 Extending the System

### Adding New Optimizers

1. **Inherit from base class**:
```python
from raytune_attention_optimizer import AttentionOptimizer

class MyCustomOptimizer(AttentionOptimizer):
    def get_search_space(self):
        return {
            "my_param": tune.uniform(0, 1),
            # Define your search space
        }
    
    def objective_function(self, config):
        # Implement your optimization logic
        pass
```

2. **Override key methods**:
   - `get_search_space()`: Define hyperparameter search space
   - `objective_function()`: Implement the evaluation logic
   - `get_config_name()`: Custom config naming (optional)

### Adding New Metrics

1. **Extend metric extraction**:
```python
def extract_metrics_from_output(self, output_dir):
    # Read from micro_metrics.jsonl
    metrics_file = output_dir / "micro_metrics.jsonl"
    
    # Add your metric extraction logic
    metrics = {}
    # ... existing logic ...
    metrics["my_custom_metric"] = self._extract_metric(metrics_file, "my_metric")
    
    return metrics
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct environment and directory
2. **GPU Memory**: Reduce `max_concurrent` if running out of GPU memory
3. **Missing Metrics**: Check that `micro_metrics.jsonl` is being generated
4. **Dashboard Issues**: Ensure results directory is persistent and accessible

### Debug Commands

```bash
# Check GPU availability
nvidia-smi

# Verify environment
which python
python -c "import ray; print(ray.__version__)"

# Check results directory
ls -la ./custom_results/raytune_detailed_results/
```

## 📋 Requirements

- Python 3.8+
- Ray[tune] 2.0+
- PyTorch (for GPU detection)
- HyperOpt
- Colorama
- Existing sparse-attention-hub dependencies

## 🎮 Production Configuration

For production runs, use these recommended settings:

```python
production_config = {
    "num_samples": 200,          # Comprehensive search
    "max_concurrent": 8,         # Full GPU utilization
    "cpu_per_trial": 8.0,        # Adequate CPU resources
    "gpu_per_trial": 1.0,        # One GPU per trial
    "use_asha": True,           # Early stopping
    "metric": "combined_score",  # Balanced optimization
    "mode": "min"               # Minimize the metric
}
```

## 🔗 Integration Points

- **Benchmarking**: Uses existing Loogle and ModelAdapterHF infrastructure
- **Metrics**: Integrates with MicroMetricLogger output
- **Visualization**: Compatible with dashboard.py
- **Environment**: Works with existing `.venv` setup

---

**Happy Optimizing!** 🚀 For questions or issues, check the troubleshooting section or examine the demo scripts for working examples.
