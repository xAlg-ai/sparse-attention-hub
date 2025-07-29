# Ray Tune Optimization for Sparse Attention

This directory contains modular Ray Tune optimization tools for finding optimal sparse attention configurations that minimize attention error while maintaining low sparsity/density.

## Features

- **Modular Design**: Easy to extend to different attention mechanisms
- **Multi-objective Optimization**: Minimize attention error and density simultaneously  
- **Comprehensive Monitoring**: GPU memory, runtime, and attention metrics tracking
- **Hyperparameter Search**: Both grid search and advanced algorithms (HyperOpt)
- **Early Stopping**: ASHA scheduler for efficient resource usage
- **Result Analysis**: Detailed reporting and visualization of optimization results

## Files

- `raytune_attention_optimizer.py`: Base class for attention optimization
- `raytune_magicpig_optimizer_clean.py`: MagicPig-specific optimizer implementation
- `demo_raytune_optimization.py`: Simple demo showing usage and extension examples
- `single_benchmark_model_example_magicpig.py`: Original script (for reference)

## Quick Start

### 1. Install Dependencies

```bash
pip install ray[tune] hyperopt pynvml colorama psutil
```

### 2. Run Demo

```bash
cd /scratch/krishna/inference/longcontext/sparse-attention-hub/benchmark/scripts/magic_pig_experiments
python demo_raytune_optimization.py
```

### 3. Real Optimization

```python
from raytune_magicpig_optimizer_clean import MagicPigOptimizer

# Create optimizer
optimizer = MagicPigOptimizer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    benchmark_tasks=["shortdep_qa"],
    max_requests=10,  # More samples for real optimization
    max_context_length=16000
)

# Run optimization
results = optimizer.run_optimization(
    num_samples=50,      # Try 50 configurations
    max_concurrent=2,    # Use 2 GPUs if available
    cpu_per_trial=1.0,
    gpu_per_trial=0.4,   # 40% GPU memory per trial
    use_asha=True        # Enable early stopping
)

# Analyze results
best_result, top_configs = optimizer.analyze_results(results, top_k=10)
```

## Extending to Other Attention Mechanisms

The base class `AttentionOptimizerBase` makes it easy to optimize other sparse attention mechanisms:

```python
from raytune_attention_optimizer import AttentionOptimizerBase
from ray import tune

class MyAttentionOptimizer(AttentionOptimizerBase):
    def __init__(self, **kwargs):
        super().__init__(attention_type="my_attention", **kwargs)
    
    def create_attention_config(self, params):
        # Create your attention configuration from hyperparameters
        config_name = f"my_config_{params['param1']}_{params['param2']}"
        config = ResearchAttentionConfig(masker_configs=[
            # Your masker configurations
        ])
        return config_name, config
    
    def get_search_space(self):
        return {
            'param1': tune.choice([1, 2, 3]),
            'param2': tune.uniform(0.1, 1.0),
            # Your hyperparameters
        }

# Use exactly like MagicPigOptimizer
my_optimizer = MyAttentionOptimizer()
results = my_optimizer.run_optimization(num_samples=30)
```

## Configuration Options

### MagicPig Hyperparameters

- `l`: Number of LSH hash functions [16, 32, 64, 96, 128]
- `k`: Number of LSH hash bits [4, 8, 16, 32]  
- `packing`: Hash packing method ['int64', 'float32']
- `center`: Whether to use centering [True, False]
- `sink_size`: Number of sink tokens [64, 128, 256]
- `window_size`: Local attention window size [64, 128, 256, 512]

### Optimization Metrics

- **attention_error**: Primary objective - minimize attention approximation error
- **density**: Secondary objective - minimize attention matrix density (maximize sparsity)
- **combined_score**: `attention_error + 0.1 * density` (tunable weights)
- **gpu_runtime_s**: Runtime performance metric
- **memory_mb**: Peak GPU memory usage

### Ray Tune Parameters

- `num_samples`: Number of configurations to try (50+ recommended)
- `max_concurrent`: Maximum parallel trials (depends on GPU memory)
- `cpu_per_trial`: CPU cores per trial
- `gpu_per_trial`: GPU memory fraction per trial (0.3-0.8 recommended)
- `use_asha`: Enable ASHA scheduler for early stopping

## Results Analysis

The optimizer automatically generates:

1. **Console Output**: Real-time progress and best configurations
2. **JSON Results**: `raytune_{attention_type}_detailed_results.json` with:
   - Best configuration and metrics
   - Top-K configurations ranked by performance
   - Full experiment data
3. **Ray Tune Logs**: Detailed logs in the results directory

### Key Metrics to Monitor

- **Combined Score**: Lower is better (target < 1.0)
- **Attention Error**: Lower is better (target < 0.1) 
- **Density**: Lower is better (target < 0.2 for high sparsity)
- **GPU Runtime**: Faster is better for production use

## Best Practices

### For Development/Testing
- Use `max_requests=2`, `max_context_length=8000` for fast iteration
- Start with `num_samples=10` to validate setup
- Use `max_concurrent=1` to avoid memory issues

### For Production Optimization  
- Use `max_requests=10+`, full context length for realistic evaluation
- Use `num_samples=50+` for thorough search
- Tune `max_concurrent` based on your GPU memory (e.g., 2-4 for 40GB+ GPUs)
- Enable ASHA scheduler for efficiency

### Resource Management
- Monitor GPU memory usage with `nvidia-smi`
- Adjust `gpu_per_trial` if you get OOM errors
- Use Ray's resource management for multi-GPU setups

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct directory and have all dependencies
2. **CUDA OOM**: Reduce `gpu_per_trial` or `max_concurrent`
3. **Ray Connection Issues**: Check Ray dashboard at http://localhost:8265
4. **Slow Optimization**: Reduce `max_requests` or `max_context_length` for faster trials

### Debug Mode

Add these for debugging:
```python
import ray
ray.init(local_mode=True)  # Single-threaded mode for debugging
```

### Performance Tips

- Use HyperOpt search algorithm for continuous parameters
- Enable ASHA scheduler for early stopping of poor configurations
- Cache model loading if possible (advanced)
- Use smaller benchmark datasets for hyperparameter search

## Future Extensions

The modular design supports easy extension to:

- **Hash Attention**: Optimize hash function parameters
- **Adaptive Sampling**: Tune adaptive thresholds  
- **Multi-head Patterns**: Optimize different sparsity per head
- **Layer-wise Optimization**: Different configurations per layer
- **Multi-benchmark Optimization**: Optimize across multiple tasks simultaneously

Each new attention mechanism only requires implementing 2-3 methods in a new subclass.
