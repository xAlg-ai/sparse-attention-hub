# Sparse Attention Hub

A comprehensive framework for sparse attention mechanisms in deep learning models. This project provides implementations of various sparse attention algorithms, benchmarking tools, and seamless integration with popular model frameworks like HuggingFace Transformers.

## 🚀 Features

- **Multiple Sparse Attention Algorithms**: Implementations of efficient attention mechanisms including Double Sparsity, Hash Attention, and various research-oriented masking strategies
- **Framework Integration**: Seamless integration with HuggingFace Transformers and other popular frameworks
- **Comprehensive Benchmarking**: Built-in support for LongBench, Loogle, InfBench, and custom benchmarks
- **Advanced Metrics**: Micro-metrics logging system for detailed performance analysis
- **Visualization Tools**: Generate plots and heatmaps for attention pattern analysis
- **Extensible Architecture**: Modular design for easy extension and customization

## 📦 Installation

### From Source

```bash
git clone https://github.com/xAlg-ai/sparse-attention-hub.git
cd sparse-attention-hub
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## 🏗️ Architecture

The framework is organized into several key modules:

### Sparse Attention
- **Base Classes**: `SparseAttention`, `EfficientAttention`, `ResearchAttention`
- **Efficient Implementations**: `DoubleSparsity`, `HashAttention`
- **Research Maskers**: Various masking strategies for attention patterns
- **Generators**: Integration interfaces for different frameworks

### Model Hub
- **ModelHub**: Abstract interface for model integration
- **ModelHubHF**: HuggingFace-specific implementation

### Pipeline
- **Pipeline**: Base pipeline interface
- **PipelineHF**: HuggingFace-compatible pipeline
- **SparseAttentionServer**: Server for hosting sparse attention models

### Benchmarking
- **Benchmark**: Abstract benchmark interface
- **Datasets**: LongBench, Loogle, InfBench implementations
- **BenchmarkExecutor**: Execution and result management
- **ResultStorage**: Persistent storage for benchmark results

### Metrics & Visualization
- **MicroMetricLogger**: Singleton logger for detailed metrics
- **MicroMetrics**: TopkRecall, LocalError, SampleVariance implementations
- **PlotGenerator**: Visualization tools with multiple granularity levels

## 🚀 Quick Start

```python
from sparse_attention_hub import (
    SparseAttentionHF, 
    ModelHubHF,
    BenchmarkExecutor,
    PlotGenerator,
    Granularity
)
from sparse_attention_hub.sparse_attention.efficient import DoubleSparsity
from sparse_attention_hub.benchmark.datasets import LongBench

# Create sparse attention mechanism
sparse_attention = DoubleSparsity()
attention_generator = SparseAttentionHF(sparse_attention)

# Set up model hub
model_hub = ModelHubHF(api_token="your_token")

# Run benchmarks
benchmark_executor = BenchmarkExecutor()
longbench = LongBench()
benchmark_executor.register_benchmark(longbench)

# Generate visualizations
plot_generator = PlotGenerator()
plot_path = plot_generator.generate_plot(Granularity.PER_HEAD)
```

## 📊 Benchmarking

The framework supports multiple benchmark datasets:

- **LongBench**: Long-context understanding tasks
- **Loogle**: Dependency tracking benchmarks  
- **InfBench**: Infinite context benchmarks

```python
from sparse_attention_hub.benchmark import BenchmarkExecutor
from sparse_attention_hub.benchmark.datasets import LongBench

executor = BenchmarkExecutor()
benchmark = LongBench()
results = executor.evaluate(benchmark, your_model)
```

## 📈 Metrics and Logging

Track detailed performance metrics:

```python
from sparse_attention_hub.metrics import MicroMetricLogger
from sparse_attention_hub.metrics.implementations import TopkRecall

logger = MicroMetricLogger()
metric = TopkRecall(k=10)
logger.register_metric(metric)
logger.enable_metric_logging(metric)

# Log metrics during model execution
logger.log("layer_1", metric, computed_value)
```

## 🎨 Visualization

Generate attention pattern visualizations:

```python
from sparse_attention_hub.plotting import PlotGenerator, Granularity

generator = PlotGenerator()

# Generate different types of plots
token_plot = generator.generate_plot(Granularity.PER_TOKEN, data)
head_plot = generator.generate_plot(Granularity.PER_HEAD, data)
layer_plot = generator.generate_plot(Granularity.PER_LAYER, data)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python scripts/run_tests.py --type all

# Run only unit tests
python scripts/run_tests.py --type unit

# Run specific test
python scripts/run_tests.py --type specific --test-path tests/unit/test_metrics.py

# Discover available tests
python scripts/run_tests.py --discover

# Run tests with coverage
make test-coverage
```

## 🔧 Development Tools

### Code Formatting and Linting

The project uses comprehensive linting and formatting tools:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Format code
bash scripts/format.sh

# Run all linting checks
bash scripts/lint.sh

# Run specific linters
bash scripts/lint.sh --flake8
bash scripts/lint.sh --mypy
bash scripts/lint.sh --pylint
bash scripts/lint.sh --bandit

# Using Make commands
make format          # Format code
make lint           # Run all linting
make dev-check      # Quick format + lint check
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Update pre-commit hooks
pre-commit autoupdate
```

### Development Workflow

```bash
# Complete development setup
make dev-setup

# Run all development checks
make dev-check

# Simulate CI pipeline
make ci
```

## 📁 Project Structure

```
sparse-attention-hub/
├── sparse_attention_hub/          # Main package
│   ├── sparse_attention/           # Sparse attention implementations
│   ├── model_hub/                  # Model integration
│   ├── pipeline/                   # Pipeline implementations
│   ├── benchmark/                  # Benchmarking tools
│   ├── metrics/                    # Metrics and logging
│   ├── plotting/                   # Visualization tools
│   └── testing/                    # Testing utilities
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
├── examples/                       # Usage examples
├── scripts/                        # Utility scripts
└── docs/                          # Documentation
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **Repository**: https://github.com/xAlg-ai/sparse-attention-hub
- **Documentation**: https://sparse-attention-hub.readthedocs.io
- **Issues**: https://github.com/xAlg-ai/sparse-attention-hub/issues

## 🙏 Acknowledgments

This project implements and extends various sparse attention mechanisms from the research community. We acknowledge the original authors of these algorithms and the open-source community for their contributions.