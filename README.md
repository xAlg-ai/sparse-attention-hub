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

### Adapter System
- **ModelAdapterHF**: Unified adapter for HuggingFace integration
- **Request/RequestResponse**: Structured request/response handling
- **ModelAdapter**: Abstract base class for model adapters
- **ModelHubAdapterInterface**: Interface for model hosting libraries
- **SparseAttentionAdapterInterface**: Interface for sparse attention integration

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
from sparse_attention_hub.adapters import ModelAdapterHF, Request, RequestResponse
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)
from sparse_attention_hub.benchmark import BenchmarkExecutor
from sparse_attention_hub.plotting import PlotGenerator, Granularity

# Create sparse attention configuration
local_config = LocalMaskerConfig(window_size=16)
sink_config = SinkMaskerConfig(sink_size=4)
sparse_attention_config = ResearchAttentionConfig(
    masker_configs=[local_config, sink_config]
)

# Create model adapter
adapter = ModelAdapterHF(
    model_name="microsoft/DialoGPT-small",
    sparse_attention_config=sparse_attention_config,
    device="auto"
)

# Create request
request = Request(
    context="The capital of France is Paris. It is known for the Eiffel Tower.",
    questions=["What is the capital of France?", "What is Paris known for?"]
)

# Process request with sparse attention
with adapter.enable_sparse_mode():
    response = adapter.process_request(request)
    print(response.responses)  # ['Paris', 'The Eiffel Tower']

# Process request with dense attention (default)
response = adapter.process_request(request)
print(response.responses)

# Run benchmarks and visualizations
benchmark_executor = BenchmarkExecutor()
plot_generator = PlotGenerator()
plot_path = plot_generator.generate_plot(Granularity.PER_HEAD)
```

## 📊 Benchmarking

The framework supports multiple benchmark datasets and integrates seamlessly with the new adapter system:

- **LongBench**: Long-context understanding tasks
- **Loogle**: Dependency tracking benchmarks  
- **InfBench**: Infinite context benchmarks

```python
from sparse_attention_hub.benchmark import BenchmarkExecutor
from sparse_attention_hub.benchmark.datasets import LongBench
from sparse_attention_hub.adapters import ModelAdapterHF

# Create adapter with sparse attention
adapter = ModelAdapterHF(
    model_name="microsoft/DialoGPT-small",
    sparse_attention_config=your_sparse_config
)

# Run benchmarks
executor = BenchmarkExecutor()
benchmark = LongBench()
results = executor.evaluate(benchmark, adapter)
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
│   ├── adapters/                   # Model adapter system
│   ├── benchmark/                  # Benchmarking tools
│   ├── metrics/                    # Metrics and logging
│   ├── plotting/                   # Visualization tools
│   └── testing/                    # Testing utilities
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
├── scripts/                        # Utility scripts
├── tutorials/                      # Tutorial notebooks and examples
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