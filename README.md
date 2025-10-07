# Sparse Attention Hub

A framework for sparse attention mechanisms with efficient algorithms, benchmarking tools, and HuggingFace integration.

## 🚀 Features

- **Sparse Attention Algorithms**: Double Sparsity, Hash Attention, and research masking strategies
- **HuggingFace Integration**: Seamless Transformers support via ModelAdapterHF
- **Benchmarking**: LongBench, Loogle, InfBench, and custom benchmarks
- **Metrics & Visualization**: Detailed logging and attention pattern visualization
- **Extensible**: Modular design for custom implementations

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

### Configuration

For HashAttention experiments, set the weights directory:

```bash
export SPARSE_ATTENTION_WEIGHTS_DIR=/path/to/hashattention/weights
```

## 🏗️ Architecture

The framework is organized into several key modules:

- **Sparse Attention**: Base classes and implementations (DoubleSparsity, HashAttention)
- **Adapters**: ModelAdapterHF for HuggingFace integration with request/response handling
- **Benchmarking**: Multiple datasets (LongBench, Loogle, InfBench) with BenchmarkExecutor
- **Metrics & Visualization**: MicroMetricLogger and PlotGenerator for analysis

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

Run benchmarks on long-context datasets:

```python
from sparse_attention_hub.benchmark import BenchmarkExecutor, LongBench

executor = BenchmarkExecutor()
results = executor.evaluate(LongBench(), adapter)
```

Available benchmarks: LongBench, Loogle, InfBench, AIME2024/2025, RULER, ZeroScrolls

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

### Development Setup

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