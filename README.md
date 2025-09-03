<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/logo.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/logo.png">
    <!-- Fallback -->
    <img alt="Sparse Attention Hub" src="./docs/logo.png" width="50%">
  </picture>
</p>

<div align="center">

**A framework for implementing and evaluating sparse attention mechanisms in transformer models**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.20+-yellow?style=flat-square)](https://huggingface.co/transformers/)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat-square)](https://github.com/psf/black)

</div>

Sparse Attention Hub provides efficient implementations of sparse attention algorithms for transformer models. It includes research-oriented attention mechanisms, production-ready implementations, comprehensive benchmarking tools, and seamless integration with HuggingFace Transformers.

## Key Features

- **Multiple Sparse Attention Algorithms**: Hash Attention, Double Sparsity, and configurable masking strategies
- **HuggingFace Integration**: Drop-in replacement for standard attention with minimal code changes  
- **Comprehensive Benchmarking**: Built-in support for LongBench, Ruler, AIME, InfiniteBench, and custom datasets
- **Research Tools**: Flexible masker system for experimenting with novel attention patterns
- **Performance Monitoring**: Detailed metrics logging and visualization capabilities
- **Modular Architecture**: Easy to extend with custom attention mechanisms

## Installation

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone and install the project
git clone https://github.com/xAlg-ai/sparse-attention-hub.git
cd sparse-attention-hub
poetry install
```

### Development Installation

```bash
# Install with development dependencies
poetry install --with dev

# Install with all optional dependencies
poetry install --with dev,docs,benchmarks
```

### From Source (pip)

```bash
git clone https://github.com/xAlg-ai/sparse-attention-hub.git
cd sparse-attention-hub
pip install -e .
```

### Requirements

The project requires Python 3.9+ and key dependencies include PyTorch, Transformers, and several benchmarking libraries. All dependencies are managed through Poetry and defined in `pyproject.toml`.

## Quick Start

### Basic Usage with HuggingFace Models

```python
from sparse_attention_hub.adapters import ModelAdapterHF, Request
from sparse_attention_hub.sparse_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)

# Configure sparse attention (StreamingLLM pattern)
sparse_config = ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=4),      # Keep first 4 tokens
    LocalMaskerConfig(window_size=64)    # Keep last 64 tokens
])

# Create model adapter
adapter = ModelAdapterHF(
    model_name="microsoft/DialoGPT-small",
    sparse_attention_config=sparse_config,
    device="auto"
)

# Create request
request = Request(
    context="Paris is the capital of France. It is famous for the Eiffel Tower and the Louvre Museum.",
    questions=["What is the capital of France?", "What is Paris famous for?"]
)

# Process with sparse attention
with adapter.enable_sparse_mode():
    response = adapter.process_request(request)
    print(response.responses)

# Process with dense attention (default)
response = adapter.process_request(request)
print(response.responses)
```

### Available Sparse Attention Methods

```python
# Hash Attention
from sparse_attention_hub.sparse_attention import HashAttentionConfig, EfficientAttentionConfig

hash_config = EfficientAttentionConfig(
    implementation_config=HashAttentionConfig(
        num_hash_functions=4,
        hash_budget=512
    )
)

# Double Sparsity  
from sparse_attention_hub.sparse_attention import DoubleSparsityConfig

double_config = EfficientAttentionConfig(
    implementation_config=DoubleSparsityConfig(
        channel_config=ChannelConfig(channels_per_head=32),
        sparsity_ratio=0.1
    )
)

# Custom masking patterns
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    TopKMaskerConfig, OracleTopK
)

custom_config = ResearchAttentionConfig(masker_configs=[
    TopKMaskerConfig(k=128),
    SinkMaskerConfig(sink_size=4)
])
```

## Benchmarking

### Running Benchmarks

```python
from benchmark import BenchmarkExecutor, BenchmarkConfig
from benchmark.executor_config import AdapterConfig

# Configure benchmark
benchmark_config = BenchmarkConfig(
    benchmark_name="longbench",
    subsets=["narrativeqa", "qasper"]
)

adapter_config = AdapterConfig(
    model_name="microsoft/DialoGPT-small",
    sparse_attention_config=sparse_config
)

# Run evaluation
executor = BenchmarkExecutor(
    result_dir="./results",
    gpus=[0, 1],
    max_concurrent_runs=2
)

results = executor.run_benchmarks(
    benchmark_configs=[benchmark_config],
    adapter_configs=[adapter_config]
)
```

### Supported Benchmarks

- **LongBench**: Long-context understanding tasks (22 standard + 13 extended datasets)
- **Ruler**: Synthetic tasks for testing context length capabilities  
- **AIME 2024/2025**: Mathematical reasoning benchmarks
- **InfiniteBench**: Infinite context evaluation tasks
- **Loogle**: Dependency tracking and retrieval tasks
- **ZeroScrolls**: Long document understanding
- **Custom benchmarks**: Easy to add new evaluation datasets

## Architecture

### Core Components

- **`sparse_attention/`**: Core sparse attention implementations
  - `base.py`: Abstract interfaces for all attention mechanisms
  - `efficient_attention/`: Production-ready algorithms (Hash Attention, Double Sparsity)
  - `research_attention/`: Experimental masking strategies for research
  - `utils/`: Common utilities and mask operations

- **`adapters/`**: Integration layer for different frameworks
  - `huggingface.py`: HuggingFace Transformers integration
  - `model_servers/`: Centralized model and tokenizer management
  - `base.py`: Abstract adapter interfaces

- **`benchmark/`**: Comprehensive benchmarking system
  - Individual benchmark implementations (LongBench, Ruler, etc.)
  - `executor.py`: Parallel execution with GPU management
  - `base.py`: Abstract benchmark interface

- **`metric_logging/`**: Performance monitoring and analysis
- **`plotting/`**: Visualization tools for attention patterns and results

## Advanced Usage

### Custom Maskers

```python
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import ResearchMasker
from sparse_attention_hub.sparse_attention.utils import Mask

class CustomMasker(ResearchMasker):
    def add_mask(self, keys, queries, values, previous_mask, **kwargs):
        # Implement custom masking logic
        custom_mask = self.create_custom_mask(queries.shape)
        return previous_mask.combine_with(custom_mask)
```

### Metrics and Monitoring

```python
from sparse_attention_hub.metric_logging import MicroMetricLogger

logger = MicroMetricLogger()
# Metrics are automatically logged when using ResearchAttention
# Access logged data for analysis
metrics_data = logger.get_logged_metrics()
```

## Testing

```bash
# Run all tests
python scripts/run_tests.py --type all

# Run specific test categories
python scripts/run_tests.py --type unit
python scripts/run_tests.py --type integration

# Run tests with coverage
make test-coverage
```

## Development

### Code Quality Tools

```bash
# Install development dependencies
poetry install --with dev

# Activate Poetry shell
poetry shell

# Format code
bash scripts/format.sh

# Run linting
bash scripts/lint.sh

# Set up pre-commit hooks
pre-commit install
```

### Project Structure

```
sparse-attention-hub/
├── sparse_attention_hub/          # Main package
│   ├── sparse_attention/          # Sparse attention implementations
│   ├── adapters/                  # Model integration adapters
│   ├── metric_logging/            # Performance monitoring
│   └── plotting/                  # Visualization tools
├── benchmark/                     # Benchmarking framework
│   ├── longbench/                 # LongBench implementation
│   ├── ruler/                     # Ruler benchmark
│   ├── AIME2024/, AIME2025/       # Mathematical reasoning
│   └── scripts/                   # Benchmark execution scripts
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── tutorials/                     # Examples and tutorials
└── docs/                          # Documentation
```

## Contributing

We welcome contributions to improve sparse attention implementations, add new benchmarks, or enhance the framework. Please ensure all code follows the project's formatting standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **Repository**: https://github.com/xAlg-ai/sparse-attention-hub
- **Documentation**: https://sparse-attention-hub.readthedocs.io
- **Issues**: https://github.com/xAlg-ai/sparse-attention-hub/issues

## Acknowledgments

This project implements sparse attention mechanisms from various research papers. We acknowledge the original authors and the open-source community for their contributions to advancing efficient attention algorithms.