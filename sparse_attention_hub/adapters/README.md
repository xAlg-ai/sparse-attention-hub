# Adapters Module

The `adapters` module provides a unified interface for integrating sparse attention mechanisms with external model frameworks, particularly HuggingFace Transformers. This module enables seamless switching between dense and sparse attention modes while maintaining compatibility with existing model architectures.

## 🏗️ Architecture Overview

The adapters module follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    ModelAdapter Interface                   │
│  (SparseAttentionAdapterInterface + ModelHubAdapterInterface) │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 ModelAdapterHF (HuggingFace)                │
│              Concrete implementation for HF models          │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    ModelServer (Singleton)                  │
│              Centralized model & tokenizer management       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **`ModelAdapter`**: Abstract base class defining the interface between sparse attention and external libraries
- **`ModelAdapterHF`**: HuggingFace-specific implementation with full transformer integration
- **`ModelServer`**: Singleton service for centralized model and tokenizer management
- **`Request`/`RequestResponse`**: Data structures for context-aware question answering
- **Utility classes**: Configuration, error handling, and GPU memory management

## 🚀 Quick Start

### Basic Usage

```python
import torch
from sparse_attention_hub.adapters import ModelAdapterHF, Request, RequestResponse
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            SinkMaskerConfig,
            LocalMaskerConfig
        )

# 1. Create sparse attention configuration
sparse_config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=256)
    ]
)

# 2. Initialize adapter
adapter = ModelAdapterHF(
    model_name="meta-llama/Llama-3.2-1B",
    sparse_attention_config=sparse_config,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)

# 3. Create and process request
request = Request(
    context="The capital of France is Paris. It is known for the Eiffel Tower.",
    questions="What is the capital of France?",
    answer_prefix="Answer: "
)

response = adapter.process_request(
    request=request,
    generation_kwargs={"max_new_tokens": 50},
    request_kwargs={"max_context_length": 1024}
)

print(response.responses)  # "Answer: The capital of France is Paris."
```

### Dense-Only Mode

```python
# For dense attention (no sparse attention)
adapter = ModelAdapterHF(
    model_name="meta-llama/Llama-3.2-1B",
    sparse_attention_config=None,  # No sparse attention = dense mode
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)
```

## 📋 Detailed Usage Examples

### 1. Multiple Question Processing

```python
# Process multiple questions with the same context
request = Request(
    context="Paris is the capital of France. It has a population of 2.1 million people.",
    questions=[
        "What is the capital of France?",
        "What is the population of Paris?",
        "Which country is Paris in?"
    ],
    answer_prefix="Answer: "
)

response = adapter.process_request(
    request=request,
    generation_kwargs={"max_new_tokens": 30},
    request_kwargs={"max_context_length": 2048}
)

# response.responses will be a list of 3 answers
for i, answer in enumerate(response.responses):
    print(f"Q{i+1}: {answer}")
```

### 2. Advanced Sparse Attention Configuration

```python
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            SinkMaskerConfig,
            LocalMaskerConfig,
            OracleTopKConfig
        )

# StreamingLLM with Oracle Top-K configuration
sparse_config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),      # Keep first 128 tokens
        LocalMaskerConfig(window_size=256),   # Local attention window
        OracleTopKConfig(heavy_size=256)      # Select top 256 most important tokens
    ]
)

adapter = ModelAdapterHF(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    sparse_attention_config=sparse_config,
    model_kwargs={"torch_dtype": torch.bfloat16}
)
```

### 3. Context Manager for Sparse Mode

```python
# Explicitly control when sparse attention is used
adapter = ModelAdapterHF(
    model_name="microsoft/Phi-4-mini-instruct",
    sparse_attention_config=sparse_config
)

# Process with sparse attention
with adapter.enable_sparse_mode():
    response = adapter.process_request(
        request=request,
        generation_kwargs={"max_new_tokens": 50},
        request_kwargs={"max_context_length": 1024}
    )

# Process with dense attention (default mode)
response_dense = adapter.process_request(
    request=request,
    generation_kwargs={"max_new_tokens": 50},
    request_kwargs={"max_context_length": 1024}
)
```

## 🔧 Configuration Options

### ModelAdapterHF Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | Required | HuggingFace model identifier |
| `sparse_attention_config` | `SparseAttentionConfig` | `None` | Sparse attention configuration (None for dense mode) |
| `model_kwargs` | `Dict[str, Any]` | `{}` | Additional model creation arguments |
| `tokenizer_kwargs` | `Dict[str, Any]` | `{}` | Additional tokenizer creation arguments |
| `device` | `str` | `"cuda"` if available else `"cpu"` | Device to run the model on (use `None` for CPU) |

### Common Model Arguments

```python
model_kwargs = {
    "torch_dtype": torch.bfloat16,  # Data type
    "device_map": "auto",           # Automatic device mapping
    "trust_remote_code": True,      # Trust remote code for custom models
    "low_cpu_mem_usage": True,      # Optimize CPU memory usage
}

tokenizer_kwargs = {
    "padding_side": "left",         # Padding direction
    "truncation": True,             # Enable truncation
    "max_length": 2048,            # Maximum sequence length
}
```

### Request Parameters

```python
# Generation parameters (passed to model.generate())
generation_kwargs = {
    "max_new_tokens": 100,          # Maximum tokens to generate
    "temperature": 0.7,             # Sampling temperature
    "top_p": 0.9,                   # Nucleus sampling
    "do_sample": True,              # Enable sampling
    "pad_token_id": tokenizer.eos_token_id,  # Padding token
}

# Request processing parameters
request_kwargs = {
    "max_context_length": 2048,     # Maximum context length
    "truncate_context": True,       # Truncate long contexts
}
```

## 🎯 Best Practices

### 1. Memory Management

```python
# Use context managers for explicit cleanup
with ModelAdapterHF(...) as adapter:
    response = adapter.process_request(request, {}, {})

# Or manually clean up
adapter = ModelAdapterHF(...)
try:
    response = adapter.process_request(request, {}, {})
finally:
    del adapter  # Triggers cleanup
```

### 2. Performance Optimization

```python
# Use appropriate data types and settings
model_kwargs = {
    "torch_dtype": torch.bfloat16,  # Faster than float32
    "low_cpu_mem_usage": True,      # Reduce CPU memory usage
}

request_kwargs = {
    "max_context_length": 1024,     # Adjust based on your needs
}
```
## 📚 API Reference

### Core Classes

- **`ModelAdapter`**: Abstract base class for model adapters
- **`ModelAdapterHF`**: HuggingFace implementation
- **`ModelServer`**: Centralized model management
- **`ModelServerHF`**: HuggingFace model server implementation
- **`Request`**: Input data structure for context + questions
- **`RequestResponse`**: Output data structure for responses

### Key Methods

- **`process_request()`**: Process a request and return response
- **`enable_sparse_mode()`**: Context manager for sparse attention
- **`get_custom_attention_function()`**: Get attention function for external libraries
- **`get_model()`**: Get or create model instance
- **`get_tokenizer()`**: Get or create tokenizer instance

For detailed API documentation, see the individual class docstrings and type hints.
