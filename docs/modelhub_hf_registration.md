# ModelHubHF Attention Registration

This document describes the new attention registration functionality in `ModelHubHF`, which allows you to register sparse attention implementations with the HuggingFace transformers library without replacing forward methods.

## Overview

The `ModelHubHF` class now provides a modern approach to integrating sparse attention mechanisms with HuggingFace models by:

1. **Registering attention implementations** with the transformers library's attention registry
2. **Using adapters** to bridge sparse attention implementations to the HuggingFace attention interface
3. **Configuring models** to use registered attention functions by name
4. **Managing the lifecycle** of attention registrations

## Key Components

### BaseAttentionFunction

Abstract base class that defines the interface for attention functions compatible with HuggingFace transformers.

```python
from sparse_attention_hub.model_hub import BaseAttentionFunction

class CustomAttention(BaseAttentionFunction):
    def attention_forward(self, module, query_states, key_states, value_states, **kwargs):
        # Your attention implementation
        pass
    
    @classmethod
    def get_attention_name(cls):
        return "custom_attention"
```

### Attention Adapters

Adapter classes that bridge sparse attention implementations to the BaseAttentionFunction interface:

- `SparseAttentionAdapter`: For base SparseAttention implementations
- `ResearchAttentionAdapter`: For ResearchAttention implementations  
- `EfficientAttentionAdapter`: For EfficientAttention implementations

### ModelHubHF

Enhanced HuggingFace model hub with attention registration capabilities.

## Usage Examples

### Basic Registration

```python
from sparse_attention_hub.model_hub import ModelHubHF
from sparse_attention_hub.sparse_attention import ResearchAttention, ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)

# Initialize ModelHubHF
model_hub = ModelHubHF()

# Create sparse attention implementation
local_config = LocalMaskerConfig(window_size=8)
sink_config = SinkMaskerConfig(sink_size=4)
research_config = ResearchAttentionConfig(masker_configs=[sink_config, local_config])
research_attention = ResearchAttention.create_from_config(research_config)

# Register with transformers
attention_name = model_hub.register_sparse_attention(
    research_attention,
    attention_name="local_sink_attention",
    config={"description": "Local attention with sink tokens"}
)
```

### Model Configuration

```python
# Configure a model to use the registered attention
model_hub.configure_model_attention(model, attention_name)

# The model will now use your sparse attention implementation
```

### Lifecycle Management

```python
# List registered attention functions
registered = model_hub.list_registered_attention_functions()
print(registered)  # {'local_sink_attention': 'ResearchAttentionAdapter'}

# Get adapter instance
adapter = model_hub.get_attention_adapter(attention_name)

# Unregister when done
model_hub.unregister_attention_function(attention_name)
```

## API Reference

### ModelHubHF Methods

#### `register_sparse_attention(sparse_attention, attention_name=None, config=None)`

Register a sparse attention implementation with the transformers library.

**Parameters:**
- `sparse_attention`: The sparse attention implementation to register
- `attention_name`: Optional custom name for the attention function
- `config`: Optional configuration dictionary

**Returns:** The registered attention function name

**Raises:** 
- `ImportError`: If transformers library is not available
- `ValueError`: If attention_name already exists

#### `unregister_attention_function(attention_name)`

Unregister an attention function.

**Parameters:**
- `attention_name`: Name of the attention function to unregister

**Raises:** `ValueError` if attention function is not registered

#### `configure_model_attention(model, attention_name, layer_indices=None)`

Configure a model to use a specific registered attention function.

**Parameters:**
- `model`: HuggingFace model instance
- `attention_name`: Name of the registered attention function
- `layer_indices`: Optional list of layer indices to apply attention to

**Raises:** `ValueError` if attention function is not registered

#### `list_registered_attention_functions()`

List all registered attention functions.

**Returns:** Dict mapping attention names to their class names

#### `get_attention_adapter(attention_name)`

Get the attention adapter instance by name.

**Parameters:**
- `attention_name`: Name of the registered attention function

**Returns:** The attention adapter instance

**Raises:** `ValueError` if attention function is not registered

## Integration with HuggingFace

The registration system integrates with HuggingFace's `ALL_ATTENTION_FUNCTIONS` registry, which means:

1. **Automatic Discovery**: Registered attention functions are automatically available to HuggingFace models
2. **Configuration Support**: Models can be configured to use specific attention implementations via `attn_implementation`
3. **Seamless Integration**: No need to modify model forward methods or monkey-patch classes

## Example: Complete Workflow

```python
import torch
from sparse_attention_hub.model_hub import ModelHubHF
from sparse_attention_hub.sparse_attention import ResearchAttention, ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)

# 1. Initialize ModelHubHF
model_hub = ModelHubHF()

# 2. Create sparse attention
local_config = LocalMaskerConfig(window_size=8)
sink_config = SinkMaskerConfig(sink_size=4)
research_config = ResearchAttentionConfig(masker_configs=[sink_config, local_config])
research_attention = ResearchAttention.create_from_config(research_config)

# 3. Register attention
attention_name = model_hub.register_sparse_attention(
    research_attention,
    attention_name="local_sink_attention"
)

# 4. Load and configure model (pseudo-code)
# model = AutoModel.from_pretrained("model_name")
# model_hub.configure_model_attention(model, attention_name)

# 5. Use model with sparse attention
# outputs = model(input_ids)

# 6. Cleanup
model_hub.unregister_attention_function(attention_name)
```

## Benefits

1. **No Forward Method Replacement**: Uses HuggingFace's native attention registry
2. **Type Safety**: Proper adapter pattern with type checking
3. **Lifecycle Management**: Clean registration and unregistration
4. **Multiple Attention Types**: Support for Research, Efficient, and base sparse attention
5. **Configuration Support**: Custom configuration for each registered attention
6. **Backward Compatibility**: Legacy methods still available with deprecation warnings

## Requirements

- `torch`: For tensor operations
- `transformers`: For HuggingFace integration (optional but recommended)

## Migration from Legacy Methods

If you're using the legacy methods (`replaceAttentionInterface`, etc.), migrate to the new registration system:

**Old:**
```python
model_hub.replaceAttentionInterface(model, attention_callable, "my_attention")
```

**New:**
```python
attention_name = model_hub.register_sparse_attention(sparse_attention, "my_attention")
model_hub.configure_model_attention(model, attention_name)
```

The new approach provides better integration, type safety, and lifecycle management.