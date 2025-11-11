# Research Attention Module

The `research_attention` module provides a flexible framework for implementing and experimenting with various sparse attention mechanisms. It's designed for research purposes and allows easy composition of different masking strategies to create novel attention patterns.

## 📁 Folder Structure

```
research_attention/
├── __init__.py                    # Module exports
├── base.py                        # ResearchAttention and ResearchAttentionConfig classes
├── maskers/                       # Masker implementations
│   ├── __init__.py
│   ├── base.py                    # Base masker classes and registry system
│   ├── fixed/                     # Fixed pattern maskers
│   │   ├── __init__.py
│   │   ├── base.py                # FixedMasker base class
│   │   └── implementations/       # Specific fixed masker implementations
│   │       ├── __init__.py
│   │       ├── basic_fixed.py     # Local, Causal, Sink maskers
│   │       ├── hashattention_top_k.py # HashAttention top-k (Desai et al. 2024)
│   │       ├── oracle_top_k.py    # Oracle-based top-K selection
│   │       └── oracle_top_p.py    # Oracle-based top-P selection
│   └── sampling/                  # Sampling-based maskers
│       ├── __init__.py
│       ├── base.py                # SamplingMasker base class
│       └── implementations/       # Specific sampling masker implementations
│           ├── __init__.py
│           ├── adaptive_sampling.py # vAttention adaptive sampling (Desai et al. 2025)
│           ├── magic_pig.py       # LSH-based sampling (Chen et al. 2024)
│           └── random_sampling.py # Random sampling
```

## 🎭 What is a Masker and Mask Object?

### Mask Object

A `Mask` object represents attention patterns which are manipulated by Masker objects. We store two main formats:

1. **Dense Representation**: Full tensor of shape `(batch_size, num_heads, seq_len_queries, seq_len_keys)`
2. **Sparse Representation**: Compressed format using indices and pointer arrays

**Special masks:**
- **Empty Mask**: All elements are 0.0 (no attention connections)
- **Full Mask**: All elements are 1.0 (dense attention, memory-optimized)
- These are stored with special flags for optimizing computation and memory usage.

**Mask Operations:**
- `merge_mask()`: Combine two masks (additive)
- `get_density()`: Calculate sparsity ratio
- `is_full_mask()`: Check if mask represents full attention
- `is_empty_mask()`: Check if mask has no active elements

### Masker

A `Masker` is a component that applies specific masking logic to attention computation. Each masker implements the `add_mask()` method which:

1. Takes attention tensors (queries, keys, values) and a previous mask
2. Applies its specific masking logic, adding more active elements to the mask
3. Returns a new mask that can be further processed by subsequent maskers

**Key Concept**: Maskers are **additive** - they add attention connections to the existing mask rather than replacing it entirely. This allows for composition of different attention patterns.

**Core Function: `add_mask()`**

```python
def add_mask(
    self,
    keys: torch.Tensor,           # Key tensor (b, h, sq, d)
    queries: torch.Tensor,        # Query tensor (b, h, sk, d)  
    values: torch.Tensor,         # Value tensor (b, h, sq, d)
    attention_mask: torch.Tensor, # Optional attention mask
    scaling: float,               # Attention scaling factor
    dropout: float,               # Dropout probability
    sparse_meta_data: Dict,       # Additional metadata
    previous_mask: Mask,          # Mask from previous masker
    **kwargs: Dict[str, Any]      # Additional arguments
) -> Mask:
    """Apply masking logic and return new mask."""
```

## 🔬 Implemented Sparse Attention Mechanisms

### Fixed Pattern Maskers

| Masker | Description |
|--------|-------------|
| **LocalMasker** | Implements local attention with configurable window size around each query position |
| **CausalMasker** | Applies causal masking to prevent future token attention |
| **SinkMasker** | Creates sink token attention pattern for efficient long sequences |
| **TopKMasker** | Selects top-K most important key positions for each query |
| **TopPMasker** | Implements nucleus sampling (top-P) for attention selection |
| **OracleTopK** | Uses ground truth attention weights to select optimal top-K positions (research only) |
| **OracleTopPMasker** | Uses ground truth attention weights for optimal top-P selection (research only) |
| **HashAttentionTopKMasker** |  top-K selection compatible with HashAttention (Desai et. al 2024) |

### Sampling Maskers

| Masker | Description |
|--------|-------------|
| **RandomSamplingMasker** | Randomly samples a fraction of key positions for each query |
| **MagicPig** | Uses Locality Sensitive Hashing (LSH) for similarity-based attention sampling (Chen et. al. 2024) |
| **AdaptiveSamplingMasker** | Adaptively adjusts sampling based on attention patterns (Sampling logic from vAttention Desai et. al 2025) |

## 📚 Research Papers and Implementations

This module implements several state-of-the-art sparse attention mechanisms from recent research (soon to support more):

- **HashAttention (Desai et al. 2024)**: Hash-based attention mechanism that uses learned hash functions to efficiently select attention positions
- **vAttention (Desai et al. 2025)**: Adaptive sampling mechanism that dynamically adjusts attention patterns based on input characteristics
- **MagicPig (Chen et al. 2024)**: Locality Sensitive Hashing (LSH) based attention sampling for similarity-based sparse attention

## ⚙️ Parameter Configuration

The examples below use parameters optimized through hyperparameter search (see `benchmark/raytune/run_optimize_configs.py`). These parameters have been tuned for optimal performance across various benchmarks and model configurations.

**Key Parameter Sources:**
- **HashAttention**: Uses production-ready parameters with `hat_bits=32`, `hat_mlp_layers=3`, `hat_mlp_hidden_size=128`
- **MagicPig**: Optimized LSH parameters with `lsh_l=8`, `lsh_k=8`
- **Sink/Local**: Standard parameters with `sink_size=128`, `window_size=128`
- **AdaptiveSampling**: Tuned parameters for vAttention implementation

## 💡 Examples: Creating Research Attention

### Example 1: HashAttention Implementation

```python
import torch
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    HashAttentionTopKMaskerConfig,
    LocalMaskerConfig,
    SinkMaskerConfig
)

# Create HashAttention configuration (Desai et al. 2024)
# Note: Requires pre-trained HashAttention weights file (.pkl format)
# it can be obtained from https://github.com/xAlg-ai/HashAttention-1.0/tree/main/artifacts/ for some models
config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),                         # Sink tokens for long sequences
        LocalMaskerConfig(window_size=128),                      # Local attention window
        HashAttentionTopKMaskerConfig(
            heavy_size=0.1,                                      # HashAttention top-K selection
            hat_bits=32,                                         # Hash attention bits
            hat_mlp_layers=3,                                    # MLP layers
            hat_mlp_hidden_size=128,                             # Hidden size
            hat_mlp_activation='silu',                           # Activation function
            hat_weight_file='path/to/hat_weights.pkl'            # Pre-trained weights file
        ),
    ]
)

# Create ResearchAttention instance
hash_attention = ResearchAttention.create_from_config(config)

# Use in attention computation
attention_output, attention_weights = hash_attention.custom_attention(
    module=attention_module,
    queries=queries,
    keys=keys,
    values=values,
    attention_mask=attention_mask,
    scaling=1.0,
    dropout=0.1,
    sparse_meta_data={},
    layer_idx=0
)
```

### Example 2: MagicPig Implementation

```python
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    MagicPigConfig
)

# Create MagicPig configuration (Chen et al. 2024)
config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),                         # Sink tokens for long sequences
        LocalMaskerConfig(window_size=128),                      # Local attention window
        # MagicPig with LSH parameters for similarity-based sampling
        MagicPigConfig(
            lsh_l=8,                  # 8 LSH tables
            lsh_k=8,                  # 8 bits per table
            center=True,              # Center keys and queries
            packing="int64",          # Use int64 packing for efficiency
            seed=42                   # Reproducible results
        )
    ]
)

magic_pig_attention = ResearchAttention.create_from_config(config)
```

### Example 3: vAttention Adaptive Sampling with Oracle Top-K

```python
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig
)

# Create vAttention adaptive sampling configuration (Desai et al. 2025)
# Combines OracleTopK with AdaptiveSampling
config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),                         # Sink tokens for long sequences
        LocalMaskerConfig(window_size=128),                      # Local attention window
        OracleTopKConfig(heavy_size=0.1),                        # Oracle top-K selection (research only)
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.1,      # Base sampling rate
            epsilon=0.25,                # Error bound
            delta=0.25,                  # Confidence bound
            init_offset=128,             # Start index for sampling
            local_offset=128             # End offset for sampling
        )
    ]
)

vattention_adaptive = ResearchAttention.create_from_config(config)
```

### Example 4: Complex Multi-Masker Configuration with HashAttention

```python
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    HashAttentionTopKMaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    MagicPigConfig
)

# Combine multiple maskers for complex attention patterns
# This example demonstrates a comprehensive sparse attention setup
config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),             # Sink tokens for long sequences
        LocalMaskerConfig(window_size=128),          # Local attention window
        HashAttentionTopKMaskerConfig(               # HashAttention top-K selection
            heavy_size=0.1,
            hat_bits=32,
            hat_mlp_layers=3,
            hat_mlp_hidden_size=128,
            hat_mlp_activation='silu',
            hat_weight_file='/data/apdesai/code/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.pt'
        ),
        MagicPigConfig(                              # LSH-based sampling
            lsh_l=75,
            lsh_k=8
        )
    ]
)

complex_attention = ResearchAttention.create_from_config(config)
```

## 🛠️ How to Add a New Masker

### Step 1: Create Configuration Class

```python
from dataclasses import dataclass, field
from typing import Any, Dict
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry
)

@dataclass
class MyCustomMaskerConfig(MaskerConfig):
    """Configuration for MyCustomMasker."""
    
    custom_param: float
    another_param: int = 10
    search_space: Dict[str, Any] = field(default_factory=dict)
```

### Step 2: Implement the Masker Class

```python
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    ResearchMasker,
    AttentionTensorDimensions
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
import torch

@MaskerRegistry.register(MyCustomMaskerConfig)
class MyCustomMasker(ResearchMasker):
    """Custom masker implementation."""
    
    def __init__(self, config: MyCustomMaskerConfig) -> None:
        """Initialize custom masker."""
        super().__init__(config)
        self.custom_param = config.custom_param
        self.another_param = config.another_param
    
    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any]
    ) -> Mask:
        """Implement your custom masking logic here."""
        
        # Extract tensor dimensions
        dims = self._extract_tensor_dimensions(keys, queries)
        
        # If previous mask is full, return it (optimization)
        if previous_mask.is_full_mask():
            return previous_mask
        
        # Implement your masking logic
        # Example: Create a custom attention pattern
        mask_shape = (dims.batch_size, dims.num_heads, 
                     dims.seq_len_queries, dims.seq_len_keys)
        
        # Your custom logic here...
        # For example, create a custom mask based on your parameters
        custom_mask = self._create_custom_mask(dims, keys, queries)
        
        # Merge with previous mask
        return previous_mask.merge_mask(custom_mask, inplace=False)
    
    def _create_custom_mask(
        self, 
        dims: AttentionTensorDimensions, 
        keys: torch.Tensor, 
        queries: torch.Tensor
    ) -> Mask:
        """Create your custom mask pattern."""
        # Implement your specific masking logic
        # This is where you define how your masker works
        
        # Example: Create a simple pattern
        mask_shape = (dims.batch_size, dims.num_heads, 
                     dims.seq_len_queries, dims.seq_len_keys)
        
        # Your implementation here...
        # Return a Mask object
        pass
    
    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "MyCustomMasker":
        """Create masker from configuration."""
        if not isinstance(config, MyCustomMaskerConfig):
            raise ValueError(f"Expected MyCustomMaskerConfig, got {type(config)}")
        return cls(config)
```

### Step 3: Register and Use

```python
# Import your masker module to register it
from your_module import MyCustomMasker, MyCustomMaskerConfig

# Use in ResearchAttention configuration
config = ResearchAttentionConfig(
    masker_configs=[
        MyCustomMaskerConfig(custom_param=0.5, another_param=20)
    ]
)

research_attention = ResearchAttention.create_from_config(config)
```

### Step 4: Add to Module Exports

Update the appropriate `__init__.py` files to export your new masker:

```python
# In maskers/__init__.py or implementations/__init__.py
from .my_custom_masker import MyCustomMasker, MyCustomMaskerConfig

__all__ = [
    # ... existing exports ...
    "MyCustomMasker",
    "MyCustomMaskerConfig",
]
```

This framework provides a foundation for experimenting with novel sparse attention mechanisms while maintaining clean, modular, and efficient code.
