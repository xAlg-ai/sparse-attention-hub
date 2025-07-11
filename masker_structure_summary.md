# Masker Class and Config Structure Summary

## 🏗️ Inheritance Hierarchy

```
ResearchMasker (ABC)
├── FixedMasker
│   ├── LocalMasker
│   ├── CausalMasker  
│   ├── SinkMasker
│   └── TopKMasker
│       ├── OracleTopK
│       ├── PQCache
│       ├── HashAttention
│       └── DoubleSparsity
└── SamplingMasker
    ├── RandomSamplingMasker
    └── MagicPig

MaskerConfig
├── FixedMaskerConfig
│   ├── LocalMaskerConfig
│   ├── SinkMaskerConfig
│   └── TopKMaskerConfig
│       ├── OracleTopKConfig
│       ├── PQCacheConfig
│       ├── HashAttentionConfig
│       └── DoubleSparsityConfig
└── SamplingMaskerConfig
    ├── RandomSamplingMaskerConfig
    └── MagicPigConfig
```

## 📋 All Masker Classes and Configs

### Fixed Pattern Maskers

| Masker Class | Config Class | Parameters |
|--------------|--------------|------------|
| `LocalMasker` | `LocalMaskerConfig` | `window_size: Union[float, int]` |
| `CausalMasker` | `FixedMaskerConfig` | None (uses base config) |
| `SinkMasker` | `SinkMaskerConfig` | `sink_size: Union[float, int]` |
| `OracleTopK` | `OracleTopKConfig` | `heavy_size: Union[float, int]` |
| `PQCache` | `PQCacheConfig` | `heavy_size`, `pq_sub_dim: int`, `pq_bits: int` |
| `HashAttention` | `HashAttentionConfig` | `heavy_size`, `hat_bits: int`, `hat_mlp_layers: int`, `hat_mlp_hidden_size: int` |
| `DoubleSparsity` | `DoubleSparsityConfig` | `heavy_size`, `group_factor: int`, `label_bits: int`, `channel_config: Any` |

### Sampling Maskers

| Masker Class | Config Class | Parameters |
|--------------|--------------|------------|
| `RandomSamplingMasker` | `RandomSamplingMaskerConfig` | `sampling_rate: Union[float, int]` |
| `MagicPig` | `MagicPigConfig` | `sampling_rate`, `lsh_l: int`, `lsh_k: int` |

## 🔧 Usage Pattern

All maskers follow the same pattern:

```python
# 1. Import the masker and config
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
    LocalMasker, LocalMaskerConfig
)

# 2. Create config with parameters
config = LocalMaskerConfig(window_size=0.5)

# 3. Create masker instance using create_from_config
masker = LocalMasker.create_from_config(config)
```

## 📁 File Organization

```
sparse_attention_hub/sparse_attention/research_attention/maskers/
├── base.py                       # ResearchMasker, MaskerConfig
├── fixed/
│   ├── base.py                  # FixedMasker, TopKMasker, TopPMasker + configs
│   └── implementations/
│       ├── basic_fixed.py       # LocalMasker, CausalMasker, SinkMasker + configs
│       ├── oracle_top_k.py      # OracleTopK + OracleTopKConfig
│       ├── pq_top_k.py          # PQCache + PQCacheConfig
│       ├── hashattention_top_k.py # HashAttention + HashAttentionConfig
│       └── double_sparsity_top_k.py # DoubleSparsity + DoubleSparsityConfig
└── sampling/
    ├── base.py                  # SamplingMasker + SamplingMaskerConfig
    └── implementations/
        ├── random_sampling.py   # RandomSamplingMasker + RandomSamplingMaskerConfig
        └── magic_pig.py         # MagicPig + MagicPigConfig
```

## ✅ Key Features

- **Consistent Interface**: All maskers have `__init__(config)` and `create_from_config(config)` methods
- **Type Safety**: All configs use dataclasses with proper type hints
- **Inheritance**: Proper inheritance hierarchy for both classes and configs
- **Modularity**: Each masker type has its own config class
- **Extensibility**: Easy to add new maskers by following the same pattern 