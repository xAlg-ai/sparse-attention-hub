# AdaptiveSamplingMasker Implementation Plan

## Overview
Adaptive sampling masker with statistical error bounds. Combines base sampling with adaptive budget allocation.

## Configuration
```python
@dataclass
class AdaptiveSamplingMaskerConfig(SamplingMaskerConfig):
    base_rate_sampling: Union[int, float]  # Base rate (0,1) if float
    epsilon: float  # Error bound (0,1)
    delta: float   # Confidence bound (0,1)
    init_offset: int  # Start index
    local_offset: int  # End offset
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if isinstance(self.base_rate_sampling, float):
            if not (0.0 < self.base_rate_sampling < 1.0):
                raise ValueError(f"base_rate_sampling must be in (0,1) if float, got {self.base_rate_sampling}")
        elif isinstance(self.base_rate_sampling, int):
            if self.base_rate_sampling <= 0:
                raise ValueError(f"base_rate_sampling must be positive if int, got {self.base_rate_sampling}")
        else:
            raise ValueError(f"base_rate_sampling must be int or float, got {type(self.base_rate_sampling)}")
        
        if not (0.0 < self.epsilon < 1.0):
            raise ValueError(f"epsilon must be in (0,1), got {self.epsilon}")
        
        if not (0.0 < self.delta < 1.0):
            raise ValueError(f"delta must be in (0,1), got {self.delta}")
        
        if self.init_offset < 0:
            raise ValueError(f"init_offset must be non-negative, got {self.init_offset}")
        
        if self.local_offset < 0:
            raise ValueError(f"local_offset must be non-negative, got {self.local_offset}")
```

## Implementation

### Class Structure
```python
@MaskerRegistry.register(AdaptiveSamplingMaskerConfig)
class AdaptiveSamplingMasker(SamplingMasker):
    def __init__(self, config: AdaptiveSamplingMaskerConfig) -> None:
        super().__init__(config)
        self.base_rate_sampling = config.base_rate_sampling
        self.epsilon = config.epsilon
        self.delta = config.delta
        self.init_offset = config.init_offset
        self.local_offset = config.local_offset
        
        # Pre-compute delta_ppf for efficiency
        from scipy.stats import norm
        self.delta_ppf = norm.ppf(1 - self.delta)
```

### Main add_mask Method Algorithm

#### Step 1: Early Exit Check
```python
if previous_mask.is_full_mask():
    return previous_mask
```

#### Step 2: Extract Dimensions and Compute Attention Scores
```python
dims = self._extract_tensor_dimensions(keys, queries)
batch_size, num_heads, seq_len_queries, seq_len_keys = dims.batch_size, dims.num_heads, dims.seq_len_queries, dims.seq_len_keys

# Compute raw attention scores with max trick for numerical stability
raw_attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
max_scores = torch.max(raw_attention_scores, dim=-1, keepdim=True)[0]
expwts = torch.exp(raw_attention_scores - max_scores)  # (b, h, q, k)
```

#### Step 3: Compute Static Denominator
```python
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import apply_inv_mask_sum
static_denominator = apply_inv_mask_sum(expwts, previous_mask)  # (b, h, q, 1)
```

#### Step 4: Create Base Sampling and Estimate Standard Deviation
```python
# Determine sampling range
start_idx = self.init_offset
end_idx = seq_len_keys - self.local_offset
sampling_range = end_idx - start_idx

# Validate sampling range
if sampling_range <= 0:
    raise ValueError(f"Invalid sampling range: {sampling_range} (start_idx={start_idx}, end_idx={end_idx})")

# Handle base_rate_sampling: if int, use as direct budget; if float, multiply by sampling_range
if isinstance(self.base_rate_sampling, int):
    num_base_samples = self.base_rate_sampling
else:
    num_base_samples = int(self.base_rate_sampling * sampling_range)

# Create base sampling indices
base_row_wise_idx = torch.randint(
    low=start_idx, 
    high=end_idx,
    size=(batch_size, num_heads, seq_len_queries, num_base_samples),
    device=keys.device
)

# Extract values at sampled indices using torch.gather
sampled_values = torch.gather(expwts, dim=-1, index=base_row_wise_idx)  # (b, h, q, num_base_samples)

# Flatten for std computation
total_rows = batch_size * num_heads * seq_len_queries
row_sampled_values = sampled_values.view(total_rows, num_base_samples)  # (total_rows, num_base_samples)

# Compute standard deviation per row (vectorized)
std_estimate = torch.std(row_sampled_values, dim=-1, keepdim=True)  # (total_rows, 1)

# Handle zero std case
std_estimate = torch.clamp(std_estimate, min=1e-8)  # Avoid division by zero

# Reshape back to original dimensions
std_estimate = std_estimate.view(batch_size, num_heads, seq_len_queries, 1)  # (b, h, q, 1)

# Create base sampling mask
if isinstance(self.base_rate_sampling, float):
    base_data = torch.full_like(base_row_wise_idx, self.base_rate_sampling, dtype=keys.dtype)
else:
    base_data = torch.full_like(base_row_wise_idx, self.base_rate_sampling / sampling_range, dtype=keys.dtype)

base_sampling_mask = Mask.create_from_row_wise_idx(
    shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
    row_wise_idx=base_row_wise_idx,
    data=base_data,
    type="index",
    dtype=previous_mask.dtype
)
```

#### Step 5: Compute Sampled Denominator
```python
sampled_denominator = apply_inv_mask_sum(expwts, base_sampling_mask)  # (b, h, q, 1)
```

#### Step 6: Estimate Total Denominator
```python
estimated_denominator = static_denominator + sampled_denominator  # (b, h, q, 1)
```

#### Step 7: Compute Adaptive Budgets
```python
# Compute error bounds
epsilon_allowable_error = self.epsilon * estimated_denominator  # (b, h, q, 1)

# Handle very small epsilon_allowable_error to prevent numerical issues
epsilon_allowable_error = torch.clamp(epsilon_allowable_error, min=1e-8)

# Budget computation: b = (delta_ppf * std_estimate * sampling_range / epsilon_error)^2
budget_numerator = self.delta_ppf * std_estimate * sampling_range
budget = torch.clamp(
    (budget_numerator / epsilon_allowable_error) ** 2,
    min=1,  # Minimum 1 sample
    max=sampling_range  # Maximum all available positions
).long()  # (b, h, q, 1)
```

#### Step 8: Create Adaptive Sampling Mask
```python
# Calculate sampling probabilities
sampling_probabilities = budget / sampling_range  # (b, h, q, 1)

# Create adaptive sampling mask using utility function
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import create_sampling_mask_with_per_head_budget

adaptive_mask = create_sampling_mask_with_per_head_budget(
    budgets=budget,  # (b, h, q, 1)
    sampling_probability=sampling_probabilities,  # (b, h, q, 1)
    seq_len_keys=seq_len_keys,
    start_idx=start_idx,
    end_idx=end_idx,
    dtype=previous_mask.dtype
)
```

#### Step 9: Merge and Return
```python
# Merge base sampling mask with adaptive mask
combined_mask = base_sampling_mask.merge_mask(adaptive_mask, inplace=False)
# Merge with previous mask
return previous_mask.merge_mask(combined_mask, inplace=False)
```

### Required Methods
```python
@classmethod
def create_from_config(cls, config: MaskerConfig) -> "AdaptiveSamplingMasker":
    """Create AdaptiveSamplingMasker instance from configuration."""
    if not isinstance(config, AdaptiveSamplingMaskerConfig):
        raise ValueError(f"Invalid config type: {type(config)}")
    return cls(config)
```

## Key Points
- **Numerical stability**: Max trick, clamp operations, zero handling
- **Efficiency**: Vectorized operations, sparse representations
- **Error handling**: Validation, edge cases, device consistency
- **Statistical correctness**: Proper bounds, probability interpretations

## Testing Strategy
- **Unit tests**: Configuration validation, edge cases, statistical properties
- **Integration tests**: End-to-end functionality, performance benchmarks
- **Statistical tests**: Error bound verification, confidence intervals

## Dependencies
- `scipy.stats.norm` for PPF computation
- `torch` for tensor operations
- Existing Mask utilities 