# OracleTopPMasker Implementation Plan (REVISED - SIMPLIFIED)

## Overview
Implement `OracleTopPMasker` and `OracleTopPMaskerConfig` classes that extend from `TopPMasker` and `TopPMaskerConfig` respectively. This masker will use oracle knowledge of attention scores to select the top-p fraction of tokens for each query position.

## Current State Analysis

### Existing Structure
- `TopPMaskerConfig` (base class) - currently has no parameters
- `TopPMasker` (base class) - abstract base class for top-P maskers
- `LocalMasker` - reference implementation showing the pattern to follow
- `Mask` class - has `create_from_dense_mask()` and `merge_mask()` methods

### Key Dependencies
- `Mask.create_from_dense_mask()` - creates mask from dense tensor
- `Mask.merge_mask()` - merges two masks together
- `FixedMasker` base class - provides common functionality
- `MaskerRegistry` - for registering the new masker
- `torch.percentile()` - for computing top-p thresholds

## Implementation Plan

### 1. Update TopPMaskerConfig
**File**: `sparse_attention_hub/sparse_attention/research_attention/maskers/fixed/base.py`

**Changes**:
- Add `top_p: float` parameter to `TopPMaskerConfig` (only float, not int)
- Add validation to ensure `top_p` is in range [0, 1] in the base class
- Add proper type annotations and docstring

```python
@dataclass
class TopPMaskerConfig(FixedMaskerConfig):
    """Base configuration for top-P maskers."""
    
    top_p: float
    
    def __post_init__(self):
        """Validate top_p parameter."""
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in range [0, 1], got {self.top_p}")
```

### 2. Create OracleTopPMaskerConfig
**File**: `sparse_attention_hub/sparse_attention/research_attention/maskers/fixed/implementations/oracle_top_p.py`

**Implementation**:
- Extend `TopPMaskerConfig`
- No additional validation needed (handled in base class)
- Follow the same pattern as `LocalMaskerConfig`

```python
@dataclass
class OracleTopPMaskerConfig(TopPMaskerConfig):
    """Configuration for OracleTopPMasker."""
    
    pass  # Inherits top_p from parent with validation
```

### 3. Create OracleTopPMasker Class
**File**: `sparse_attention_hub/sparse_attention/research_attention/maskers/fixed/implementations/oracle_top_p.py`

**Implementation**:
- Extend `TopPMasker`
- Register with `@MaskerRegistry.register(OracleTopPMaskerConfig)`
- Follow `LocalMasker` pattern for structure

### 4. Core Algorithm Implementation (SIMPLIFIED)

#### 4.1 Main `add_mask` Method
**Logic Flow**:
1. Check if `previous_mask.is_full_mask()` - if so, return full mask
2. Extract tensor dimensions using `_extract_tensor_dimensions()`
3. Check if sequence is small enough for full attention using `_should_use_full_attention()`
4. If small enough, return full mask using `_create_full_mask()`
5. Otherwise, create oracle top-p mask using `_create_oracle_top_p_mask()`
6. Merge with previous mask using `previous_mask.merge_mask()`

#### 4.2 Score Computation
**Method**: `_compute_attention_scores()`
- Compute raw attention scores: `queries @ keys.transpose(-2, -1)`
- Shape: `(batch_size, num_heads, seq_len_queries, seq_len_keys)`
- Return: `torch.Tensor`

#### 4.3 Top-P Threshold Computation (VECTORIZED)
**Method**: `_compute_top_p_thresholds()`
- Input: attention scores tensor of shape `(batch_size, num_heads, seq_len_queries, seq_len_keys)`, top_p value
- **Fully vectorized operations**:
  1. Sort scores in descending order along last dimension: `torch.sort(scores, dim=-1, descending=True)`
  2. Compute cumulative sum along last dimension: `torch.cumsum(sorted_scores, dim=-1)`
  3. Normalize by total sum (last element of cumsum): `cumsum / cumsum[..., -1:]`
  4. Find smallest position where normalized cumsum >= top_p using `torch.searchsorted()` or `torch.where()`
  5. Extract threshold values using advanced indexing
- Return: threshold tensor of shape `(batch_size, num_heads, seq_len_queries, 1)`

**Vectorized Implementation**:
```python
def _compute_top_p_thresholds(self, scores: torch.Tensor, top_p: float) -> torch.Tensor:
    """Compute top-p thresholds using vectorized operations."""
    # Sort scores in descending order along last dimension
    sorted_scores, sort_indices = torch.sort(scores, dim=-1, descending=True)
    
    # Compute cumulative sum
    cumsum = torch.cumsum(sorted_scores, dim=-1)
    
    # Normalize by total sum (last element)
    total_sum = cumsum[..., -1:]
    normalized_cumsum = cumsum / total_sum
    
    # Find positions where normalized_cumsum >= top_p
    # Use torch.searchsorted for efficient binary search
    threshold_positions = torch.searchsorted(normalized_cumsum, top_p, side='left')
    
    # Extract threshold values using advanced indexing
    batch_size, num_heads, seq_len_queries, seq_len_keys = scores.shape
    batch_indices = torch.arange(batch_size, device=scores.device).view(-1, 1, 1, 1)
    head_indices = torch.arange(num_heads, device=scores.device).view(1, -1, 1, 1)
    query_indices = torch.arange(seq_len_queries, device=scores.device).view(1, 1, -1, 1)
    
    thresholds = sorted_scores[batch_indices, head_indices, query_indices, threshold_positions]
    
    return thresholds
```

**Example**:
```python
# For tensor of shape (batch_size, num_heads, seq_len_queries, seq_len_keys):
# All operations are vectorized across all dimensions
# No loops needed - pure PyTorch tensor operations
```

#### 4.4 Dense Mask Creation (SIMPLIFIED)
**Method**: `_create_oracle_top_p_mask()`
- Get attention scores using `_compute_attention_scores()`
- Get previous dense mask and mask out already active positions
- Compute thresholds using `_compute_top_p_thresholds()`
- Create dense mask: `scores >= thresholds`
- Use `Mask.create_from_dense_mask()` to create the mask
- Return: `Mask` object

### 5. Helper Methods

#### 5.1 Tensor Dimension Extraction
- Use existing `_extract_tensor_dimensions()` from base class
- Returns `AttentionTensorDimensions` with batch_size, num_heads, seq_len_queries, seq_len_keys

#### 5.2 Full Attention Check
**Method**: `_should_use_full_attention()`
- Calculate effective top-p size: `int(top_p * seq_len_keys)`
- Return `True` if `seq_len_keys <= effective_size`
- This ensures we don't create sparse masks when full attention is more efficient

#### 5.3 Full Mask Creation
- Use existing `_create_full_mask()` from base class
- Creates full attention mask when sequence is small

### 6. Registration and Integration

#### 6.1 Registry Registration
- Add `@MaskerRegistry.register(OracleTopPMaskerConfig)` decorator
- This enables automatic masker creation from config

#### 6.2 Import Updates
**File**: `sparse_attention_hub/sparse_attention/research_attention/maskers/fixed/__init__.py`
- Add imports for new classes
- Add to `__all__` list

#### 6.3 Main Package Import
**File**: `sparse_attention_hub/sparse_attention/__init__.py`
- Add imports for new classes
- Add to `__all__` list

### 7. Error Handling and Validation

#### 7.1 Config Validation
- Validate `top_p` is in range [0, 1] (handled in base class)
- Add appropriate error messages

#### 7.2 Runtime Validation
- Check tensor shapes are compatible
- Handle edge cases (empty tensors, single tokens, etc.)
- Ensure device consistency

### 8. Testing Strategy

#### 8.1 Unit Tests
**File**: `tests/unit/sparse_attention/research_attention/maskers/fixed/test_oracle_top_p.py`
- Test config creation and validation
- Test masker creation from config
- Test `add_mask` method with various scenarios
- Test edge cases (full mask, small sequences, etc.)
- Test percentile threshold computation
- Test dense mask creation

#### 8.2 Integration Tests
- Test with existing masker pipeline
- Test mask merging functionality
- Test performance characteristics

## Implementation Details

### Key Design Decisions

1. **Inheritance Structure**: Follow existing pattern of extending base classes
2. **Parameter Handling**: Only support float top_p values in range [0, 1]
3. **Efficiency**: Use full attention when sequence is small enough
4. **Compatibility**: Ensure works with existing mask merging pipeline
5. **Type Safety**: Use strong type annotations throughout
6. **Simplified Approach**: Use torch.percentile + dense mask creation

### Performance Considerations

1. **Memory Usage**: Avoid unnecessary tensor copies
2. **Computation**: Use vectorized operations with torch.percentile
3. **Device Handling**: Ensure tensors are on correct device
4. **Efficiency**: Dense mask creation is simpler than sparse representation

### Code Quality Standards

1. **Documentation**: Add comprehensive docstrings following Google style
2. **Type Annotations**: Use strong typing for all methods and variables
3. **Error Handling**: Provide clear error messages
4. **Testing**: Ensure good test coverage
5. **Linting**: Follow project linting standards

## File Structure

```
sparse_attention_hub/sparse_attention/research_attention/maskers/fixed/
├── base.py (update TopPMaskerConfig)
├── implementations/
│   └── oracle_top_p.py (new file)
└── __init__.py (update imports)

tests/unit/sparse_attention/research_attention/maskers/fixed/
└── test_oracle_top_p.py (new file)
```

## Success Criteria

1. ✅ OracleTopPMasker extends TopPMasker correctly
2. ✅ OracleTopPMaskerConfig extends TopPMaskerConfig with top_p parameter (float only)
3. ✅ add_mask method implements the specified algorithm with percentile approach
4. ✅ Proper integration with existing masker pipeline
5. ✅ Comprehensive test coverage
6. ✅ Follows project coding standards
7. ✅ Registered with MaskerRegistry
8. ✅ Proper imports and exports
9. ✅ Uses torch.percentile for efficient top-p computation

## Next Steps

1. Implement the plan step by step
2. Create unit tests alongside implementation
3. Verify integration with existing codebase
4. Run linting and type checking
5. Test with sample data
6. Document usage examples 