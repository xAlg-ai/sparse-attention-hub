# MagicPig Implementation Plan

## Overview
MagicPig is a sampling-based masker that uses Locality Sensitive Hashing (LSH) to efficiently approximate maximum inner product search for attention computation. It combines LSH-based similarity matching with probability-based sampling to create sparse attention patterns.

## Algorithm Pseudo Code

### 1. Configuration and Initialization
```python
@dataclass
class MagicPigConfig(SamplingMaskerConfig):
    lsh_l: int  # number of LSH tables
    lsh_k: int  # number of bits per LSH table
    # Note: sampling_rate is inherited but not used in MagicPig
```

### 2. Main add_mask Method Pseudo Code

```python
def add_mask(self, keys, queries, values, attention_mask, sparse_meta_data, previous_mask, **kwargs):
    # Step 1: Check if previous_mask is full mask
    if previous_mask.is_full_mask():
        return previous_mask
    
    # Step 2: Extract tensor dimensions
    batch_size, num_heads, seq_len_queries, seq_len_keys = extract_dimensions(keys, queries)
    
    # Step 3: Compute probabilities using LSH collision probability
    probabilities = compute_probabilities(keys, queries, self.lsh_k, self.lsh_l)
    # Shape: (batch_size, num_heads, seq_len_queries, seq_len_keys)
    
    # Step 4: Compute LSH matches
    matches = compute_lsh_matches(keys, queries, self.lsh_k, self.lsh_l)
    # Shape: (batch_size, num_heads, seq_len_queries, seq_len_keys) - binary tensor
    
    # Step 5: Combine matches and probabilities
    dense_mask = matches * probabilities
    # Shape: (batch_size, num_heads, seq_len_queries, seq_len_keys)
    
    # Step 6: Create Mask from dense mask
    this_mask = Mask.create_mask_from_dense_mask(
        shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
        mask=dense_mask,
        dtype=previous_mask.dtype
    )
    
    # Step 7: Merge with previous mask and return
    return previous_mask.merge_mask(this_mask, inplace=False)
```

### 3. Key Helper Functions

#### 3.1 Transform for Maximum Inner Product Search
```python
def transform_for_mips(keys, queries):
    """
    Transform keys and queries for maximum inner product search using the technique from:
    Neyshabur, Behnam and Srebro, Nathan. "On symmetric and asymmetric LSHs for inner product search."
    
    Args:
        keys: (batch_size, num_heads, seq_len_keys, head_dim)
        queries: (batch_size, num_heads, seq_len_queries, head_dim)
    
    Returns:
        transformed_keys, transformed_queries
    """
    # Normalize queries
    query_norms = torch.norm(queries, dim=-1, keepdim=True)
    queries_normalized = queries / (query_norms + 1e-8)
    
    # For keys, use the augmentation technique
    key_norms = torch.norm(keys, dim=-1, keepdim=True)
    max_key_norm = torch.max(key_norms)
    
    # Scale keys by max norm
    keys_scaled = keys / max_key_norm
    
    # Compute augmentation terms
    key_augmentation = torch.sqrt(max_key_norm**2 - key_norms**2) / max_key_norm
    query_augmentation = torch.zeros_like(queries_normalized[..., :1])
    
    # Concatenate augmentation terms
    keys_transformed = torch.cat([keys_scaled, key_augmentation], dim=-1)
    queries_transformed = torch.cat([queries_normalized, query_augmentation], dim=-1)
    
    return keys_transformed, queries_transformed
```

#### 3.2 Compute LSH Collision Probabilities
```python
def compute_probabilities(keys, queries, lsh_k, lsh_l):
    """
    Compute LSH collision probabilities using cosine similarity.
    
    Args:
        keys: (batch_size, num_heads, seq_len_keys, head_dim)
        queries: (batch_size, num_heads, seq_len_queries, head_dim)
        lsh_k: number of bits per LSH table
        lsh_l: number of LSH tables
    
    Returns:
        probabilities: (batch_size, num_heads, seq_len_queries, seq_len_keys)
    """
    # Transform for MIPS
    keys_transformed, queries_transformed = transform_for_mips(keys, queries)
    
    # Compute cosine similarities
    # Normalize the transformed vectors
    keys_norm = torch.norm(keys_transformed, dim=-1, keepdim=True)
    queries_norm = torch.norm(queries_transformed, dim=-1, keepdim=True)
    
    keys_normalized = keys_transformed / (keys_norm + 1e-8)
    queries_normalized = queries_transformed / (queries_norm + 1e-8)
    
    # Compute cosine similarities
    cosine_similarities = torch.matmul(queries_normalized, keys_normalized.transpose(-2, -1))
    cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)
    
    # Convert to angles
    angles = torch.acos(cosine_similarities)
    
    # Compute LSH collision probability
    # P(collision) = (1 - theta/pi)^k for single table
    # P(collision across l tables) = 1 - (1 - p)^l
    single_table_prob = (1 - angles / torch.pi) ** lsh_k
    collision_prob = 1 - (1 - single_table_prob) ** lsh_l
    
    return collision_prob
```

#### 3.3 Compute LSH Matches
```python
def compute_lsh_matches(keys, queries, lsh_k, lsh_l):
    """
    Compute LSH matches using random signed projections.
    
    Args:
        keys: (batch_size, num_heads, seq_len_keys, head_dim)
        queries: (batch_size, num_heads, seq_len_queries, head_dim)
        lsh_k: number of bits per LSH table
        lsh_l: number of LSH tables
    
    Returns:
        matches: (batch_size, num_heads, seq_len_queries, seq_len_keys) - binary tensor
    """
    # Transform for MIPS
    keys_transformed, queries_transformed = transform_for_mips(keys, queries)
    
    batch_size, num_heads, seq_len_queries, seq_len_keys = keys.shape
    head_dim = keys_transformed.shape[-1]
    
    # Generate random projection matrix
    total_bits = lsh_k * lsh_l
    projection = torch.randn(head_dim, total_bits, device=keys.device, dtype=keys.dtype)
    
    # Compute signatures
    # Reshape for batch processing
    keys_flat = keys_transformed.view(-1, head_dim)  # (batch*heads*seq_len_keys, head_dim)
    queries_flat = queries_transformed.view(-1, head_dim)  # (batch*heads*seq_len_queries, head_dim)
    
    # Compute signed projections
    keys_signatures = torch.sign(torch.matmul(keys_flat, projection))  # (batch*heads*seq_len_keys, total_bits)
    queries_signatures = torch.sign(torch.matmul(queries_flat, projection))  # (batch*heads*seq_len_queries, total_bits)
    
    # Reshape back to original dimensions
    keys_signatures = keys_signatures.view(batch_size, num_heads, seq_len_keys, total_bits)
    queries_signatures = queries_signatures.view(batch_size, num_heads, seq_len_queries, total_bits)
    
    # Compute matches for each query-key pair
    # Expand dimensions for broadcasting
    keys_signatures_expanded = keys_signatures.unsqueeze(2)  # (batch, heads, 1, seq_len_keys, total_bits)
    queries_signatures_expanded = queries_signatures.unsqueeze(3)  # (batch, heads, seq_len_queries, 1, total_bits)
    
    # Compute element-wise product
    signature_matches = keys_signatures_expanded * queries_signatures_expanded
    # Shape: (batch, heads, seq_len_queries, seq_len_keys, total_bits)
    
    # Reshape to group by LSH tables
    signature_matches_grouped = signature_matches.view(batch_size, num_heads, seq_len_queries, seq_len_keys, lsh_l, lsh_k)
    
    # Check if at least one group (table) has all bits matching
    # Sum within each group - if sum == lsh_k, all bits match
    group_matches = (signature_matches_grouped.sum(dim=-1) == lsh_k).int()
    # Shape: (batch, heads, seq_len_queries, seq_len_keys, lsh_l)
    
    # Check if at least one table has a match
    matches = (group_matches.sum(dim=-1) > 0).int()
    # Shape: (batch, heads, seq_len_queries, seq_len_keys)
    
    return matches
```

### 4. Implementation Notes

#### 4.1 Key Design Decisions
1. **MIPS Transformation**: Using the technique from Neyshabur & Srebro to transform inner product search to cosine similarity
2. **LSH Parameters**: 
   - `lsh_k`: Controls precision of each hash table (higher = more precise but fewer collisions)
   - `lsh_l`: Controls number of hash tables (higher = more chances for collision)
3. **Probability Computation**: Using theoretical LSH collision probability based on cosine similarity
4. **Match Computation**: Using random signed projections for efficient LSH implementation

#### 4.2 Memory and Computational Considerations
1. **Memory Usage**: 
   - Need to store transformed keys/queries
   - Need to store random projection matrix
   - Need to store signatures during computation
2. **Computational Complexity**:
   - O(batch_size * num_heads * seq_len * head_dim * total_bits) for signature computation
   - O(batch_size * num_heads * seq_len_queries * seq_len_keys * lsh_l) for match computation
3. **Optimizations**:
   - Use vectorized operations where possible
   - Consider caching random projections if used multiple times
   - Use efficient broadcasting for signature comparisons

#### 4.3 Integration with Existing Framework
1. **Mask Creation**: Use `Mask.create_mask_from_dense_mask()` to create mask from computed dense tensor
2. **Mask Merging**: Use `previous_mask.merge_mask()` to combine with existing masks
3. **Configuration**: Extend `SamplingMaskerConfig` but ignore `sampling_rate`
4. **Registry**: Register with `@MaskerRegistry.register(MagicPigConfig)`

### 5. Testing Strategy

#### 5.1 Unit Tests
1. Test MIPS transformation correctness
2. Test LSH probability computation
3. Test LSH match computation
4. Test mask creation and merging
5. Test edge cases (full masks, empty masks)

#### 5.2 Integration Tests
1. Test with different LSH parameters
2. Test with different tensor shapes
3. Test with existing masker pipeline
4. Test memory usage and performance

### 6. Expected Behavior

#### 6.1 Sparsity Pattern
- MagicPig should create sparser attention patterns than random sampling
- Similar keys should have higher probability of being attended to
- The sparsity should be controlled by LSH parameters

#### 6.2 Performance Characteristics
- Should be more computationally expensive than random sampling due to LSH computation
- Should provide better attention quality for similar content
- Memory usage should scale with LSH parameters

#### 6.3 Quality Metrics
- Compare attention quality with random sampling
- Measure sparsity vs. performance trade-off
- Evaluate similarity preservation in attention patterns 