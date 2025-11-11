# Profiling Scripts

This directory contains profiling scripts for analyzing performance bottlenecks in the sparse attention framework.

## 📊 Available Scripts

### `profile_research_attention.py`

Profiles the `custom_attention` function of `ResearchAttention` using PyTorch profiler. This script helps to analyse the masker overhead for specific configurations.


#### Features:
- **Multi-masker Configuration**: Profiles a realistic sparse attention setup with:
  - SinkMasker (128 sink tokens)
  - LocalMasker (512 window size)
  - OracleTopK (10% of sequence)
  - RandomSampling (5% sampling rate) [optional]
- **Long Context Testing**: Uses 32K+ sequence length for realistic profiling
- **CUDA Support**: Automatically uses GPU if available
- **Comprehensive Metrics**: Reports CPU time, CUDA time, and memory usage
- **Chrome Tracing**: Outputs trace file compatible with `https://ui.perfetto.dev/`
- **Simple Timing**: Clean timing measurements with statistics (average, min, max, std dev)
- **Baseline Comparisons**: Compare against empty maskers and Flash Attention
- **Overhead Analysis**: Automatic calculation of masker overhead and performance insights

#### Usage:
1. Add your sparse attention config that you want to profile
2. run 
```bash
# Run profiling
python profile/profile_research_attention.py
```
3. You can look at the masker overhead. Generally a overhead of upto 150% is acceptable i.e. custom attention is within $2.5\times$ of baseline attention. This is specifically kept loose to enable quick iteration on ideasa in pytorch implementation. Beyond this, it will start affecting benchmarking.

#### Sample Output:

The script generates:
1. **Console Statistics**: Top operations by CPU time, CUDA time, and memory usage
2. **Trace File**: `sample_trace.json` for detailed visualization # use it to debug your implementation.
3. **Validation Info**: Output shapes and configuration details
4. **Timing Measurements**: Clean timing statistics for sparse attention
5. **Baseline Timings**: Empty maskers and Flash Attention comparisons
6. **Comparative Analysis**: Overhead analysis and performance insights

**Example Output:**
```
📊 Timing Results (Sparse Attention):
   - Average time: 2.368 ms
   - Median time:  2.362 ms
   - Min time:     2.350 ms
   - Max time:     2.477 ms
   - Std dev:      0.021 ms

📊 Baseline Results (Empty Maskers):
   - Average time: 1.154 ms
   - Median time:  1.167 ms
   - Min time:     1.092 ms
   - Max time:     1.190 ms
   - Std dev:      0.031 ms

🔍 Comparative Analysis:
📈 Masker Overhead Analysis:
   - Sparse Attention:     2.368 ms
   - Baseline (no maskers): 1.154 ms
   - Masker overhead:      1.214 ms (105.2%)
```

#### Configuration:

You can modify the script to profile different configurations:

```python
# Change sequence length
seq_len = 32678  # For longer context

# Modify masker configuration
masker_configs = [
    SinkMaskerConfig(sink_size=256),     # More sink tokens
    LocalMaskerConfig(window_size=1024), # Larger window
    # Add/remove maskers as needed
]

# Adjust profiling parameters
profile_custom_attention(
    num_warmup_runs=10,    # More warmup
    num_profile_runs=20,   # More profiling runs
    trace_file="custom_trace.json"
)

# Adjust timing parameters
time_custom_attention(
    num_timing_runs=100,   # More timing runs for better statistics
    num_warmup_runs=20     # More warmup runs
)
```