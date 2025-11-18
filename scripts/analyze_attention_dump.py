#!/usr/bin/env python3
"""Script to analyze dumped attention data.

This script loads a pickle file containing queries, keys, and values tensors,
and performs various analyses on the attention mechanism.

Available analyses:
    - computation: Computes attention output using scaled dot-product attention
    - denominator-estimation: Analyzes convergence of denominator variance estimation
                              (uses unbiased variance estimator with Bessel's correction)
    - denominator-budget: Analyzes convergence of budget computation for adaptive sampling
                          (examines how estimated budget converges to true budget)

Examples:
    # Compute attention output for head 0 (default)
    $ python scripts/analyze_attention_dump.py /workspace/attention_data/0.a1b2c3d4.pkl
    
    # Compute attention for a specific head with verbose output
    $ python scripts/analyze_attention_dump.py /workspace/attention_data/0.a1b2c3d4.pkl \
        --head-id 5 --verbose
    
    # Analyze denominator variance estimation convergence
    $ python scripts/analyze_attention_dump.py /workspace/attention_data/0.a1b2c3d4.pkl \
        --analysis denominator-estimation --head-id 0 --init-offset 4 --local-offset 64
    
    # Analyze budget computation convergence
    $ python scripts/analyze_attention_dump.py /workspace/attention_data/0.a1b2c3d4.pkl \
        --analysis denominator-budget --head-id 0 --init-offset 4 --local-offset 64 \
        --epsilon 0.1 --delta 0.05
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def load_attention_data(filepath: Path) -> Dict[str, torch.Tensor]:
    """Load attention data from a pickle file.
    
    Args:
        filepath: Path to the pickle file containing attention data.
        
    Returns:
        Dictionary containing 'queries', 'keys', and 'values' tensors.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain required keys.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        data: Dict[str, torch.Tensor] = pickle.load(f)
    
    required_keys = {"queries", "keys", "values"}
    if not required_keys.issubset(data.keys()):
        raise ValueError(
            f"Pickle file must contain {required_keys}, but got {set(data.keys())}"
        )
    
    return data


def repeat_kv(tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors to match the number of query heads for GQA.
    
    This is used for Grouped Query Attention where keys and values have fewer
    heads than queries.
    
    Args:
        tensor: Key or value tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each head
        
    Returns:
        Repeated tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return tensor
    
    batch_size: int = tensor.shape[0]
    num_kv_heads: int = tensor.shape[1]
    seq_len: int = tensor.shape[2]
    head_dim: int = tensor.shape[3]
    
    # Expand and reshape to repeat heads
    tensor = tensor[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    tensor = tensor.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    
    return tensor


def compute_attention_output(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    head_id: int,
    plot_cumulative: bool = False,
    output_path: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention output for a specific head.
    
    Note: Assumes keys and values are already repeated to match query heads (if needed for GQA).
    
    Args:
        queries: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        keys: Key tensor of shape (batch, num_heads, seq_len_k, head_dim)
        values: Value tensor of shape (batch, num_heads, seq_len_v, head_dim)
        head_id: Head ID to compute attention for
        plot_cumulative: Whether to plot cumulative attention scores (default: False)
        output_path: Path to save the cumulative attention plot (default: None)
        
    Returns:
        Tuple containing:
            - attention_output: Output tensor of shape (batch, seq_len_q, head_dim)
            - attention_weights: Attention weights of shape (batch, seq_len_q, seq_len_k)
    """
    # Extract specific head
    query: torch.Tensor = queries[:, head_id, :, :]  # (batch, seq_len_q, head_dim)
    key: torch.Tensor = keys[:, head_id, :, :]       # (batch, seq_len_k, head_dim)
    value: torch.Tensor = values[:, head_id, :, :]   # (batch, seq_len_v, head_dim)
    
    # Get head dimension for scaling
    head_dim: int = query.shape[-1]
    scaling: float = head_dim ** -0.5
    
    # Compute attention scores: Q @ K^T
    attention_scores: torch.Tensor = torch.matmul(query, key.transpose(-2, -1))  # (batch, seq_len_q, seq_len_k)
    
    # Scale the scores
    attention_scores = attention_scores * scaling
    
    # Apply softmax to get attention weights
    attention_weights: torch.Tensor = F.softmax(attention_scores, dim=-1)  # (batch, seq_len_q, seq_len_k)
    
    # Plot cumulative attention if requested
    if plot_cumulative and output_path is not None:
        _plot_cumulative_attention(attention_weights, output_path, head_id)
    
    # Compute attention output: attention_weights @ V
    attention_output: torch.Tensor = torch.matmul(attention_weights, value)  # (batch, seq_len_q, head_dim)
    
    return attention_output, attention_weights


def _plot_cumulative_attention(
    attention_weights: torch.Tensor,
    output_path: Path,
    head_id: int,
) -> None:
    """Plot cumulative post-softmax attention scores for a specific head.
    
    Sorts attention weights in descending order before computing cumulative sum
    to show how quickly the top-k positions capture most of the attention.
    
    Args:
        attention_weights: Attention weights of shape (batch, seq_len_q, seq_len_k)
        output_path: Path to save the plot
        head_id: Head ID being analyzed (for title)
    """
    batch_size: int = attention_weights.shape[0]
    seq_len_q: int = attention_weights.shape[1]
    seq_len_k: int = attention_weights.shape[2]
    
    # Sort attention weights in descending order
    sorted_attention_weights, _ = torch.sort(attention_weights, dim=-1, descending=True)
    
    # Compute cumulative attention across sorted keys for each query
    cumulative_attention: torch.Tensor = torch.cumsum(sorted_attention_weights, dim=-1)
    
    # Move to CPU for plotting
    cumulative_attention = cumulative_attention.cpu()
    
    # Create figure - single plot for the specific head
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot for first query in batch (assuming batch_size=1, seq_len_q=1 for decoding)
    batch_idx: int = 0
    query_idx: int = 0
    
    # Get cumulative attention
    cum_attn: torch.Tensor = cumulative_attention[batch_idx, query_idx, :]
    
    # Plot
    positions = list(range(seq_len_k))
    ax.plot(positions, cum_attn.float().numpy(), linewidth=2, color='blue', label='Cumulative Attention')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Complete (1.0)')
    
    ax.set_xlabel('Top-K Positions (Sorted by Attention)', fontsize=14)
    ax.set_ylabel('Cumulative Attention', fontsize=14)
    ax.set_title(f'Cumulative Post-Softmax Attention Scores - Head {head_id}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=12)
    
    # Add text showing where 50%, 90%, 95%, 99% attention is reached
    cum_attn_np = cum_attn.float().numpy()
    for threshold in [0.5, 0.9, 0.95, 0.99]:
        if (cum_attn_np >= threshold).any():
            idx = int((cum_attn_np >= threshold).nonzero()[0][0])
            ax.axvline(x=idx, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
            ax.text(idx, threshold, f'{int(threshold*100)}%@{idx}', fontsize=10, 
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add info text box
    info_text = f"Sequence Length: {seq_len_k}\nBatch: {batch_idx}, Query: {query_idx}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Cumulative attention plot saved to: {output_path}")
    
    plt.close()


def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """Print information about a tensor.
    
    Args:
        name: Name of the tensor.
        tensor: The tensor to print information about.
    """
    print(f"\n{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")


def analyse_attention_computation(
    filepath: Path,
    head_id: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    """Analyze attention computation from dumped data for a specific head.
    
    This function loads attention data, computes the attention output,
    and prints detailed analysis.
    
    Args:
        filepath: Path to the pickle file containing attention data.
        head_id: Head ID to analyze (default: 0).
        device: Device to use for computation (default: "cpu").
        verbose: Whether to print detailed information (default: False).
    """
    # Load the attention data
    print(f"Loading attention data from: {filepath}")
    
    data: Dict[str, torch.Tensor] = load_attention_data(filepath)
    
    queries: torch.Tensor = data["queries"].to(device)
    keys: torch.Tensor = data["keys"].to(device)
    values: torch.Tensor = data["values"].to(device)
    
    # Handle GQA: repeat keys and values to match query heads if necessary
    num_q_heads: int = queries.shape[1]
    num_kv_heads: int = keys.shape[1]
    
    print(f"\nNumber of query heads: {num_q_heads}")
    print(f"Number of key/value heads: {num_kv_heads}")
    
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, (
            f"Number of query heads ({num_q_heads}) must be divisible by "
            f"number of key/value heads ({num_kv_heads})"
        )
        n_rep: int = num_q_heads // num_kv_heads
        print(f"GQA detected: repeating keys and values {n_rep} times")
        keys = repeat_kv(keys, n_rep)
        values = repeat_kv(values, n_rep)
    
    # Validate head_id
    if head_id >= num_q_heads:
        raise ValueError(f"head_id {head_id} is out of range. Available heads: 0-{num_q_heads-1}")
    
    print(f"\nAnalyzing head: {head_id}")
    
    # Print input information
    print("\n" + "=" * 80)
    print("INPUT TENSORS")
    print("=" * 80)
    print_tensor_info("Queries", queries)
    print_tensor_info("Keys", keys)
    print_tensor_info("Values", values)
    
    # Compute attention output
    print("\n" + "=" * 80)
    print(f"COMPUTING ATTENTION OUTPUT FOR HEAD {head_id}")
    print("=" * 80)
    
    # Prepare output path for cumulative attention plot
    output_dir: Path = filepath.parent
    cumulative_plot_path: Path = output_dir / f"cumulative_attention_head{head_id}.png"
    
    attention_output: torch.Tensor
    attention_weights: torch.Tensor
    attention_output, attention_weights = compute_attention_output(
        queries, keys, values, 
        head_id=head_id,
        plot_cumulative=True,
        output_path=cumulative_plot_path
    )
    
    # Print output information
    print("\n" + "=" * 80)
    print("OUTPUT TENSORS")
    print("=" * 80)
    print_tensor_info("Attention Output", attention_output)
    print_tensor_info("Attention Weights", attention_weights)
    
    if verbose:
        print("\n" + "=" * 80)
        print("DETAILED INFORMATION")
        print("=" * 80)
        
        num_q_heads: int = queries.shape[1]
        num_kv_heads: int = keys.shape[1]
        
        print(f"\nBatch size: {queries.shape[0]}")
        print(f"Number of query heads: {num_q_heads}")
        print(f"Number of key/value heads: {num_kv_heads}")
        
        if num_q_heads != num_kv_heads:
            print(f"GQA: Using Grouped Query Attention (ratio: {num_q_heads // num_kv_heads}:1)")
        else:
            print(f"MHA: Using Multi-Head Attention")
        
        print(f"Query sequence length: {queries.shape[2]}")
        print(f"Key/Value sequence length: {keys.shape[2]}")
        print(f"Head dimension: {queries.shape[3]}")
        
        print(f"\nAttention weights sum per query (should be ~1.0):")
        weights_sum: torch.Tensor = attention_weights.sum(dim=-1)
        print(f"  Min: {weights_sum.min().item():.6f}")
        print(f"  Max: {weights_sum.max().item():.6f}")
        print(f"  Mean: {weights_sum.mean().item():.6f}")
        
        print(f"\nAttention weights sparsity:")
        threshold: float = 1e-6
        sparse_ratio: float = (attention_weights < threshold).float().mean().item()
        print(f"  Ratio of weights < {threshold}: {sparse_ratio:.4f} ({sparse_ratio*100:.2f}%)")
        
        print(f"\nTop-5 attention weights (first query, first head):")
        first_query_weights: torch.Tensor = attention_weights[0, 0, 0, :]
        top_values: torch.Tensor
        top_indices: torch.Tensor
        top_values, top_indices = torch.topk(first_query_weights, min(5, first_query_weights.shape[0]))
        for idx, (val, pos) in enumerate(zip(top_values, top_indices)):
            print(f"  {idx+1}. Position {pos.item()}: {val.item():.6f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def analyse_denominator_estimation(
    filepath: Path,
    head_id: int,
    init_offset: int,
    local_offset: int,
    device: str = "cpu",
) -> None:
    """Analyze how fast the denominator variance estimation converges.
    
    This analysis examines the convergence of unbiased variance estimation
    (using Bessel's correction with N-1 denominator) for the attention 
    denominator (sum of exponential weights) as a function of sample size. 
    For each sample size, it runs 100 trials with different random samples 
    and computes the relative MSE to measure convergence quality.
    
    The analysis produces two plots:
        1. Relative MSE vs Sample Size (log scale)
        2. Mean Estimated Variance vs Sample Size with true variance reference
    
    Args:
        filepath: Path to the pickle file containing attention data.
        head_id: ID of the attention head to analyze.
        init_offset: Initial offset to trim from the start of the sequence.
        local_offset: Local offset to trim from the end of the sequence.
        device: Device to use for computation (default: "cpu").
    """
    print(f"Loading attention data from: {filepath}")
    
    data: Dict[str, torch.Tensor] = load_attention_data(filepath)
    
    queries: torch.Tensor = data["queries"].to(device)
    keys: torch.Tensor = data["keys"].to(device)
    values: torch.Tensor = data["values"].to(device)
    
    # Handle GQA: repeat keys to match query heads if necessary
    num_q_heads: int = queries.shape[1]
    num_kv_heads: int = keys.shape[1]
    
    print("\n" + "=" * 80)
    print(f"DENOMINATOR ESTIMATION CONVERGENCE ANALYSIS - HEAD {head_id}")
    print("=" * 80)
    
    print(f"\nNumber of query heads: {num_q_heads}")
    print(f"Number of key/value heads: {num_kv_heads}")
    
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, (
            f"Number of query heads ({num_q_heads}) must be divisible by "
            f"number of key/value heads ({num_kv_heads})"
        )
        n_rep: int = num_q_heads // num_kv_heads
        print(f"GQA detected: repeating keys {n_rep} times")
        keys = repeat_kv(keys, n_rep)
        print(f"Keys shape after repeat_kv: {keys.shape}")
    
    # Validate head_id after potential key repetition
    num_heads: int = queries.shape[1]
    if head_id >= num_heads:
        raise ValueError(f"head_id {head_id} is out of range. Available heads: 0-{num_heads-1}")
    
    # Extract data for the specific head
    # Assuming batch size = 1 and single query (decoding)
    query: torch.Tensor = queries[0, head_id, 0, :]  # Shape: (head_dim,)
    keys_head: torch.Tensor = keys[0, head_id, :, :]  # Shape: (seq_len, head_dim)
    
    print(f"\nQuery shape: {query.shape}")
    print(f"Keys shape: {keys_head.shape}")
    print(f"Sequence length: {keys_head.shape[0]}")
    
    # Get head dimension for scaling
    head_dim: int = query.shape[-1]
    scaling: float = head_dim ** -0.5
    
    # Compute raw attention scores: query @ keys^T
    raw_attention_scores: torch.Tensor = torch.matmul(query, keys_head.T)  # Shape: (seq_len,)
    
    # Apply scaling factor
    raw_attention_scores = raw_attention_scores * scaling
    
    print(f"Raw attention scores shape: {raw_attention_scores.shape}")
    print(f"Scaling factor applied: {scaling:.6f}")
    
    # Trim the raw attention scores first (before computing max)
    seq_len: int = raw_attention_scores.shape[0]
    
    # Handle edge cases for trimming
    if local_offset == 0:
        end_idx: int = seq_len
    else:
        end_idx = seq_len - local_offset
    
    start_idx: int = init_offset
    
    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid offsets: init_offset={init_offset}, local_offset={local_offset} "
            f"result in empty trimmed range [{start_idx}:{end_idx}] for seq_len={seq_len}"
        )
    
    print(f"\nTrimming: [{start_idx}:{end_idx}]")
    trimmed_scores: torch.Tensor = raw_attention_scores[start_idx:end_idx]
    trimmed_len: int = trimmed_scores.shape[0]
    print(f"Trimmed sequence length: {trimmed_len}")
    
    # Compute exponential weights (without max normalization)
    max_score: torch.Tensor = torch.max(trimmed_scores)
    trimmed_expwts: torch.Tensor = torch.exp(trimmed_scores - max_score)
    
    # print(f"Max score (from trimmed): {max_score.item():.6f}")
    print(f"Trimmed expwts shape: {trimmed_expwts.shape}")
    print(f"Trimmed expwts min: {trimmed_expwts.min().item():.6e}, max: {trimmed_expwts.max().item():.6e}")
    
    # Compute true variance (using unbiased estimation with N-1 denominator)
    true_var: float = torch.var(trimmed_expwts, unbiased=True).item()
    print(f"True variance (unbiased): {true_var:.6e}")
    
    # Generate sample sizes as multiples of 1024
    sample_sizes: List[int] = []
    multiple: int = 1024
    current_size: int = multiple
    
    while current_size <= trimmed_len:
        sample_sizes.append(current_size)
        current_size += multiple
    
    # Ensure we include the full length if it's not already in the list
    if not sample_sizes or sample_sizes[-1] != trimmed_len:
        sample_sizes.append(trimmed_len)
    
    print(f"\nSample sizes: {sample_sizes}")
    
    # Number of random samples to estimate MSE
    num_trials: int = 100
    
    print(f"Number of trials per sample size: {num_trials}")
    
    # Compute relative MSE for each sample size
    relative_mses: List[float] = []
    mean_estimated_vars: List[float] = []
    std_estimated_vars: List[float] = []
    
    print("\n" + "=" * 80)
    print("CONVERGENCE RESULTS (100 trials per sample size)")
    print("=" * 80)
    print(f"{'Sample Size':<15} {'Mean Est. Var':<20} {'Std Est. Var':<20} {'Relative MSE':<20} {'RMSE (%)'}")
    print("-" * 100)
    
    torch.manual_seed(42)  # For reproducibility
    
    for sample_size in sample_sizes:
        estimated_vars_trials: List[float] = []
        
        # Run multiple trials with different random samples
        for trial in range(num_trials):
            # Create a different random permutation for each trial
            perm: torch.Tensor = torch.randperm(trimmed_len, device=device)
            sample: torch.Tensor = trimmed_expwts[perm[:sample_size]]
            # Use unbiased variance estimation (Bessel's correction: divide by N-1)
            estimated_var: float = torch.var(sample, unbiased=True).item()
            estimated_vars_trials.append(estimated_var)
        
        # Compute statistics across trials
        estimated_vars_tensor: torch.Tensor = torch.tensor(estimated_vars_trials)
        mean_estimated_var: float = torch.mean(estimated_vars_tensor).item()
        mean_estimated_vars.append(mean_estimated_var)
        
        # Compute standard deviation of the estimates (for error bars)
        std_estimated_var: float = torch.std(estimated_vars_tensor, unbiased=True).item()
        std_estimated_vars.append(std_estimated_var)
        
        # Compute relative MSE: MSE / true_var^2
        squared_errors: torch.Tensor = (estimated_vars_tensor - true_var) ** 2
        mse: float = torch.mean(squared_errors).item()
        relative_mse: float = mse / (true_var ** 2)
        relative_mses.append(relative_mse)
        
        # RMSE as percentage of true variance
        rmse_pct: float = (mse ** 0.5) / true_var * 100
        
        print(f"{sample_size:<15} {mean_estimated_var:<20.6e} {std_estimated_var:<20.6e} {relative_mse:<20.6e} {rmse_pct:.4f}")
    
    # Create plot
    print("\n" + "=" * 80)
    print("GENERATING PLOT")
    print("=" * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Relative MSE vs Sample Size
    ax1.plot(sample_sizes, relative_mses, 'o-', linewidth=2, markersize=8, 
             color='blue', label='Relative MSE')
    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('Relative MSE (MSE / True Var²)', fontsize=12)
    ax1.set_title(f'Relative MSE Convergence - Head {head_id}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    
    # Plot 2: Mean Estimated Variance vs Sample Size with error bars
    ax2.errorbar(sample_sizes, mean_estimated_vars, yerr=std_estimated_vars,
                 fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                 color='green', ecolor='green', alpha=0.7,
                 label='Mean Estimated Var ± 1σ (100 trials, unbiased)')
    ax2.axhline(y=true_var, color='r', linestyle='--', linewidth=2, 
                label=f'True Var (unbiased) = {true_var:.6e}')
    ax2.set_xlabel('Sample Size', fontsize=12)
    ax2.set_ylabel('Variance (unbiased estimator)', fontsize=12)
    ax2.set_title(f'Mean Estimated Variance with ±1σ Error Bars - Head {head_id}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_dir: Path = filepath.parent
    plot_filename: str = f"denominator_var_convergence_head{head_id}.png"
    plot_path: Path = output_dir / plot_filename
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def analyse_denominator_budget(
    filepath: Path,
    head_id: int,
    init_offset: int,
    local_offset: int,
    epsilon: float,
    delta: float,
    device: str = "cpu",
) -> None:
    """Analyze how fast the denominator budget computation converges.
    
    This analysis examines the convergence of budget computation used in
    adaptive sampling. The budget is computed as:
        budget = (delta_ppf * std * sampling_range / (epsilon * estimated_sum))^2
    
    For each sample size, it runs 100 trials and compares:
    - True budget: Using exact variance and exact sum from full data
    - Estimated budget: Using sampled variance and estimated sum
      (estimated_sum = init_part + local_part + middle_estimate)
    
    Args:
        filepath: Path to the pickle file containing attention data.
        head_id: ID of the attention head to analyze.
        init_offset: Initial offset to trim from the start of the sequence.
        local_offset: Local offset to trim from the end of the sequence.
        epsilon: Error bound for budget computation (0 < epsilon < 1).
        delta: Confidence bound for budget computation (0 < delta < 1).
        device: Device to use for computation (default: "cpu").
    """
    from scipy.stats import norm
    
    print(f"Loading attention data from: {filepath}")
    
    data: Dict[str, torch.Tensor] = load_attention_data(filepath)
    
    queries: torch.Tensor = data["queries"].to(device)
    keys: torch.Tensor = data["keys"].to(device)
    values: torch.Tensor = data["values"].to(device)
    
    # Handle GQA: repeat keys to match query heads if necessary
    num_q_heads: int = queries.shape[1]
    num_kv_heads: int = keys.shape[1]
    
    print("\n" + "=" * 80)
    print(f"DENOMINATOR BUDGET CONVERGENCE ANALYSIS - HEAD {head_id}")
    print("=" * 80)
    
    print(f"\nNumber of query heads: {num_q_heads}")
    print(f"Number of key/value heads: {num_kv_heads}")
    print(f"Epsilon: {epsilon}")
    print(f"Delta: {delta}")
    
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, (
            f"Number of query heads ({num_q_heads}) must be divisible by "
            f"number of key/value heads ({num_kv_heads})"
        )
        n_rep: int = num_q_heads // num_kv_heads
        print(f"GQA detected: repeating keys {n_rep} times")
        keys = repeat_kv(keys, n_rep)
        print(f"Keys shape after repeat_kv: {keys.shape}")
    
    # Validate head_id after potential key repetition
    num_heads: int = queries.shape[1]
    if head_id >= num_heads:
        raise ValueError(f"head_id {head_id} is out of range. Available heads: 0-{num_heads-1}")
    
    # Extract data for the specific head
    query: torch.Tensor = queries[0, head_id, 0, :]
    keys_head: torch.Tensor = keys[0, head_id, :, :]
    
    print(f"\nQuery shape: {query.shape}")
    print(f"Keys shape: {keys_head.shape}")
    print(f"Sequence length: {keys_head.shape[0]}")
    
    # Get head dimension for scaling
    head_dim: int = query.shape[-1]
    scaling: float = head_dim ** -0.5
    
    # Compute raw attention scores
    raw_attention_scores: torch.Tensor = torch.matmul(query, keys_head.T)
    
    # Apply scaling factor
    raw_attention_scores = raw_attention_scores * scaling
    
    print(f"Scaling factor applied: {scaling:.6f}")
    
    # Compute exponential weights with max normalization
    max_score: torch.Tensor = torch.max(raw_attention_scores)
    expwts: torch.Tensor = torch.exp(raw_attention_scores - max_score)

    # Plot and store the cumulative expwts normalized by sum
    seq_len: int = expwts.shape[0]
    expwts_sum: float = torch.sum(expwts).item()
    normalized_expwts: torch.Tensor = expwts / expwts_sum
    sorted_normalized_expwts, _ = torch.sort(normalized_expwts, descending=True)
    cumulative_expwts: torch.Tensor = torch.cumsum(sorted_normalized_expwts, dim=0)
    
    plt.figure(figsize=(12, 8))
    positions = list(range(seq_len))
    plt.plot(positions, cumulative_expwts.float().cpu().numpy(), linewidth=2, color='green', label='Cumulative expwts')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Complete (1.0)')
    plt.xlabel('Top-K Positions (Sorted by expwts)', fontsize=14)
    plt.ylabel('Cumulative Normalized expwts', fontsize=14)
    plt.title(f'Cumulative Softmax Pre-Normalized Scores - Head {head_id}', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.legend(fontsize=12)
    plt.tight_layout()
    plot_path: Path = Path(f"XYZ.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Cumulative expwts plot saved to: {plot_path}")
    plt.close()
    
    # Compute offsets
    if local_offset == 0:
        end_idx: int = seq_len
    else:
        end_idx = seq_len - local_offset
    
    start_idx: int = init_offset
    
    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid offsets: init_offset={init_offset}, local_offset={local_offset} "
            f"result in empty trimmed range [{start_idx}:{end_idx}] for seq_len={seq_len}"
        )
    
    print(f"\nTrimming for sampling: [{start_idx}:{end_idx}]")
    
    # Separate into three parts: init, middle (sampling region), local
    init_expwts: torch.Tensor = expwts[:start_idx]
    middle_expwts: torch.Tensor = expwts[start_idx:end_idx]
    local_expwts: torch.Tensor = expwts[end_idx:]
    
    sampling_range: int = middle_expwts.shape[0]
    
    print(f"Init part size: {init_expwts.shape[0]}")
    print(f"Middle (sampling) part size: {sampling_range}")
    print(f"Local part size: {local_expwts.shape[0]}")
    
    # Compute true values using full data
    true_var: float = torch.var(middle_expwts, unbiased=True).item()
    true_std: float = (true_var ** 0.5)
    
    init_sum: float = torch.sum(init_expwts).item()
    middle_sum: float = torch.sum(middle_expwts).item()
    local_sum: float = torch.sum(local_expwts).item()
    total_sum: float = init_sum + middle_sum + local_sum
    
    print(f"\nTrue variance (middle region): {true_var:.6e}")
    print(f"True std (middle region): {true_std:.6e}")
    print(f"Init sum: {init_sum:.6e}")
    print(f"Middle sum: {middle_sum:.6e}")
    print(f"Local sum: {local_sum:.6e}")
    print(f"Total sum: {total_sum:.6e}")
    
    # Compute true budget
    delta_ppf: float = float(norm.ppf(1 - delta))
    epsilon_allowable_error: float = epsilon * total_sum
    budget_numerator: float = delta_ppf * true_std * sampling_range
    true_budget: float = (budget_numerator / epsilon_allowable_error) ** 2
    true_budget = max(1.0, min(true_budget, float(sampling_range)))
    
    print(f"\nDelta PPF: {delta_ppf:.6f}")
    print(f"True budget: {true_budget:.2f}")
    
    # Generate sample sizes
    sample_sizes: List[int] = []
    multiple: int = 1024
    current_size: int = multiple
    
    while current_size <= sampling_range:
        sample_sizes.append(current_size)
        current_size += multiple
    
    if not sample_sizes or sample_sizes[-1] != sampling_range:
        sample_sizes.append(sampling_range)
    
    print(f"\nSample sizes: {sample_sizes}")
    
    # Run trials for each sample size
    num_trials: int = 100
    print(f"Number of trials per sample size: {num_trials}")
    
    mean_estimated_budgets: List[float] = []
    std_estimated_budgets: List[float] = []
    
    print("\n" + "=" * 80)
    print("CONVERGENCE RESULTS (100 trials per sample size)")
    print("=" * 80)
    print(f"{'Sample Size':<15} {'Mean Est. Budget':<20} {'Std Est. Budget':<20} {'True Budget':<15}")
    print("-" * 80)
    
    torch.manual_seed(42)
    
    for sample_size in sample_sizes:
        estimated_budgets_trials: List[float] = []
        
        for trial in range(num_trials):
            # Sample from middle region
            perm: torch.Tensor = torch.randperm(sampling_range, device=device)
            sample: torch.Tensor = middle_expwts[perm[:sample_size]]
            
            # Estimate variance from sample
            estimated_var: float = torch.var(sample, unbiased=True).item()
            estimated_std: float = (estimated_var ** 0.5)
            
            # Estimate middle sum from sample
            sample_mean: float = torch.mean(sample).item()
            estimated_middle_sum: float = sample_mean * sampling_range
            
            # Compute estimated total sum
            estimated_total_sum: float = init_sum + estimated_middle_sum + local_sum
            
            # Compute estimated budget
            est_epsilon_error: float = epsilon * estimated_total_sum
            est_epsilon_error = max(est_epsilon_error, 1e-8)
            est_budget_numerator: float = delta_ppf * estimated_std * sampling_range
            estimated_budget: float = (est_budget_numerator / est_epsilon_error) ** 2
            estimated_budget = max(1.0, min(estimated_budget, float(sampling_range)))
            
            estimated_budgets_trials.append(estimated_budget)
        
        # Compute statistics
        budgets_tensor: torch.Tensor = torch.tensor(estimated_budgets_trials)
        mean_budget: float = torch.mean(budgets_tensor).item()
        std_budget: float = torch.std(budgets_tensor, unbiased=True).item()
        
        mean_estimated_budgets.append(mean_budget)
        std_estimated_budgets.append(std_budget)
        
        print(f"{sample_size:<15} {mean_budget:<20.2f} {std_budget:<20.2f} {true_budget:<15.2f}")
    
    # Create plot
    print("\n" + "=" * 80)
    print("GENERATING PLOT")
    print("=" * 80)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot estimated budget with error bars
    ax.errorbar(sample_sizes, mean_estimated_budgets, yerr=std_estimated_budgets,
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                color='blue', ecolor='blue', alpha=0.7,
                label='Estimated Budget ± 1σ (100 trials)')
    ax.axhline(y=true_budget, color='r', linestyle='--', linewidth=2,
               label=f'True Budget = {true_budget:.2f}')
    
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('Budget', fontsize=12)
    ax.set_title(f'Budget Convergence (ε={epsilon}, δ={delta}) - Head {head_id}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_dir: Path = filepath.parent
    plot_filename: str = f"denominator_budget_convergence_head{head_id}_eps{epsilon}_delta{delta}.png"
    plot_path: Path = output_dir / plot_filename
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def main() -> None:
    """Main function to parse arguments and run analysis."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Analyze dumped attention data and compute attention output"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the pickle file containing attention data",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="computation",
        choices=["computation", "denominator-estimation", "denominator-budget"],
        help="Type of analysis to perform (default: computation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (default: cpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about tensors",
    )
    
    # Arguments for head-specific analyses
    parser.add_argument(
        "--head-id",
        type=int,
        default=0,
        help="Head ID to analyze (default: 0, used by all analyses)",
    )
    parser.add_argument(
        "--init-offset",
        type=int,
        default=0,
        help="Initial offset to trim from the start (required for denominator-estimation and denominator-budget)",
    )
    parser.add_argument(
        "--local-offset",
        type=int,
        default=0,
        help="Local offset to trim from the end (required for denominator-estimation and denominator-budget)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Error bound for budget computation (required for denominator-budget, default: 0.1)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Confidence bound for budget computation (required for denominator-budget, default: 0.05)",
    )
    
    args: argparse.Namespace = parser.parse_args()
    
    filepath: Path = Path(args.filepath)
    
    # Route to appropriate analysis function
    if args.analysis == "computation":
        analyse_attention_computation(
            filepath=filepath,
            head_id=args.head_id,
            device=args.device,
            verbose=args.verbose,
        )
    elif args.analysis == "denominator-estimation":
        analyse_denominator_estimation(
            filepath=filepath,
            head_id=args.head_id,
            init_offset=args.init_offset,
            local_offset=args.local_offset,
            device=args.device,
        )
    elif args.analysis == "denominator-budget":
        analyse_denominator_budget(
            filepath=filepath,
            head_id=args.head_id,
            init_offset=args.init_offset,
            local_offset=args.local_offset,
            epsilon=args.epsilon,
            delta=args.delta,
            device=args.device,
        )


if __name__ == "__main__":
    main()

