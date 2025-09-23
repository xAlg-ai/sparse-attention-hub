"""Data analysis utilities for experiment logs.

This module provides functionality to load and analyze experiment log data,
particularly for AIME VATT experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


def load_experiment_data(file_path: str) -> pd.DataFrame:
    """Load experiment data from a log file into a DataFrame.
    
    The expected format is a space-separated file with columns:
    layer, density, error
    
    Args:
        file_path: Path to the experiment data file
        
    Returns:
        DataFrame with columns ['layer', 'density', 'error']
        
    Example:
        >>> df = load_experiment_data('extracted_aime_vatt_data')
        >>> print(df.head())
    """
    file_path_obj: Path = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the data, skipping the header row
    df: pd.DataFrame = pd.read_csv(
        file_path, 
        sep=' ', 
        names=['layer', 'density', 'error'],
        skiprows=1,
        dtype={'layer': int, 'density': float, 'error': float}
    )
    
    return df


def split_into_examples(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split experiment data into separate examples.
    
    A new example starts when the density switches to 1.0 from a different value.
    This function identifies these transition points and splits the data accordingly.
    
    Args:
        df: DataFrame containing experiment data with columns ['layer', 'density', 'error']
        
    Returns:
        List of DataFrames, each representing data for one example
        
    Example:
        >>> df = load_experiment_data('extracted_aime_vatt_data')
        >>> examples = split_into_examples(df)
        >>> print(f"Found {len(examples)} examples")
        >>> print(f"First example has {len(examples[0])} rows")
    """
    if df.empty:
        return []
    
    # Find transition points where density switches to 1.0
    density_is_one: pd.Series = df['density'] == 1.0
    
    # Find where density changes from non-1.0 to 1.0
    # This marks the start of a new example
    transition_points: List[int] = []
    
    # Check if the first row starts with density 1.0 (first example)
    if density_is_one.iloc[0]:
        transition_points.append(0)
    
    # Find subsequent transitions from non-1.0 to 1.0
    for i in range(1, len(df)):
        if density_is_one.iloc[i] and not density_is_one.iloc[i-1]:
            transition_points.append(i)
    
    # If no transitions found, return the entire dataset as one example
    if not transition_points:
        return [df.copy()]
    
    # Split the dataframe at transition points
    examples: List[pd.DataFrame] = []
    
    for i, start_idx in enumerate(transition_points):
        # Determine end index for this example
        if i < len(transition_points) - 1:
            end_idx: int = transition_points[i + 1]
            example_df: pd.DataFrame = df.iloc[start_idx:end_idx].copy()
        else:
            # Last example goes to the end
            example_df = df.iloc[start_idx:].copy()
        
        # Reset index for each example
        example_df.reset_index(drop=True, inplace=True)
        examples.append(example_df)
    
    return examples


def analyze_example_stats(examples: List[pd.DataFrame]) -> Dict[str, Any]:
    """Analyze statistics across all examples.
    
    Args:
        examples: List of DataFrames representing different examples
        
    Returns:
        Dictionary containing analysis statistics
        
    Example:
        >>> df = load_experiment_data('extracted_aime_vatt_data')
        >>> examples = split_into_examples(df)
        >>> stats = analyze_example_stats(examples)
        >>> print(f"Number of examples: {stats['num_examples']}")
    """
    if not examples:
        return {
            'num_examples': 0,
            'total_rows': 0,
            'avg_rows_per_example': 0,
            'min_rows_per_example': 0,
            'max_rows_per_example': 0,
            'unique_densities': [],
            'avg_error_overall': 0
        }
    
    num_examples: int = len(examples)
    example_lengths: List[int] = [len(example) for example in examples]
    total_rows: int = sum(example_lengths)
    
    # Collect all unique density values
    all_densities: set = set()
    all_errors: List[float] = []
    
    for example in examples:
        all_densities.update(example['density'].unique())
        all_errors.extend(example['error'].tolist())
    
    stats: Dict[str, Any] = {
        'num_examples': num_examples,
        'total_rows': total_rows,
        'avg_rows_per_example': total_rows / num_examples if num_examples > 0 else 0,
        'min_rows_per_example': min(example_lengths) if example_lengths else 0,
        'max_rows_per_example': max(example_lengths) if example_lengths else 0,
        'unique_densities': sorted(list(all_densities)),
        'avg_error_overall': sum(all_errors) / len(all_errors) if all_errors else 0
    }
    
    return stats


def load_and_split_experiment_data(file_path: str) -> tuple[pd.DataFrame, List[pd.DataFrame], Dict[str, Any]]:
    """Load experiment data and split into examples in one convenient function.
    
    Args:
        file_path: Path to the experiment data file
        
    Returns:
        Tuple containing:
        - Original DataFrame
        - List of DataFrames split by examples
        - Dictionary with analysis statistics
        
    Example:
        >>> df, examples, stats = load_and_split_experiment_data('extracted_aime_vatt_data')
        >>> print(f"Loaded {stats['num_examples']} examples")
    """
    # Load the data
    df: pd.DataFrame = load_experiment_data(file_path)
    
    # Split into examples
    examples: List[pd.DataFrame] = split_into_examples(df)
    
    # Analyze statistics
    stats: Dict[str, Any] = analyze_example_stats(examples)
    
    return df, examples, stats


def plot_example_metrics(examples: List[pd.DataFrame], 
                        example_id: int,
                        metric: str = 'both',
                        layer_mode: str = 'average',
                        specific_layers: Optional[List[int]] = None,
                        figsize: tuple = (15, 8),
                        save_path: Optional[str] = None) -> None:
    """Plot density and/or errors vs sequence length for a specific example.
    
    Args:
        examples: List of DataFrames representing different examples
        example_id: ID of the example to plot (0-indexed)
        metric: Which metric to plot ('density', 'error', or 'both')
        layer_mode: How to handle layers ('average', 'layer_wise', or 'specific')
        specific_layers: List of specific layer numbers to plot (used when layer_mode='specific')
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        
    Example:
        >>> df, examples, stats = load_and_split_experiment_data('extracted_aime_vatt_data')
        >>> # Plot average across all layers
        >>> plot_example_metrics(examples, 0, metric='both', layer_mode='average')
        >>> # Plot specific layers
        >>> plot_example_metrics(examples, 0, metric='density', layer_mode='specific', specific_layers=[0, 15, 31])
        >>> # Plot all layers separately
        >>> plot_example_metrics(examples, 1, metric='error', layer_mode='layer_wise')
    """
    if example_id >= len(examples) or example_id < 0:
        raise ValueError(f"Invalid example_id {example_id}. Must be between 0 and {len(examples)-1}")
    
    example_df: pd.DataFrame = examples[example_id]
    
    if metric not in ['density', 'error', 'both']:
        raise ValueError("metric must be 'density', 'error', or 'both'")
    
    if layer_mode not in ['average', 'layer_wise', 'specific']:
        raise ValueError("layer_mode must be 'average', 'layer_wise', or 'specific'")
    
    # Determine subplot configuration
    if metric == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    
    # Create sequence position based on order in the example
    # Since we have layer 0-31 cycling, sequence position is row_index // 32
    example_df = example_df.copy()
    example_df['sequence_pos'] = example_df.index // 32
    
    # Plot based on layer mode
    if layer_mode == 'average':
        _plot_average_layers(example_df, axes, metric)
    elif layer_mode == 'layer_wise':
        _plot_all_layers(example_df, axes, metric)
    elif layer_mode == 'specific':
        if specific_layers is None:
            raise ValueError("specific_layers must be provided when layer_mode='specific'")
        _plot_specific_layers(example_df, axes, metric, specific_layers)
    
    # Set common properties
    if metric == 'both':
        axes[-1].set_xlabel('Sequence Position')
        axes[0].set_title(f'Example {example_id} - Density and Error vs Sequence Length')
    else:
        axes[0].set_xlabel('Sequence Position')
        metric_name = 'Density' if metric == 'density' else 'Error'
        axes[0].set_title(f'Example {example_id} - {metric_name} vs Sequence Length')
    
    # Add grid and layout
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def _plot_average_layers(df: pd.DataFrame, axes: List, metric: str) -> None:
    """Plot metrics averaged across all layers."""
    # Group by sequence position and calculate mean
    avg_data: pd.DataFrame = df.groupby('sequence_pos').agg({
        'density': 'mean',
        'error': 'mean'
    }).reset_index()
    
    if metric in ['density', 'both']:
        ax_idx = 0 if metric == 'both' else 0
        axes[ax_idx].plot(avg_data['sequence_pos'], avg_data['density'], 
                         'b-', linewidth=2, label='Average Density', alpha=0.8)
        axes[ax_idx].set_ylabel('Density')
        axes[ax_idx].set_ylim(0, 1.05)
    
    if metric in ['error', 'both']:
        ax_idx = 1 if metric == 'both' else 0
        axes[ax_idx].plot(avg_data['sequence_pos'], avg_data['error'], 
                         'r-', linewidth=2, label='Average Error', alpha=0.8)
        axes[ax_idx].set_ylabel('Error')


def _plot_all_layers(df: pd.DataFrame, axes: List, metric: str) -> None:
    """Plot metrics for all layers separately."""
    layers: np.ndarray = df['layer'].unique()
    layers.sort()
    
    # Use a colormap for better distinction
    colors = plt.cm.tab20(np.linspace(0, 1, len(layers)))
    
    for i, layer in enumerate(layers):
        layer_data: pd.DataFrame = df[df['layer'] == layer].copy()
        layer_data = layer_data.sort_values('sequence_pos')
        
        if metric in ['density', 'both']:
            ax_idx = 0 if metric == 'both' else 0
            axes[ax_idx].plot(layer_data['sequence_pos'], layer_data['density'], 
                             color=colors[i], alpha=0.7, linewidth=1, 
                             label=f'Layer {layer}' if i < 10 else '')
            axes[ax_idx].set_ylabel('Density')
            axes[ax_idx].set_ylim(0, 1.05)
        
        if metric in ['error', 'both']:
            ax_idx = 1 if metric == 'both' else 0
            axes[ax_idx].plot(layer_data['sequence_pos'], layer_data['error'], 
                             color=colors[i], alpha=0.7, linewidth=1,
                             label=f'Layer {layer}' if i < 10 else '')
            axes[ax_idx].set_ylabel('Error')
    
    # Only show legend for first 10 layers to avoid clutter
    if len(layers) > 10:
        if metric in ['density', 'both']:
            axes[0].text(0.02, 0.98, f'Showing all {len(layers)} layers', 
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if metric in ['error', 'both']:
            ax_idx = 1 if metric == 'both' else 0
            axes[ax_idx].text(0.02, 0.98, f'Showing all {len(layers)} layers', 
                             transform=axes[ax_idx].transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _plot_specific_layers(df: pd.DataFrame, axes: List, metric: str, specific_layers: List[int]) -> None:
    """Plot metrics for specific layers."""
    available_layers: set = set(df['layer'].unique())
    
    # Filter to only available layers
    valid_layers: List[int] = [layer for layer in specific_layers if layer in available_layers]
    if not valid_layers:
        raise ValueError(f"None of the specified layers {specific_layers} are available in the data")
    
    # Use distinct colors for each layer
    colors = plt.cm.Set1(np.linspace(0, 1, len(valid_layers)))
    
    for i, layer in enumerate(valid_layers):
        layer_data: pd.DataFrame = df[df['layer'] == layer].copy()
        layer_data = layer_data.sort_values('sequence_pos')
        
        if metric in ['density', 'both']:
            ax_idx = 0 if metric == 'both' else 0
            axes[ax_idx].plot(layer_data['sequence_pos'], layer_data['density'], 
                             color=colors[i], linewidth=2, marker='o', markersize=3,
                             label=f'Layer {layer}', alpha=0.8)
            axes[ax_idx].set_ylabel('Density')
            axes[ax_idx].set_ylim(0, 1.05)
        
        if metric in ['error', 'both']:
            ax_idx = 1 if metric == 'both' else 0
            axes[ax_idx].plot(layer_data['sequence_pos'], layer_data['error'], 
                             color=colors[i], linewidth=2, marker='s', markersize=3,
                             label=f'Layer {layer}', alpha=0.8)
            axes[ax_idx].set_ylabel('Error')


def plot_multiple_examples_comparison(examples: List[pd.DataFrame],
                                    example_ids: List[int],
                                    metric: str = 'density',
                                    layer_mode: str = 'average',
                                    figsize: tuple = (15, 8),
                                    save_path: Optional[str] = None) -> None:
    """Compare metrics across multiple examples.
    
    Args:
        examples: List of DataFrames representing different examples
        example_ids: List of example IDs to compare
        metric: Which metric to plot ('density' or 'error')
        layer_mode: How to handle layers ('average' only supported for now)
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        
    Example:
        >>> df, examples, stats = load_and_split_experiment_data('extracted_aime_vatt_data')
        >>> plot_multiple_examples_comparison(examples, [0, 1, 2], metric='density')
    """
    if metric not in ['density', 'error']:
        raise ValueError("metric must be 'density' or 'error'")
    
    if layer_mode != 'average':
        raise ValueError("Only 'average' layer_mode is currently supported for comparison plots")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(example_ids)))
    
    for i, example_id in enumerate(example_ids):
        if example_id >= len(examples) or example_id < 0:
            print(f"Warning: Skipping invalid example_id {example_id}")
            continue
            
        example_df: pd.DataFrame = examples[example_id]
        example_df = example_df.copy()
        example_df['sequence_pos'] = example_df.index // 32
        
        # Calculate average across layers
        avg_data: pd.DataFrame = example_df.groupby('sequence_pos').agg({
            'density': 'mean',
            'error': 'mean'
        }).reset_index()
        
        ax.plot(avg_data['sequence_pos'], avg_data[metric], 
                color=colors[i], linewidth=2, label=f'Example {example_id}', alpha=0.8)
    
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison Across Examples (Average of All Layers)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if metric == 'density':
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
