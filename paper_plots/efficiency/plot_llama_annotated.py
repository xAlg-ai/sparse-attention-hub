#!/usr/bin/env python3
"""Plot efficiency data for Llama3-8B model with annotated points.

This module creates a clean plot showing the relationship between sparsity and speedup
for the Llama3-8B model, with logarithmic scales on both axes and annotated data points.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def load_and_filter_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and filter for Llama3-8B model.
    
    Args:
        csv_path: Path to the CSV file containing efficiency data
        
    Returns:
        DataFrame containing only Llama3-8B data
    """
    df: pd.DataFrame = pd.read_csv(csv_path)
    llama3_data: pd.DataFrame = df[df['Model'] == 'Llama3-8B'].copy()
    return llama3_data


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sparsity and speedup metrics.
    
    Args:
        df: DataFrame containing the raw benchmark data
        
    Returns:
        DataFrame with added sparsity and speedup columns
    """
    # Calculate sparsity = 1048576 / remote_tokens_CPU
    df['Sparsity'] = 1048576 / df['Remote_Tokens_CPU']
    
    # Calculate speedup = 30927.049 / Latency_ms (baseline latency for max tokens)
    baseline_latency: float = 30927.049
    df['Speedup'] = baseline_latency / df['Latency_ms']
    
    return df


def create_plot(df: pd.DataFrame, output_path: str) -> None:
    """Create and save the efficiency plot with logarithmic scales and annotations.
    
    Args:
        df: DataFrame containing sparsity and speedup data
        output_path: Path to save the output plot
    """
    # Set up the plot
    plt.figure(figsize=(12, 9))
    
    # Create the scatter plot with line connection
    plt.plot(df['Sparsity'], df['Speedup'], 'o-', 
             linewidth=2, markersize=8, color='#2E86AB', 
             markerfacecolor='#A23B72', markeredgecolor='#2E86AB', markeredgewidth=2)
    
    # Add annotations for each point
    for idx, row in df.iterrows():
        sparsity: float = row['Sparsity']
        speedup: float = row['Speedup']
        
        # Create annotation text showing both sparsity and speedup
        annotation_text: str = f'({sparsity:.0f}, {speedup:.1f}x)'
        
        # Position annotations slightly offset from points
        # Use different offsets based on point position to avoid overlap
        if sparsity <= 4:
            xytext_offset = (10, 10)  # Right and up
        elif sparsity <= 16:
            xytext_offset = (10, -15)  # Right and down
        else:
            xytext_offset = (-10, 10)  # Left and up
        
        plt.annotate(annotation_text,
                    xy=(sparsity, speedup),
                    xytext=xytext_offset,
                    textcoords='offset points',
                    fontsize=9,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.6))
    
    # Customize the plot
    plt.xlabel('Sparsity (1048576 / Remote Tokens CPU)', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup (Baseline Latency / Current Latency)', fontsize=12, fontweight='bold')
    plt.title('Llama3-8B: Speedup vs Sparsity (Log-Log Scale, Annotated Points)', fontsize=14, fontweight='bold')
    
    # Use logarithmic scale on both axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Set x-axis limit to 64 (but since we're using log scale, let it auto-adjust within the data range)
    plt.xlim(0.8, 70)  # Slightly wider range for better visualization on log scale
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Customize ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def main() -> None:
    """Main function to create the efficiency plot."""
    csv_path: str = "/data/apdesai/code/sparse-attention-hub/paper_plots/efficiency/efficiency_data.csv"
    output_path: str = "/data/apdesai/code/sparse-attention-hub/paper_plots/efficiency/llama.png"
    
    # Load and process data
    llama3_data: pd.DataFrame = load_and_filter_data(csv_path)
    llama3_data = calculate_metrics(llama3_data)
    
    # Filter to sparsity <= 64
    llama3_data = llama3_data[llama3_data['Sparsity'] <= 64]
    
    # Sort by sparsity for better line plotting
    llama3_data = llama3_data.sort_values('Sparsity')
    
    # Display the calculated metrics
    print("Llama3-8B Efficiency Data (Sparsity ≤ 64, Log-Log Scale, Annotated):")
    print(llama3_data[['Remote_Tokens_CPU', 'Latency_ms', 'Sparsity', 'Speedup']].to_string(index=False))
    print()
    
    # Create the plot
    create_plot(llama3_data, output_path)


if __name__ == "__main__":
    main()

