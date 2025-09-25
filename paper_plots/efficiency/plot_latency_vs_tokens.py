#!/usr/bin/env python3
"""Plot latency vs remote tokens for Llama3-8B model.

This module creates a plot showing the relationship between remote tokens
and latency for the Llama3-8B model.
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


def create_plot(df: pd.DataFrame, output_path: str) -> None:
    """Create and save the latency vs remote tokens plot.
    
    Args:
        df: DataFrame containing remote tokens and latency data
        output_path: Path to save the output plot
    """
    # Set up the plot
    plt.figure(figsize=(12, 9))
    
    # Create the scatter plot with line connection
    plt.plot(df['Remote_Tokens_CPU'], df['Latency_ms'], 'o-', 
             linewidth=2, markersize=8, color='#2E86AB', 
             markerfacecolor='#A23B72', markeredgecolor='#2E86AB', markeredgewidth=2)
    
    # Add annotations for each point
    for idx, row in df.iterrows():
        tokens: int = row['Remote_Tokens_CPU']
        latency: float = row['Latency_ms']
        
        # Create annotation text showing both values
        if tokens >= 1000:
            tokens_label = f"{tokens//1000}K" if tokens % 1000 == 0 else f"{tokens/1000:.1f}K"
        else:
            tokens_label = str(tokens)
            
        if latency >= 1000:
            latency_label = f"{latency/1000:.1f}s"
        else:
            latency_label = f"{latency:.1f}ms"
            
        annotation_text: str = f'({tokens_label}, {latency_label})'
        
        # Position annotations to avoid overlap
        if tokens <= 4096:
            xytext_offset = (15, 15)  # Right and up
        elif tokens <= 32768:
            xytext_offset = (15, -20)  # Right and down
        elif tokens <= 131072:
            xytext_offset = (-15, 15)  # Left and up
        else:
            xytext_offset = (-15, -20)  # Left and down
        
        plt.annotate(annotation_text,
                    xy=(tokens, latency),
                    xytext=xytext_offset,
                    textcoords='offset points',
                    fontsize=9,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.6))
    
    # Customize the plot
    plt.xlabel('Remote Tokens CPU', fontsize=12, fontweight='bold')
    plt.ylabel('Single-token Decode Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Llama3-8B: Latency vs Remote Tokens', fontsize=14, fontweight='bold')
    
    # Use logarithmic scale on both axes since the data spans orders of magnitude
    plt.xscale('log')
    plt.yscale('log')
    
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
    """Main function to create the latency vs tokens plot."""
    csv_path: str = "/data/apdesai/code/sparse-attention-hub/paper_plots/efficiency/efficiency_data.csv"
    output_path: str = "/data/apdesai/code/sparse-attention-hub/paper_plots/efficiency/latency_vs_tokens.png"
    
    # Load and process data
    llama3_data: pd.DataFrame = load_and_filter_data(csv_path)
    
    # Sort by remote tokens for better line plotting
    llama3_data = llama3_data.sort_values('Remote_Tokens_CPU')
    
    # Display the data
    print("Llama3-8B Latency vs Remote Tokens Data:")
    print(llama3_data[['Remote_Tokens_CPU', 'Latency_ms']].to_string(index=False))
    print()
    
    # Create the plot
    create_plot(llama3_data, output_path)


if __name__ == "__main__":
    main()

