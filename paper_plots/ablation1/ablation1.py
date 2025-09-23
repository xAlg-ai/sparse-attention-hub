#!/usr/bin/env python3
"""Generate matrix images showing accuracy vs epsilon and delta parameters.

This script creates heatmap matrices with epsilon on x-axis and delta on y-axis,
showing accuracy values for qa_1, qa_2, and their average.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import pandas as pd

# Raw data provided
data_lines = """
ruler32k_qa_1  12.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.4_delta_0.4
ruler32k_qa_1  12.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.5_delta_0.5
ruler32k_qa_1  20.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.5
ruler32k_qa_1  58.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.3_delta_0.3
ruler32k_qa_1  66.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.25_delta_0.25
ruler32k_qa_1  72.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.5_delta_0.05
ruler32k_qa_1  74.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.4_delta_0.05
ruler32k_qa_1  76.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.2_delta_0.2
ruler32k_qa_1  78.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.05
ruler32k_qa_1  78.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.2_delta_0.05
ruler32k_qa_1  78.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.3_delta_0.05
ruler32k_qa_1  80.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.01_delta_0.05
ruler32k_qa_1  80.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.1
ruler32k_qa_1  80.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.2
ruler32k_qa_1  80.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.3
ruler32k_qa_1  80.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.4
ruler32k_qa_1  80.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.1_delta_0.05
ruler32k_qa_1  80.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.1_delta_0.1
ruler32k_qa_1  82.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.01_delta_0.01
ruler32k_qa_1  82.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.01
ruler32k_qa_2  18.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.5
ruler32k_qa_2  26.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.4_delta_0.4
ruler32k_qa_2  28.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.5_delta_0.5
ruler32k_qa_2  48.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.2_delta_0.05
ruler32k_qa_2  48.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.3_delta_0.3
ruler32k_qa_2  50.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.01_delta_0.05
ruler32k_qa_2  50.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.1_delta_0.05
ruler32k_qa_2  50.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.5_delta_0.05
ruler32k_qa_2  52.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.01_delta_0.01
ruler32k_qa_2  52.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.01
ruler32k_qa_2  52.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.1
ruler32k_qa_2  52.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.3
ruler32k_qa_2  54.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.05
ruler32k_qa_2  54.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.2
ruler32k_qa_2  54.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.1_delta_0.1
ruler32k_qa_2  54.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.2_delta_0.2
ruler32k_qa_2  54.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.3_delta_0.05
ruler32k_qa_2  56.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.25_delta_0.25
ruler32k_qa_2  56.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.4_delta_0.05
ruler32k_qa_2  58.0,  sink_local_oracle_top_k_adaptive_sampling_other_epsilon_0.05_delta_0.4
""".strip()


def parse_data(data_lines: str) -> Tuple[Dict[Tuple[float, float], float], Dict[Tuple[float, float], float]]:
    """Parse the raw data and extract accuracy values for qa_1 and qa_2.
    
    Args:
        data_lines: Raw data string containing accuracy measurements
        
    Returns:
        Tuple of dictionaries mapping (epsilon, delta) to accuracy for qa_1 and qa_2
    """
    qa_1_data: Dict[Tuple[float, float], float] = {}
    qa_2_data: Dict[Tuple[float, float], float] = {}
    
    pattern = r'ruler32k_(qa_\d+)\s+(\d+\.?\d*),\s+.*epsilon_(\d+\.?\d*)_delta_(\d+\.?\d*)'
    
    for line in data_lines.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(pattern, line)
        if match:
            qa_type: str = match.group(1)
            accuracy: float = float(match.group(2))
            epsilon: float = float(match.group(3))
            delta: float = float(match.group(4))
            
            # Skip data points with delta = 0.25
            if delta == 0.25:
                continue
            
            key: Tuple[float, float] = (epsilon, delta)
            
            if qa_type == 'qa_1':
                qa_1_data[key] = accuracy
            elif qa_type == 'qa_2':
                qa_2_data[key] = accuracy
    
    return qa_1_data, qa_2_data


def create_matrix_from_data(data: Dict[Tuple[float, float], float]) -> Tuple[np.ndarray, List[float], List[float]]:
    """Create a matrix from the parsed data with proper indexing.
    
    Args:
        data: Dictionary mapping (epsilon, delta) to accuracy values
        
    Returns:
        Tuple of (matrix, epsilon_values, delta_values)
    """
    # Extract unique epsilon and delta values
    epsilons: List[float] = sorted(list(set(key[0] for key in data.keys())))
    deltas: List[float] = sorted(list(set(key[1] for key in data.keys())))
    
    # Create matrix
    matrix: np.ndarray = np.full((len(deltas), len(epsilons)), np.nan)
    
    for (epsilon, delta), accuracy in data.items():
        epsilon_idx: int = epsilons.index(epsilon)
        delta_idx: int = deltas.index(delta)
        matrix[delta_idx, epsilon_idx] = accuracy
    
    return matrix, epsilons, deltas


def create_heatmap(matrix: np.ndarray, epsilons: List[float], deltas: List[float], 
                   title: str, filename: str) -> None:
    """Create and save a heatmap visualization.
    
    Args:
        matrix: 2D array of accuracy values
        epsilons: List of epsilon values for x-axis
        deltas: List of delta values for y-axis
        title: Plot title
        filename: Output filename for the image
    """
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with custom colormap
    mask: np.ndarray = np.isnan(matrix)
    sns.heatmap(matrix, 
                xticklabels=[f"{eps:.2f}" for eps in epsilons],
                yticklabels=[f"{delta:.2f}" for delta in deltas],
                annot=True, 
                fmt='.1f',
                cmap='RdYlBu_r',
                mask=mask,
                cbar_kws={'label': 'Accuracy (%)'},
                square=True)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epsilon', fontsize=14)
    plt.ylabel('Delta', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved heatmap to {filename}")


def compute_average_matrix(qa_1_data: Dict[Tuple[float, float], float], 
                          qa_2_data: Dict[Tuple[float, float], float]) -> Dict[Tuple[float, float], float]:
    """Compute average accuracy for common (epsilon, delta) pairs.
    
    Args:
        qa_1_data: QA1 accuracy data
        qa_2_data: QA2 accuracy data
        
    Returns:
        Dictionary with average accuracy values
    """
    avg_data: Dict[Tuple[float, float], float] = {}
    
    # Find common keys
    common_keys: set = set(qa_1_data.keys()) & set(qa_2_data.keys())
    
    for key in common_keys:
        avg_data[key] = (qa_1_data[key] + qa_2_data[key]) / 2.0
    
    return avg_data


def main() -> None:
    """Main function to generate all matrix visualizations."""
    print("Parsing data...")
    qa_1_data, qa_2_data = parse_data(data_lines)
    
    print(f"Found {len(qa_1_data)} data points for QA1")
    print(f"Found {len(qa_2_data)} data points for QA2")
    
    # Update todo status
    print("Creating QA1 matrix...")
    qa_1_matrix, qa_1_epsilons, qa_1_deltas = create_matrix_from_data(qa_1_data)
    create_heatmap(qa_1_matrix, qa_1_epsilons, qa_1_deltas, 
                   'QA1 Accuracy vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/qa_1_accuracy_matrix.png')
    
    print("Creating QA2 matrix...")
    qa_2_matrix, qa_2_epsilons, qa_2_deltas = create_matrix_from_data(qa_2_data)
    create_heatmap(qa_2_matrix, qa_2_epsilons, qa_2_deltas, 
                   'QA2 Accuracy vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/qa_2_accuracy_matrix.png')
    
    print("Computing average matrix...")
    avg_data = compute_average_matrix(qa_1_data, qa_2_data)
    print(f"Found {len(avg_data)} common data points for average")
    
    avg_matrix, avg_epsilons, avg_deltas = create_matrix_from_data(avg_data)
    create_heatmap(avg_matrix, avg_epsilons, avg_deltas, 
                   'Average QA Accuracy vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/qa_average_accuracy_matrix.png')
    
    print("\nSummary:")
    print("========")
    print(f"QA1 matrix shape: {qa_1_matrix.shape}")
    print(f"QA2 matrix shape: {qa_2_matrix.shape}")
    print(f"Average matrix shape: {avg_matrix.shape}")
    print(f"Epsilon range: {min(qa_1_epsilons + qa_2_epsilons)} - {max(qa_1_epsilons + qa_2_epsilons)}")
    print(f"Delta range: {min(qa_1_deltas + qa_2_deltas)} - {max(qa_1_deltas + qa_2_deltas)}")
    print("\nAll matrix images have been generated successfully!")


if __name__ == "__main__":
    main()
