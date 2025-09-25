#!/usr/bin/env python3
"""Generate matrix images showing density and error vs epsilon and delta parameters.

This script creates heatmap matrices with epsilon on x-axis and delta on y-axis,
showing density and error values for qa_1, qa_2, and their average.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import pandas as pd


def parse_density_error_data(filename: str) -> Tuple[Dict[Tuple[float, float], float], 
                                                   Dict[Tuple[float, float], float],
                                                   Dict[Tuple[float, float], float], 
                                                   Dict[Tuple[float, float], float]]:
    """Parse the density and error data file.
    
    Args:
        filename: Path to the error_density_data file
        
    Returns:
        Tuple of dictionaries for (qa_1_density, qa_1_error, qa_2_density, qa_2_error)
        Each dict maps (epsilon, delta) to the respective value
    """
    qa_1_density: Dict[Tuple[float, float], float] = {}
    qa_1_error: Dict[Tuple[float, float], float] = {}
    qa_2_density: Dict[Tuple[float, float], float] = {}
    qa_2_error: Dict[Tuple[float, float], float] = {}
    
    # Pattern to extract epsilon, delta, and QA type from filename
    pattern = r'epsilon_(\d+\.?\d*)_delta_(\d+\.?\d*).*/(ruler32k_qa_\d+)/'
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num == 1:  # Skip header
                continue
                
            line = line.strip()
            if not line:
                continue
            
            parts: List[str] = line.split()
            if len(parts) < 3:
                continue
                
            filepath: str = parts[0]
            density: float = float(parts[1])
            error: float = float(parts[2])
            
            # Extract epsilon, delta, and QA type from filepath
            match = re.search(pattern, filepath)
            if match:
                epsilon: float = float(match.group(1))
                delta: float = float(match.group(2))
                qa_type: str = match.group(3)
                
                # Skip data points with delta = 0.25 (as done in ablation1.py)
                if delta == 0.25:
                    continue
                
                key: Tuple[float, float] = (epsilon, delta)
                
                if qa_type == 'ruler32k_qa_1':
                    qa_1_density[key] = density
                    qa_1_error[key] = error
                elif qa_type == 'ruler32k_qa_2':
                    qa_2_density[key] = density
                    qa_2_error[key] = error
    
    return qa_1_density, qa_1_error, qa_2_density, qa_2_error


def create_matrix_from_data(data: Dict[Tuple[float, float], float]) -> Tuple[np.ndarray, List[float], List[float]]:
    """Create a matrix from the parsed data with proper indexing.
    
    Args:
        data: Dictionary mapping (epsilon, delta) to values
        
    Returns:
        Tuple of (matrix, epsilon_values, delta_values)
    """
    if not data:
        return np.array([]), [], []
        
    # Extract unique epsilon and delta values
    epsilons: List[float] = sorted(list(set(key[0] for key in data.keys())))
    deltas: List[float] = sorted(list(set(key[1] for key in data.keys())))
    
    # Create matrix
    matrix: np.ndarray = np.full((len(deltas), len(epsilons)), np.nan)
    
    for (epsilon, delta), value in data.items():
        epsilon_idx: int = epsilons.index(epsilon)
        delta_idx: int = deltas.index(delta)
        matrix[delta_idx, epsilon_idx] = value
    
    return matrix, epsilons, deltas


def create_heatmap(matrix: np.ndarray, epsilons: List[float], deltas: List[float], 
                   title: str, filename: str, value_label: str, 
                   colormap: str = 'viridis', fmt: str = '.3f') -> None:
    """Create and save a heatmap visualization.
    
    Args:
        matrix: 2D array of values
        epsilons: List of epsilon values for x-axis
        deltas: List of delta values for y-axis
        title: Plot title
        filename: Output filename for the image
        value_label: Label for the colorbar
        colormap: Matplotlib colormap name
        fmt: Format string for annotations
    """
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with custom colormap
    mask: np.ndarray = np.isnan(matrix)
    sns.heatmap(matrix, 
                xticklabels=[f"{eps:.2f}" for eps in epsilons],
                yticklabels=[f"{delta:.2f}" for delta in deltas],
                annot=True, 
                fmt=fmt,
                cmap=colormap,
                mask=mask,
                cbar_kws={'label': value_label},
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
    """Compute average values for common (epsilon, delta) pairs.
    
    Args:
        qa_1_data: QA1 data
        qa_2_data: QA2 data
        
    Returns:
        Dictionary with average values
    """
    avg_data: Dict[Tuple[float, float], float] = {}
    
    # Find common keys
    common_keys: set = set(qa_1_data.keys()) & set(qa_2_data.keys())
    
    for key in common_keys:
        avg_data[key] = (qa_1_data[key] + qa_2_data[key]) / 2.0
    
    return avg_data


def main() -> None:
    """Main function to generate all matrix visualizations."""
    data_file: str = '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/error_density_data'
    
    print("Parsing density and error data...")
    qa_1_density, qa_1_error, qa_2_density, qa_2_error = parse_density_error_data(data_file)
    
    print(f"Found {len(qa_1_density)} density data points for QA1")
    print(f"Found {len(qa_1_error)} error data points for QA1")
    print(f"Found {len(qa_2_density)} density data points for QA2")
    print(f"Found {len(qa_2_error)} error data points for QA2")
    
    # Create density matrices
    print("\nCreating density matrices...")
    
    # QA1 Density
    qa_1_density_matrix, qa_1_d_epsilons, qa_1_d_deltas = create_matrix_from_data(qa_1_density)
    create_heatmap(qa_1_density_matrix, qa_1_d_epsilons, qa_1_d_deltas, 
                   'QA1 Density vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/qa_1_density_matrix.png',
                   'Density', 'viridis')
    
    # QA2 Density
    qa_2_density_matrix, qa_2_d_epsilons, qa_2_d_deltas = create_matrix_from_data(qa_2_density)
    create_heatmap(qa_2_density_matrix, qa_2_d_epsilons, qa_2_d_deltas, 
                   'QA2 Density vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/qa_2_density_matrix.png',
                   'Density', 'viridis')
    
    # Average Density
    avg_density = compute_average_matrix(qa_1_density, qa_2_density)
    print(f"Found {len(avg_density)} common data points for average density")
    
    avg_density_matrix, avg_d_epsilons, avg_d_deltas = create_matrix_from_data(avg_density)
    create_heatmap(avg_density_matrix, avg_d_epsilons, avg_d_deltas, 
                   'Average Density vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/qa_average_density_matrix.png',
                   'Density', 'viridis')
    
    # Create error matrices
    print("\nCreating error matrices...")
    
    # QA1 Error
    qa_1_error_matrix, qa_1_e_epsilons, qa_1_e_deltas = create_matrix_from_data(qa_1_error)
    create_heatmap(qa_1_error_matrix, qa_1_e_epsilons, qa_1_e_deltas, 
                   'QA1 Error vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/qa_1_error_matrix.png',
                   'Error', 'Reds')
    
    # QA2 Error  
    qa_2_error_matrix, qa_2_e_epsilons, qa_2_e_deltas = create_matrix_from_data(qa_2_error)
    create_heatmap(qa_2_error_matrix, qa_2_e_epsilons, qa_2_e_deltas, 
                   'QA2 Error vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/qa_2_error_matrix.png',
                   'Error', 'Reds')
    
    # Average Error
    avg_error = compute_average_matrix(qa_1_error, qa_2_error)
    print(f"Found {len(avg_error)} common data points for average error")
    
    avg_error_matrix, avg_e_epsilons, avg_e_deltas = create_matrix_from_data(avg_error)
    create_heatmap(avg_error_matrix, avg_e_epsilons, avg_e_deltas, 
                   'Average Error vs Epsilon and Delta', 
                   '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/qa_average_error_matrix.png',
                   'Error', 'Reds')
    
    print("\nSummary:")
    print("========")
    print(f"QA1 density matrix shape: {qa_1_density_matrix.shape}")
    print(f"QA2 density matrix shape: {qa_2_density_matrix.shape}")
    print(f"Average density matrix shape: {avg_density_matrix.shape}")
    print(f"QA1 error matrix shape: {qa_1_error_matrix.shape}")
    print(f"QA2 error matrix shape: {qa_2_error_matrix.shape}")
    print(f"Average error matrix shape: {avg_error_matrix.shape}")
    
    if qa_1_d_epsilons:
        all_epsilons = qa_1_d_epsilons + qa_2_d_epsilons + qa_1_e_epsilons + qa_2_e_epsilons
        all_deltas = qa_1_d_deltas + qa_2_d_deltas + qa_1_e_deltas + qa_2_e_deltas
        print(f"Epsilon range: {min(all_epsilons)} - {max(all_epsilons)}")
        print(f"Delta range: {min(all_deltas)} - {max(all_deltas)}")
    
    print("\nAll density and error matrix images have been generated successfully!")


if __name__ == "__main__":
    main()
