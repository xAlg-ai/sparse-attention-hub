#!/usr/bin/env python3
"""Generate combined 1x3 grid showing average density, error, and accuracy matrices.

This script creates a single figure with three subplots showing the average
density, error, and accuracy values across epsilon and delta parameters.
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
    """
    qa_1_density: Dict[Tuple[float, float], float] = {}
    qa_1_error: Dict[Tuple[float, float], float] = {}
    qa_2_density: Dict[Tuple[float, float], float] = {}
    qa_2_error: Dict[Tuple[float, float], float] = {}
    
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
            
            match = re.search(pattern, filepath)
            if match:
                epsilon: float = float(match.group(1))
                delta: float = float(match.group(2))
                qa_type: str = match.group(3)
                
                # Skip data points with delta = 0.25
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


def parse_accuracy_data() -> Tuple[Dict[Tuple[float, float], float], Dict[Tuple[float, float], float]]:
    """Parse the accuracy data from ablation1.py format.
    
    Returns:
        Tuple of dictionaries for (qa_1_accuracy, qa_2_accuracy)
    """
    # Raw data from ablation1.py
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


def create_combined_heatmap(density_matrix: np.ndarray, error_matrix: np.ndarray, 
                           accuracy_matrix: np.ndarray, epsilons: List[float], 
                           deltas: List[float], filename: str) -> None:
    """Create a combined 1x3 heatmap visualization.
    
    Args:
        density_matrix: 2D array of density values
        error_matrix: 2D array of error values 
        accuracy_matrix: 2D array of accuracy values
        epsilons: List of epsilon values for x-axis
        deltas: List of delta values for y-axis
        filename: Output filename for the image
    """
    # Create figure with 1 row, 3 columns (adjusted height for horizontal colorbars)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Common settings
    epsilon_labels: List[str] = [f"{eps:.2f}" for eps in epsilons]
    delta_labels: List[str] = [f"{delta:.2f}" for delta in deltas]
    
    # Density plot (left)
    mask_density: np.ndarray = np.isnan(density_matrix)
    sns.heatmap(density_matrix, 
                ax=axes[0],
                xticklabels=epsilon_labels,
                yticklabels=delta_labels,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                mask=mask_density,
                cbar_kws={'label': 'Density', 'orientation': 'horizontal', 'pad': 0.1},
                square=True)
    axes[0].set_title('Average Density', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epsilon', fontsize=12)
    axes[0].set_ylabel('Delta', fontsize=12)
    
    # Error plot (middle)
    mask_error: np.ndarray = np.isnan(error_matrix)
    sns.heatmap(error_matrix, 
                ax=axes[1],
                xticklabels=epsilon_labels,
                yticklabels=delta_labels,
                annot=True, 
                fmt='.3f',
                cmap='Reds',
                mask=mask_error,
                cbar_kws={'label': 'Error', 'orientation': 'horizontal', 'pad': 0.1},
                square=True)
    axes[1].set_title('Average Layer Error', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epsilon', fontsize=12)
    axes[1].set_ylabel('Delta', fontsize=12)
    
    # Accuracy plot (right)
    mask_accuracy: np.ndarray = np.isnan(accuracy_matrix)
    sns.heatmap(accuracy_matrix, 
                ax=axes[2],
                xticklabels=epsilon_labels,
                yticklabels=delta_labels,
                annot=True, 
                fmt='.1f',
                cmap='RdYlBu_r',
                mask=mask_accuracy,
                cbar_kws={'label': 'Accuracy (%)', 'orientation': 'horizontal', 'pad': 0.1},
                square=True)
    axes[2].set_title('Average Accuracy', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epsilon', fontsize=12)
    axes[2].set_ylabel('Delta', fontsize=12)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved combined heatmap to {filename}")


def main() -> None:
    """Main function to generate the combined matrix visualization."""
    data_file: str = '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/error_density_data'
    
    print("Parsing density and error data...")
    qa_1_density, qa_1_error, qa_2_density, qa_2_error = parse_density_error_data(data_file)
    
    print("Parsing accuracy data...")
    qa_1_accuracy, qa_2_accuracy = parse_accuracy_data()
    
    print("Computing average matrices...")
    
    # Compute averages
    avg_density = compute_average_matrix(qa_1_density, qa_2_density)
    avg_error = compute_average_matrix(qa_1_error, qa_2_error)
    avg_accuracy = compute_average_matrix(qa_1_accuracy, qa_2_accuracy)
    
    print(f"Found {len(avg_density)} common data points for density")
    print(f"Found {len(avg_error)} common data points for error")
    print(f"Found {len(avg_accuracy)} common data points for accuracy")
    
    # Create matrices
    density_matrix, density_epsilons, density_deltas = create_matrix_from_data(avg_density)
    error_matrix, error_epsilons, error_deltas = create_matrix_from_data(avg_error)
    accuracy_matrix, accuracy_epsilons, accuracy_deltas = create_matrix_from_data(avg_accuracy)
    
    # Use the same epsilon/delta ordering for all matrices (from density data)
    print("Creating combined 1x3 heatmap...")
    create_combined_heatmap(density_matrix, error_matrix, accuracy_matrix,
                           density_epsilons, density_deltas,
                           '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/combined_average_matrices.png')
    
    print("\nSummary:")
    print("========")
    print(f"Density matrix shape: {density_matrix.shape}")
    print(f"Error matrix shape: {error_matrix.shape}")
    print(f"Accuracy matrix shape: {accuracy_matrix.shape}")
    print(f"Epsilon range: {min(density_epsilons)} - {max(density_epsilons)}")
    print(f"Delta range: {min(density_deltas)} - {max(density_deltas)}")
    print("\nCombined matrix visualization has been generated successfully!")


if __name__ == "__main__":
    main()
