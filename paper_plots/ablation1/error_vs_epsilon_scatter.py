#!/usr/bin/env python3
"""Create scatter plot of Average Layer Error vs Epsilon for delta = 0.05.

This script extracts data for delta = 0.05 and creates a scatter plot showing
the relationship between epsilon and average layer error, with a fitted line.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from scipy import stats
from sklearn.linear_model import LinearRegression


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


def extract_delta_data(data: Dict[Tuple[float, float], float], target_delta: float) -> Tuple[List[float], List[float]]:
    """Extract epsilon and error values for a specific delta.
    
    Args:
        data: Dictionary mapping (epsilon, delta) to error values
        target_delta: The delta value to filter for
        
    Returns:
        Tuple of (epsilon_values, error_values) for the target delta
    """
    epsilons: List[float] = []
    errors: List[float] = []
    
    for (epsilon, delta), error in data.items():
        if delta == target_delta:
            epsilons.append(epsilon)
            errors.append(error)
    
    # Sort by epsilon for better visualization
    sorted_pairs = sorted(zip(epsilons, errors))
    epsilons_sorted, errors_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    return list(epsilons_sorted), list(errors_sorted)


def create_scatter_plot_with_fit(epsilons: List[float], errors: List[float], 
                                target_delta: float, filename: str) -> None:
    """Create scatter plot with fitted line.
    
    Args:
        epsilons: List of epsilon values
        errors: List of error values
        target_delta: The delta value being plotted
        filename: Output filename for the image
    """
    if len(epsilons) == 0:
        print(f"No data found for delta = {target_delta}")
        return
    
    # Convert to numpy arrays for fitting
    x: np.ndarray = np.array(epsilons)
    y: np.ndarray = np.array(errors)
    
    # Fit linear regression
    regressor = LinearRegression()
    x_reshaped: np.ndarray = x.reshape(-1, 1)
    regressor.fit(x_reshaped, y)
    
    # Generate line points
    x_line: np.ndarray = np.linspace(min(x), max(x), 100)
    y_line: np.ndarray = regressor.predict(x_line.reshape(-1, 1))
    
    # Calculate R-squared and other statistics
    r_squared: float = regressor.score(x_reshaped, y)
    slope: float = regressor.coef_[0]
    intercept: float = regressor.intercept_
    
    # Calculate correlation coefficient
    correlation, p_value = stats.pearsonr(x, y)
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(x, y, color='blue', alpha=0.7, s=120, edgecolors='darkblue', marker='*', linewidth=1)
    
    # Fitted line
    plt.plot(x_line, y_line, color='red', linewidth=3, alpha=0.8, linestyle='--',
             label=f'Linear fit: y = {slope:.4f}x + {intercept:.4f}')
    
    # Formatting
    plt.xlabel('Epsilon', fontsize=20)
    plt.ylabel(r'Average Layer Error ($\frac{||o - \hat{o}||}{||o||}$)', fontsize=20)
    #plt.title(r'Average Layer Error ($\frac{||o - \hat{o}||}{||o||}$) vs Epsilon (δ = ' + f'{target_delta})', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    
    # Add statistics text box
    textstr = f'R² = {r_squared:.4f}\nCorrelation = {correlation:.4f}\np-value = {p_value:.4e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=20,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved scatter plot to {filename}")
    
    # Print statistics
    print(f"\nStatistics for delta = {target_delta}:")
    print(f"Number of data points: {len(epsilons)}")
    print(f"Epsilon range: {min(epsilons):.3f} - {max(epsilons):.3f}")
    print(f"Error range: {min(errors):.6f} - {max(errors):.6f}")
    print(f"Linear fit: Average Layer Error = {slope:.6f} × Epsilon + {intercept:.6f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")


def main() -> None:
    """Main function to generate the scatter plot."""
    data_file: str = '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/error_density_data'
    target_delta: float = 0.05
    
    print("Parsing density and error data...")
    qa_1_density, qa_1_error, qa_2_density, qa_2_error = parse_density_error_data(data_file)
    
    print("Computing average error...")
    avg_error = compute_average_matrix(qa_1_error, qa_2_error)
    
    print(f"Extracting data for delta = {target_delta}...")
    epsilons, errors = extract_delta_data(avg_error, target_delta)
    
    print(f"Creating scatter plot with fitted line...")
    create_scatter_plot_with_fit(epsilons, errors, target_delta,
                                '/data/apdesai/code/sparse-attention-hub/paper_plots/ablation1/error_vs_epsilon_delta_0.05.png')


if __name__ == "__main__":
    main()
