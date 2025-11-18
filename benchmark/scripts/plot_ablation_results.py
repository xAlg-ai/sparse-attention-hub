#!/usr/bin/env python3
"""
Plot Ablation Results

This script analyzes the ablation results from the benchmark experiments and generates
plots comparing research_attention_density and research_attention_output_error across
different epsilon values for both denominator and numerator modes.

Usage:
    python plot_ablation_results.py
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Change to directory and add to Python path
os.chdir('/workspace/sparse-attention-hub')
sys.path.insert(0, '/workspace/sparse-attention-hub')


def parse_directory_name(dir_name: str) -> Tuple[str, float, float]:
    """
    Parse directory name to extract mode, epsilon, and delta values.
    
    Args:
        dir_name: Directory name (e.g., 'denominator.eps_0.01.delta_0.025')
    
    Returns:
        Tuple of (mode, epsilon, delta)
    """
    pattern = r'(denominator|numerator)\.eps_([\d\.]+)\.delta_([\d\.]+)'
    match = re.match(pattern, dir_name)
    
    if match:
        mode: str = match.group(1)
        epsilon: float = float(match.group(2))
        delta: float = float(match.group(3))
        return mode, epsilon, delta
    
    # Handle old format (den.eps_0.01)
    old_pattern = r'den\.eps_([\d\.]+)'
    old_match = re.match(old_pattern, dir_name)
    if old_match:
        epsilon_val: float = float(old_match.group(1))
        return "denominator", epsilon_val, 0.025
    
    raise ValueError(f"Cannot parse directory name: {dir_name}")


def load_metrics_from_jsonl(file_path: Path) -> Dict[str, List[float]]:
    """
    Load metrics from a JSONL file.
    
    Args:
        file_path: Path to the micro_metrics.jsonl file
    
    Returns:
        Dictionary with metric names as keys and lists of values
    """
    metrics: Dict[str, List[float]] = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            data: dict = json.loads(line.strip())
            metric_name: str = data['metric']
            value: float = data['value']
            metrics[metric_name].append(value)
    
    return metrics


def aggregate_ablation_results(ablations_dir: Path) -> Dict[str, Dict[float, Dict[float, Dict[str, float]]]]:
    """
    Aggregate all ablation results by mode, delta, and epsilon.
    
    Args:
        ablations_dir: Path to the ablations directory
    
    Returns:
        Nested dictionary: {mode: {delta: {epsilon: {metric_name: avg_value}}}}
    """
    results: Dict[str, Dict[float, Dict[float, Dict[str, float]]]] = {
        'denominator': defaultdict(dict),
        'numerator': defaultdict(dict)
    }
    
    for dir_path in ablations_dir.iterdir():
        if not dir_path.is_dir():
            continue
        
        try:
            mode, epsilon, delta = parse_directory_name(dir_path.name)
        except ValueError:
            print(f"Skipping directory: {dir_path.name}")
            continue
        
        metrics_file: Path = dir_path / 'micro_metrics.jsonl'
        if not metrics_file.exists():
            print(f"No metrics file found in: {dir_path.name}")
            continue
        
        print(f"Processing: {dir_path.name} (mode={mode}, epsilon={epsilon}, delta={delta})")
        
        # Load and aggregate metrics
        metrics: Dict[str, List[float]] = load_metrics_from_jsonl(metrics_file)
        
        avg_metrics: Dict[str, float] = {}
        for metric_name, values in metrics.items():
            avg_metrics[metric_name] = float(np.mean(values))
            print(f"  {metric_name}: {avg_metrics[metric_name]:.6f} (n={len(values)})")
        
        results[mode][delta][epsilon] = avg_metrics
    
    return results


def create_plots(results: Dict[str, Dict[float, Dict[float, Dict[str, float]]]], output_dir: Path) -> None:
    """
    Create plots for each mode showing density and error vs epsilon, with one line per delta.
    
    Args:
        results: Aggregated results from aggregate_ablation_results
        output_dir: Directory to save the plots
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    modes: List[str] = ['denominator', 'numerator']
    colors: List[str] = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    markers: List[str] = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for mode in modes:
        if not results[mode]:
            print(f"No data for mode: {mode}")
            continue
        
        # Sort delta values
        deltas: List[float] = sorted(results[mode].keys())
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot for each delta value
        for idx, delta in enumerate(deltas):
            color: str = colors[idx % len(colors)]
            marker: str = markers[idx % len(markers)]
            
            # Sort epsilon values for this delta
            epsilons: List[float] = sorted(results[mode][delta].keys())
            
            # Extract metric values
            densities: List[float] = []
            errors: List[float] = []
            
            for eps in epsilons:
                densities.append(results[mode][delta][eps].get('research_attention_density', 0.0))
                errors.append(results[mode][delta][eps].get('research_attention_output_error', 0.0))
            
            # Plot 1: Density vs Epsilon
            ax1.plot(epsilons, densities, marker=marker, linewidth=2, markersize=8, 
                    color=color, label=f'δ = {delta}', alpha=0.8)
            
            # Plot 2: Error vs Epsilon (scatter with fitted line)
            # Scatter plot
            ax2.scatter(epsilons, errors, s=100, color=color, alpha=0.6, 
                       marker=marker, edgecolors='black', linewidth=1, zorder=3)
            
            # Fit a line to the data
            if len(epsilons) >= 2:
                coefficients: np.ndarray = np.polyfit(epsilons, errors, 1)
                poly_function = np.poly1d(coefficients)
                
                # Calculate R² and correlation coefficient
                y_pred: np.ndarray = poly_function(epsilons)
                ss_res: float = float(np.sum((np.array(errors) - y_pred) ** 2))
                ss_tot: float = float(np.sum((np.array(errors) - np.mean(errors)) ** 2))
                r_squared: float = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calculate Pearson correlation coefficient
                correlation: float = float(np.corrcoef(epsilons, errors)[0, 1])
                
                # Generate points for the fitted line
                x_fit: np.ndarray = np.linspace(min(epsilons), max(epsilons), 100)
                y_fit: np.ndarray = poly_function(x_fit)
                
                # Plot the fitted line
                ax2.plot(x_fit, y_fit, linestyle='--', linewidth=2, color=color, alpha=0.7, 
                        label=f'δ={delta}: y={coefficients[0]:.4f}x+{coefficients[1]:.4f}, r={correlation:.3f}, R²={r_squared:.3f}', 
                        zorder=2)
        
        # Configure Plot 1: Density
        ax1.set_xlabel('Epsilon (ε)', fontsize=12)
        ax1.set_ylabel('Average Research Attention Density', fontsize=12)
        ax1.set_title(f'Attention Density vs Epsilon\n(Mode: {mode.capitalize()})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.legend(fontsize=10, loc='best', framealpha=0.9)
        
        # Configure Plot 2: Error
        ax2.set_xlabel('Epsilon (ε)', fontsize=12)
        ax2.set_ylabel('Average Research Attention Output Error', fontsize=12)
        ax2.set_title(f'Attention Output Error vs Epsilon\n(Mode: {mode.capitalize()})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='best', framealpha=0.9)
        # Keep both axes non-logarithmic for error plot
        
        plt.tight_layout()
        
        # Save plot
        output_file: Path = output_dir / f'ablation_{mode}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: {output_file}")
        
        plt.close()


def main() -> None:
    """Main function to orchestrate the plotting."""
    ablations_dir: Path = Path('./ablations')
    output_dir: Path = Path('./benchmark/scripts/plots')
    
    if not ablations_dir.exists():
        print(f"Error: Ablations directory not found: {ablations_dir}")
        sys.exit(1)
    
    print("="*80)
    print("Analyzing Ablation Results")
    print("="*80)
    
    # Aggregate results
    results: Dict[str, Dict[float, Dict[str, float]]] = aggregate_ablation_results(ablations_dir)
    
    print("\n" + "="*80)
    print("Creating Plots")
    print("="*80)
    
    # Create plots
    create_plots(results, output_dir)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for mode in ['denominator', 'numerator']:
        if results[mode]:
            print(f"\n{mode.capitalize()} mode:")
            for delta in sorted(results[mode].keys()):
                epsilon_values: List[float] = sorted(results[mode][delta].keys())
                print(f"  Delta {delta}: {len(epsilon_values)} epsilon values {epsilon_values}")
    
    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

