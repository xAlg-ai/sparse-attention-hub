#!/usr/bin/env python3
"""Aggregate micro metrics from JSONL files.

This script reads micro_metrics.jsonl files and computes aggregate statistics
for density and error metrics across different layers.

Example:
    $ python aggregate_micro_metrics.py path/to/micro_metrics.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics
from collections import defaultdict


def load_metrics(file_path: Path) -> List[Dict[str, Any]]:
    """Load metrics from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing metrics
        
    Returns:
        List of metric dictionaries
    """
    metrics: List[Dict[str, Any]] = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metric = json.loads(line)
                    metrics.append(metric)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
                    
    return metrics


def separate_metrics(metrics: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """Separate density and error metrics from the list.
    
    Args:
        metrics: List of metric dictionaries
        
    Returns:
        Tuple of (density_values, error_values)
    """
    density_values: List[float] = []
    error_values: List[float] = []
    
    for metric in metrics:
        if metric['metric'] == 'research_attention_density':
            density_values.append(metric['value'])
        elif metric['metric'] == 'research_attention_output_error':
            error_values.append(metric['value'])
            
    return density_values, error_values


def get_layer_metrics(metrics: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[float]]]:
    """Group metrics by layer index.
    
    Args:
        metrics: List of metric dictionaries
        
    Returns:
        Dictionary mapping layer_idx to metrics dictionary
    """
    layer_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {'density': [], 'error': []})
    
    for metric in metrics:
        layer_idx = metric.get('metadata', {}).get('layer_idx')
        if layer_idx is not None:
            if metric['metric'] == 'research_attention_density':
                layer_metrics[layer_idx]['density'].append(metric['value'])
            elif metric['metric'] == 'research_attention_output_error':
                layer_metrics[layer_idx]['error'].append(metric['value'])
                
    return dict(layer_metrics)


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute aggregate statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with mean, median, std, min, max, count
    """
    if not values:
        return {'count': 0, 'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
        
    return {
        'count': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values)
    }


def print_stats(name: str, stats: Dict[str, float]) -> None:
    """Print formatted statistics.
    
    Args:
        name: Name of the metric
        stats: Dictionary containing statistics
    """
    print(f"\n{name} Statistics:")
    print(f"  Count:  {stats['count']:>8}")
    print(f"  Mean:   {stats['mean']:>8.6f}")
    print(f"  Median: {stats['median']:>8.6f}")
    print(f"  Std:    {stats['std']:>8.6f}")
    print(f"  Min:    {stats['min']:>8.6f}")
    print(f"  Max:    {stats['max']:>8.6f}")


def print_layer_stats(layer_metrics: Dict[int, Dict[str, List[float]]]) -> None:
    """Print per-layer statistics.
    
    Args:
        layer_metrics: Dictionary mapping layer indices to metrics
    """
    print(f"\n{'='*60}")
    print("PER-LAYER STATISTICS")
    print(f"{'='*60}")
    
    for layer_idx in sorted(layer_metrics.keys()):
        print(f"\nLayer {layer_idx}:")
        
        density_stats = compute_stats(layer_metrics[layer_idx]['density'])
        error_stats = compute_stats(layer_metrics[layer_idx]['error'])
        
        print(f"  Density - Mean: {density_stats['mean']:>8.6f}, "
              f"Std: {density_stats['std']:>8.6f}, "
              f"Count: {density_stats['count']:>4}")
        print(f"  Error   - Mean: {error_stats['mean']:>8.6f}, "
              f"Std: {error_stats['std']:>8.6f}, "
              f"Count: {error_stats['count']:>4}")


def main() -> None:
    """Main function to run the aggregation."""
    parser = argparse.ArgumentParser(description='Aggregate micro metrics from JSONL files')
    parser.add_argument('file_path', type=str, help='Path to the micro_metrics.jsonl file')
    parser.add_argument('--per-layer', action='store_true', 
                       help='Show per-layer statistics in addition to overall stats')
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return
    
    print(f"Loading metrics from: {file_path}")
    metrics = load_metrics(file_path)
    print(f"Loaded {len(metrics)} metrics")
    
    # Overall statistics
    density_values, error_values = separate_metrics(metrics)
    
    print(f"\n{'='*60}")
    print("OVERALL AGGREGATE STATISTICS")
    print(f"{'='*60}")
    
    density_stats = compute_stats(density_values)
    error_stats = compute_stats(error_values)
    
    print_stats("Attention Density", density_stats)
    print_stats("Attention Output Error", error_stats)
    
    # Per-layer statistics
    if args.per_layer:
        layer_metrics = get_layer_metrics(metrics)
        print_layer_stats(layer_metrics)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total metrics processed: {len(metrics)}")
    print(f"Density samples: {len(density_values)}")
    print(f"Error samples: {len(error_values)}")
    if args.per_layer:
        layer_metrics = get_layer_metrics(metrics)
        print(f"Layers analyzed: {len(layer_metrics)}")


if __name__ == "__main__":
    main()
