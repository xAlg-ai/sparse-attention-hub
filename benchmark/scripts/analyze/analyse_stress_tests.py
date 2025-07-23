#!/usr/bin/env python3
"""
Analysis script for stress test results.

This script processes the stress_test_adaptive.matrix directory and generates:
1. vector.tsv - Contains average error and density values for each configuration
2. metadata.tsv - Contains sparse attention configuration settings

Usage:
    python benchmark/scripts/analyse_stress_tests.py
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse


def parse_config_name(config_name: str) -> Dict[str, float]:
    """Parse configuration name to extract parameters.
    
    Args:
        config_name: Configuration name like "adaptive_sampling.sink_0.001_window_0.001_heavy_0.005_base_0.01_epsilon_0.01_delta_0.01"
                     or "oracle_top_k_0.5.sink_0.02_window_0.02"
                     or "oracle_top_p_0.9999.sink_0.001_window_0.001"
        
    Returns:
        Dictionary with parsed parameters
    """
    # Extract parameters using regex for different configuration types
    
    # Pattern for adaptive_sampling
    adaptive_pattern = r"adaptive_sampling\.sink_([\d.]+)_window_([\d.]+)_heavy_([\d.]+)_base_([\d.]+)_epsilon_([\d.]+)_delta_([\d.]+)"
    adaptive_match = re.match(adaptive_pattern, config_name)
    
    if adaptive_match:
        return {
            "config_type": "adaptive_sampling",
            "sink_size": float(adaptive_match.group(1)),
            "window_size": float(adaptive_match.group(2)),
            "heavy_size": float(adaptive_match.group(3)),
            "base_rate_sampling": float(adaptive_match.group(4)),
            "epsilon": float(adaptive_match.group(5)),
            "delta": float(adaptive_match.group(6))
        }
    
    # Pattern for oracle_top_k
    top_k_pattern = r"oracle_top_k_([\d.]+)\.sink_([\d.]+)_window_([\d.]+)"
    top_k_match = re.match(top_k_pattern, config_name)
    
    if top_k_match:
        return {
            "config_type": "oracle_top_k",
            "top_k": float(top_k_match.group(1)),
            "sink_size": float(top_k_match.group(2)),
            "window_size": float(top_k_match.group(3))
        }
    
    # Pattern for oracle_top_p
    top_p_pattern = r"oracle_top_p_([\d.]+)\.sink_([\d.]+)_window_([\d.]+)"
    top_p_match = re.match(top_p_pattern, config_name)
    
    if top_p_match:
        return {
            "config_type": "oracle_top_p",
            "top_p": float(top_p_match.group(1)),
            "sink_size": float(top_p_match.group(2)),
            "window_size": float(top_p_match.group(3))
        }
    
    # If no pattern matches, return empty dict
    return {"config_type": "unknown"}


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def load_micro_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    """Load micro metrics from JSONL file.
    
    Args:
        metrics_path: Path to micro_metrics.jsonl file
        
    Returns:
        List of metric entries
    """
    metrics = []
    with open(metrics_path, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def process_experiment_directory(exp_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process a single experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Tuple of (metrics_data, config_data)
    """
    # Find the benchmark subdirectory (e.g., longbench_passage_retrieval_en)
    benchmark_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if not benchmark_dirs:
        return [], {}
    
    benchmark_dir = benchmark_dirs[0]  # Take the first benchmark directory
    
    # Load configuration
    config_path = benchmark_dir / "config.json"
    if not config_path.exists():
        return [], {}
    
    config = load_config_file(config_path)
    
    # Load micro metrics
    metrics_path = benchmark_dir / "micro_metrics.jsonl"
    if not metrics_path.exists():
        return [], {}
    
    metrics = load_micro_metrics(metrics_path)
    
    return metrics, config


def extract_sparse_config_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sparse attention configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with sparse attention parameters
    """
    sparse_config = config.get("sparse_attention_config", {})
    masker_configs = sparse_config.get("masker_configs", [])
    
    params = {}
    
    # Extract parameters from masker configs
    for masker_config in masker_configs:
        if "sink_size" in masker_config:
            params["sink_size"] = masker_config["sink_size"]
        elif "window_size" in masker_config:
            params["window_size"] = masker_config["window_size"]
        elif "heavy_size" in masker_config:
            params["heavy_size"] = masker_config["heavy_size"]
        elif "base_rate_sampling" in masker_config:
            params["base_rate_sampling"] = masker_config["base_rate_sampling"]
            params["epsilon"] = masker_config.get("epsilon")
            params["delta"] = masker_config.get("delta")
            params["init_offset"] = masker_config.get("init_offset")
            params["local_offset"] = masker_config.get("local_offset")
        elif "top_k" in masker_config:
            params["top_k"] = masker_config["top_k"]
        elif "top_p" in masker_config:
            params["top_p"] = masker_config["top_p"]
    
    return params


def organize_metrics_by_layer(metrics: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Organize metrics by layer index and average multiple measurements.
    
    Args:
        metrics: List of metric entries
        
    Returns:
        Dictionary mapping layer_idx to averaged metrics
    """
    layer_metrics = {}
    
    # First pass: collect all values for each layer
    for metric in metrics:
        layer_idx = metric.get("metadata", {}).get("layer_idx")
        if layer_idx is None:
            continue
            
        if layer_idx not in layer_metrics:
            layer_metrics[layer_idx] = {"density": [], "error": []}
        
        metric_name = metric.get("metric")
        value = metric.get("value")
        
        if metric_name == "research_attention_density":
            layer_metrics[layer_idx]["density"].append(value)
        elif metric_name == "research_attention_output_error":
            layer_metrics[layer_idx]["error"].append(value)
    
    # Second pass: average the collected values
    averaged_metrics = {}
    for layer_idx, values in layer_metrics.items():
        averaged_metrics[layer_idx] = {}
        
        if values["density"]:
            averaged_metrics[layer_idx]["density"] = sum(values["density"]) / len(values["density"])
        
        if values["error"]:
            averaged_metrics[layer_idx]["error"] = sum(values["error"]) / len(values["error"])
    
    return averaged_metrics


def analyze_stress_tests(results_dir: str, output_dir: str) -> None:
    """Analyze stress test results and generate TSV files.
    
    Args:
        results_dir: Path to stress_test_adaptive.matrix directory
        output_dir: Output directory for TSV files
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find model directories
    model_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    all_vector_data = []
    all_metadata = []
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        # Find configuration directories
        config_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        
        for config_dir in config_dirs:
            config_name = config_dir.name
            
            # Parse configuration name
            parsed_params = parse_config_name(config_name)
            
            # Process experiment directory
            metrics, config = process_experiment_directory(config_dir)
            
            if not metrics or not config:
                continue
            
            # Extract sparse attention parameters
            sparse_params = extract_sparse_config_params(config)
            
            # Organize metrics by layer
            layer_metrics = organize_metrics_by_layer(metrics)
            
            # Generate vector data
            for layer_idx, layer_data in layer_metrics.items():
                if "density" in layer_data and "error" in layer_data:
                    vector_entry = {
                        "model": model_name,
                        "config": config_name,
                        "layer_idx": layer_idx,
                        "density": layer_data["density"],
                        "error": layer_data["error"]
                    }
                    all_vector_data.append(vector_entry)
            
            # Generate metadata entry
            metadata_entry = {
                "model": model_name,
                "config": config_name,
                "layer_idx": "all",  # This will be expanded for each layer
                **parsed_params,
                **sparse_params
            }
            
            # Add metadata for each layer
            for layer_idx in layer_metrics.keys():
                layer_metadata = metadata_entry.copy()
                layer_metadata["layer_idx"] = layer_idx
                all_metadata.append(layer_metadata)
    
    # Write vector.tsv
    vector_path = output_path / "vector.tsv"
    metadata_path = output_path / "metadata.tsv"
    
    # Write vector data
    with open(vector_path, 'w') as f:
        f.write("density\terror\n")
        for entry in all_vector_data:
            f.write(f"{entry['density']}\t{entry['error']}\n")
    
    # Write metadata
    with open(metadata_path, 'w') as f:
        if all_metadata:
            # Get all unique keys from all metadata entries
            all_keys = set()
            for entry in all_metadata:
                all_keys.update(entry.keys())
            
            # Sort keys for consistent output
            sorted_keys = sorted(all_keys)
            
            # Write header
            f.write("\t".join(sorted_keys) + "\n")
            
            # Write data
            for entry in all_metadata:
                row = [str(entry.get(key, "")) for key in sorted_keys]
                f.write("\t".join(row) + "\n")


    
    print(f"Analysis complete!")
    print(f"Vector data written to: {vector_path}")
    print(f"Metadata written to: {metadata_path}")
    print(f"Total vector entries: {len(all_vector_data)}")
    print(f"Total metadata entries: {len(all_metadata)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze stress test results")
    parser.add_argument(
        "--results-dir", 
        default="./stress_test_adaptive.matrix",
        help="Path to stress test results directory"
    )
    parser.add_argument(
        "--output-dir", 
        default="./analysis_output",
        help="Output directory for TSV files"
    )
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist")
        return
    
    analyze_stress_tests(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main() 