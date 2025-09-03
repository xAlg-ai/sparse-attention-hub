#!/usr/bin/env python3
"""
Script to generate a results table from benchmark data.

This script processes benchmark results to create a comprehensive table showing
performance metrics across different sparse attention configurations.

Usage:
    python generate_results_table.py <dataset_name> <base_folder>
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from collections import defaultdict


def find_dataset_files(base_folder: str, dataset_name: str) -> List[Path]:
    """
    Find all files corresponding to the given dataset in the base folder.

    Args:
        base_folder: Path to the base folder containing benchmark results
        dataset_name: Name of the dataset to search for

    Returns:
        List of paths to dataset directories
    """
    base_path = Path(base_folder)
    dataset_paths = []

    # Walk through the directory structure
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        for config_dir in model_dir.iterdir():
            if not config_dir.is_dir():
                continue

            dataset_dir = config_dir / dataset_name
            if dataset_dir.exists():
                dataset_paths.append(dataset_dir)

    return dataset_paths


def extract_config_name(dataset_path: Path) -> str:
    """
    Extract the configuration name from the dataset path.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Configuration name
    """
    # The config name is the parent directory of the dataset
    return dataset_path.parent.name


def parse_metrics_json(metrics_file: Path) -> Dict[str, Any]:
    """
    Parse the metrics.json file to extract macro metrics.

    Args:
        metrics_file: Path to the metrics.json file

    Returns:
        Dictionary containing the parsed metrics
    """
    try:
        with open(metrics_file, "r") as f:
            data = json.load(f)

        metrics = {}

        # Extract overall score
        if "overall_score" in data:
            metrics["overall_score"] = data["overall_score"]

        # Extract task-specific scores
        if "task_scores" in data:
            for task_name, task_metrics in data["task_scores"].items():
                for metric_name, metric_value in task_metrics.items():
                    # Create a unique metric name if there are multiple tasks
                    if len(data["task_scores"]) > 1:
                        metric_key = f"{task_name}_{metric_name}"
                    else:
                        metric_key = metric_name
                    metrics[metric_key] = metric_value

        return metrics

    except Exception as e:
        print(f"Warning: Could not parse {metrics_file}: {e}")
        return {}


def parse_micro_metrics_jsonl(
    micro_metrics_file: Path, max_lines: int = 1000
) -> Dict[str, float]:
    """
    Parse the micro_metrics.jsonl file to extract average density and attention error.

    Args:
        micro_metrics_file: Path to the micro_metrics.jsonl file
        max_lines: Maximum number of lines to read

    Returns:
        Dictionary containing average density and attention error
    """
    try:
        density_values = []
        error_values = []
        line_count = 0

        with open(micro_metrics_file, "r") as f:
            for line in f:
                if line_count >= max_lines:
                    break

                try:
                    data = json.loads(line.strip())
                    metric_name = data.get("metric", "")
                    value = data.get("value", 0.0)

                    if metric_name == "research_attention_density":
                        density_values.append(value)
                    elif metric_name == "research_attention_output_error":
                        error_values.append(value)

                except json.JSONDecodeError:
                    continue

                line_count += 1

        micro_metrics = {}

        if density_values:
            micro_metrics["avg_density"] = sum(density_values) / len(density_values)

        if error_values:
            micro_metrics["avg_attention_error"] = sum(error_values) / len(error_values)

        return micro_metrics

    except Exception as e:
        print(f"Warning: Could not parse {micro_metrics_file}: {e}")
        return {}


def get_all_available_metrics(dataset_paths: List[Path]) -> List[str]:
    """
    Get all available metrics across all dataset paths.

    Args:
        dataset_paths: List of paths to dataset directories

    Returns:
        List of all available metric names
    """
    all_metrics = set()

    for dataset_path in dataset_paths:
        metrics_file = dataset_path / "metrics.json"
        if metrics_file.exists():
            metrics = parse_metrics_json(metrics_file)
            all_metrics.update(metrics.keys())

        micro_metrics_file = dataset_path / "micro_metrics.jsonl"
        if micro_metrics_file.exists():
            micro_metrics = parse_micro_metrics_jsonl(micro_metrics_file)
            all_metrics.update(micro_metrics.keys())

    return sorted(list(all_metrics))


def generate_results_table(dataset_name: str, base_folder: str) -> pd.DataFrame:
    """
    Generate a results table from benchmark data.

    Args:
        dataset_name: Name of the dataset to analyze
        base_folder: Path to the base folder containing benchmark results

    Returns:
        DataFrame with configs as rows and metrics as columns
    """
    print(f"🔍 Finding dataset files for '{dataset_name}' in '{base_folder}'...")
    dataset_paths = find_dataset_files(base_folder, dataset_name)

    if not dataset_paths:
        print(f"❌ No dataset files found for '{dataset_name}' in '{base_folder}'")
        return pd.DataFrame()

    print(f"✅ Found {len(dataset_paths)} dataset directories")

    # Get all available metrics
    print("📊 Discovering available metrics...")
    all_metrics = get_all_available_metrics(dataset_paths)
    print(f"✅ Found {len(all_metrics)} metrics: {', '.join(all_metrics)}")

    # Initialize results dictionary
    results = []

    print("📈 Processing each configuration...")
    for dataset_path in dataset_paths:
        config_name = extract_config_name(dataset_path)
        print(f"  Processing: {config_name}")

        row_data = {"config": config_name}

        # Parse macro metrics
        metrics_file = dataset_path / "metrics.json"
        if metrics_file.exists():
            macro_metrics = parse_metrics_json(metrics_file)
            row_data.update(macro_metrics)

        # Parse micro metrics
        micro_metrics_file = dataset_path / "micro_metrics.jsonl"
        if micro_metrics_file.exists():
            micro_metrics = parse_micro_metrics_jsonl(micro_metrics_file)
            row_data.update(micro_metrics)

        results.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(results)

    if df.empty:
        print("❌ No results found")
        return df

    # Reorder columns to put config first, then metrics
    if "config" in df.columns:
        other_cols = [col for col in df.columns if col != "config"]
        df = df[["config"] + other_cols]

    return df


def save_results_table(df: pd.DataFrame, dataset_name: str, base_folder: str) -> None:
    """
    Save the results table to files.

    Args:
        df: DataFrame containing the results
        dataset_name: Name of the dataset
        base_folder: Base folder path for output
    """
    if df.empty:
        print("❌ No data to save")
        return

    # Create output directory
    output_dir = Path(base_folder) / "results_tables"
    output_dir.mkdir(exist_ok=True)

    # Save as CSV
    csv_file = output_dir / f"{dataset_name}_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"💾 Saved CSV results to: {csv_file}")

    # Save as Excel
    excel_file = output_dir / f"{dataset_name}_results.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"💾 Saved Excel results to: {excel_file}")

    # Print summary
    print(f"\n📊 Results Summary:")
    print(f"  - Configurations: {len(df)}")
    print(f"  - Metrics: {len(df.columns) - 1}")  # Exclude config column
    print(f"  - Output files: {csv_file}, {excel_file}")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Generate a results table from benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_results_table.py loogle_shortdep_cloze full_benchmark.matrix
  python generate_results_table.py longbench_passage_retrieval_en ./results/
        """,
    )

    parser.add_argument(
        "dataset_name",
        help="Name of the dataset to analyze (e.g., loogle_shortdep_cloze)",
    )

    parser.add_argument(
        "base_folder", help="Path to the base folder containing benchmark results"
    )

    args = parser.parse_args()

    print("🚀 Starting Results Table Generation")
    print("=" * 50)
    print(f"Dataset: {args.dataset_name}")
    print(f"Base folder: {args.base_folder}")
    print()

    # Generate the results table
    df = generate_results_table(args.dataset_name, args.base_folder)

    if not df.empty:
        # Display the table
        print("\n📋 Results Table:")
        print("=" * 80)
        print(df.to_string(index=False))

        # Save the results
        save_results_table(df, args.dataset_name, args.base_folder)

        print(f"\n✅ Results table generation completed successfully!")
    else:
        print(f"\n❌ No results found for dataset '{args.dataset_name}'")


if __name__ == "__main__":
    main()
