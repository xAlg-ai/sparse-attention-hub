#!/usr/bin/env python3
"""
Interactive scatter plot for stress test results.

This script creates an interactive scatter plot showing error vs density
with hover information displaying configuration details for each point.

Usage:
    python benchmark/scripts/plot_stress_test_results.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path
from typing import Optional


def load_tsv_data(vector_path: str, metadata_path: str) -> pd.DataFrame:
    """Load and merge vector and metadata TSV files.

    Args:
        vector_path: Path to vector.tsv file
        metadata_path: Path to metadata.tsv file

    Returns:
        Merged DataFrame with all data
    """
    # Load the data
    vector_df = pd.read_csv(vector_path, sep="\t")
    metadata_df = pd.read_csv(metadata_path, sep="\t")

    # Merge the dataframes
    # Since both files have the same number of rows in the same order,
    # we can simply concatenate them
    merged_df = pd.concat([vector_df, metadata_df], axis=1)

    return merged_df


def create_interactive_scatter_plot(
    df: pd.DataFrame, output_path: str, dataset_name: str = ""
) -> None:
    """Create an interactive scatter plot of error vs density.

    Args:
        df: DataFrame containing the data
        output_path: Path to save the HTML plot
        dataset_name: Name of the dataset for the plot title
    """
    # Create the scatter plot
    fig = go.Figure()

    # Define colors for different configuration types
    config_colors = {
        "adaptive_sampling": "#1f77b4",  # Blue
        "oracle_top_k": "#ff7f0e",  # Orange
        "oracle_top_p": "#2ca02c",  # Green
        "hashattention": "#d62728",  # Red
        "adaptive_sampling_hat": "#9467bd",  # Purple
        "random_sampling": "#8c564b",  # Brown
    }

    # Add scatter traces for each configuration type
    for config_type in [
        "adaptive_sampling",
        "oracle_top_k",
        "oracle_top_p",
        "hashattention",
        "adaptive_sampling_hat",
        "random_sampling",
    ]:
        config_data = df[df["config_type"] == config_type]

        if len(config_data) > 0:
            # Create hover text for this configuration type
            hover_text = []
            for _, row in config_data.iterrows():
                config_info = f"""
                <b>Model:</b> {row.get('model', 'N/A')}<br>
                <b>Config Type:</b> {row.get('config_type', 'N/A')}<br>
                <b>Config:</b> {row.get('config', 'N/A')}<br>
                <b>Layer:</b> {row.get('layer_idx', 'N/A')}<br>
                <b>Sink Size:</b> {row.get('sink_size', 'N/A')}<br>
                <b>Window Size:</b> {row.get('window_size', 'N/A')}<br>
                """

                # Add configuration-specific parameters
                if config_type == "adaptive_sampling":
                    config_info += f"""
                <b>Heavy Size:</b> {row.get('heavy_size', 'N/A')}<br>
                <b>Base Rate:</b> {row.get('base_rate_sampling', 'N/A')}<br>
                <b>Epsilon:</b> {row.get('epsilon', 'N/A')}<br>
                <b>Delta:</b> {row.get('delta', 'N/A')}<br>
                """
                elif config_type == "oracle_top_k":
                    config_info += f"<b>Top-K:</b> {row.get('top_k', 'N/A')}<br>"
                elif config_type == "oracle_top_p":
                    config_info += f"<b>Top-P:</b> {row.get('top_p', 'N/A')}<br>"
                elif config_type == "hashattention":
                    config_info += f"""
                <b>Hash Top-K:</b> {row.get('hat_top_k', 'N/A')}<br>
                <b>Hash Heavy Size:</b> {row.get('hat_heavy_size', 'N/A')}<br>
                <b>Hash Bits:</b> {row.get('hat_bits', 'N/A')}<br>
                <b>Hash MLP Layers:</b> {row.get('hat_mlp_layers', 'N/A')}<br>
                <b>Hash MLP Hidden Size:</b> {row.get('hat_mlp_hidden_size', 'N/A')}<br>
                <b>Hash MLP Activation:</b> {row.get('hat_mlp_activation', 'N/A')}<br>
                """
                elif config_type == "adaptive_sampling_hat":
                    config_info += f"""
                <b>Heavy Size:</b> {row.get('heavy_size', 'N/A')}<br>
                <b>Base Rate:</b> {row.get('base_rate_sampling', 'N/A')}<br>
                <b>Epsilon:</b> {row.get('epsilon', 'N/A')}<br>
                <b>Delta:</b> {row.get('delta', 'N/A')}<br>
                <b>Hash Heavy Size:</b> {row.get('hat_heavy_size', 'N/A')}<br>
                <b>Hash Bits:</b> {row.get('hat_bits', 'N/A')}<br>
                <b>Hash MLP Layers:</b> {row.get('hat_mlp_layers', 'N/A')}<br>
                <b>Hash MLP Hidden Size:</b> {row.get('hat_mlp_hidden_size', 'N/A')}<br>
                <b>Hash MLP Activation:</b> {row.get('hat_mlp_activation', 'N/A')}<br>
                """
                elif config_type == "random_sampling":
                    config_info += (
                        f"<b>Sampling Rate:</b> {row.get('sampling_rate', 'N/A')}<br>"
                    )

                config_info += f"""
                <b>Density:</b> {row.get('density', 'N/A'):.4f}<br>
                <b>Error:</b> {row.get('error', 'N/A'):.4f}
                """
                hover_text.append(config_info)

            fig.add_trace(
                go.Scatter(
                    x=config_data["density"],
                    y=config_data["error"],
                    mode="markers",
                    marker=dict(size=6, color=config_colors[config_type], opacity=0.7),
                    text=hover_text,
                    hoverinfo="text",
                    hovertemplate="%{text}<extra></extra>",
                    name=config_type.replace("_", " ").title(),
                )
            )

    # Create title with dataset name if provided
    title_text = "Sparse Attention: Error vs Density (All Configurations)"
    if dataset_name:
        title_text = f"Sparse Attention: Error vs Density - {dataset_name}"

    # Update layout
    fig.update_layout(
        title={"text": title_text, "x": 0.5, "xanchor": "center", "font": {"size": 20}},
        xaxis_title="Density",
        yaxis_title="Error",
        xaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=12),
            gridcolor="lightgray",
            zeroline=False,
        ),
        yaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=12),
            gridcolor="lightgray",
            zeroline=False,
        ),
        plot_bgcolor="white",
        hovermode="closest",
        width=1000,
        height=700,
        showlegend=True,
    )

    # Save the plot
    fig.write_html(output_path)
    print(f"Interactive plot saved to: {output_path}")


def create_configuration_analysis_plots(
    df: pd.DataFrame, output_dir: str, dataset_name: str = ""
) -> None:
    """Create additional analysis plots for different configurations.

    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the plots
        dataset_name: Name of the dataset for the plot titles
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. Plot by configuration type with subplots
    config_types = df["config_type"].unique()

    fig = make_subplots(
        rows=1,
        cols=len(config_types),
        subplot_titles=[f"{ct.replace('_', ' ').title()}" for ct in config_types],
        specs=[[{"secondary_y": False}] * len(config_types)],
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]  # Blue, Orange, Green, Red, Purple, Brown

    for i, config_type in enumerate(config_types):
        config_data = df[df["config_type"] == config_type]

        fig.add_trace(
            go.Scatter(
                x=config_data["density"],
                y=config_data["error"],
                mode="markers",
                marker=dict(size=8, color=colors[i], opacity=0.7),
                name=config_type.replace("_", " ").title(),
                text=[
                    f"Layer: {layer}<br>Density: {d:.4f}<br>Error: {e:.4f}"
                    for layer, d, e in zip(
                        config_data["layer_idx"],
                        config_data["density"],
                        config_data["error"],
                    )
                ],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )

    # Create title with dataset name if provided
    title_text = "Error vs Density by Configuration Type"
    if dataset_name:
        title_text = f"Error vs Density by Configuration Type - {dataset_name}"

    fig.update_layout(title=title_text, showlegend=False, width=1200, height=500)

    fig.write_html(output_path / "config_type_analysis.html")
    print(
        f"Configuration type analysis plot saved to: {output_path / 'config_type_analysis.html'}"
    )

    # 2. Layer-wise analysis with configuration type colors
    layer_groups = df.groupby("layer_idx")

    fig2 = go.Figure()

    # Define colors for different configuration types
    config_colors = {
        "adaptive_sampling": "#1f77b4",  # Blue
        "oracle_top_k": "#ff7f0e",  # Orange
        "oracle_top_p": "#2ca02c",  # Green
        "hashattention": "#d62728",  # Red
        "adaptive_sampling_hat": "#9467bd",  # Purple
        "random_sampling": "#8c564b",  # Brown
    }

    for layer_idx, group in layer_groups:
        # Color by configuration type
        colors_for_layer = [
            config_colors.get(ct, "#000000") for ct in group["config_type"]
        ]

        fig2.add_trace(
            go.Scatter(
                x=group["density"],
                y=group["error"],
                mode="markers",
                marker=dict(size=6, color=colors_for_layer, opacity=0.7),
                name=f"Layer {layer_idx}",
                text=[
                    f"Config: {config}<br>Type: {ct}<br>Density: {d:.4f}<br>Error: {e:.4f}"
                    for config, ct, d, e in zip(
                        group["config"],
                        group["config_type"],
                        group["density"],
                        group["error"],
                    )
                ],
                hoverinfo="text",
            )
        )

    # Create title with dataset name if provided
    title_text2 = "Error vs Density by Layer (Colored by Config Type)"
    if dataset_name:
        title_text2 = (
            f"Error vs Density by Layer (Colored by Config Type) - {dataset_name}"
        )

    fig2.update_layout(
        title=title_text2,
        xaxis_title="Density",
        yaxis_title="Error",
        width=1000,
        height=700,
    )

    fig2.write_html(output_path / "layer_analysis.html")
    print(f"Layer analysis plot saved to: {output_path / 'layer_analysis.html'}")

    # 3. Configuration comparison plot
    fig3 = go.Figure()

    for config_type in [
        "adaptive_sampling",
        "oracle_top_k",
        "oracle_top_p",
        "hashattention",
        "adaptive_sampling_hat",
        "random_sampling",
    ]:
        config_data = df[df["config_type"] == config_type]

        if len(config_data) > 0:
            # Calculate average error and density for each configuration
            config_avg = (
                config_data.groupby("config")
                .agg({"error": "mean", "density": "mean"})
                .reset_index()
            )

            fig3.add_trace(
                go.Scatter(
                    x=config_avg["density"],
                    y=config_avg["error"],
                    mode="markers",
                    marker=dict(size=10, color=config_colors[config_type], opacity=0.8),
                    name=config_type.replace("_", " ").title(),
                    text=[
                        f"Config: {config}<br>Avg Density: {d:.4f}<br>Avg Error: {e:.4f}"
                        for config, d, e in zip(
                            config_avg["config"],
                            config_avg["density"],
                            config_avg["error"],
                        )
                    ],
                    hoverinfo="text",
                )
            )

    # Create title with dataset name if provided
    title_text3 = "Average Error vs Density by Configuration"
    if dataset_name:
        title_text3 = f"Average Error vs Density by Configuration - {dataset_name}"

    fig3.update_layout(
        title=title_text3,
        xaxis_title="Average Density",
        yaxis_title="Average Error",
        width=1000,
        height=700,
        showlegend=True,
    )

    fig3.write_html(output_path / "config_comparison.html")
    print(
        f"Configuration comparison plot saved to: {output_path / 'config_comparison.html'}"
    )


def main():
    """Main function to create interactive plots per dataset."""
    parser = argparse.ArgumentParser(
        description="Create interactive plots for stress test results"
    )
    parser.add_argument(
        "--vector-file",
        default="./sparse-attention-hub-share/docs/micro_tests/vector.tsv",
        help="Path to vector.tsv file",
    )
    parser.add_argument(
        "--metadata-file",
        default="./sparse-attention-hub-share/docs/micro_tests/metadata.tsv",
        help="Path to metadata.tsv file",
    )
    parser.add_argument(
        "--output-dir",
        default="./sparse-attention-hub-share/docs/micro_tests/plots/",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df = load_tsv_data(args.vector_file, args.metadata_file)
    print(f"Loaded {len(df)} data points")

    # Check if dataset column exists
    if "dataset" not in df.columns:
        print(
            "Warning: No 'dataset' column found in metadata. Creating plots for all data combined."
        )
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)

        # Create main interactive scatter plot
        print("Creating main interactive scatter plot...")
        create_interactive_scatter_plot(df, output_path / "error_vs_density.html")

        # Create additional analysis plots
        print("Creating additional analysis plots...")
        create_configuration_analysis_plots(df, args.output_dir)

        print("All plots created successfully!")
        print(f"Main plot: {output_path / 'error_vs_density.html'}")
        print(f"Additional plots: {output_path / 'config_analysis.html'}")
        print(f"Layer analysis: {output_path / 'layer_analysis.html'}")
        return
    # Group data by dataset
    datasets = df["dataset"].unique()
    # Filter out NaN values
    datasets = [d for d in datasets if pd.notna(d)]
    print(f"Found {len(datasets)} datasets: {datasets}")

    # Create plots for each dataset
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")

        # Filter data for this dataset
        dataset_df = df[df["dataset"] == dataset_name]
        print(f"  Found {len(dataset_df)} data points for this dataset")

        # Create dataset-specific output directory
        dataset_output_dir = Path(args.output_dir) / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Create main interactive scatter plot for this dataset
        print(f"  Creating main interactive scatter plot...")
        create_interactive_scatter_plot(
            dataset_df, dataset_output_dir / "error_vs_density.html", dataset_name
        )

        # Create additional analysis plots for this dataset
        print(f"  Creating additional analysis plots...")
        create_configuration_analysis_plots(
            dataset_df, str(dataset_output_dir), dataset_name
        )

        print(f"  Plots for {dataset_name} saved to: {dataset_output_dir}")

    print(f"\nAll plots created successfully!")
    print(f"Plots organized by dataset in: {args.output_dir}/")
    for dataset_name in datasets:
        print(f"  - {dataset_name}: {Path(args.output_dir) / dataset_name}/")


if __name__ == "__main__":
    main()
