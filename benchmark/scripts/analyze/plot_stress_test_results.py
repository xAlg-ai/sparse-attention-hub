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
    vector_df = pd.read_csv(vector_path, sep='\t')
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    
    # Merge the dataframes
    # Since both files have the same number of rows in the same order,
    # we can simply concatenate them
    merged_df = pd.concat([vector_df, metadata_df], axis=1)
    
    return merged_df


def create_interactive_scatter_plot(df: pd.DataFrame, output_path: str) -> None:
    """Create an interactive scatter plot of error vs density.
    
    Args:
        df: DataFrame containing the data
        output_path: Path to save the HTML plot
    """
    # Create hover text with configuration details
    hover_text = []
    for _, row in df.iterrows():
        config_info = f"""
        <b>Model:</b> {row.get('model', 'N/A')}<br>
        <b>Config Type:</b> {row.get('config_type', 'N/A')}<br>
        <b>Config:</b> {row.get('config', 'N/A')}<br>
        <b>Layer:</b> {row.get('layer_idx', 'N/A')}<br>
        <b>Sink Size:</b> {row.get('sink_size', 'N/A')}<br>
        <b>Window Size:</b> {row.get('window_size', 'N/A')}<br>
        """
        
        # Add configuration-specific parameters
        config_type = row.get('config_type', '')
        if config_type == 'adaptive_sampling':
            config_info += f"""
        <b>Heavy Size:</b> {row.get('heavy_size', 'N/A')}<br>
        <b>Base Rate:</b> {row.get('base_rate_sampling', 'N/A')}<br>
        <b>Epsilon:</b> {row.get('epsilon', 'N/A')}<br>
        <b>Delta:</b> {row.get('delta', 'N/A')}<br>
        """
        elif config_type == 'oracle_top_k':
            config_info += f"<b>Top-K:</b> {row.get('top_k', 'N/A')}<br>"
        elif config_type == 'oracle_top_p':
            config_info += f"<b>Top-P:</b> {row.get('top_p', 'N/A')}<br>"
        
        config_info += f"""
        <b>Density:</b> {row.get('density', 'N/A'):.4f}<br>
        <b>Error:</b> {row.get('error', 'N/A'):.4f}
        """
        hover_text.append(config_info)
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Define colors for different configuration types
    config_colors = {
        'adaptive_sampling': '#1f77b4',  # Blue
        'oracle_top_k': '#ff7f0e',       # Orange
        'oracle_top_p': '#2ca02c'        # Green
    }
    
    # Add scatter traces for each configuration type
    for config_type in ['adaptive_sampling', 'oracle_top_k', 'oracle_top_p']:
        config_data = df[df['config_type'] == config_type]
        
        if len(config_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=config_data['density'],
                    y=config_data['error'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=config_colors[config_type],
                        opacity=0.7
                    ),
                    text=[hover_text[i] for i in config_data.index],
                    hoverinfo='text',
                    hovertemplate='%{text}<extra></extra>',
                    name=config_type.replace('_', ' ').title()
                )
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Sparse Attention: Error vs Density (All Configurations)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Density',
        yaxis_title='Error',
        xaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            zeroline=False
        ),
        plot_bgcolor='white',
        hovermode='closest',
        width=1000,
        height=700,
        showlegend=True
    )
    
    # Save the plot
    fig.write_html(output_path)
    print(f"Interactive plot saved to: {output_path}")


def create_configuration_analysis_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Create additional analysis plots for different configurations.
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Plot by configuration type with subplots
    config_types = df['config_type'].unique()
    
    fig = make_subplots(
        rows=1, cols=len(config_types),
        subplot_titles=[f"{ct.replace('_', ' ').title()}" for ct in config_types],
        specs=[[{"secondary_y": False}] * len(config_types)]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, config_type in enumerate(config_types):
        config_data = df[df['config_type'] == config_type]
        
        fig.add_trace(
            go.Scatter(
                x=config_data['density'],
                y=config_data['error'],
                mode='markers',
                marker=dict(size=8, color=colors[i], opacity=0.7),
                name=config_type.replace('_', ' ').title(),
                text=[f"Layer: {layer}<br>Density: {d:.4f}<br>Error: {e:.4f}" 
                      for layer, d, e in zip(config_data['layer_idx'], config_data['density'], config_data['error'])],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title='Error vs Density by Configuration Type',
        showlegend=False,
        width=1200,
        height=500
    )
    
    fig.write_html(output_path / "config_type_analysis.html")
    print(f"Configuration type analysis plot saved to: {output_path / 'config_type_analysis.html'}")
    
    # 2. Layer-wise analysis with configuration type colors
    layer_groups = df.groupby('layer_idx')
    
    fig2 = go.Figure()
    
    # Define colors for different configuration types
    config_colors = {
        'adaptive_sampling': '#1f77b4',  # Blue
        'oracle_top_k': '#ff7f0e',       # Orange
        'oracle_top_p': '#2ca02c'        # Green
    }
    
    for layer_idx, group in layer_groups:
        # Color by configuration type
        colors_for_layer = [config_colors.get(ct, '#000000') for ct in group['config_type']]
        
        fig2.add_trace(
            go.Scatter(
                x=group['density'],
                y=group['error'],
                mode='markers',
                marker=dict(size=6, color=colors_for_layer, opacity=0.7),
                name=f'Layer {layer_idx}',
                text=[f"Config: {config}<br>Type: {ct}<br>Density: {d:.4f}<br>Error: {e:.4f}" 
                      for config, ct, d, e in zip(group['config'], group['config_type'], group['density'], group['error'])],
                hoverinfo='text'
            )
        )
    
    fig2.update_layout(
        title='Error vs Density by Layer (Colored by Config Type)',
        xaxis_title='Density',
        yaxis_title='Error',
        width=1000,
        height=700
    )
    
    fig2.write_html(output_path / "layer_analysis.html")
    print(f"Layer analysis plot saved to: {output_path / 'layer_analysis.html'}")
    
    # 3. Configuration comparison plot
    fig3 = go.Figure()
    
    for config_type in ['adaptive_sampling', 'oracle_top_k', 'oracle_top_p']:
        config_data = df[df['config_type'] == config_type]
        
        if len(config_data) > 0:
            # Calculate average error and density for each configuration
            config_avg = config_data.groupby('config').agg({
                'error': 'mean',
                'density': 'mean'
            }).reset_index()
            
            fig3.add_trace(
                go.Scatter(
                    x=config_avg['density'],
                    y=config_avg['error'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=config_colors[config_type],
                        opacity=0.8
                    ),
                    name=config_type.replace('_', ' ').title(),
                    text=[f"Config: {config}<br>Avg Density: {d:.4f}<br>Avg Error: {e:.4f}" 
                          for config, d, e in zip(config_avg['config'], config_avg['density'], config_avg['error'])],
                    hoverinfo='text'
                )
            )
    
    fig3.update_layout(
        title='Average Error vs Density by Configuration',
        xaxis_title='Average Density',
        yaxis_title='Average Error',
        width=1000,
        height=700,
        showlegend=True
    )
    
    fig3.write_html(output_path / "config_comparison.html")
    print(f"Configuration comparison plot saved to: {output_path / 'config_comparison.html'}")


def main():
    """Main function to create interactive plots."""
    parser = argparse.ArgumentParser(description='Create interactive plots for stress test results')
    parser.add_argument('--vector-file', default='analysis_output/vector.tsv',
                       help='Path to vector.tsv file')
    parser.add_argument('--metadata-file', default='analysis_output/metadata.tsv',
                       help='Path to metadata.tsv file')
    parser.add_argument('--output-dir', default='analysis_output/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = load_tsv_data(args.vector_file, args.metadata_file)
    print(f"Loaded {len(df)} data points")
    
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


if __name__ == "__main__":
    main() 