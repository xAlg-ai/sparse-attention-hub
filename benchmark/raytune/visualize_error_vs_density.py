#!/usr/bin/env python3
"""
Interactive HTML visualization for error vs density across benchmarks and configurations.

This script creates an interactive Plotly dashboard to visualize the relationship
between error and density metrics across different models, benchmarks, and attention
configurations from Ray Tune optimization results.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_experiment_name(experiment_dir: str) -> Tuple[str, str, str, str]:
    """
    Parse experiment directory name to extract model, benchmark, task, and config.
    
    Args:
        experiment_dir: Directory name like 'meta-llama_Llama-3.1-8B-Instruct_loogle_shortdep_qa_sink_local_random_sampling'
    
    Returns:
        Tuple of (model, benchmark, task, config_type)
    """
    parts = experiment_dir.split('_')
    
    # Handle model name with underscores
    if parts[0] == 'meta-llama':
        model = f"{parts[0]}/{parts[1]}"
        remaining = parts[2:]
    else:
        model = parts[0]
        remaining = parts[1:]
    
    # Extract benchmark
    benchmark = remaining[0] if len(remaining) > 0 else "unknown"
    
    # The task is everything between benchmark and sink_local
    # Find where 'sink_local' starts
    sink_idx = -1
    for i in range(1, len(remaining)):
        if remaining[i] == 'sink' and i+1 < len(remaining) and remaining[i+1] == 'local':
            sink_idx = i
            break
    
    if sink_idx > 1:
        # Task is everything between benchmark and sink_local
        task = '_'.join(remaining[1:sink_idx])
        # Config type is everything from sink_local onwards
        config_type = '_'.join(remaining[sink_idx:])
    else:
        # Fallback parsing
        task = remaining[1] if len(remaining) > 1 else "unknown"
        config_type = '_'.join(remaining[2:]) if len(remaining) > 2 else "unknown"
    
    return model, benchmark, task, config_type


def extract_config_params(config: Dict) -> str:
    """
    Extract and format configuration parameters for display.
    
    Args:
        config: Configuration dictionary from result.json
    
    Returns:
        Formatted string of configuration parameters
    """
    params = []
    for key, value in sorted(config.items()):
        # Shorten parameter names for display
        short_key = key.replace('masker_', '').replace('_size', '').replace('_rate', '')
        if isinstance(value, float):
            params.append(f"{short_key}={value:.3f}")
        else:
            params.append(f"{short_key}={value}")
    return ", ".join(params)


def collect_results(ray_results_dir: Path) -> pd.DataFrame:
    """
    Collect all results from ray_results directory.
    
    Args:
        ray_results_dir: Path to ray_results directory
    
    Returns:
        DataFrame with columns: model, benchmark, task, config_type, density, error, config_params, trial_id
    """
    results = []
    
    for experiment_dir in ray_results_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        # Parse experiment name
        model, benchmark, task, config_type = parse_experiment_name(experiment_dir.name)
        
        # Process each trial in the experiment
        for trial_dir in experiment_dir.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith('objective_'):
                continue
                
            result_file = trial_dir / 'result.json'
            if not result_file.exists():
                continue
                
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Extract metrics
                density = data.get('density', None)
                error = data.get('error', None)
                
                if density is None or error is None:
                    continue
                
                # Extract trial ID
                trial_id = data.get('trial_id', trial_dir.name.split('_')[1])
                
                # Format config parameters
                config_params = extract_config_params(data.get('config', {}))
                
                results.append({
                    'model': model,
                    'benchmark': benchmark,
                    'task': task,
                    'config_type': config_type,
                    'density': density,
                    'error': error,
                    'config_params': config_params,
                    'trial_id': trial_id,
                    'combined_score': data.get('combined_score', None)
                })
                
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                continue
    
    return pd.DataFrame(results)


def create_interactive_dashboard(df: pd.DataFrame, output_file: str = "error_vs_density_dashboard.html", output_dir: Path = None):
    """
    Create an interactive Plotly dashboard for error vs density visualization.
    
    Args:
        df: DataFrame with results
        output_file: Output HTML file name
        output_dir: Output directory for additional files
    """
    # Get unique tasks
    tasks = sorted(df['task'].unique())
    n_tasks = len(tasks)
    
    # Force 2x2 layout for better presentation
    n_cols = 2
    n_rows = 2
    
    # Create subplot titles with better formatting
    subplot_titles = [f"{task.replace('_', ' ').title()}" for task in tasks]
    
    # Create the main figure with subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    # Define dark color palette for config types
    config_types = sorted(df['config_type'].unique())
    # Using dark, vibrant colors for better visibility
    dark_colors = [
        '#1f77b4',  # dark blue
        '#ff7f0e',  # dark orange
        '#2ca02c',  # dark green
        '#d62728',  # dark red
        '#9467bd',  # dark purple
        '#8c564b',  # dark brown
        '#e377c2',  # dark pink
        '#7f7f7f',  # dark gray
        '#bcbd22',  # dark olive
        '#17becf',  # dark cyan
        '#393b79',  # midnight blue
        '#637939',  # dark olive green
        '#8c6d31',  # dark tan
        '#843c39',  # dark maroon
        '#7b4173',  # dark magenta
        '#5254a3',  # dark indigo
        '#6b6ecf',  # dark lavender
        '#9c9ede',  # dark periwinkle
        '#bd9e39',  # dark gold
        '#ad494a',  # dark coral
        '#a55194',  # dark orchid
    ]
    color_map = {config: dark_colors[i % len(dark_colors)] for i, config in enumerate(config_types)}
    
    # Define marker symbols for better distinction
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
               'triangle-left', 'triangle-right', 'pentagon', 'hexagon', 'star']
    symbol_map = {config: symbols[i % len(symbols)] for i, config in enumerate(config_types)}
    
    # Track if we've added each config type to legend
    added_to_legend = set()
    
    # For each task, create a subplot
    for idx, task in enumerate(tasks):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        task_df = df[df['task'] == task]
        
        # Find best configs for this task at different density levels
        best_configs = []
        for density_threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            subset = task_df[task_df['density'] <= density_threshold]
            if not subset.empty:
                best_idx = subset['error'].idxmin()
                best_configs.append(subset.loc[best_idx])
        
        # Add traces for each config type in this task
        for config_type in config_types:
            config_task_df = task_df[task_df['config_type'] == config_type]
            
            if config_task_df.empty:
                continue
            
            # Check if we should show in legend
            show_legend = config_type not in added_to_legend
            if show_legend:
                added_to_legend.add(config_type)
            
            fig.add_trace(
                go.Scatter(
                    x=config_task_df['density'],
                    y=config_task_df['error'],
                    mode='markers',
                    name=config_type.replace('sink_local_', '').replace('_', ' '),
                    marker=dict(
                        size=10,
                        color=color_map[config_type],
                        symbol=symbol_map[config_type],
                        line=dict(width=1, color='white'),
                        opacity=0.9
                    ),
                    customdata=config_task_df[['model', 'benchmark', 'task', 'config_params', 'trial_id', 'combined_score']],
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>" +
                        "Density: %{x:.3f}<br>" +
                        "Error: %{y:.3f}<br>" +
                        "Model: %{customdata[0]}<br>" +
                        "Benchmark: %{customdata[1]}<br>" +
                        "Task: %{customdata[2]}<br>" +
                        "Config: %{customdata[3]}<br>" +
                        "Trial ID: %{customdata[4]}<br>" +
                        "Combined Score: %{customdata[5]:.3f}<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=show_legend,
                    legendgroup=config_type
                ),
                row=row, col=col
            )
        
        # Highlight best performers with larger markers
        if best_configs:
            best_df = pd.DataFrame(best_configs)
            fig.add_trace(
                go.Scatter(
                    x=best_df['density'],
                    y=best_df['error'],
                    mode='markers',
                    name='Best at density level',
                    marker=dict(
                        size=16,
                        color='#8B0000',  # dark red
                        symbol='star',
                        line=dict(width=2, color='#4B0000')  # even darker red
                    ),
                    customdata=best_df[['config_type', 'config_params']],
                    hovertemplate=(
                        "<b>BEST at density %.1f</b><br>" +
                        "Config: %{customdata[0]}<br>" +
                        "Params: %{customdata[1]}<br>" +
                        "Density: %{x:.3f}<br>" +
                        "Error: %{y:.3f}<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=(idx == 0),  # Only show in legend once
                    legendgroup='best'
                ),
                row=row, col=col
            )
    
    # Update all axes
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            # Update x-axis
            fig.update_xaxes(
                title=dict(text="Density", font={'size': 14}),
                tickfont={'size': 12},
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=False,
                range=[-0.05, 1.05],  # Fixed range for better comparison
                row=i, col=j
            )
            # Update y-axis
            fig.update_yaxes(
                title=dict(text="Error", font={'size': 14}),
                tickfont={'size': 12},
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=False,
                range=[-0.05, 0.9],  # Fixed range for better comparison
                row=i, col=j
            )
    
    # Update layout with aesthetic styling
    fig.update_layout(
        title={
            'text': f"Error vs Density Analysis by Task<br><sub>{df['benchmark'].iloc[0]} benchmark on {df['model'].iloc[0]}</sub>",
            'font': {'size': 24, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            title=dict(text="Configuration Type", font={'size': 14}),
            font={'size': 11},
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            itemsizing='constant',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        height=400 * n_rows,
        width=1400,
        margin=dict(l=80, r=250, t=120, b=80),
        showlegend=True
    )
    
    # Save the figure
    fig.write_html(
        output_file,
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    print(f"Dashboard saved to {output_file}")
    
    # Also create separate plots by benchmark and task
    if output_dir:
        create_faceted_plots(df, str(output_dir / "error_vs_density_by_benchmark.html"))
    else:
        create_faceted_plots(df, "error_vs_density_by_benchmark.html")


def create_faceted_plots(df: pd.DataFrame, output_file: str):
    """
    Create faceted plots showing error vs density grouped by benchmark and task.
    
    Args:
        df: DataFrame with results
        output_file: Output HTML file name
    """
    # Create a more detailed visualization with facets
    fig = px.scatter(
        df,
        x='density',
        y='error',
        color='config_type',
        facet_col='task',
        facet_row='benchmark',
        hover_data=['model', 'config_params', 'trial_id', 'combined_score'],
        title="Error vs Density by Benchmark and Task",
        labels={
            'density': 'Density',
            'error': 'Error',
            'config_type': 'Configuration Type'
        },
        height=1200,
        width=1600
    )
    
    # Update styling
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
    
    fig.update_layout(
        font={'family': 'Arial, sans-serif'},
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='rgba(128, 128, 128, 0.2)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(128, 128, 128, 0.2)', zeroline=False)
    
    # Save the figure
    fig.write_html(
        output_file,
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    print(f"Faceted plots saved to {output_file}")


def create_best_config_summary(df: pd.DataFrame, output_file: str):
    """
    Create a summary visualization showing best configurations for each task.
    
    Args:
        df: DataFrame with results
        output_file: Output HTML file name
    """
    tasks = sorted(df['task'].unique())
    density_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    
    # Create summary data
    summary_data = []
    for task in tasks:
        task_df = df[df['task'] == task]
        for density_level in density_levels:
            subset = task_df[task_df['density'] <= density_level]
            if not subset.empty:
                best_idx = subset['error'].idxmin()
                best_row = subset.loc[best_idx]
                summary_data.append({
                    'task': task,
                    'density_level': density_level,
                    'best_config': best_row['config_type'].replace('sink_local_', ''),
                    'error': best_row['error'],
                    'actual_density': best_row['density'],
                    'params': best_row['config_params']
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create a heatmap-style visualization
    fig = go.Figure()
    
    # Create a trace for each config type
    config_types = summary_df['best_config'].unique()
    colors = px.colors.qualitative.Set3
    color_map = {config: colors[i % len(colors)] for i, config in enumerate(config_types)}
    
    for task in tasks:
        task_data = summary_df[summary_df['task'] == task]
        
        # Add bar chart showing best config at each density level
        fig.add_trace(
            go.Bar(
                name=task,
                x=[f"≤{d:.0%}" for d in task_data['density_level']],
                y=task_data['error'],
                text=[f"{row['best_config']}<br>Error: {row['error']:.3f}" 
                      for _, row in task_data.iterrows()],
                textposition='auto',
                marker_color=[color_map[config] for config in task_data['best_config']],
                customdata=task_data[['best_config', 'actual_density', 'params']],
                hovertemplate=(
                    "<b>Task: %{fullData.name}</b><br>" +
                    "Density Level: %{x}<br>" +
                    "Best Config: %{customdata[0]}<br>" +
                    "Error: %{y:.3f}<br>" +
                    "Actual Density: %{customdata[1]:.3f}<br>" +
                    "Parameters: %{customdata[2]}<br>" +
                    "<extra></extra>"
                )
            )
        )
    
    fig.update_layout(
        title={
            'text': "Best Configurations by Task and Density Level",
            'font': {'size': 20, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="Maximum Density Level",
            tickfont={'size': 12}
        ),
        yaxis=dict(
            title="Error",
            tickfont={'size': 12}
        ),
        barmode='group',
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            title="Task",
            font={'size': 12},
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(r=150),  # Add right margin for legend
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white'
    )
    
    fig.write_html(output_file, config={'displayModeBar': True, 'displaylogo': False})
    print(f"Best config summary saved to {output_file}")


def main():
    """Main function to generate the visualization."""
    # Get the ray_results directory
    ray_results_dir = Path(__file__).parent.parent.parent / "ray_results"
    
    if not ray_results_dir.exists():
        print(f"Error: ray_results directory not found at {ray_results_dir}")
        return
    
    print("Collecting results from ray_results directory...")
    df = collect_results(ray_results_dir)
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"Found {len(df)} results across {df['model'].nunique()} models, "
          f"{df['benchmark'].nunique()} benchmarks, and {df['config_type'].nunique()} configuration types")
    
    # Create output directory
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating interactive dashboard...")
    create_interactive_dashboard(df, str(output_dir / "error_vs_density_by_task.html"), output_dir)
    
    # Create best config summary
    print("\nGenerating best config summary...")
    create_best_config_summary(df, str(output_dir / "best_configs_summary.html"))
    
    # Print a clean summary
    print("\nVisualization complete! Generated files:")
    print(f"  - error_vs_density_by_task.html (task-wise subplots)")
    print(f"  - best_configs_summary.html (best configs at each density level)")
    print(f"  - error_vs_density_by_benchmark.html (faceted by benchmark/task)")
    
    print(f"\nAnalyzed {len(df)} configurations across {len(df['task'].unique())} tasks")


if __name__ == "__main__":
    main()
