#!/usr/bin/env python3
"""
Create specific plots for sparse attention benchmark results.

Plot 1: Density vs Performance per task (subplots)
Plot 2: Dashboard with task-based comparisons
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def load_benchmark_data(results_dir: Path) -> pd.DataFrame:
    """Load benchmark results into a DataFrame."""
    results = []
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        for config_dir in model_dir.iterdir():
            if not config_dir.is_dir():
                continue
                
            config_name = config_dir.name
            
            for task_dir in config_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_name = task_dir.name
                
                # Load metrics
                metrics_file = task_dir / "metrics.json"
                if not metrics_file.exists():
                    continue
                    
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                result = {
                    'model': model_name,
                    'config': config_name,
                    'task': task_name,
                    'performance': metrics.get('overall_score', 0)
                }
                
                # Load density and error for sparse configs
                if config_name != 'dense':
                    micro_metrics_file = task_dir / "micro_metrics.jsonl"
                    if micro_metrics_file.exists():
                        densities = []
                        errors = []
                        
                        with open(micro_metrics_file, 'r') as f:
                            for line in f:
                                try:
                                    entry = json.loads(line.strip())
                                    if entry.get("metric") == "research_attention_density":
                                        densities.append(entry["value"])
                                    elif entry.get("metric") == "research_attention_output_error":
                                        errors.append(entry["value"])
                                except:
                                    continue
                        
                        result['density'] = np.mean(densities) if densities else np.nan
                        result['error'] = np.mean(errors) if errors else np.nan
                else:
                    # Dense baseline
                    result['density'] = 1.0
                    result['error'] = 0.0
                
                results.append(result)
    
    return pd.DataFrame(results)


def create_density_performance_subplots(data: pd.DataFrame, output_path: Path):
    """Create density vs performance plot with subplots per task."""
    # Get unique tasks
    tasks = sorted(data['task'].unique())
    
    # Define markers for different configs
    config_markers = {
        'dense': 'square',
        'sink_local_random_sampling': 'circle',
        'sink_local_oracle_top_k_adaptive_sampling': 'diamond',
        'sink_local_hash_attention_top_k_adaptive_sampling': 'cross',
        'sink_local_oracle_top_p': 'x',
        'sink_local_oracle_top_k': 'triangle-up',
        'sink_local_hash_attention_top_k': 'triangle-down',
        'sink_local_magic_pig': 'star'
    }
    
    # Define colors - blue to green gradient (dark to light)
    config_colors = {
        'dense': '#08519c',  # Dark blue
        'sink_local_random_sampling': '#2171b5',  # Medium blue
        'sink_local_oracle_top_k_adaptive_sampling': '#4292c6',  # Light blue
        'sink_local_hash_attention_top_k_adaptive_sampling': '#6baed6',  # Lighter blue
        'sink_local_oracle_top_p': '#4eb3a6',  # Blue-green
        'sink_local_oracle_top_k': '#41ab5d',  # Medium green
        'sink_local_hash_attention_top_k': '#238b45',  # Dark green
        'sink_local_magic_pig': '#005a32'  # Darkest green
    }
    
    # Calculate grid size
    n_tasks = len(tasks)
    n_cols = 3
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[task.replace('_', ' ').title() for task in tasks],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add traces for each task
    for idx, task in enumerate(tasks):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        task_data = data[data['task'] == task]
        
        # Add scatter points for each config
        for config in sorted(task_data['config'].unique()):
            config_data = task_data[task_data['config'] == config]
            
            fig.add_trace(
                go.Scatter(
                    x=config_data['density'],
                    y=config_data['performance'],
                    mode='markers',
                    name=config.replace('_', ' ').title() if idx == 0 else None,  # Only show legend for first subplot
                    showlegend=(idx == 0),
                    legendgroup=config,  # Link legend across all subplots
                    marker=dict(
                        symbol=config_markers.get(config, 'circle'),
                        size=12,
                        color=config_colors.get(config, '#000000'),
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'<b>{config.replace("_", " ").title()}</b><br>Density: %{{x:.3f}}<br>Performance: %{{y:.3f}}<extra></extra>'
                ),
                row=row,
                col=col
            )
        
        # Update axes
        fig.update_xaxes(title_text="Density", range=[0, 1.05], row=row, col=col)
        fig.update_yaxes(title_text="Performance", row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title="Density vs Performance by Task",
        height=300 * n_rows,
        width=1400,  # Increased width to accommodate legend
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        margin=dict(r=200)  # Add right margin for legend
    )
    
    # Ensure subplot titles are horizontal
    for annotation in fig['layout']['annotations']:
        annotation['textangle'] = 0
    
    # Save
    output_file = output_path / "density_vs_performance_by_task.html"
    fig.write_html(output_file)
    print(f"Saved: {output_file}")


def create_task_comparison_dashboard(data: pd.DataFrame, output_path: Path):
    """Create dashboard with three plots comparing metrics across tasks."""
    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Performance Delta from Dense Baseline",
            "Average Density by Task",
            "Average Error by Task"
        ],
        vertical_spacing=0.12,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # Get unique tasks and configs
    tasks = sorted(data['task'].unique())
    configs = sorted(data['config'].unique())
    
    # Define colors - blue to green gradient (dark to light)
    config_colors = {
        'dense': '#08519c',  # Dark blue
        'sink_local_random_sampling': '#2171b5',  # Medium blue
        'sink_local_oracle_top_k_adaptive_sampling': '#4292c6',  # Light blue
        'sink_local_hash_attention_top_k_adaptive_sampling': '#6baed6',  # Lighter blue
        'sink_local_oracle_top_p': '#4eb3a6',  # Blue-green
        'sink_local_oracle_top_k': '#41ab5d',  # Medium green
        'sink_local_hash_attention_top_k': '#238b45',  # Dark green
        'sink_local_magic_pig': '#005a32'  # Darkest green
    }
    
    # Get dense baseline performance for each task
    dense_performance = {}
    dense_data = data[data['config'] == 'dense']
    for task in tasks:
        task_data = dense_data[dense_data['task'] == task]
        dense_performance[task] = task_data['performance'].mean() if not task_data.empty else 0
    
    # Plot 1: Performance difference from dense baseline
    for config in configs:
        if config == 'dense':
            continue  # Skip dense since we're showing delta from dense
            
        config_data = data[data['config'] == config]
        
        # Calculate mean performance difference per task
        task_performance = []
        for task in tasks:
            task_data = config_data[config_data['task'] == task]
            perf = task_data['performance'].mean() if not task_data.empty else 0
            # Calculate difference from dense baseline
            perf_diff = perf - dense_performance.get(task, 0)
            task_performance.append(perf_diff)
        
        fig.add_trace(
            go.Bar(
                name=config.replace('_', ' ').title(),
                x=tasks,
                y=task_performance,
                marker_color=config_colors.get(config, '#000000'),
                hovertemplate=f'<b>{config.replace("_", " ").title()}</b><br>Task: %{{x}}<br>Performance Delta: %{{y:.3f}}<extra></extra>',
                legendgroup=config  # Link legend across all plots
            ),
            row=1,
            col=1
        )
    
    # Plot 2: Density by task (only sparse configs)
    sparse_configs = [c for c in configs if c != 'dense']
    for config in sparse_configs:
        config_data = data[data['config'] == config]
        
        # Calculate mean density per task
        task_density = []
        for task in tasks:
            task_data = config_data[config_data['task'] == task]
            density = task_data['density'].mean() if not task_data.empty else np.nan
            task_density.append(density)
        
        fig.add_trace(
            go.Bar(
                name=config.replace('_', ' ').title(),
                x=tasks,
                y=task_density,
                marker_color=config_colors.get(config, '#000000'),
                hovertemplate=f'<b>{config.replace("_", " ").title()}</b><br>Task: %{{x}}<br>Density: %{{y:.3f}}<extra></extra>',
                showlegend=False,  # Use same legend as plot 1
                legendgroup=config  # Link legend across all plots
            ),
            row=2,
            col=1
        )
    
    # Plot 3: Error by task (only sparse configs)
    for config in sparse_configs:
        config_data = data[data['config'] == config]
        
        # Calculate mean error per task
        task_error = []
        for task in tasks:
            task_data = config_data[config_data['task'] == task]
            error = task_data['error'].mean() if not task_data.empty else np.nan
            task_error.append(error)
        
        fig.add_trace(
            go.Bar(
                name=config.replace('_', ' ').title(),
                x=tasks,
                y=task_error,
                marker_color=config_colors.get(config, '#000000'),
                hovertemplate=f'<b>{config.replace("_", " ").title()}</b><br>Task: %{{x}}<br>Error: %{{y:.3f}}<extra></extra>',
                showlegend=False,  # Use same legend as plot 1
                legendgroup=config  # Link legend across all plots
            ),
            row=3,
            col=1
        )
    
    # Update axes
    fig.update_xaxes(title_text="Task", row=3, col=1)
    fig.update_xaxes(tickangle=0)
    
    fig.update_yaxes(title_text="Performance Delta", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_yaxes(title_text="Error", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title="Task-wise Comparison Dashboard",
        height=1200,
        width=1200,
        barmode='group',
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        margin=dict(r=200)  # Add right margin for legend
    )
    
    # Ensure subplot titles are horizontal
    for annotation in fig['layout']['annotations']:
        annotation['textangle'] = 0
    
    # Save
    output_file = output_path / "task_comparison_dashboard.html"
    fig.write_html(output_file)
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create specific plots for benchmark results")
    parser.add_argument("--results-dir", type=str, default="benchmark_results_ray",
                       help="Directory containing benchmark results")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading benchmark data...")
    data = load_benchmark_data(results_dir)
    
    if data.empty:
        print("No data found!")
        sys.exit(1)
    
    print(f"Loaded {len(data)} benchmark results")
    
    # Create plots
    print("\nCreating density vs performance subplots...")
    create_density_performance_subplots(data, output_dir)
    
    print("\nCreating task comparison dashboard...")
    create_task_comparison_dashboard(data, output_dir)
    
    print("\nAll plots created successfully!")


if __name__ == "__main__":
    main()


