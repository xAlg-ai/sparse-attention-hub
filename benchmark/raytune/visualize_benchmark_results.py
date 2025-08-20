#!/usr/bin/env python3
"""
Production-quality interactive visualization dashboard for sparse attention benchmark results.

This script creates professional-grade interactive plots using Plotly to visualize:
- Model performance across different tasks
- Sparse attention density vs error trade-offs
- Comparative analysis across different sparse attention methods

Usage:
    python visualize_benchmark_results.py --results-dir benchmark_results_ray --output dashboard.html
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.colors import qualitative


class BenchmarkResultsVisualizer:
    """Production-grade visualizer for sparse attention benchmark results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.data = self._load_all_results()
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup consistent styling for all plots."""
        self.colors = {
            'dense': '#1f77b4',  # Blue
            'sink_local_random_sampling': '#ff7f0e',  # Orange
            'sink_local_oracle_top_k_adaptive_sampling': '#2ca02c',  # Green
            'sink_local_hash_attention_top_k_adaptive_sampling': '#d62728',  # Red
            'sink_local_oracle_top_p': '#9467bd',  # Purple
            'sink_local_oracle_top_k': '#8c564b',  # Brown
            'sink_local_hash_attention_top_k': '#e377c2',  # Pink
            'sink_local_magic_pig': '#7f7f7f',  # Gray
        }
        
        self.plot_config = {
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'sparse_attention_benchmark',
                'height': 1200,
                'width': 1600,
                'scale': 2
            }
        }
        
        self.layout_template = {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'title_font': {'size': 20, 'family': 'Arial Black, sans-serif'},
            'hovermode': 'x unified',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 80, 'r': 80, 't': 100, 'b': 80}
        }
    
    def _load_all_results(self) -> pd.DataFrame:
        """Load all benchmark results into a structured DataFrame."""
        results = []
        
        for model_dir in self.results_dir.iterdir():
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
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        # Load micro metrics for sparse configs
                        density = None
                        attention_error = None
                        micro_metrics_file = task_dir / "micro_metrics.jsonl"
                        
                        if micro_metrics_file.exists() and config_name != "dense":
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
                            
                            if densities:
                                density = np.mean(densities)
                            if errors:
                                attention_error = np.mean(errors)
                        
                        # Extract performance metrics
                        result = {
                            'model': model_name,
                            'config': config_name,
                            'task': task_name,
                            'overall_score': metrics.get('overall_score', 0),
                            'density': density,
                            'attention_error': attention_error,
                            'total_samples': metrics.get('summary', {}).get('total_samples', 0)
                        }
                        
                        # Add task-specific scores
                        task_scores = metrics.get('task_scores', {})
                        if task_scores:
                            first_task = list(task_scores.values())[0]
                            for metric, value in first_task.items():
                                result[f'metric_{metric}'] = value
                        
                        results.append(result)
        
        return pd.DataFrame(results)
    
    def create_performance_heatmap(self) -> go.Figure:
        """Create a heatmap showing performance across tasks and configs."""
        # Pivot data for heatmap
        pivot_data = self.data.pivot_table(
            index='config',
            columns='task',
            values='overall_score',
            aggfunc='mean'
        )
        
        # Sort configs by average performance
        config_order = pivot_data.mean(axis=1).sort_values(ascending=False).index
        pivot_data = pivot_data.loc[config_order]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdBu_r',
            text=np.round(pivot_data.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Overall Score"),
            hovertemplate='Config: %{y}<br>Task: %{x}<br>Score: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Performance Heatmap: Sparse Attention Methods vs Tasks',
            xaxis_title='Benchmark Task',
            yaxis_title='Sparse Attention Configuration',
            height=600,
            **self.layout_template
        )
        
        return fig
    
    def create_density_vs_performance_scatter(self) -> go.Figure:
        """Create scatter plot showing density vs performance trade-off."""
        # Filter out dense baseline
        sparse_data = self.data[self.data['config'] != 'dense'].copy()
        
        fig = go.Figure()
        
        # Add scatter points for each config
        for config in sparse_data['config'].unique():
            config_data = sparse_data[sparse_data['config'] == config]
            
            fig.add_trace(go.Scatter(
                x=config_data['density'],
                y=config_data['overall_score'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.colors.get(config, '#000000'),
                    line=dict(width=1, color='white')
                ),
                name=config.replace('_', ' ').title(),
                text=config_data['task'],
                hovertemplate='<b>%{text}</b><br>Density: %{x:.3f}<br>Score: %{y:.3f}<extra></extra>'
            ))
        
        # Add dense baseline as horizontal line
        dense_scores = self.data[self.data['config'] == 'dense']['overall_score']
        if not dense_scores.empty:
            fig.add_hline(
                y=dense_scores.mean(),
                line_dash="dash",
                line_color="gray",
                annotation_text="Dense Baseline",
                annotation_position="right"
            )
        
        fig.update_layout(
            title='Density vs Performance Trade-off',
            xaxis_title='Average Attention Density',
            yaxis_title='Overall Score',
            height=600,
            xaxis=dict(range=[0, 1]),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            margin=dict(r=150),  # Add right margin for legend
            **self.layout_template
        )
        
        return fig
    
    def create_error_vs_density_scatter(self) -> go.Figure:
        """Create scatter plot showing attention error vs density."""
        # Filter out dense baseline and data without error metrics
        sparse_data = self.data[
            (self.data['config'] != 'dense') & 
            (self.data['attention_error'].notna())
        ].copy()
        
        fig = go.Figure()
        
        # Add scatter points for each task
        for task in sparse_data['task'].unique():
            task_data = sparse_data[sparse_data['task'] == task]
            
            fig.add_trace(go.Scatter(
                x=task_data['density'],
                y=task_data['attention_error'],
                mode='markers',
                marker=dict(
                    size=10,
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                name=task.replace('_', ' ').title(),
                text=task_data['config'],
                hovertemplate='<b>%{text}</b><br>Density: %{x:.3f}<br>Error: %{y:.3f}<extra></extra>'
            ))
        
        # Add ideal line (y=0)
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="green",
            annotation_text="Perfect Attention",
            annotation_position="right"
        )
        
        fig.update_layout(
            title='Attention Error vs Density by Task',
            xaxis_title='Average Attention Density',
            yaxis_title='Average Attention Error',
            height=600,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, max(0.5, sparse_data['attention_error'].max() * 1.1)]),
            showlegend=True,
            **self.layout_template
        )
        
        return fig
    
    def create_performance_by_task_bar(self) -> go.Figure:
        """Create grouped bar chart showing performance by task."""
        fig = go.Figure()
        
        # Get unique tasks and configs
        tasks = sorted(self.data['task'].unique())
        configs = sorted(self.data['config'].unique())
        
        # Create grouped bars
        for config in configs:
            config_data = self.data[self.data['config'] == config]
            
            # Calculate mean score per task
            task_scores = []
            for task in tasks:
                task_data = config_data[config_data['task'] == task]
                score = task_data['overall_score'].mean() if not task_data.empty else 0
                task_scores.append(score)
            
            fig.add_trace(go.Bar(
                name=config.replace('_', ' ').title(),
                x=tasks,
                y=task_scores,
                marker_color=self.colors.get(config, '#000000'),
                hovertemplate='Task: %{x}<br>Score: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Performance Comparison by Task',
            xaxis_title='Benchmark Task',
            yaxis_title='Overall Score',
            barmode='group',
            height=600,
            xaxis_tickangle=-45,
            **self.layout_template
        )
        
        return fig
    
    def create_dashboard(self, output_file: str = "benchmark_dashboard.html"):
        """Create a comprehensive dashboard with all visualizations."""
        # Create subplots with specific layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Heatmap',
                'Density vs Performance Trade-off',
                'Performance by Task',
                'Attention Error vs Density'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Create individual plots
        heatmap = self.create_performance_heatmap()
        density_perf = self.create_density_vs_performance_scatter()
        task_bars = self.create_performance_by_task_bar()
        error_density = self.create_error_vs_density_scatter()
        
        # Add traces to subplots
        for trace in heatmap.data:
            fig.add_trace(trace, row=1, col=1)
        
        for trace in density_perf.data:
            fig.add_trace(trace, row=1, col=2)
        
        for trace in task_bars.data:
            fig.add_trace(trace, row=2, col=1)
        
        for trace in error_density.data:
            fig.add_trace(trace, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Sparse Attention Benchmark Results Dashboard",
            title_font_size=24,
            height=1200,
            showlegend=False,  # Individual plots have their own legends
            **self.layout_template
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Benchmark Task", row=1, col=1)
        fig.update_yaxes(title_text="Configuration", row=1, col=1)
        
        fig.update_xaxes(title_text="Density", row=1, col=2)
        fig.update_yaxes(title_text="Overall Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Task", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Density", row=2, col=2)
        fig.update_yaxes(title_text="Attention Error", row=2, col=2)
        
        # Save dashboard
        fig.write_html(
            output_file,
            config=self.plot_config,
            include_plotlyjs='cdn'
        )
        
        # Also create individual plots
        output_dir = Path(output_file).parent
        
        # Save individual plots
        heatmap.write_html(output_dir / "performance_heatmap.html", config=self.plot_config)
        density_perf.write_html(output_dir / "density_vs_performance.html", config=self.plot_config)
        task_bars.write_html(output_dir / "performance_by_task.html", config=self.plot_config)
        error_density.write_html(output_dir / "error_vs_density.html", config=self.plot_config)
        
        print(f"Dashboard saved to: {output_file}")
        print(f"Individual plots saved to: {output_dir}/")
        
        return fig
    
    def generate_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics for the benchmark results."""
        summary = []
        
        for config in self.data['config'].unique():
            config_data = self.data[self.data['config'] == config]
            
            stats = {
                'config': config,
                'avg_score': config_data['overall_score'].mean(),
                'std_score': config_data['overall_score'].std(),
                'avg_density': config_data['density'].mean() if config != 'dense' else 1.0,
                'avg_error': config_data['attention_error'].mean() if config != 'dense' else 0.0,
                'num_tasks': len(config_data),
                'best_task': config_data.loc[config_data['overall_score'].idxmax(), 'task'] if not config_data.empty else None,
                'worst_task': config_data.loc[config_data['overall_score'].idxmin(), 'task'] if not config_data.empty else None
            }
            
            summary.append(stats)
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('avg_score', ascending=False)
        
        # Save summary
        summary_df.to_csv(self.results_dir.parent / "benchmark_summary.csv", index=False)
        
        return summary_df


def main():
    parser = argparse.ArgumentParser(description="Visualize sparse attention benchmark results")
    parser.add_argument("--results-dir", type=str, default="benchmark_results_ray",
                       help="Directory containing benchmark results")
    parser.add_argument("--output", type=str, default="benchmark_dashboard.html",
                       help="Output HTML file for dashboard")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        sys.exit(1)
    
    # Create visualizer and generate dashboard
    visualizer = BenchmarkResultsVisualizer(results_dir)
    
    # Generate dashboard
    visualizer.create_dashboard(args.output)
    
    # Generate summary statistics
    summary = visualizer.generate_summary_stats()
    print("\nBenchmark Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()



