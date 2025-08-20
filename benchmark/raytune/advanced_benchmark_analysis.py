#!/usr/bin/env python3
"""
Advanced analysis and visualization for sparse attention benchmarks.

This script provides:
- Statistical analysis with confidence intervals
- Pareto frontier analysis
- Performance regression analysis
- Detailed breakdowns by metric type
- Export capabilities for publication-ready figures

Usage:
    python advanced_benchmark_analysis.py --results-dir benchmark_results_ray
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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff


class AdvancedBenchmarkAnalyzer:
    """Advanced analysis for sparse attention benchmarks."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.data = self._load_comprehensive_results()
        self._compute_statistics()
        self._setup_professional_styling()
    
    def _setup_professional_styling(self):
        """Setup publication-quality styling."""
        # Professional color palette
        self.colors = px.colors.qualitative.D3
        self.config_colors = {
            'dense': '#1f77b4',
            'sink_local_random_sampling': '#ff7f0e',
            'sink_local_oracle_top_k_adaptive_sampling': '#2ca02c',
            'sink_local_hash_attention_top_k_adaptive_sampling': '#d62728',
            'sink_local_oracle_top_p': '#9467bd',
            'sink_local_oracle_top_k': '#8c564b',
            'sink_local_hash_attention_top_k': '#e377c2',
            'sink_local_magic_pig': '#7f7f7f',
        }
        
        self.layout_template = go.layout.Template(
            layout=go.Layout(
                font=dict(family="Arial, sans-serif", size=14),
                title_font=dict(size=22, family="Arial Black, sans-serif"),
                hovermode='closest',
                plot_bgcolor='rgba(240,240,240,0.1)',
                paper_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    zeroline=False
                ),
                margin=dict(l=100, r=100, t=120, b=100)
            )
        )
    
    def _load_comprehensive_results(self) -> pd.DataFrame:
        """Load results with detailed metrics and metadata."""
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
                    
                    # Load all available data
                    result = self._load_task_result(
                        model_name, config_name, task_name, task_dir
                    )
                    
                    if result:
                        results.append(result)
        
        df = pd.DataFrame(results)
        
        # Add derived metrics
        if not df.empty:
            df['efficiency_score'] = df.apply(
                lambda x: x['overall_score'] / x['density'] if x['density'] > 0 else 0,
                axis=1
            )
            
            # Normalize scores for comparison
            if 'overall_score' in df.columns:
                df['normalized_score'] = (df['overall_score'] - df['overall_score'].min()) / \
                                        (df['overall_score'].max() - df['overall_score'].min())
        
        return df
    
    def _load_task_result(self, model: str, config: str, task: str, 
                          task_dir: Path) -> Optional[Dict]:
        """Load comprehensive result data for a single task."""
        result = {
            'model': model,
            'config': config,
            'task': task,
            'config_type': 'sparse' if config != 'dense' else 'dense'
        }
        
        # Load metrics
        metrics_file = task_dir / "metrics.json"
        if not metrics_file.exists():
            return None
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        result['overall_score'] = metrics.get('overall_score', 0)
        result['total_samples'] = metrics.get('summary', {}).get('total_samples', 0)
        
        # Extract all individual metrics
        task_scores = metrics.get('task_scores', {})
        if task_scores:
            first_task = list(task_scores.values())[0]
            for metric, value in first_task.items():
                result[f'metric_{metric}'] = value
        
        # Load micro metrics for sparse configs
        if config != 'dense':
            micro_stats = self._compute_micro_statistics(task_dir / "micro_metrics.jsonl")
            result.update(micro_stats)
        else:
            # Dense baseline values
            result['density'] = 1.0
            result['attention_error'] = 0.0
            result['density_std'] = 0.0
            result['error_std'] = 0.0
        
        return result
    
    def _compute_micro_statistics(self, micro_metrics_file: Path) -> Dict:
        """Compute statistics from micro metrics."""
        stats = {
            'density': np.nan,
            'attention_error': np.nan,
            'density_std': np.nan,
            'error_std': np.nan,
            'density_percentiles': {},
            'error_percentiles': {}
        }
        
        if not micro_metrics_file.exists():
            return stats
        
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
            stats['density'] = np.mean(densities)
            stats['density_std'] = np.std(densities)
            stats['density_percentiles'] = {
                'p25': np.percentile(densities, 25),
                'p50': np.percentile(densities, 50),
                'p75': np.percentile(densities, 75)
            }
        
        if errors:
            stats['attention_error'] = np.mean(errors)
            stats['error_std'] = np.std(errors)
            stats['error_percentiles'] = {
                'p25': np.percentile(errors, 25),
                'p50': np.percentile(errors, 50),
                'p75': np.percentile(errors, 75)
            }
        
        return stats
    
    def _compute_statistics(self):
        """Compute statistical summaries and comparisons."""
        if self.data.empty:
            return
        
        # Compute config-level statistics
        self.config_stats = self.data.groupby('config').agg({
            'overall_score': ['mean', 'std', 'count'],
            'density': ['mean', 'std'],
            'attention_error': ['mean', 'std']
        }).round(4)
        
        # Compute task-level statistics
        self.task_stats = self.data.groupby('task').agg({
            'overall_score': ['mean', 'std', 'count']
        }).round(4)
        
        # Statistical comparisons vs dense baseline
        self.comparisons = self._compute_statistical_comparisons()
    
    def _compute_statistical_comparisons(self) -> pd.DataFrame:
        """Compute statistical comparisons against dense baseline."""
        comparisons = []
        
        dense_data = self.data[self.data['config'] == 'dense']
        if dense_data.empty:
            return pd.DataFrame()
        
        for config in self.data['config'].unique():
            if config == 'dense':
                continue
            
            config_data = self.data[self.data['config'] == config]
            
            # Perform t-test for each task
            for task in self.data['task'].unique():
                dense_task = dense_data[dense_data['task'] == task]['overall_score']
                config_task = config_data[config_data['task'] == task]['overall_score']
                
                if len(dense_task) > 0 and len(config_task) > 0:
                    t_stat, p_value = stats.ttest_ind(dense_task, config_task)
                    
                    comparisons.append({
                        'config': config,
                        'task': task,
                        'dense_mean': dense_task.mean(),
                        'config_mean': config_task.mean(),
                        'difference': config_task.mean() - dense_task.mean(),
                        'percent_change': ((config_task.mean() - dense_task.mean()) / dense_task.mean() * 100),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
        
        return pd.DataFrame(comparisons)
    
    def create_pareto_frontier(self) -> go.Figure:
        """Create Pareto frontier plot for density vs performance."""
        sparse_data = self.data[self.data['config'] != 'dense'].copy()
        
        # Compute Pareto frontier
        pareto_points = []
        sorted_data = sparse_data.sort_values('density')
        
        max_score = -np.inf
        for _, row in sorted_data.iterrows():
            if row['overall_score'] >= max_score:
                max_score = row['overall_score']
                pareto_points.append(row)
        
        pareto_df = pd.DataFrame(pareto_points)
        
        # Create figure
        fig = go.Figure()
        
        # Add all points
        for config in sparse_data['config'].unique():
            config_data = sparse_data[sparse_data['config'] == config]
            
            fig.add_trace(go.Scatter(
                x=config_data['density'],
                y=config_data['overall_score'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=self.config_colors.get(config, '#000000'),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name=config.replace('_', ' ').title(),
                text=config_data['task'],
                hovertemplate='<b>%{text}</b><br>Density: %{x:.3f}<br>Score: %{y:.3f}<extra></extra>'
            ))
        
        # Add Pareto frontier
        if not pareto_df.empty:
            fig.add_trace(go.Scatter(
                x=pareto_df['density'],
                y=pareto_df['overall_score'],
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name='Pareto Frontier',
                showlegend=True
            ))
        
        # Add dense baseline
        dense_score = self.data[self.data['config'] == 'dense']['overall_score'].mean()
        fig.add_hline(
            y=dense_score,
            line_dash="dot",
            line_color="black",
            annotation_text="Dense Baseline",
            annotation_position="right"
        )
        
        fig.update_layout(
            template=self.layout_template,
            title='Pareto Frontier: Density vs Performance Trade-off',
            xaxis_title='Attention Density',
            yaxis_title='Overall Performance Score',
            height=700,
            width=1000,
            xaxis=dict(range=[0, 1.05]),
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        )
        
        return fig
    
    def create_statistical_comparison_plot(self) -> go.Figure:
        """Create plot showing statistical comparisons vs baseline."""
        if self.comparisons.empty:
            return go.Figure()
        
        # Aggregate by config
        config_comparison = self.comparisons.groupby('config').agg({
            'percent_change': 'mean',
            'significant': 'sum',
            'task': 'count'
        }).reset_index()
        
        config_comparison.columns = ['config', 'avg_percent_change', 'num_significant', 'num_tasks']
        config_comparison['percent_significant'] = config_comparison['num_significant'] / config_comparison['num_tasks'] * 100
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=config_comparison['config'],
            y=config_comparison['avg_percent_change'],
            marker_color=[self.config_colors.get(c, '#000000') for c in config_comparison['config']],
            text=config_comparison['percent_significant'].round(1),
            texttemplate='%{text}% significant',
            textposition='outside',
            hovertemplate='Config: %{x}<br>Avg Change: %{y:.1f}%<br>Significant Tests: %{text}<extra></extra>'
        ))
        
        # Add significance threshold
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
        
        fig.update_layout(
            template=self.layout_template,
            title='Performance Change vs Dense Baseline<br><sub>Percentage of statistically significant differences shown</sub>',
            xaxis_title='Sparse Attention Configuration',
            yaxis_title='Average Performance Change (%)',
            height=600,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, output_dir: str = "benchmark_analysis"):
        """Create comprehensive analysis dashboard with multiple views."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create main dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Pareto Frontier Analysis',
                'Statistical Comparisons',
                'Performance Distribution by Config',
                'Error vs Density Correlation',
                'Task Difficulty Analysis',
                'Efficiency Scores'
            ),
            row_heights=[0.35, 0.35, 0.3],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "violin"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Pareto Frontier
        pareto = self.create_pareto_frontier()
        for trace in pareto.data:
            fig.add_trace(trace, row=1, col=1)
        
        # 2. Statistical Comparisons
        stats_comp = self.create_statistical_comparison_plot()
        for trace in stats_comp.data:
            fig.add_trace(trace, row=1, col=2)
        
        # 3. Performance Distribution
        sparse_data = self.data[self.data['config'] != 'dense']
        for config in sparse_data['config'].unique():
            config_data = sparse_data[sparse_data['config'] == config]
            fig.add_trace(go.Violin(
                y=config_data['overall_score'],
                name=config.replace('_', ' ').title(),
                marker_color=self.config_colors.get(config, '#000000'),
                box_visible=True,
                meanline_visible=True
            ), row=2, col=1)
        
        # 4. Error vs Density
        if 'attention_error' in sparse_data.columns:
            fig.add_trace(go.Scatter(
                x=sparse_data['density'],
                y=sparse_data['attention_error'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=sparse_data['overall_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score", x=1.02)
                ),
                text=sparse_data['config'],
                hovertemplate='Config: %{text}<br>Density: %{x:.3f}<br>Error: %{y:.3f}<extra></extra>'
            ), row=2, col=2)
        
        # 5. Task Difficulty
        task_avg = self.data.groupby('task')['overall_score'].mean().sort_values()
        fig.add_trace(go.Bar(
            x=task_avg.values,
            y=task_avg.index,
            orientation='h',
            marker_color='lightblue'
        ), row=3, col=1)
        
        # 6. Efficiency Scores
        if 'efficiency_score' in self.data.columns:
            efficiency_data = self.data[self.data['efficiency_score'] > 0]
            for config in efficiency_data['config'].unique():
                config_data = efficiency_data[efficiency_data['config'] == config]
                fig.add_trace(go.Scatter(
                    x=config_data['density'],
                    y=config_data['efficiency_score'],
                    mode='markers',
                    name=config,
                    marker=dict(size=10)
                ), row=3, col=2)
        
        # Update layout
        fig.update_layout(
            template=self.layout_template,
            title_text="Comprehensive Sparse Attention Benchmark Analysis",
            title_font_size=26,
            height=1800,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_file = output_path / "comprehensive_dashboard.html"
        fig.write_html(
            dashboard_file,
            include_plotlyjs='cdn'
        )
        
        # Generate additional analyses
        self._generate_detailed_reports(output_path)
        
        print(f"Analysis complete. Results saved to: {output_path}")
        
        return fig
    
    def _generate_detailed_reports(self, output_path: Path):
        """Generate detailed reports and additional visualizations."""
        # 1. Summary statistics
        summary_stats = pd.DataFrame({
            'Configuration': self.config_stats.index,
            'Avg Score': self.config_stats[('overall_score', 'mean')],
            'Std Score': self.config_stats[('overall_score', 'std')],
            'Avg Density': self.config_stats[('density', 'mean')],
            'Avg Error': self.config_stats[('attention_error', 'mean')]
        })
        summary_stats.to_csv(output_path / "summary_statistics.csv", index=False)
        
        # 2. Detailed comparisons
        if not self.comparisons.empty:
            self.comparisons.to_csv(output_path / "statistical_comparisons.csv", index=False)
        
        # 3. Best configurations per task
        best_configs = []
        for task in self.data['task'].unique():
            task_data = self.data[self.data['task'] == task]
            best = task_data.loc[task_data['overall_score'].idxmax()]
            best_configs.append({
                'task': task,
                'best_config': best['config'],
                'score': best['overall_score'],
                'density': best.get('density', 1.0)
            })
        
        pd.DataFrame(best_configs).to_csv(output_path / "best_configs_per_task.csv", index=False)
        
        # 4. Performance correlation matrix
        if len(self.data.columns) > 10:
            metric_cols = [col for col in self.data.columns if col.startswith('metric_')]
            if metric_cols:
                corr_matrix = self.data[metric_cols].corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig_corr.update_layout(
                    title='Metric Correlation Matrix',
                    height=800,
                    width=800
                )
                
                fig_corr.write_html(output_path / "metric_correlations.html")


def main():
    parser = argparse.ArgumentParser(description="Advanced analysis of sparse attention benchmarks")
    parser.add_argument("--results-dir", type=str, default="benchmark_results_ray",
                       help="Directory containing benchmark results")
    parser.add_argument("--output-dir", type=str, default="benchmark_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        sys.exit(1)
    
    # Run analysis
    analyzer = AdvancedBenchmarkAnalyzer(results_dir)
    analyzer.create_comprehensive_dashboard(args.output_dir)
    
    # Print summary
    print("\nConfiguration Performance Summary:")
    print(analyzer.config_stats)


if __name__ == "__main__":
    main()



