#!/usr/bin/env python3
"""Plot combined quality and error data in a 2x3 grid.

This script creates a combined plot showing quality vs density in the top row
and error vs density in the bottom row for three specific datasets:
- QA_2 RULER 32K / ruler_qa_2
- HotpotQA LongBench / hotpotqa  
- ShortDep Cloze Loogle / shortdep_cloze
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from matplotlib.ticker import FormatStrFormatter


def get_quality_data() -> Dict[str, pd.DataFrame]:
    """Extract quality data from plot_pareto.py for the three required datasets.
    
    Returns:
        Dict mapping dataset names to DataFrames with quality data.
    """
    datasets = {}
    
    # QA_2 RULER 32K
    qa2_data = {
        'density': [0.05, 0.1, 0.15, 0.2],
        'Full': [51.5, 51.5, 51.5, 51.5],
        'Oracle Top-k': [49, 48, 48, 49],
        'Oracle Top-p': [47.5, 48.5, 50.5, 50.5],
        'HashAttention Top-k': [45.5, 45.5, 48.5, 47.5],
        'vAttention(oracle-top-k)': [51, 51.5, 51, 51],
        'vAttention(hashattention)': [47, 48, 49.5, 50]
    }
    datasets['QA_2 RULER 32K'] = pd.DataFrame(qa2_data)
    
    # HotpotQA LongBench
    hotpotqa_data = {
        'density': [0.05, 0.1, 0.15, 0.2],
        'Full': [55.75, 55.75, 55.75, 55.75],
        'Oracle Top-k': [52.47, 51.98, 52.75, 52.92],
        'Oracle Top-p': [52.39, 54.72, 56.61, 56.0],
        'HashAttention Top-k': [50.91, 52.57, 51.82, 52.48],
        'vAttention(oracle-top-k)': [56.1, 55.95, 55.57, 55.45],
        'vAttention(hashattention)': [50.91, 55.59, 56.02, 56.64]
    }
    datasets['HotpotQA LongBench'] = pd.DataFrame(hotpotqa_data)
    
    # ShortDep Cloze Loogle (ignore match columns)
    shortdep_cloze_data = {
        'density': [0.05, 0.1, 0.15, 0.2],
        'Full': [0.5521, 0.5521, 0.5521, 0.5521],
        'Oracle Top-k': [0.3507, 0.5047, 0.5213, 0.5403],
        'Oracle Top-p': [0.5498, 0.5379, 0.5616, 0.5498],
        'HashAttention Top-k': [0.4455, 0.4834, 0.5095, 0.5213],
        'vAttention(oracle-top-k)': [0.5332, 0.5498, 0.5498, 0.5427],
        'vAttention(hashattention)': [0.4455, 0.4692, 0.5095, 0.5213]
    }
    datasets['ShortDep Cloze Loogle'] = pd.DataFrame(shortdep_cloze_data)
    
    return datasets


def get_error_data() -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """Extract error data from plot_attention_results.py for the three required datasets.
    
    Returns:
        Dict mapping dataset names to method data (density, error) tuples.
    """
    datasets = {}
    
    # ruler_qa_2 data
    ruler_qa_2_raw = """
    sink_local_hash_attention_top_k,0.058157,0.243065
    sink_local_hash_attention_top_k,0.103150,0.175304
    sink_local_hash_attention_top_k,0.158159,0.134139
    sink_local_hash_attention_top_k,0.208169,0.106352
    sink_local_hash_attention_top_k_adaptive_sampling,0.051537,0.209314
    sink_local_hash_attention_top_k_adaptive_sampling,0.078947,0.187638
    sink_local_hash_attention_top_k_adaptive_sampling,0.153942,0.076740
    sink_local_hash_attention_top_k_adaptive_sampling,0.196851,0.057732
    sink_local_oracle_top_k,0.058164,0.105703
    sink_local_oracle_top_k,0.103121,0.068634
    sink_local_oracle_top_k,0.158173,0.050621
    sink_local_oracle_top_k,0.208168,0.039016
    sink_local_oracle_top_k_adaptive_sampling,0.049467,0.043303
    sink_local_oracle_top_k_adaptive_sampling,0.090930,0.022806
    sink_local_oracle_top_k_adaptive_sampling,0.152515,0.013741
    sink_local_oracle_top_k_adaptive_sampling,0.168363,0.011796
    sink_local_oracle_top_p,0.052304,0.130287
    sink_local_oracle_top_p,0.097983,0.070763
    sink_local_oracle_top_p,0.126857,0.055570
    sink_local_oracle_top_p,0.179323,0.036691
    """
    
    # hotpotqa data
    hotpotqa_raw = """
    sink_local_oracle_top_k_adaptive_sampling,0.152574,0.025505
    sink_local_oracle_top_k_adaptive_sampling,0.055908,0.089229
    sink_local_hash_attention_top_k_adaptive_sampling,0.198917,0.061421
    sink_local_oracle_top_k,0.221413,0.048596
    sink_local_hash_attention_top_k_adaptive_sampling,0.105509,0.141730
    sink_local_hash_attention_top_k,0.221787,0.102106
    sink_local_hash_attention_top_k,0.073072,0.222028
    sink_local_oracle_top_k,0.121856,0.083511
    sink_local_oracle_top_k,0.171683,0.062462
    sink_local_oracle_top_k_adaptive_sampling,0.104538,0.036789
    sink_local_hash_attention_top_k,0.171797,0.126884
    sink_local_hash_attention_top_k_adaptive_sampling,0.154702,0.085973
    sink_local_hash_attention_top_k_adaptive_sampling,0.055090,0.265673
    sink_local_oracle_top_k,0.072654,0.119449
    sink_local_oracle_top_k_adaptive_sampling,0.207016,0.017826
    sink_local_hash_attention_top_k,0.122292,0.157777
    sink_local_oracle_top_p,0.053907,0.193479
    sink_local_oracle_top_p,0.092051,0.116004
    sink_local_oracle_top_p,0.131323,0.079087
    sink_local_oracle_top_p,0.204828,0.046133
    """
    
    # shortdep_cloze data
    shortdep_cloze_raw = """
    sink_local_oracle_top_k_adaptive_sampling,0.145063,0.019633
    sink_local_oracle_top_k_adaptive_sampling,0.048608,0.071137
    sink_local_hash_attention_top_k_adaptive_sampling,0.194726,0.158095
    sink_local_oracle_top_k,0.215420,0.040531
    sink_local_hash_attention_top_k_adaptive_sampling,0.093704,0.196768
    sink_local_hash_attention_top_k,0.215414,0.147568
    sink_local_hash_attention_top_k,0.065413,0.250572
    sink_local_oracle_top_k,0.115413,0.072269
    sink_local_oracle_top_k,0.165402,0.051934
    sink_local_oracle_top_k_adaptive_sampling,0.093562,0.028051
    sink_local_hash_attention_top_k,0.165401,0.168105
    sink_local_hash_attention_top_k_adaptive_sampling,0.145577,0.168549
    sink_local_hash_attention_top_k_adaptive_sampling,0.047984,0.277436
    sink_local_oracle_top_k,0.065427,0.118486
    sink_local_oracle_top_k_adaptive_sampling,0.193612,0.013235
    sink_local_hash_attention_top_k,0.115420,0.198599
    sink_local_oracle_top_p,0.298482,0.014339
    sink_local_oracle_top_p,0.065194,0.100925
    sink_local_oracle_top_p,0.169924,0.034923
    sink_local_oracle_top_p,0.297403,0.014353
    """
    
    def parse_raw_data(raw_data: str) -> Dict[str, List[Tuple[float, float]]]:
        """Parse raw CSV data into structured format."""
        data: Dict[str, List[Tuple[float, float]]] = {}
        
        for line in raw_data.strip().split('\n'):
            if line.strip():
                parts: List[str] = line.strip().split(',')
                if len(parts) >= 3:
                    method_name: str = parts[0]
                    density: float = float(parts[1])
                    error: float = float(parts[2])
                    
                    if method_name not in data:
                        data[method_name] = []
                    data[method_name].append((density, error))
        
        # Sort each method's data by density
        for method in data:
            data[method].sort(key=lambda x: x[0])
        
        return data
    
    datasets['QA_2 RULER 32K'] = parse_raw_data(ruler_qa_2_raw)
    datasets['HotpotQA LongBench'] = parse_raw_data(hotpotqa_raw)
    datasets['ShortDep Cloze Loogle'] = parse_raw_data(shortdep_cloze_raw)
    
    return datasets


def create_main_paper_plot() -> None:
    """Create the main paper plot with 2x3 grid showing quality and error vs density."""
    # Get data
    quality_datasets = get_quality_data()
    error_datasets = get_error_data()
    
    # Dataset names in order
    dataset_names = ['QA_2 RULER 32K', 'HotpotQA LongBench', 'ShortDep Cloze Loogle']
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(32, 16))
    
    # Define colors matching both scripts exactly
    colors = {
        'Full': '#000000',  # black for Full attention
        'Oracle Top-k': '#17becf',  # cyan/light blue
        'Oracle Top-p': '#9467bd',  # purple
        'HashAttention Top-k': '#ff7f0e',  # orange
        'vAttention(oracle-top-k)': '#1f77b4',  # blue
        'vAttention(hashattention)': '#8b0000'  # dark red/maroon
    }
    
    # Map error data method names to quality data method names
    error_to_quality_mapping = {
        'sink_local_oracle_top_k': 'Oracle Top-k',
        'sink_local_oracle_top_k_adaptive_sampling': 'vAttention(oracle-top-k)',
        'sink_local_hash_attention_top_k': 'HashAttention Top-k',
        'sink_local_hash_attention_top_k_adaptive_sampling': 'vAttention(hashattention)',
        'sink_local_oracle_top_p': 'Oracle Top-p'
    }
    
    # Pentagon markers for all methods
    marker_style = 'p'
    markersize = 12
    linewidth = 3
    
    # Track all methods for legend
    all_methods = set()
    
    # Plot quality data (top row)
    for col, dataset_name in enumerate(dataset_names):
        ax = axes[0, col]
        df = quality_datasets[dataset_name]
        
        # Plot each method
        for column in df.columns:
            if column == 'density':
                continue
            
            x = df['density']
            y = df[column]
            
            # Handle missing data
            mask = ~pd.isna(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) > 0:
                color = colors.get(column, '#000000')
                
                # For Full attention, use solid line with no markers
                if column == 'Full':
                    ax.plot(x_clean, y_clean, 
                           color=color, linewidth=linewidth, 
                           linestyle='-', label=column)
                else:
                    ax.plot(x_clean, y_clean, 
                           color=color, marker=marker_style, linewidth=linewidth, 
                           markersize=markersize, linestyle='--', label=column)
                
                all_methods.add(column)
        
        ax.set_xlabel('Density', fontsize=27)
        ax.set_ylabel('Quality', fontsize=27)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.grid(True, alpha=0.3)
        
        # Add dataset name as column title for top row
        ax.set_title(dataset_name, fontsize=30, fontweight='bold', pad=20)
        
        # Set axis limits with padding
        density_values = df['density'].dropna()
        if len(density_values) > 0:
            x_min = density_values.min() - 0.01
            x_max = density_values.max() + 0.01
            ax.set_xlim(x_min, x_max)
        
        # Format x-axis to show 2 decimal places (0.00 precision)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        numeric_data = df.select_dtypes(include=[np.number]).drop(columns=['density'])
        if not numeric_data.empty:
            y_min = numeric_data.min().min()
            y_max = numeric_data.max().max()
            y_range = y_max - y_min
            y_padding = y_range * 0.05 if y_range > 0 else 0.1
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Plot error data (bottom row)
    for col, dataset_name in enumerate(dataset_names):
        ax = axes[1, col]
        data = error_datasets[dataset_name]
        
        # Plot each method
        for method, points in data.items():
            densities: List[float] = [p[0] for p in points]
            errors: List[float] = [p[1] for p in points]
            
            # Map method name to quality data naming
            quality_method_name = error_to_quality_mapping.get(method, method)
            color = colors.get(quality_method_name, '#000000')
            
            ax.plot(densities, errors, 
                   color=color, marker=marker_style, linewidth=linewidth, 
                   markersize=markersize, linestyle='--', label=quality_method_name)
            
            all_methods.add(quality_method_name)
        
        ax.set_xlabel('Density', fontsize=27)
        ax.set_ylabel('Error', fontsize=27)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        if data:
            max_density = max([max([p[0] for p in points]) for points in data.values()])
            max_error = max([max([p[1] for p in points]) for points in data.values()])
            ax.set_xlim(0, max_density * 1.05)
            ax.set_ylim(0, max_error * 1.05)
    
    # Create legend at the top of the plot
    legend_methods = {
        'Full': ('Full', '#000000', '', '-'),  # Black, no marker, solid line
        'Oracle Top-k': ('Oracle Top-k', '#17becf', 'p', '--'),  # Pentagon, dotted
        'Oracle Top-p': ('Oracle Top-p', '#9467bd', 'p', '--'),  # Purple, pentagon, dotted
        'HashAttention Top-k': ('HashAttention Top-k', '#ff7f0e', 'p', '--'),  # Pentagon, dotted
        'vAttention(oracle-top-k)': ('vAttention(oracle-top-k)', '#1f77b4', 'p', '--'),  # Pentagon, dotted
        'vAttention(hashattention)': ('vAttention(hashattention)', '#8b0000', 'p', '--')  # Pentagon, dotted
    }
    
    legend_elements = []
    for method_name in ['Full', 'Oracle Top-k', 'Oracle Top-p', 'HashAttention Top-k', 
                       'vAttention(oracle-top-k)', 'vAttention(hashattention)']:
        if method_name in legend_methods:
            label, color, marker, linestyle = legend_methods[method_name]
            marker_size = 0 if marker == '' else markersize
            legend_elements.append(plt.Line2D([0], [0], color=color, marker=marker, 
                                            linewidth=linewidth, markersize=marker_size, 
                                            linestyle=linestyle, label=label))
    
    # Add legend at the top
    fig.legend(handles=legend_elements, 
              loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=len(legend_elements), fontsize=27, frameon=True,
              borderaxespad=0, columnspacing=1.5, handletextpad=0.8)
    
    # Adjust layout to make room for legend with reduced vertical spacing
    plt.subplots_adjust(hspace=0.15, wspace=0.25, top=0.90, bottom=0.1)
    
    # Save the plot
    plt.savefig('/data/apdesai/code/sparse-attention-hub/paper_plots/pareto/main_paper_pareto.png', 
                dpi=300, bbox_inches='tight')
    print("Main paper plot saved successfully as 'main_paper_pareto.png'")
    plt.close()


if __name__ == "__main__":
    print("Creating main paper pareto plot...")
    create_main_paper_plot()
    print("Main paper plot generation completed!")
