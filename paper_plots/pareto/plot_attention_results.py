#!/usr/bin/env python3
"""Plot attention method performance data.

This script creates a plot showing the relationship between density and error
for different attention methods.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def parse_data(data_name: str) -> Tuple[Dict[str, List[Tuple[float, float]]], str]:
    """Parse the raw data into organized format.
    
    Returns:
        Tuple containing:
        - Dict mapping method names to list of (density, error) tuples
        - String with the plot name for output file
    """
    # Raw data as provided
    shortdep_qa = """
    sink_local_oracle_top_k_adaptive_sampling,0.146384,0.019202
    sink_local_oracle_top_k_adaptive_sampling,0.048310,0.070139
    sink_local_hash_attention_top_k_adaptive_sampling,0.194078,0.079443
    sink_local_oracle_top_k,0.215946,0.042876
    sink_local_hash_attention_top_k_adaptive_sampling,0.099103,0.137432
    sink_local_hash_attention_top_k,0.215524,0.115667
    sink_local_hash_attention_top_k,0.065669,0.232989
    sink_local_oracle_top_k,0.115739,0.073275
    sink_local_oracle_top_k,0.165825,0.058167
    sink_local_oracle_top_k_adaptive_sampling,0.100929,0.027899
    sink_local_hash_attention_top_k,0.165527,0.136591
    sink_local_hash_attention_top_k_adaptive_sampling,0.141602,0.095972
    sink_local_hash_attention_top_k_adaptive_sampling,0.049462,0.259293
    sink_local_oracle_top_k,0.065675,0.110791
    sink_local_oracle_top_k_adaptive_sampling,0.189660,0.012876
    sink_local_hash_attention_top_k,0.115615,0.169794
    sink_local_oracle_top_p,0.046264,0.195296
    sink_local_oracle_top_p,0.088468,0.112782
    sink_local_oracle_top_p,0.128702,0.074335
    sink_local_oracle_top_p,0.203514,0.041692
    """
    shortdep_cloze = """
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
    hotpotqa = """
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

    multifield = """
    sink_local_oracle_top_k_adaptive_sampling,0.136479,0.044981
    sink_local_oracle_top_k_adaptive_sampling,0.068225,0.163068
    sink_local_hash_attention_top_k_adaptive_sampling,0.182615,0.100923
    sink_local_oracle_top_k,0.249299,0.049829
    sink_local_hash_attention_top_k_adaptive_sampling,0.095709,0.219421
    sink_local_hash_attention_top_k,0.248065,0.101054
    sink_local_hash_attention_top_k,0.099783,0.217910
    sink_local_oracle_top_k,0.145444,0.081787
    sink_local_oracle_top_k,0.197457,0.062538
    sink_local_oracle_top_k_adaptive_sampling,0.098285,0.069518
    sink_local_hash_attention_top_k,0.198146,0.126159
    sink_local_hash_attention_top_k_adaptive_sampling,0.141125,0.146414
    sink_local_hash_attention_top_k_adaptive_sampling,0.067695,0.376394
    sink_local_oracle_top_k,0.098179,0.121627
    sink_local_oracle_top_k_adaptive_sampling,0.190115,0.031201
    sink_local_hash_attention_top_k,0.150312,0.156784
    sink_local_oracle_top_p,0.057302,0.337931
    sink_local_oracle_top_p,0.085191,0.186692
    sink_local_oracle_top_p,0.141493,0.110598
    sink_local_oracle_top_p,0.186919,0.074550
    """

    ruler_qa_1 ="""
    sink_local_hash_attention_top_k,0.058163,0.235393
    sink_local_hash_attention_top_k,0.103137,0.170459
    sink_local_hash_attention_top_k,0.158154,0.130000
    sink_local_hash_attention_top_k,0.208165,0.105360
    sink_local_hash_attention_top_k_adaptive_sampling,0.050596,0.215980
    sink_local_hash_attention_top_k_adaptive_sampling,0.150443,0.086075
    sink_local_hash_attention_top_k_adaptive_sampling,0.202130,0.056033
    sink_local_oracle_top_k,0.058165,0.105583
    sink_local_oracle_top_k,0.103157,0.069357
    sink_local_oracle_top_k,0.158174,0.048120
    sink_local_oracle_top_k,0.208157,0.039362
    sink_local_oracle_top_k_adaptive_sampling,0.049701,0.043761
    sink_local_oracle_top_k_adaptive_sampling,0.097586,0.022588
    sink_local_oracle_top_k_adaptive_sampling,0.150255,0.013158
    sink_local_oracle_top_k_adaptive_sampling,0.191966,0.010042
    sink_local_oracle_top_p,0.051830,0.127600
    sink_local_oracle_top_p,0.098654,0.070521
    sink_local_oracle_top_p,0.127560,0.053414
    sink_local_oracle_top_p,0.179152,0.034741
    """

    ruler_qa_2 = """
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
    if data_name == 'ruler_qa_1':
        raw_data = ruler_qa_1
    elif data_name == 'ruler_qa_2':
        raw_data = ruler_qa_2
    elif data_name == 'shortdep_qa':
        raw_data = shortdep_qa
    elif data_name == 'shortdep_cloze':
        raw_data = shortdep_cloze
    elif data_name == 'hotpotqa':
        raw_data = hotpotqa
    elif data_name == 'multifield':
        raw_data = multifield
    else:
        raise ValueError(f"Invalid data name: {data_name}")

    plot_name = data_name
    data: Dict[str, List[Tuple[float, float]]] = {}
    
    for line in raw_data.strip().split('\n'):
        if line.strip():
            # Handle CSV format (comma-separated values)
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
    
    return data, plot_name

def parse_data_averaged(data_name: str) -> Tuple[Dict[str, List[Tuple[float, float]]], str]:
    if data_name == 'avg-ruler':
        data1 = parse_data('ruler_qa_1')[0]
        data2 = parse_data('ruler_qa_2')[0]
    elif data_name == 'avg-loogle':
        data1 = parse_data('shortdep_qa')[0]
        data2 = parse_data('shortdep_cloze')[0]
    elif data_name == 'avg-longbench':
        data1 = parse_data('hotpotqa')[0]
        data2 = parse_data('multifield')[0]
    else:
        raise ValueError(f"Invalid data name: {data_name}")
    
    

def create_plot(data_name: str) -> None:
    """Create and display the plot."""
    # Parse data
    result = parse_data(data_name)
    data: Dict[str, List[Tuple[float, float]]] = result[0]
    plot_name: str = result[1]
    
    # Legend mapping as specified - matching plot_pareto.py style
    legend_mapping: Dict[str, str] = {
        'sink_local_oracle_top_k': 'Oracle Top-k',
        'sink_local_oracle_top_k_adaptive_sampling': 'vAttention(oracle-top-k)',
        'sink_local_hash_attention_top_k': 'HashAttention Top-k',
        'sink_local_hash_attention_top_k_adaptive_sampling': 'vAttention(hashattention)',
        'sink_local_magic_pig': 'magic_pig',
        'sink_local_oracle_top_p': 'Oracle Top-p',
    }
    
    # Colors matching plot_pareto.py exactly
    color_mapping: Dict[str, str] = {
        'sink_local_oracle_top_k': '#17becf',  # cyan/light blue - matches Oracle Top-k
        'sink_local_oracle_top_k_adaptive_sampling': '#1f77b4',  # blue - matches vAttention(oracle-top-k)
        'sink_local_hash_attention_top_k': '#ff7f0e',  # orange - matches HashAttention Top-k
        'sink_local_hash_attention_top_k_adaptive_sampling': '#8b0000',  # dark red/maroon - matches vAttention(hashattention)
        'sink_local_magic_pig': '#2ca02c',  # green
        'sink_local_oracle_top_p': '#9467bd',  # purple - matches Oracle Top-p
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each method
    markers: List[str] = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, (method, points) in enumerate(data.items()):
        densities: List[float] = [p[0] for p in points]
        errors: List[float] = [p[1] for p in points]
        
        legend_name: str = legend_mapping.get(method, method)
        color: str = color_mapping.get(method, '#000000')  # default to black if not found
        
        # Plot with dotted lines and markers (7px = ~5pt markersize)
        ax.plot(densities, errors, 
                linestyle='--', 
                marker=markers[i % len(markers)], 
                markersize=7,
                linewidth=2,
                color=color,
                label=legend_name)
    
    # Create a descriptive title based on dataset name
    title_mapping: Dict[str, str] = {
        'ruler_qa_1': 'qa_1: Ruler 32K',
        'ruler_qa_2': 'qa_2: Ruler 32K', 
        'shortdep_qa': 'shortdep_qa: Loogle (16K)',
        'shortdep_cloze': 'shortdep_cloze: Loogle (16k)',
        'hotpotqa': 'hotpotqa: Longbench',
        'multifield': 'multifield: Longbench'
    }
    
    plot_title: str = title_mapping.get(data_name, f'{data_name}: Attention Method Performance')
    
    # Customize plot
    ax.set_xlabel('Density', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set reasonable axis limits
    ax.set_xlim(0, max([max([p[0] for p in points]) for points in data.values()]) * 1.05)
    ax.set_ylim(0, max([max([p[1] for p in points]) for points in data.values()]) * 1.05)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_name + '.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{plot_name}.png'")
    plt.show()

def create_combined_plot() -> None:
    """Create a combined plot with all datasets in a 2x3 grid."""
    # List of all datasets
    datasets: List[str] = ['ruler_qa_1', 'ruler_qa_2', 'shortdep_qa', 'shortdep_cloze', 'hotpotqa', 'multifield']
    
    # Create figure with subplots (2 rows, 3 columns) - larger figure for better readability
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()  # Flatten for easy indexing
    
    # Legend mapping as specified - matching plot_pareto.py style
    legend_mapping: Dict[str, str] = {
        'sink_local_oracle_top_k': 'Oracle Top-k',
        'sink_local_oracle_top_k_adaptive_sampling': 'vAttention(oracle-top-k)',
        'sink_local_hash_attention_top_k': 'HashAttention Top-k',
        'sink_local_hash_attention_top_k_adaptive_sampling': 'vAttention(hashattention)',
        'sink_local_magic_pig': 'magic_pig',
        'sink_local_oracle_top_p': 'Oracle Top-p'
    }
    
    # Colors matching plot_pareto.py exactly
    color_mapping: Dict[str, str] = {
        'sink_local_oracle_top_k': '#17becf',  # cyan/light blue - matches Oracle Top-k
        'sink_local_oracle_top_k_adaptive_sampling': '#1f77b4',  # blue - matches vAttention(oracle-top-k)
        'sink_local_hash_attention_top_k': '#ff7f0e',  # orange - matches HashAttention Top-k
        'sink_local_hash_attention_top_k_adaptive_sampling': '#8b0000',  # dark red/maroon - matches vAttention(hashattention)
        'sink_local_magic_pig': '#2ca02c',  # green
        'sink_local_oracle_top_p': '#9467bd',  # purple - matches Oracle Top-p
    }
    
    # Title mapping
    title_mapping: Dict[str, str] = {
        'ruler_qa_1': 'QA_1 RULER 32K',
        'ruler_qa_2': 'QA_2 RULER 32K', 
        'shortdep_qa': 'ShortDep QA Loogle',
        'shortdep_cloze': 'ShortDep Cloze Loogle',
        'hotpotqa': 'HotpotQA LongBench',
        'multifield': 'MultiFieldQA LongBench'
    }
    
    # Use pentagon markers for all methods like in plot_pareto.py
    marker_style: str = 'p'  # Pentagon marker for all methods
    
    # Plot each dataset
    for idx, data_name in enumerate(datasets):
        ax = axes[idx]
        
        # Parse data for this dataset
        result = parse_data(data_name)
        data: Dict[str, List[Tuple[float, float]]] = result[0]
        
        # Plot each method
        for method, points in data.items():
            densities: List[float] = [p[0] for p in points]
            errors: List[float] = [p[1] for p in points]
            
            legend_name: str = legend_mapping.get(method, method)
            color: str = color_mapping.get(method, '#000000')  # default to black if not found
            
            # Plot with dotted lines and pentagon markers (matching plot_pareto.py style)
            ax.plot(densities, errors, 
                    linestyle='--', 
                    marker=marker_style, 
                    markersize=10,  # Larger markers like plot_pareto.py
                    linewidth=3,    # Thicker lines like plot_pareto.py
                    color=color,
                    label=legend_name)
        
        # Customize subplot with larger fonts
        ax.set_xlabel('Density', fontsize=16)  # Increased from 10
        ax.set_ylabel('Error', fontsize=16)    # Increased from 10
        ax.set_title(title_mapping.get(data_name, data_name), fontsize=18, fontweight='bold')  # Increased from 12, added bold
        ax.tick_params(axis='both', which='major', labelsize=14)  # Larger tick labels
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        if data:
            max_density = max([max([p[0] for p in points]) for points in data.values()])
            max_error = max([max([p[1] for p in points]) for points in data.values()])
            ax.set_xlim(0, max_density * 1.05)
            ax.set_ylim(0, max_error * 1.05)
    
    # Add a global legend with better styling
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
              loc='lower center', bbox_to_anchor=(0.5, 0.02),
              ncol=min(4, len(handles)), fontsize=16,  # Larger legend font, max 4 columns
              frameon=True, borderaxespad=0, 
              columnspacing=1.5, handletextpad=0.8)
    
    # Overall title with larger font
    fig.suptitle('Sparse Attention Methods Performance Comparison', 
                fontsize=24, fontweight='bold', y=0.95)
    
    # Use tight layout with proper spacing for legend (matching plot_pareto.py)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    # Save the combined plot
    plt.savefig('combined_attention_results.png', dpi=300, bbox_inches='tight')
    print("Combined plot saved as 'combined_attention_results.png'")
    plt.show()

if __name__ == "__main__":
    print("Starting plot generation...")
    # Create the combined plot with all datasets in a 2x3 grid
    create_combined_plot()
    print("Plot generation completed!")
    
    # Uncomment below lines if you want individual plots as well
    # create_plot('ruler_qa_1')
    # create_plot('ruler_qa_2')
    # create_plot('shortdep_qa')
    # create_plot('shortdep_cloze')
    # create_plot('hotpotqa')
    # create_plot('multifield')
