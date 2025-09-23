import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def parse_data() -> Dict[str, pd.DataFrame]:
    """Parse the embedded data and return structured datasets."""
    datasets = {}
    
    # QA_1 RULER 32K (first dataset)
    qa1_data1 = {
        'density': [0.05, 0.1, 0.15, 0.2],
        'Full': [80.5, 80.5, 80.5, 80.5],
        'Oracle Top-k': [73.5, 73.5, 75, 75],
        'Oracle Top-p': [78.5, 77.0, 78.5, 78.0],
        'HashAttention Top-k': [68, 73, 75.5, 76],
        'vAttention(oracle-top-k)': [77.5, 79.5, 80.5, 79],
        'vAttention(hashattention)': [67.5, 76, 78, 77]
    }
    datasets['QA_1 RULER 32K'] = pd.DataFrame(qa1_data1)
    
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
    
    # MultiFieldQA LongBench
    multifieldqa_data = {
        'density': [0.1, 0.15, 0.2],
        'Full': [54.25, 54.25, 54.25],
        'Oracle Top-k': [52.95, 52.59, 53.22],
        'Oracle Top-p': [54.38, 53.97, 54.19],
        'HashAttention Top-k': [55.35, 53.79, 52.95],
        'vAttention(oracle-top-k)': [54.46, 53.83, 54.09],
        'vAttention(hashattention)': [55.35, 55.43, 53.25]
    }
    datasets['MultiFieldQA LongBench'] = pd.DataFrame(multifieldqa_data)
    
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
        'HashAttention': [0.4455, 0.4834, 0.5095, 0.5213],
        'vAttention(oracle-top-k)': [0.5332, 0.5498, 0.5498, 0.5427],
        'vAttention(hashattention)': [0.4455, 0.4692, 0.5095, 0.5213]
    }
    datasets['ShortDep Cloze Loogle'] = pd.DataFrame(shortdep_cloze_data)
    
    # ShortDep QA Loogle
    shortdep_qa_data = {
        'density': [0.05, 0.1, 0.15, 0.2],
        'Full': [0.8771018982, 0.8771018982, 0.8771018982, 0.8771018982],
        'HashAttention': [0.8727242351, 0.8725495934, 0.872217536, 0.8721134663],
        'Oracle Top-k': [0.8736855388, 0.8763585091, 0.8757869601, 0.8744517565],
        'Oracle Top-p': [0.8760424852, 0.8794152141, 0.8785768747, 0.8788005114],
        'vAttention(hashattention)': [0.8727242351, 0.8782834411, 0.8739673495, 0.8750917315],
        'vAttention(oracle-top-k)': [0.8767986894, 0.8764945269, 0.8764416575, 0.8771838546]
    }
    datasets['ShortDep QA Loogle'] = pd.DataFrame(shortdep_qa_data)
    
    return datasets

def plot_quality_vs_density():
    """Create a 2x3 grid plot showing quality vs density for all baselines."""
    datasets = parse_data()
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Quality vs Density Across Different Datasets', fontsize=24, fontweight='bold', y=0.99)  # Move title higher
    
    # Reduce spacing between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.25)  # Tighter subplot spacing
    
    # Define colors matching plot_attention_results.py
    colors = {
        'Full': '#000000',  # black for Full attention (no markers)
        'Oracle Top-k': '#17becf',  # cyan/light blue (matching sink_local_oracle_top_k)
        'Oracle Top-p': '#9467bd',  # purple (matching sink_local_oracle_top_p)
        'HashAttention Top-k': '#ff7f0e',  # orange (matching sink_local_hash_attention_top_k)
        'HashAttention': '#ff7f0e',  # same as HashAttention Top-k
        'vAttention(oracle-top-k)': '#1f77b4',  # blue (matching sink_local_oracle_top_k_adaptive_sampling)
        'vAttention(hashattention)': '#8b0000'  # dark red/maroon (matching sink_local_hash_attention_top_k_adaptive_sampling)
    }
    
    # Define markers for different methods (pentagon for most, none for Full)
    markers = {
        'Full': '',  # No marker for Full
        'Oracle Top-k': 'p',  # Pentagon
        'Oracle Top-p': 'p',  # Pentagon
        'HashAttention Top-k': 'p',  # Pentagon
        'HashAttention': 'p',  # Pentagon
        'vAttention(oracle-top-k)': 'p',  # Pentagon
        'vAttention(hashattention)': 'p'  # Pentagon
    }
    
    # Collect unique methods across all datasets for consistent legend
    all_methods = set()
    for df in datasets.values():
        for col in df.columns:
            if col != 'density':
                all_methods.add(col)
    
    # Plot each dataset
    for i, (dataset_name, df) in enumerate(datasets.items()):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
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
                marker = markers.get(column, 'o')
                
                # Use dotted lines for all methods
                linestyle = '--'
                markersize = 0 if marker == '' else 10
                
                ax.plot(x_clean, y_clean, 
                       color=color, marker=marker, linewidth=3, 
                       markersize=markersize, linestyle=linestyle, label=column)
                
                # Add point labels for all methods except Full
                if column != 'Full':
                    for x_val, y_val in zip(x_clean, y_clean):
                        ax.annotate(f'{y_val:.2f}', 
                                  xy=(x_val, y_val), 
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=10, color=color, fontweight='bold',
                                  ha='left', va='bottom')
        
        ax.set_title(dataset_name, fontsize=18, fontweight='bold')
        ax.set_xlabel('Density', fontsize=16)
        ax.set_ylabel('Quality', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits based on actual density range for this dataset
        density_values = df['density'].dropna()
        if len(density_values) > 0:
            x_min = density_values.min() - 0.01
            x_max = density_values.max() + 0.01
            ax.set_xlim(x_min, x_max)
        
        # Set y-axis limits based on actual data range for this dataset with padding
        numeric_data = df.select_dtypes(include=[np.number]).drop(columns=['density'])
        if not numeric_data.empty:
            y_min = numeric_data.min().min()
            y_max = numeric_data.max().max()
            y_range = y_max - y_min
            y_padding = y_range * 0.05 if y_range > 0 else 0.1
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Create legend with consistent methods - combine similar methods for cleaner legend
    legend_methods = {
        'Full': ('Full', '#000000', '', '--'),  # Black, no marker, dotted line
        'Oracle Top-k': ('Oracle Top-k', '#17becf', 'p', '--'),  # Pentagon, dotted
        'Oracle Top-p': ('Oracle Top-p', '#9467bd', 'p', '--'),  # Purple, pentagon, dotted
        'HashAttention': ('HashAttention', '#ff7f0e', 'p', '--'),  # Pentagon, dotted
        'vAttention(oracle-top-k)': ('vAttention(oracle-top-k)', '#1f77b4', 'p', '--'),  # Pentagon, dotted
        'vAttention(hashattention)': ('vAttention(hashattention)', '#8b0000', 'p', '--')  # Pentagon, dotted
    }
    
    legend_elements = []
    for method_name, (label, color, marker, linestyle) in legend_methods.items():
        markersize = 0 if marker == '' else 10
        legend_elements.append(plt.Line2D([0], [0], color=color, marker=marker, 
                                        linewidth=3, markersize=markersize, 
                                        linestyle=linestyle, label=label))
    
    # Add single legend at the bottom with minimal whitespace - single line
    fig.legend(handles=legend_elements, 
              loc='lower center', bbox_to_anchor=(0.5, 0.005),  # Move legend closer to bottom
              ncol=len(legend_elements), fontsize=16, frameon=True,  # All items in single line
              borderaxespad=0, columnspacing=1.0, handletextpad=0.5)  # Tighter spacing
    
    # Use very tight layout with minimal margins
    plt.tight_layout(rect=[0, 0.05, 1, 0.97], pad=0.5)  # Reduced margins like combined plot
    
    # Save the plot
    plt.savefig('quality_vs_density_grid.png', 
                dpi=300, bbox_inches='tight')
    print("Plot saved successfully as 'quality_vs_density_grid.png'")
    plt.close()

def plot_average_quality_vs_density():
    """Create a plot showing average quality vs density across all datasets except MultiFieldQA."""
    datasets = parse_data()
    
    # Exclude MultiFieldQA LongBench
    datasets_to_average = {k: v for k, v in datasets.items() if k != 'MultiFieldQA LongBench'}
    
    # Get all unique densities and methods
    all_densities = set()
    all_methods = set()
    
    for df in datasets_to_average.values():
        all_densities.update(df['density'].dropna())
        for col in df.columns:
            if col != 'density':
                # Normalize method names (handle HashAttention variants)
                normalized_method = col
                if col == 'HashAttention':
                    normalized_method = 'HashAttention Top-k'
                all_methods.add(normalized_method)
    
    all_densities = sorted(all_densities)
    
    # Calculate averages for each method and density
    averaged_data = {'density': all_densities}
    
    for method in all_methods:
        method_averages = []
        
        for density in all_densities:
            values = []
            
            for dataset_name, df in datasets_to_average.items():
                # Check if this density exists in this dataset
                density_mask = df['density'] == density
                if not density_mask.any():
                    continue
                
                # Find the method column (handle naming variations)
                method_col = None
                if method in df.columns:
                    method_col = method
                elif method == 'HashAttention Top-k' and 'HashAttention' in df.columns:
                    method_col = 'HashAttention'
                
                if method_col is not None:
                    value = df.loc[density_mask, method_col].iloc[0]
                    if pd.notna(value):
                        values.append(value)
            
            # Calculate average if we have values
            if values:
                method_averages.append(np.mean(values))
            else:
                method_averages.append(np.nan)
        
        averaged_data[method] = method_averages
    
    # Create DataFrame with averaged data
    avg_df = pd.DataFrame(averaged_data)
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors matching original plot
    colors = {
        'Full': '#000000',  # black for Full attention (no markers)
        'Oracle Top-k': '#17becf',  # cyan/light blue
        'Oracle Top-p': '#9467bd',  # purple
        'HashAttention Top-k': '#ff7f0e',  # orange
        'vAttention(oracle-top-k)': '#1f77b4',  # blue
        'vAttention(hashattention)': '#8b0000'  # dark red/maroon
    }
    
    # Define markers for scatter plot
    markers = {
        'Full': 'o',  # Circle for Full
        'Oracle Top-k': 's',  # Square
        'Oracle Top-p': '^',  # Triangle up
        'HashAttention Top-k': 'D',  # Diamond
        'vAttention(oracle-top-k)': 'v',  # Triangle down
        'vAttention(hashattention)': 'p'  # Pentagon
    }
    
    # Plot each method
    for column in avg_df.columns:
        if column == 'density':
            continue
        
        if column == 'Full':
            # For Full method, place a single point at inverse fraction 1/1.0 = 1.0
            # Use the average of all Full values
            full_values = avg_df[column].dropna()
            if len(full_values) > 0:
                y_val = full_values.mean()
                color = colors.get(column, '#000000')
                marker = markers.get(column, 'o')
                markersize = 12
                
                ax.scatter([1.0], [y_val], 
                          color=color, marker=marker, s=markersize**2, 
                          label=column, zorder=5)
                
                # Add label for Full
                ax.annotate(f'{y_val:.2f}', 
                          xy=(1.0, y_val), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=18, color=color, fontweight='bold',
                          ha='left', va='bottom')
        else:
            # For other methods, use scatter points at inverse of density values (1/density)
            x = avg_df['density']
            y = avg_df[column]
            
            # Handle missing data
            mask = ~pd.isna(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) > 0:
                # Calculate inverse of density values for KV Cache movement reduction
                x_inverse = 1.0 / x_clean
                
                color = colors.get(column, '#000000')
                marker = markers.get(column, 'o')
                markersize = 10
                
                ax.scatter(x_inverse, y_clean, 
                          color=color, marker=marker, s=markersize**2, 
                          label=column, zorder=5)
                
                # Add point labels
                for x_val, y_val in zip(x_inverse, y_clean):
                    ax.annotate(f'{y_val:.2f}', 
                              xy=(x_val, y_val), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=18, color=color, fontweight='bold',
                              ha='left', va='bottom')
    
    ax.set_xlabel('KV Cache movement reduction', fontsize=18)
    ax.set_ylabel('Average Quality', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.3)
    
    # Set custom x-axis ticks and labels
    if len(all_densities) > 0:
        # Calculate all unique inverse density values
        inverse_densities = [1.0 / d for d in all_densities]
        inverse_densities.append(1.0)  # Add 1.0 for Full method
        inverse_densities = sorted(set(inverse_densities))
        
        # Create tick positions and labels
        tick_positions = []
        tick_labels = []
        
        for inv_val in inverse_densities:
            tick_positions.append(inv_val)
            if inv_val == 1.0:
                tick_labels.append('1x')
            else:
                # Round to nearest integer for cleaner labels
                rounded_val = round(inv_val)
                tick_labels.append(f'{rounded_val}x')
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    
    # Set x-axis limits to include both inverse density values and 1.0 for Full
    if len(all_densities) > 0:
        inverse_densities = [1.0 / d for d in all_densities]
        all_inverse_values = inverse_densities + [1.0]  # Include Full method
        x_min = min(all_inverse_values) * 0.9  # Add 10% padding below minimum
        x_max = max(all_inverse_values) * 1.1  # Add 10% padding above maximum
        ax.set_xlim(x_min, x_max)
    
    # Set y-axis limits based on data range with padding
    numeric_data = avg_df.select_dtypes(include=[np.number]).drop(columns=['density'])
    if not numeric_data.empty:
        y_min = numeric_data.min().min()
        y_max = numeric_data.max().max()
        y_range = y_max - y_min
        y_padding = y_range * 0.05 if y_range > 0 else 0.1
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add legend
    ax.legend(fontsize=18, loc='best')
    
    # Use tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('/data/apdesai/code/sparse-attention-hub/paper_plots/pareto/average.png', 
                dpi=300, bbox_inches='tight')
    print("Average plot saved successfully as 'average.png'")
    plt.close()

if __name__ == "__main__":
    plot_quality_vs_density()
    plot_average_quality_vs_density()
