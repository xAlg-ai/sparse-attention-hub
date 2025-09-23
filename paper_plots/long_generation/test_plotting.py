#!/usr/bin/env python3
"""Demo script showing how to use the data analysis and plotting functionality.

This script demonstrates all the available plotting options for analyzing
the extracted AIME VATT experiment data.
"""

from data_analysis import load_and_split_experiment_data, plot_example_metrics, plot_multiple_examples_comparison

def main() -> None:
    """Demonstrate the plotting functionality with various examples."""
    
    # Path to the data file (relative to script location)
    data_file: str = "extracted_aime_vatt_data"
    
    print("=== AIME VATT Data Analysis Demo ===")
    print("Loading and analyzing experiment data...")
    
    try:
        # Load and split the data
        df, examples, stats = load_and_split_experiment_data(data_file)
        
        print(f"\nDataset Summary:")
        print(f"- Total examples: {stats['num_examples']}")
        print(f"- Total data points: {stats['total_rows']:,}")
        print(f"- Average points per example: {stats['avg_rows_per_example']:.0f}")
        print(f"- Layers per example: 32 (0-31)")
        
        print(f"\nExample sizes:")
        for i, example in enumerate(examples):
            print(f"  Example {i}: {len(example):,} data points")
        
        print("\n" + "="*60)
        print("Available Plotting Functions:")
        print("="*60)
        
        print("\n1. plot_example_metrics() - Main plotting function")
        print("   Parameters:")
        print("   - example_id: Which example to plot (0, 1, 2, ...)")
        print("   - metric: 'density', 'error', or 'both'")
        print("   - layer_mode: 'average', 'layer_wise', or 'specific'")
        print("   - specific_layers: List of layers when layer_mode='specific'")
        print("   - save_path: Optional path to save the plot")
        
        print("\n2. plot_multiple_examples_comparison() - Compare across examples")
        print("   Parameters:")
        print("   - example_ids: List of example IDs to compare")
        print("   - metric: 'density' or 'error'")
        print("   - layer_mode: 'average' (currently only option)")
        
        print("\n" + "="*60)
        print("Usage Examples:")
        print("="*60)
        
        print("\n# Load the data")
        print("df, examples, stats = load_and_split_experiment_data('extracted_aime_vatt_data')")
        
        print("\n# Plot density and error for Example 0 (averaged across all layers)")
        print("plot_example_metrics(examples, example_id=0, metric='both', layer_mode='average')")
        
        print("\n# Plot density for Example 1 (specific layers)")
        print("plot_example_metrics(examples, example_id=1, metric='density',")
        print("                   layer_mode='specific', specific_layers=[0, 15, 31])")
        
        print("\n# Plot error for Example 0 (all layers separately)")
        print("plot_example_metrics(examples, example_id=0, metric='error', layer_mode='layer_wise')")
        
        print("\n# Compare density across multiple examples")
        print("plot_multiple_examples_comparison(examples, example_ids=[0, 1, 2], metric='density')")
        
        print("\n# Save plot to file")
        print("plot_example_metrics(examples, example_id=0, metric='density',")
        print("                   layer_mode='average', save_path='my_plot.png')")
        
        print("\n" + "="*60)
        print("Layer Modes Explained:")
        print("="*60)
        print("- 'average': Show average across all 32 layers")
        print("- 'layer_wise': Show all 32 layers as separate lines")
        print("- 'specific': Show only specified layers (e.g., [0, 15, 31])")
        
        print("\n" + "="*60)
        print("Interactive Example:")
        print("="*60)
        
        # Simple interactive example
        print("\nWould you like to see a demo plot? (y/n)")
        try:
            response = input().lower().strip()
            if response == 'y' or response == 'yes':
                print("\nCreating demo plot: Example 0 density (averaged across layers)...")
                plot_example_metrics(examples, example_id=0, metric='density', layer_mode='average')
                print("Demo plot created! Close the plot window to continue.")
        except KeyboardInterrupt:
            print("\nDemo cancelled by user.")
        except:
            print("\nSkipping interactive demo.")
        
        print(f"\nData analysis module ready for use!")
        print(f"All functions are available in the data_analysis.py module.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
