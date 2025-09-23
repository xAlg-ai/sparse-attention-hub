#!/usr/bin/env python3
"""Generate average density and error plots for all examples.

This script creates plots showing density and error vs sequence length
(averaged across all layers) for each example in the dataset, and saves
them to the average_plots/ folder.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
from data_analysis import load_and_split_experiment_data, plot_example_metrics
from pathlib import Path

def generate_all_average_plots(data_file: str = "extracted_aime_vatt_data", 
                              output_dir: str = "average_plots") -> None:
    """Generate average plots for all examples in the dataset.
    
    Args:
        data_file: Path to the experiment data file
        output_dir: Directory to save the plots
    """
    # Ensure output directory exists
    output_path: Path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=== Generating Average Plots for All Examples ===")
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load and split the data
    print("Loading experiment data...")
    df, examples, stats = load_and_split_experiment_data(data_file)
    
    print(f"Loaded {stats['num_examples']} examples")
    print(f"Total data points: {stats['total_rows']:,}")
    print()
    
    # Generate plots for each example
    for example_id in range(len(examples)):
        example_size: int = len(examples[example_id])
        sequence_length: int = example_size // 32  # Each sequence position has 32 layers
        
        print(f"Generating plots for Example {example_id}:")
        print(f"  - Data points: {example_size:,}")
        print(f"  - Sequence length: {sequence_length:,}")
        
        # Generate plot filename
        plot_filename: str = f"example_{example_id}_average_density_error.png"
        plot_path: str = str(output_path / plot_filename)
        
        try:
            # Generate the plot (both density and error, averaged across layers)
            plot_example_metrics(
                examples=examples,
                example_id=example_id,
                metric='both',
                layer_mode='average',
                figsize=(15, 10),
                save_path=plot_path
            )
            
            print(f"  ✓ Saved: {plot_filename}")
            
        except Exception as e:
            print(f"  ✗ Error generating plot for Example {example_id}: {e}")
        
        print()
    
    print("=== Plot Generation Complete ===")
    print(f"All plots saved in: {output_dir}/")
    
    # Generate a summary file
    summary_path: str = str(output_path / "plot_summary.txt")
    generate_summary_file(examples, stats, summary_path)
    print(f"Summary saved in: plot_summary.txt")


def generate_summary_file(examples, stats: dict, summary_path: str) -> None:
    """Generate a summary file with information about all examples and plots.
    
    Args:
        examples: List of example DataFrames
        stats: Statistics dictionary from analyze_example_stats
        summary_path: Path to save the summary file
    """
    with open(summary_path, 'w') as f:
        f.write("AIME VATT Data - Average Plots Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Dataset Overview:\n")
        f.write(f"- Total examples: {stats['num_examples']}\n")
        f.write(f"- Total data points: {stats['total_rows']:,}\n")
        f.write(f"- Average points per example: {stats['avg_rows_per_example']:.0f}\n")
        f.write(f"- Layers per example: 32 (0-31)\n\n")
        
        f.write("Example Details:\n")
        f.write("-" * 30 + "\n")
        
        for i, example in enumerate(examples):
            example_size = len(example)
            sequence_length = example_size // 32
            density_range = f"{example['density'].min():.6f} - {example['density'].max():.6f}"
            error_range = f"{example['error'].min():.6f} - {example['error'].max():.6f}"
            
            f.write(f"Example {i}:\n")
            f.write(f"  Data points: {example_size:,}\n")
            f.write(f"  Sequence length: {sequence_length:,}\n")
            f.write(f"  Density range: {density_range}\n")
            f.write(f"  Error range: {error_range}\n")
            f.write(f"  Plot file: example_{i}_average_density_error.png\n\n")
        
        f.write("Plot Description:\n")
        f.write("-" * 20 + "\n")
        f.write("Each plot shows two subplots:\n")
        f.write("1. Top: Average density vs sequence position\n")
        f.write("2. Bottom: Average error vs sequence position\n")
        f.write("\nAll values are averaged across the 32 layers (0-31) at each sequence position.\n")
        f.write("Sequence position represents the progression through the generation process.\n")


def generate_comparison_plot(examples, stats: dict, output_dir: str = "average_plots") -> None:
    """Generate a comparison plot showing all examples on the same axes.
    
    Args:
        examples: List of example DataFrames
        stats: Statistics dictionary
        output_dir: Directory to save the plot
    """
    output_path: Path = Path(output_dir)
    
    print("Generating comparison plots...")
    
    try:
        # Import the comparison function
        from data_analysis import plot_multiple_examples_comparison
        
        # Generate density comparison
        density_comparison_path: str = str(output_path / "all_examples_density_comparison.png")
        plot_multiple_examples_comparison(
            examples=examples,
            example_ids=list(range(len(examples))),
            metric='density',
            layer_mode='average',
            figsize=(15, 8),
            save_path=density_comparison_path
        )
        print(f"  ✓ Saved: all_examples_density_comparison.png")
        
        # Generate error comparison  
        error_comparison_path: str = str(output_path / "all_examples_error_comparison.png")
        plot_multiple_examples_comparison(
            examples=examples,
            example_ids=list(range(len(examples))),
            metric='error', 
            layer_mode='average',
            figsize=(15, 8),
            save_path=error_comparison_path
        )
        print(f"  ✓ Saved: all_examples_error_comparison.png")
        
    except Exception as e:
        print(f"  ✗ Error generating comparison plots: {e}")


def main() -> None:
    """Main function to generate all plots."""
    try:
        # Generate individual example plots
        generate_all_average_plots()
        
        # Load data again for comparison plots
        df, examples, stats = load_and_split_experiment_data("extracted_aime_vatt_data")
        
        # Generate comparison plots
        generate_comparison_plot(examples, stats)
        
        print("\n🎉 All plots generated successfully!")
        print("📁 Check the average_plots/ folder for your plots.")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
