import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def flatten_results(data: dict) -> pd.DataFrame:
    """Converts the possibly nested results dictionary into a fully flat pandas DataFrame."""
    records = []
    for config_name, results in data.items():
        if "error" in results:
            continue

        perf = results.get('performance', {})
        timing = perf.get('timing', {})
        memory = perf.get('memory', {})
        
        record = {
            'config': config_name,
            'GPU Runtime (s)': timing.get('benchmark_gpu_runtime_ms', 0) / 1000,
            'Peak GPU Memory (MB)': memory.get('peak_benchmark_torch_gpu_mem_mb', 0),
            'CPU Memory (MB)': memory.get('model_load_cpu_mem_mb', 0),
        }

        def flatten_dict_recursive(d: dict, parent_key: str = '', sep: str = '_'):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict_recursive(v, new_key, sep=sep))
                else:
                    items.append((new_key, v))
            return items

        bench_res = results.get('benchmark_results', {})
        if bench_res:
            for key, value in flatten_dict_recursive(bench_res):
                record[key] = value
        
        records.append(record)
        
    return pd.DataFrame(records)

def plot_grid_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Generates a single 1x3 plot comparing the three most important metrics.
    """
    if df.empty:
        print("DataFrame is empty. No plots will be generated.")
        return

    base_metrics = ['config', 'GPU Runtime (s)', 'Peak GPU Memory (MB)', 'CPU Memory (MB)']
    benchmark_metric_name = 'Not Found'
    for col in df.columns:
        if col not in base_metrics and pd.api.types.is_numeric_dtype(df[col]):
            benchmark_metric_name = col
            break

    metrics_to_plot = ['GPU Runtime (s)', 'Peak GPU Memory (MB)', benchmark_metric_name]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    fig.suptitle('Benchmark Comparison Across Key Metrics', fontsize=20, weight='bold')

    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'Metric "{metric}"\nnot found in results', ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sns.barplot(ax=ax, data=df, x='config', y=metric, hue='config', palette='viridis', legend=False)
        ax.set_title(metric, fontsize=16, weight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric.split(' ')[-1], fontsize=12)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis='x', labelsize=11)
        
        for p in ax.patches:
            label = f'{p.get_height():.1f}' if p.get_height() >= 10 else f'{p.get_height():.3f}'
            ax.annotate(label,
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plot_filename = output_dir / "plot_grid_comparison.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved grid plot: {plot_filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from a JSON file.")
    parser.add_argument(
        "json_file", 
        type=Path,
        help="Path to the benchmark_results.json file."
    )
    args = parser.parse_args()

    if not args.json_file.is_file():
        print(f"Error: File not found at {args.json_file}")
        return

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    df = flatten_results(data)
    plot_grid_comparison(df, args.json_file.parent)

if __name__ == "__main__":
    main()