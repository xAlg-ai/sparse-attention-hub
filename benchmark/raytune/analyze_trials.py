#!/usr/bin/env python3
"""
Utility script to analyze Ray Tune trial results from Phase 1.

This script demonstrates how to access and analyze the metadata from Ray trials
for post-analysis purposes.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_trial_data(optimal_configs_dir: Path):
    """Load all trial data from the optimal configs directory."""
    all_trials = []
    
    # Find all trial JSON files
    trial_files = list(optimal_configs_dir.glob("*_trials.json"))
    
    for trial_file in trial_files:
        with open(trial_file, 'r') as f:
            data = json.load(f)
            
        # Add metadata to each trial
        for trial in data['trials']:
            trial['model'] = data['model']
            trial['task'] = data['task']
            trial['masker_name'] = data['masker_name']
            trial['objective_function'] = data['objective_function']
            trial['is_best'] = trial['trial_id'] == data['best_trial_id']
            
        all_trials.extend(data['trials'])
        
        # Also check if CSV exists
        csv_path = Path(data.get('analysis_dataframe_path', ''))
        if csv_path.exists():
            print(f"  → Found detailed analysis CSV: {csv_path}")
    
    return pd.DataFrame(all_trials)


def analyze_objective_performance(df: pd.DataFrame):
    """Analyze performance across different objective functions."""
    print("\n" + "="*60)
    print("OBJECTIVE FUNCTION ANALYSIS")
    print("="*60)
    
    # Group by objective function
    obj_stats = df.groupby('objective_function')['score'].agg(['mean', 'min', 'max', 'count'])
    print("\nScore statistics by objective function:")
    print(obj_stats)
    
    # Best trials only
    best_trials = df[df['is_best']]
    best_by_obj = best_trials.groupby('objective_function')['score'].agg(['mean', 'count'])
    print("\nBest trial scores by objective function:")
    print(best_by_obj)


def analyze_hyperparameter_impact(df: pd.DataFrame):
    """Analyze impact of different hyperparameters on scores."""
    print("\n" + "="*60)
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print("="*60)
    
    # Extract hyperparameters from config
    hyperparam_cols = []
    for idx, row in df.iterrows():
        config = row['config']
        for key, value in config.items():
            if key not in hyperparam_cols:
                hyperparam_cols.append(key)
                df.loc[idx, f'hp_{key}'] = value
            else:
                df.loc[idx, f'hp_{key}'] = value
    
    # Analyze each hyperparameter's impact
    for hp in hyperparam_cols:
        hp_col = f'hp_{hp}'
        if hp_col in df.columns:
            print(f"\nImpact of {hp}:")
            hp_stats = df.groupby(hp_col)['score'].agg(['mean', 'count', 'std'])
            print(hp_stats.sort_values('mean').head(10))


def analyze_sparsity_achievement(optimal_configs_dir: Path):
    """Analyze how well different configs achieve target sparsity."""
    print("\n" + "="*60)
    print("SPARSITY ACHIEVEMENT ANALYSIS")
    print("="*60)
    
    # Load optimal configs to get actual densities
    config_files = list(optimal_configs_dir.glob("*.json"))
    config_files = [f for f in config_files if not f.name.endswith("_trials.json")]
    
    sparsity_data = []
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        if 'score' in config:
            sparsity_data.append({
                'model': config['model'],
                'task': config['task'],
                'masker_name': config['masker_name'],
                'score': config['score'],
                'num_trials': config.get('num_trials', 0)
            })
    
    if sparsity_data:
        sparsity_df = pd.DataFrame(sparsity_data)
        print("\nConfiguration performance summary:")
        print(sparsity_df.groupby('masker_name')['score'].agg(['mean', 'min', 'max', 'count']))


def plot_trial_scores(df: pd.DataFrame, output_dir: Path):
    """Create visualizations of trial scores."""
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Score distribution by objective function
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='objective_function', y='score')
    plt.xticks(rotation=45)
    plt.title('Score Distribution by Objective Function')
    plt.tight_layout()
    plt.savefig(output_dir / 'scores_by_objective.png')
    plt.close()
    
    # Plot 2: Score vs trial for each task
    tasks = df['task'].unique()
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10, 4*len(tasks)))
    if len(tasks) == 1:
        axes = [axes]
    
    for ax, task in zip(axes, tasks):
        task_df = df[df['task'] == task]
        for masker in task_df['masker_name'].unique():
            masker_df = task_df[task_df['masker_name'] == masker]
            ax.scatter(range(len(masker_df)), masker_df['score'], label=masker[:20], alpha=0.6)
        ax.set_title(f'Trial Scores for {task}')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trial_progression.png')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Ray Tune trial results")
    parser.add_argument("--optimal-configs-dir", default="./optimal_configs",
                       help="Directory containing optimal configs and trial data")
    parser.add_argument("--output-dir", default="./trial_analysis",
                       help="Directory for output plots and analysis")
    parser.add_argument("--run", type=str,
                       help="Specific run directory to analyze (e.g., 'run_20240315_143022')")
    args = parser.parse_args()
    
    base_optimal_configs_dir = Path(args.optimal_configs_dir)
    output_dir = Path(args.output_dir)
    
    if not base_optimal_configs_dir.exists():
        print(f"Error: Directory {base_optimal_configs_dir} does not exist")
        return
    
    # Handle timestamped directories
    if args.run:
        optimal_configs_dir = base_optimal_configs_dir / args.run
        if not optimal_configs_dir.exists():
            print(f"Error: Specified run {optimal_configs_dir} does not exist")
            return
    else:
        # Find the most recent run_* directory
        run_dirs = sorted([d for d in base_optimal_configs_dir.glob("run_*") if d.is_dir()])
        if run_dirs:
            optimal_configs_dir = run_dirs[-1]  # Most recent
            print(f"Using most recent run: {optimal_configs_dir.name}")
        else:
            # Fallback to base directory for backward compatibility
            optimal_configs_dir = base_optimal_configs_dir
    
    print(f"Loading trial data from {optimal_configs_dir}")
    df = load_trial_data(optimal_configs_dir)
    
    if df.empty:
        print("No trial data found!")
        return
    
    print(f"\nLoaded {len(df)} trials")
    print(f"Models: {df['model'].unique()}")
    print(f"Tasks: {df['task'].unique()}")
    print(f"Masker types: {df['masker_name'].unique()[:5]}...")  # Show first 5
    print(f"Objective functions: {df['objective_function'].unique()}")
    
    # Run analyses
    analyze_objective_performance(df)
    analyze_hyperparameter_impact(df)
    analyze_sparsity_achievement(optimal_configs_dir)
    
    # Create plots
    try:
        plot_trial_scores(df, output_dir)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    # Save combined dataframe
    output_file = output_dir / "all_trials_data.csv"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nAll trial data saved to {output_file}")


if __name__ == "__main__":
    main()
