#!/usr/bin/env python3
"""
List all benchmark tasks from optimal configs for easy inspection.

Usage:
    python benchmark/raytune/list_benchmark_tasks.py --config-run run_20250818_203531
    python benchmark/raytune/list_benchmark_tasks.py --config-run run_20250818_203531 --format csv > tasks.csv
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import csv


def main():
    parser = argparse.ArgumentParser(description="List benchmark tasks from optimal configs")
    parser.add_argument("--config-run", type=str, required=True,
                       help="Config run directory name")
    parser.add_argument("--optimal-configs-dir", default="./optimal_configs",
                       help="Base directory for optimal configurations")
    parser.add_argument("--format", choices=["table", "csv", "json", "simple"], default="table",
                       help="Output format")
    parser.add_argument("--group-by", choices=["model", "task", "masker", "none"], default="none",
                       help="Group tasks by field")
    parser.add_argument("--filter-model", type=str, help="Filter by model name (substring match)")
    parser.add_argument("--filter-task", type=str, help="Filter by task name (substring match)")
    parser.add_argument("--filter-masker", type=str, help="Filter by masker name (substring match)")
    
    args = parser.parse_args()
    
    # Load configurations
    config_dir = Path(args.optimal_configs_dir) / args.config_run
    if not config_dir.exists():
        print(f"Error: Config directory {config_dir} not found", file=sys.stderr)
        sys.exit(1)
    
    tasks = []
    for config_file in sorted(config_dir.glob("*.json")):
        if config_file.name.endswith(("_trials.json", "_analysis.csv")):
            continue
        
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
            
            # Apply filters
            if args.filter_model and args.filter_model not in data["model"]:
                continue
            if args.filter_task and args.filter_task not in data["task"]:
                continue
            if args.filter_masker and args.filter_masker not in data["masker_name"]:
                continue
            
            tasks.append({
                "model": data["model"],
                "task": data["task"],
                "masker": data["masker_name"],
                "score": data.get("score", "N/A"),
                "search_time": data.get("search_time", 0),
                "num_trials": data.get("num_trials", 0),
                "file": config_file.name
            })
        except Exception as e:
            print(f"Warning: Failed to load {config_file}: {e}", file=sys.stderr)
    
    if not tasks:
        print("No tasks found matching criteria", file=sys.stderr)
        sys.exit(1)
    
    # Sort tasks
    tasks.sort(key=lambda x: (x["model"], x["task"], x["masker"]))
    
    # Output based on format
    if args.format == "json":
        print(json.dumps(tasks, indent=2))
        
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=["model", "task", "masker", "score", "search_time", "num_trials", "file"])
        writer.writeheader()
        writer.writerows(tasks)
        
    elif args.format == "simple":
        for task in tasks:
            print(f"{task['model']} | {task['task']} | {task['masker']}")
            
    else:  # table format
        # Group if requested
        if args.group_by != "none":
            groups = defaultdict(list)
            for task in tasks:
                key = task[args.group_by]
                groups[key].append(task)
            
            print(f"Tasks grouped by {args.group_by}:")
            print("=" * 80)
            
            for key in sorted(groups.keys()):
                print(f"\n{args.group_by.upper()}: {key}")
                print("-" * 80)
                
                for task in groups[key]:
                    if args.group_by == "model":
                        print(f"  {task['task']:30} | {task['masker']:30} | Score: {task['score']}")
                    elif args.group_by == "task":
                        print(f"  {task['model']:30} | {task['masker']:30} | Score: {task['score']}")
                    else:  # masker
                        print(f"  {task['model']:30} | {task['task']:30} | Score: {task['score']}")
                
                print(f"  Total: {len(groups[key])} configurations")
        
        else:
            # Regular table
            print(f"Benchmark Tasks from {args.config_run}")
            print("=" * 120)
            print(f"{'Model':35} | {'Task':25} | {'Masker':30} | {'Score':8} | {'Trials':6}")
            print("-" * 120)
            
            for task in tasks:
                score_str = f"{task['score']:.4f}" if isinstance(task['score'], (int, float)) else str(task['score'])
                print(f"{task['model']:35} | {task['task']:25} | {task['masker']:30} | {score_str:8} | {task['num_trials']:6}")
            
            print("-" * 120)
            print(f"Total: {len(tasks)} configurations")
            
            # Summary statistics
            print(f"\nSummary:")
            models = set(t["model"] for t in tasks)
            tasks_set = set(t["task"] for t in tasks)
            maskers = set(t["masker"] for t in tasks)
            
            print(f"  Models: {len(models)}")
            for model in sorted(models):
                count = sum(1 for t in tasks if t["model"] == model)
                print(f"    - {model}: {count} configs")
            
            print(f"  Tasks: {len(tasks_set)}")
            for task in sorted(tasks_set):
                count = sum(1 for t in tasks if t["task"] == task)
                print(f"    - {task}: {count} configs")
            
            print(f"  Maskers: {len(maskers)}")
            for masker in sorted(maskers):
                count = sum(1 for t in tasks if t["masker"] == masker)
                print(f"    - {masker}: {count} configs")


if __name__ == "__main__":
    main()
