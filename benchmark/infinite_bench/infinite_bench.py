"""InfiniteBench benchmark implementation for long context evaluation."""

from typing import Dict, Any, List
import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("infinite_bench")
class InfiniteBench(Benchmark):
    """InfiniteBench benchmark for evaluating long context understanding.

    InfiniteBench is a comprehensive benchmark for evaluating the ability of large language
    models to understand extremely long contexts. It includes 9 tasks across various domains:
    retrieval, code execution, mathematical reasoning, and long document understanding.

    The benchmark evaluates different capabilities:
    - Retrieval: passkey, kv_retrieval, number_string
    - Code: code_run, code_debug
    - Math: math_find, math_calc
    - Long Document: longbook_qa_eng, longdialogue_qa_eng, longbook_choice_eng

    Example:
        >>> infinite_bench = InfiniteBench(subsets_to_run=["passkey", "kv_retrieval"])
        >>> results = infinite_bench.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"Average score: {results}")
    """

    # All available InfiniteBench datasets
    all_datasets: List[str] = [
        "passkey",
        "kv_retrieval", 
        "number_string",
        "code_run",
        "code_debug",
        "math_find",
        "longbook_qa_eng",
        "longdialogue_qa_eng",
        "longbook_choice_eng"
    ]
    
    benchmark_name: str = "infinite_bench"
    huggingface_dataset_id: str = "MaxJeblick/InfiniteBench"

    def _load_datasets(self) -> pd.DataFrame:
        """Load InfiniteBench datasets by individual configs.
        
        InfiniteBench requires loading each subset as a separate config.
        
        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading InfiniteBench datasets: {self.subsets_to_run}")
        dfs = []
        
        for subset in self.subsets_to_run:
            try:
                from datasets import load_dataset
                subset_dataset = load_dataset(self.huggingface_dataset_id, subset, split="test")
                subset_df = subset_dataset.to_pandas()
                subset_df["task"] = subset  # Ensure task column exists
                dfs.append(subset_df)
                print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")
            except Exception as subset_error:
                print(f"  ❌ Failed to load {subset}: {str(subset_error)}")
                continue
        
        if not dfs:
            raise Exception("No InfiniteBench subsets could be loaded successfully")
        
        # Combine all subset DataFrames
        import pandas as pd
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")
        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for InfiniteBench results.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - task: Dataset/task name
                - predicted_answer: Model's predicted answer
                - answer: Ground truth answers (list)

        Returns:
            Dictionary containing computed metrics:
            - overall_score: Average score across all tasks
            - task_scores: Individual scores for each task
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Group results by task
        task_groups = results_df.groupby("task")
        task_scores: Dict[str, float] = {}
        all_scores: List[float] = []
        
        for task_name, task_df in task_groups:
            try:
                # Use the calculate_metrics function from HashAttention evaluation
                score: float = calculate_metrics(task_df)
                task_scores[task_name] = score
                all_scores.append(score)
                print(f"  ✓ {task_name}: {score:.4f}")
                    
            except Exception as e:
                print(f"Error evaluating task {task_name}: {str(e)}")
                task_scores[task_name] = 0.0

        # Compute overall metrics
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        overall_metrics: Dict[str, Any] = {
            "overall_score": round(overall_score, 4),
            "task_scores": {task: round(score, 4) for task, score in task_scores.items()},
            "summary": {
                "total_tasks": len(task_scores),
                "total_samples": len(results_df)
            }
        }
        
        return overall_metrics 