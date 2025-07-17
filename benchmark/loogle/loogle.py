"""Loogle benchmark implementation for long context evaluation."""

from typing import Dict, Any, List
import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("loogle")
class Loogle(Benchmark):
    """Loogle benchmark for evaluating long context understanding.

    Loogle is a benchmark for evaluating the ability of large language models to understand
    both short and long dependency content. It includes 4 major tasks across different domains.

    The benchmark evaluates different capabilities:
    - Question Answering: shortdep_qa, longdep_qa
    - Cloze Completion: shortdep_cloze
    - Summarization: longdep_summarization

    Example:
        >>> loogle = Loogle(subsets_to_run=["shortdep_qa", "longdep_qa"])
        >>> results = loogle.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"Average score: {results}")
    """

    # All available Loogle datasets
    all_datasets: List[str] = [
        "shortdep_qa",
        "longdep_qa",
        "shortdep_cloze",
        "longdep_summarization"
    ]
    
    benchmark_name: str = "loogle"
    huggingface_dataset_id: str = "simonjegou/loogle"

    def _load_datasets(self) -> pd.DataFrame:
        """Load Loogle datasets by individual configs.
        
        Loogle requires loading each subset as a separate config.
        
        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading Loogle datasets: {self.subsets_to_run}")
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
            raise Exception("No Loogle subsets could be loaded successfully")
        
        # Combine all subset DataFrames
        import pandas as pd
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")
        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for Loogle results.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - task: Dataset/task name
                - predicted_answer: Model's predicted answer
                - answer: Ground truth answers

        Returns:
            Dictionary containing computed metrics:
            - overall_score: Average score across all tasks
            - task_scores: Individual scores for each task with detailed metrics
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Use the calculate_metrics function from HashAttention evaluation
        task_scores: Dict[str, Dict[str, float]] = calculate_metrics(results_df)
        
        # Compute overall score by averaging task scores
        all_scores: List[float] = []
        for task, scores in task_scores.items():
            # For cloze tasks, use exact_match as primary metric
            if task == "shortdep_cloze" and "exact_match" in scores:
                all_scores.append(scores["exact_match"])
            # For other tasks, use BERT score as primary metric
            elif "bert" in scores:
                all_scores.append(scores["bert"])
            # Fallback to first available metric
            elif scores:
                all_scores.append(list(scores.values())[0])
        
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        overall_metrics: Dict[str, Any] = {
            "overall_score": round(overall_score, 4),
            "task_scores": task_scores,
            "summary": {
                "total_tasks": len(task_scores),
                "total_samples": len(results_df)
            }
        }
        
        return overall_metrics 