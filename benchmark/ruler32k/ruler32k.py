"""Ruler benchmark implementation for long context evaluation."""

from typing import Dict, Any, List
import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("ruler32k")
class Ruler32K(Benchmark):
    """Ruler benchmark for evaluating long context understanding.
    """

    # All available Ruler datasets (context lengths)
    all_datasets: List[str] = [
            "cwe",
            "fwe",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multiquery",
            "niah_multivalue",
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "qa_1",
            "qa_2",
            "vt"
        ]
    
    benchmark_name: str = "ruler32k"
    huggingface_dataset_id: str = "xAlg-AI/att-hub-ruler-32k"

    def _load_datasets(self) -> pd.DataFrame:
        """Load Ruler datasets by individual configs.
        
        Ruler requires loading each context length as a separate config.
        
        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading Ruler datasets: {self.subsets_to_run}")
        dfs = []
        
        for subset in self.subsets_to_run:
            try:
                from datasets import load_dataset
                subset_dataset = load_dataset(self.huggingface_dataset_id, subset, split=subset)
                subset_df = subset_dataset.to_pandas()
                # Add context length as a column for analysis
                subset_df["context_length"] = 32768
                dfs.append(subset_df)
                print(f"  ✓ Loaded {len(subset_df)} samples from {subset} || context_length = 32768")
            except Exception as subset_error:
                print(f"  ❌ Failed to load {subset}: {str(subset_error)}")
                continue
        
        if not dfs:
            raise Exception("No Ruler subsets could be loaded successfully")
        
        # Combine all subset DataFrames
        import pandas as pd
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets || context_length = 32768")
        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for Ruler results.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - task: Dataset/task name
                - predicted_answer: Model's predicted answer
                - answer: Ground truth answers
                - context_length: Context length used

        Returns:
            Dictionary containing computed metrics:
            - overall_score: Average score across all tasks
            - task_scores: Individual scores for each task
            - context_length_scores: Scores grouped by context length
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Use the calculate_metrics function from HashAttention evaluation
        task_scores: Dict[str, Dict[str, float]] = calculate_metrics(results_df)
        
        # Extract string_match scores and compute overall
        all_scores: List[float] = []
        for task, scores in task_scores.items():
            if "string_match" in scores:
                all_scores.append(scores["string_match"])
        
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Group by context length if available
        context_length_scores: Dict[str, float] = {}
        if "context_length" in results_df.columns:
            for context_length in results_df["context_length"].unique():
                length_df = results_df[results_df["context_length"] == context_length]
                if len(length_df) > 0:
                    length_scores = calculate_metrics(length_df)
                    length_overall = sum(score.get("string_match", 0) for score in length_scores.values()) / len(length_scores) if length_scores else 0.0
                    context_length_scores[str(context_length)] = round(length_overall, 2)
        
        overall_metrics: Dict[str, Any] = {
            "overall_score": round(overall_score, 2),
            "task_scores": task_scores,
            "context_length_scores": context_length_scores,
            "summary": {
                "total_tasks": len(task_scores),
                "total_samples": len(results_df),
                "context_lengths": list(context_length_scores.keys())
            }
        }
        
        return overall_metrics 
