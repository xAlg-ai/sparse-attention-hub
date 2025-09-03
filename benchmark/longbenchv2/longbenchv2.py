"""LongBenchv2 benchmark implementation for long context evaluation."""

from typing import Dict, Any, List
import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("longbenchv2")
class LongBenchv2(Benchmark):
    """LongBenchv2 benchmark for evaluating long context understanding.

    LongBenchv2 is an extension of LongBench that focuses on multiple choice questions
    with different prompting strategies. It includes 2 main tasks.

    The benchmark evaluates different capabilities:
    - Zero-shot reasoning: 0shot
    - Chain-of-thought reasoning: cot

    Example:
        >>> longbenchv2 = LongBenchv2(subsets_to_run=["0shot", "cot"])
        >>> results = longbenchv2.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"Average score: {results}")
    """

    # All available LongBenchv2 datasets
    all_datasets: List[str] = ["0shot", "cot"]

    benchmark_name: str = "longbenchv2"
    huggingface_dataset_id: str = "Xnhyacinth/LongBench-v2"

    def _load_datasets(self) -> pd.DataFrame:
        """Load LongBenchv2 datasets by individual configs.

        LongBenchv2 requires loading each subset as a separate config.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading LongBenchv2 datasets: {self.subsets_to_run}")
        dfs = []

        for subset in self.subsets_to_run:
            try:
                from datasets import load_dataset

                subset_dataset = load_dataset(
                    self.huggingface_dataset_id, subset, split="test"
                )
                subset_df = subset_dataset.to_pandas()
                subset_df["task"] = subset  # Ensure task column exists
                dfs.append(subset_df)
                print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")
            except Exception as subset_error:
                print(f"  ❌ Failed to load {subset}: {str(subset_error)}")
                continue

        if not dfs:
            raise Exception("No LongBenchv2 subsets could be loaded successfully")

        # Combine all subset DataFrames
        import pandas as pd

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")
        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for LongBenchv2 results.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - task: Dataset/task name
                - predicted_answer: Model's predicted answer
                - answer: Ground truth answers
                - length: Context length category
                - difficulty: Question difficulty

        Returns:
            Dictionary containing computed metrics:
            - overall_score: Average score across all tasks
            - task_scores: Individual scores for each task
            - breakdown_scores: Scores broken down by length and difficulty
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Group results by task
        task_groups = results_df.groupby("task")
        task_scores: Dict[str, Any] = {}
        all_scores: List[float] = []

        for task_name, task_df in task_groups:
            try:
                # Use the calculate_metrics function from HashAttention evaluation
                scores = calculate_metrics(task_df)

                # Parse the scores string to extract overall accuracy
                if len(scores) >= 2:
                    overall_score_str = scores[1].split("\t")[0]
                    try:
                        overall_score = float(overall_score_str)
                        task_scores[task_name] = {
                            "overall_accuracy": overall_score,
                            "detailed_scores": scores,
                        }
                        all_scores.append(overall_score)
                        print(f"  ✓ {task_name}: {overall_score:.1f}%")
                    except ValueError:
                        task_scores[task_name] = {"error": "Could not parse score"}
                else:
                    task_scores[task_name] = {"error": "No scores returned"}

            except Exception as e:
                print(f"Error evaluating task {task_name}: {str(e)}")
                task_scores[task_name] = {"error": str(e)}

        # Compute overall metrics
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        overall_metrics: Dict[str, Any] = {
            "overall_score": round(overall_score, 1),
            "task_scores": task_scores,
            "summary": {
                "total_tasks": len(task_scores),
                "total_samples": len(results_df),
            },
        }

        return overall_metrics
