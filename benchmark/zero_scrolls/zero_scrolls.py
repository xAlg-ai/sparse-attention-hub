"""ZeroScrolls benchmark implementation for long context evaluation."""

from typing import Dict, Any, List
import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark


@register_benchmark("zero_scrolls")
class ZeroScrolls(Benchmark):
    """ZeroScrolls benchmark for evaluating long context understanding.

    ZeroScrolls is a benchmark for evaluating the ability of large language models to understand
    and process long documents. It includes 10 tasks across different domains.

    The benchmark evaluates different capabilities:
    - Summarization: gov_report, summ_screen_fd, qmsum, squality
    - Question Answering: qasper, narrative_qa, musique
    - Quality Assessment: quality
    - Document Processing: space_digest, book_sum_sort

    Example:
        >>> zero_scrolls = ZeroScrolls(subsets_to_run=["gov_report", "qasper"])
        >>> results = zero_scrolls.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"Results: {results}")
    """

    # All available ZeroScrolls datasets
    # all_datasets: List[str] = [
    #     "gov_report",
    #     "summ_screen_fd",
    #     "qmsum",
    #     "qasper",
    #     "narrative_qa",
    #     "quality",
    #     "musique",
    #     "squality",
    #     "space_digest",
    #     "book_sum_sort"
    # ]
    # all tasks are clubbed into a single subset on huggingface
    all_datasets: List[str] = ["default"]

    benchmark_name: str = "zero_scrolls"
    huggingface_dataset_id: str = "simonjegou/zero_scrolls"

    def _load_datasets(self) -> pd.DataFrame:
        """Load ZeroScrolls datasets by individual configs.

        ZeroScrolls requires loading each subset as a separate config.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading ZeroScrolls datasets: {self.subsets_to_run}")
        dfs = []

        for subset in self.subsets_to_run:
            try:
                from datasets import load_dataset

                subset_dataset = load_dataset(
                    self.huggingface_dataset_id, subset, split="test"
                )
                subset_df = subset_dataset.to_pandas()

                # Process the data according to ZeroScrolls format
                subset_df["context"] = subset_df.apply(
                    lambda x: x["input"][: x["document_end_index"]], axis=1
                )
                subset_df["question"] = subset_df.apply(
                    lambda x: x["input"][
                        x["document_end_index"] : x["query_end_index"]
                    ],
                    axis=1,
                )
                subset_df["answer_prefix"] = subset_df.apply(
                    lambda x: x["input"][x["query_end_index"] :], axis=1
                ).str.strip()
                subset_df["answer"] = (
                    ""  # ZeroScrolls doesn't provide ground truth answers
                )
                subset_df["task"] = subset

                dfs.append(subset_df)
                print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")
            except Exception as subset_error:
                print(f"  ❌ Failed to load {subset}: {str(subset_error)}")
                continue

        if not dfs:
            raise Exception("No ZeroScrolls subsets could be loaded successfully")

        # Combine all subset DataFrames
        import pandas as pd

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")
        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for ZeroScrolls results.

        Note: ZeroScrolls doesn't provide ground truth answers, so this is a placeholder
        for future evaluation metrics.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - task: Dataset/task name
                - predicted_answer: Model's predicted answer

        Returns:
            Dictionary containing basic statistics since no ground truth is available.
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Group results by task
        task_groups = results_df.groupby("task")
        task_stats: Dict[str, Dict[str, Any]] = {}

        for task_name, task_df in task_groups:
            # Calculate basic statistics since no ground truth is available
            avg_length = (
                task_df["predicted_answer"].str.len().mean() if len(task_df) > 0 else 0
            )
            task_stats[task_name] = {
                "num_samples": len(task_df),
                "avg_response_length": round(avg_length, 2),
                "note": "No ground truth available for evaluation",
            }

        overall_metrics: Dict[str, Any] = {
            "task_stats": task_stats,
            "summary": {
                "total_tasks": len(task_stats),
                "total_samples": len(results_df),
                "note": "ZeroScrolls benchmark completed. No ground truth available for scoring.",
            },
        }

        return overall_metrics
