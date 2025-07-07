"""LongBench benchmark implementation."""
# pylint: disable=fixme

from typing import Any, Dict, List

from ..base import Benchmark


class LongBench(Benchmark):
    """LongBench benchmark implementation."""

    def __init__(self) -> None:
        subsets = [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ]
        super().__init__("LongBench", subsets)

    def create_hugging_face_dataset(self) -> Any:
        """Create LongBench HuggingFace dataset.

        Returns:
            HuggingFace dataset for LongBench
        """
        # TODO: Implement LongBench dataset loading
        raise NotImplementedError("LongBench dataset loading not yet implemented")

    def run_benchmark(self, model: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run LongBench on a model.

        Args:
            model: Model to benchmark
            **kwargs: Additional parameters

        Returns:
            LongBench results
        """
        # TODO: Implement LongBench execution
        raise NotImplementedError("LongBench execution not yet implemented")

    def calculate_metrics(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Calculate LongBench metrics.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth answers

        Returns:
            Dictionary of metrics
        """
        # TODO: Implement LongBench metrics
        raise NotImplementedError("LongBench metrics not yet implemented")
