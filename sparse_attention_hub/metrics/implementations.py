"""Concrete metric implementations."""

from typing import Any, List, Union

import numpy as np
import torch

from .base import MicroMetric


class TopkRecall(MicroMetric):
    """Top-k recall metric implementation."""

    def __init__(self, k: int = 10):
        super().__init__(f"topk_recall_k{k}")
        self.k = k

    def compute(  # pylint: disable=arguments-differ
        self,
        predictions: Union[List, torch.Tensor, np.ndarray],
        ground_truth: Union[List, torch.Tensor, np.ndarray],
        **kwargs: Any,
    ) -> float:
        """Compute top-k recall.

        Args:
            predictions: Predicted values or rankings
            ground_truth: Ground truth values or rankings
            **kwargs: Additional parameters

        Returns:
            Top-k recall score
        """
        # TODO: Implement top-k recall computation  # pylint: disable=fixme
        # Convert inputs to appropriate format
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()

        # Compute top-k recall
        raise NotImplementedError("TopkRecall computation not yet implemented")


class LocalError(MicroMetric):
    """Local error metric implementation."""

    def __init__(self) -> None:
        super().__init__("local_error")

    def compute(  # pylint: disable=arguments-differ
        self,
        predicted_attention: Union[torch.Tensor, np.ndarray],
        true_attention: Union[torch.Tensor, np.ndarray],
        **kwargs: Any,
    ) -> float:
        """Compute local error between attention patterns.

        Args:
            predicted_attention: Predicted attention weights
            true_attention: True attention weights
            **kwargs: Additional parameters

        Returns:
            Local error score
        """
        # TODO: Implement local error computation  # pylint: disable=fixme
        # Convert inputs to appropriate format
        if isinstance(predicted_attention, torch.Tensor):
            predicted_attention = predicted_attention.cpu().numpy()
        if isinstance(true_attention, torch.Tensor):
            true_attention = true_attention.cpu().numpy()

        # Compute local error
        raise NotImplementedError("LocalError computation not yet implemented")


class SampleVariance(MicroMetric):
    """Sample variance metric implementation."""

    def __init__(self) -> None:
        super().__init__("sample_variance")

    def compute(  # pylint: disable=arguments-differ
        self, samples: Union[List, torch.Tensor, np.ndarray], **kwargs: Any
    ) -> float:
        """Compute sample variance.

        Args:
            samples: Sample values
            **kwargs: Additional parameters

        Returns:
            Sample variance
        """
        # Convert inputs to appropriate format
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        elif isinstance(samples, list):
            samples = np.array(samples)

        # Compute sample variance
        if len(samples) == 0:
            return 0.0
        return float(np.var(samples, ddof=1) if len(samples) > 1 else 0.0)
