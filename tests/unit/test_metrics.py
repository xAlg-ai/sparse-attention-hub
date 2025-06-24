"""Unit tests for metrics module."""

import os
import tempfile

import numpy as np

from sparse_attention_hub.metrics.base import MicroMetric
from sparse_attention_hub.metrics.implementations import SampleVariance
from sparse_attention_hub.metrics.logger import MicroMetricLogger


class TestMicroMetric(MicroMetric):
    """Test metric implementation."""

    def __init__(self):
        super().__init__("test_metric")

    def compute(self, *args, **kwargs):
        return sum(args) if args else 0


class TestMicroMetricLogger:
    """Test cases for MicroMetricLogger."""

    def setup_method(self):
        """Setup for each test method."""
        # Reset singleton
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

    def test_singleton_behavior(self):
        """Test singleton pattern."""
        logger1 = MicroMetricLogger()
        logger2 = MicroMetricLogger()
        assert logger1 is logger2

    def test_metric_registration(self):
        """Test metric registration."""
        logger = MicroMetricLogger()
        metric = TestMicroMetric()

        logger.register_metric(metric)
        assert metric in logger.get_available_metrics()

        # Test duplicate registration
        logger.register_metric(metric)
        assert logger.get_available_metrics().count(metric) == 1

    def test_metric_logging_control(self):
        """Test enabling/disabling metric logging."""
        logger = MicroMetricLogger()
        metric = TestMicroMetric()

        # Register metric
        logger.register_metric(metric)
        assert not logger.should_log_metric(metric)

        # Enable logging
        logger.enable_metric_logging(metric)
        assert logger.should_log_metric(metric)
        assert metric in logger.get_enabled_metrics()

        # Disable logging
        logger.disable_metric_logging(metric)
        assert not logger.should_log_metric(metric)
        assert metric not in logger.get_enabled_metrics()

    def test_log_path_setting(self):
        """Test setting log path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MicroMetricLogger()
            logger.set_log_path(temp_dir)
            assert logger.path_to_log == temp_dir
            assert os.path.exists(temp_dir)


class TestSampleVariance:
    """Test cases for SampleVariance metric."""

    def test_initialization(self):
        """Test SampleVariance initialization."""
        metric = SampleVariance()
        assert metric.name == "sample_variance"

    def test_compute_with_list(self):
        """Test computation with list input."""
        metric = SampleVariance()
        samples = [1, 2, 3, 4, 5]
        result = metric.compute(samples)
        expected = np.var(samples, ddof=1)
        assert abs(result - expected) < 1e-10

    def test_compute_with_numpy_array(self):
        """Test computation with numpy array."""
        metric = SampleVariance()
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = metric.compute(samples)
        expected = np.var(samples, ddof=1)
        assert abs(result - expected) < 1e-10

    def test_compute_empty_samples(self):
        """Test computation with empty samples."""
        metric = SampleVariance()
        result = metric.compute([])
        assert result == 0.0

    def test_compute_single_sample(self):
        """Test computation with single sample."""
        metric = SampleVariance()
        result = metric.compute([5.0])
        assert result == 0.0
