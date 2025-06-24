"""Basic integration tests for the sparse attention hub skeleton."""

import tempfile
from unittest.mock import Mock

import pytest

from sparse_attention_hub.benchmark.datasets import InfBench, LongBench, Loogle
from sparse_attention_hub.benchmark.executor import BenchmarkExecutor
from sparse_attention_hub.benchmark.storage import ResultStorage
from sparse_attention_hub.metrics.implementations import SampleVariance
from sparse_attention_hub.metrics.logger import MicroMetricLogger
from sparse_attention_hub.model_hub.huggingface import ModelHubHF
from sparse_attention_hub.pipeline.huggingface import PipelineHF
from sparse_attention_hub.sparse_attention.efficient import DoubleSparsity
from sparse_attention_hub.testing.tester import Tester


class TestBasicIntegration:
    """Test basic integration between components."""

    def test_storage_operations(self):
        """Test that storage can store and retrieve results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(storage_path=temp_dir)

            # Test storing results
            test_results = ["accuracy: 0.95", "latency: 100ms"]
            result_id = storage.store(test_results)

            assert isinstance(result_id, str)
            assert len(result_id) > 0

            # Test loading results
            loaded_data = storage.load(result_id)
            assert loaded_data is not None
            assert loaded_data["results"] == test_results

            # Test listing results
            results_list = storage.list_results()
            assert isinstance(results_list, list)
            assert len(results_list) == 1

            # Test deleting results
            deleted = storage.delete(result_id)
            assert deleted is True

            # Verify deletion
            loaded_after_delete = storage.load(result_id)
            assert loaded_after_delete is None

    def test_dataset_interfaces(self):
        """Test that all dataset classes have required interfaces."""
        datasets = [LongBench(), Loogle(), InfBench()]

        for dataset in datasets:
            # Test that all datasets have required methods from base class
            assert hasattr(dataset, "create_hugging_face_dataset")
            assert hasattr(dataset, "run_benchmark")
            assert hasattr(dataset, "name")
            assert hasattr(dataset, "subsets")

            # Test that calling methods raises NotImplementedError (expected for skeleton)
            with pytest.raises(NotImplementedError):
                dataset.create_hugging_face_dataset()

            with pytest.raises(NotImplementedError):
                dataset.run_benchmark(None)

    def test_model_hub_initialization(self):
        """Test that model hub can be initialized and has required attributes."""
        model_hub = ModelHubHF(api_token="test_token")

        assert model_hub.api_token == "test_token"
        assert hasattr(model_hub, "_original_attention_interfaces")
        assert hasattr(model_hub, "_registered_hooks")
        assert isinstance(model_hub._original_attention_interfaces, dict)
        assert isinstance(model_hub._registered_hooks, dict)

    def test_pipeline_initialization(self):
        """Test that pipeline can be initialized with a model."""
        mock_model = Mock()
        pipeline = PipelineHF(model=mock_model, device="cpu")

        assert pipeline.model == mock_model
        assert pipeline.device == "cpu"
        assert callable(pipeline)

    def test_metrics_logger_functionality(self):
        """Test that metrics logger works correctly."""
        logger = MicroMetricLogger()
        variance_metric = SampleVariance()

        # Test registration
        logger.register_metric(variance_metric)
        assert variance_metric in logger.available_metrics

        # Test that logger has required attributes
        assert hasattr(logger, "available_metrics")
        assert hasattr(logger, "metrics_to_log")
        assert hasattr(logger, "path_to_log")
        assert isinstance(logger.available_metrics, list)
        assert isinstance(logger.metrics_to_log, list)

    def test_sample_variance_metric(self):
        """Test that sample variance metric works correctly."""
        metric = SampleVariance()

        assert metric.name == "sample_variance"

        # Test with list data
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = metric.compute(test_data)

        assert isinstance(result, (int, float))
        assert result >= 0  # Variance should be non-negative

        # Test with single value
        single_value = [5.0]
        result_single = metric.compute(single_value)
        assert result_single == 0.0  # Variance of single value should be 0

    def test_efficient_attention_interface(self):
        """Test that efficient attention has the required interface."""
        double_sparsity = DoubleSparsity()

        # Test that it has the required method
        assert hasattr(double_sparsity, "custom_attention")

        # Test that calling it raises NotImplementedError (expected for skeleton)
        with pytest.raises(NotImplementedError):
            double_sparsity.custom_attention()

    def test_benchmark_executor_interface(self):
        """Test that benchmark executor has required interface."""
        executor = BenchmarkExecutor()

        # Test that it has required methods
        assert hasattr(executor, "register_benchmark")
        assert hasattr(executor, "evaluate")
        assert hasattr(executor, "result_storage")
        assert hasattr(executor, "_registered_benchmarks")

        # Test that attributes are properly initialized
        assert isinstance(executor._registered_benchmarks, dict)
        assert executor.result_storage is not None

    def test_tester_interface(self):
        """Test that tester has required interface."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = Tester(test_directory=temp_dir)

            # Test that it has required methods
            assert hasattr(tester, "execute_all_tests")
            assert hasattr(tester, "execute_unit_tests")
            assert hasattr(tester, "execute_integration_tests")
            assert hasattr(tester, "execute_specific_test")

            # Test directory setup
            assert tester.test_directory == temp_dir
            assert tester.unit_test_dir == f"{temp_dir}/unit"
            assert tester.integration_test_dir == f"{temp_dir}/integration"

    def test_cross_component_compatibility(self):
        """Test that components can be created together without conflicts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create all major components
            storage = ResultStorage(storage_path=temp_dir)
            model_hub = ModelHubHF(api_token="test")
            mock_model = Mock()
            pipeline = PipelineHF(model=mock_model)
            logger = MicroMetricLogger()
            metric = SampleVariance()
            tester = Tester(test_directory=temp_dir)

            # Test that all components are properly initialized
            assert storage.storage_path == temp_dir
            assert model_hub.api_token == "test"
            assert pipeline.model == mock_model
            assert hasattr(logger, "available_metrics")
            assert metric.name == "sample_variance"
            assert tester.test_directory == temp_dir

            # Test that components can interact
            logger.register_metric(metric)
            assert metric in logger.available_metrics

            # Test storage operations
            test_results = ["integration test results"]
            result_id = storage.store(test_results)
            loaded_results = storage.load(result_id)
            assert loaded_results["results"] == test_results

    def test_end_to_end_workflow_simulation(self):
        """Test a simulated end-to-end workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Initialize components
            storage = ResultStorage(storage_path=temp_dir)
            logger = MicroMetricLogger()
            metric = SampleVariance()

            # 2. Setup metrics logging
            logger.register_metric(metric)

            # 3. Simulate benchmark results
            benchmark_data = [0.85, 0.87, 0.83, 0.89, 0.86]
            variance_result = metric.compute(benchmark_data)

            # 4. Store results
            results = [
                f"benchmark_variance: {variance_result}",
                f"data_points: {len(benchmark_data)}",
                "status: completed",
            ]
            result_id = storage.store(results)

            # 5. Verify workflow
            assert isinstance(result_id, str)
            assert variance_result >= 0

            # 6. Retrieve and validate stored results
            stored_results = storage.load(result_id)
            assert stored_results is not None
            assert len(stored_results["results"]) == 3
            assert "benchmark_variance" in stored_results["results"][0]
            assert "status: completed" in stored_results["results"]
