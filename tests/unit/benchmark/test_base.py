"""Tests for benchmark base functionality."""

from unittest.mock import Mock, patch
from typing import Dict, Any

import pytest
import pandas as pd

from benchmark import Benchmark


class MockBenchmark(Benchmark):
    """Mock benchmark class for testing the base functionality."""
    
    all_datasets = ["test_task1", "test_task2"]
    benchmark_name = "test_benchmark"
    huggingface_dataset_id = "mock/test_dataset"
    
    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Mock evaluation that just counts samples."""
        return {
            "total_samples": len(results_df),
            "mock_score": 85.5
        }


@pytest.fixture
def mock_dataset_df() -> pd.DataFrame:
    """Create a mock dataset DataFrame for testing."""
    return pd.DataFrame({
        "context": ["Context 1", "Context 1", "Context 2"],
        "question": ["Question 1a", "Question 1b", "Question 2"],
        "task": ["test_task1", "test_task1", "test_task2"],
        "answers": [["Answer 1a"], ["Answer 1b"], ["Answer 2"]],
        "all_classes": [[], [], []]
    })


@pytest.fixture
def mock_adapter() -> Mock:
    """Create a mock adapter for testing."""
    adapter = Mock()
    
    def mock_process_request(request, generation_kwargs, request_kwargs):
        """Mock processing that returns responses based on questions."""
        if isinstance(request.questions, list):
            responses = [f"Response to {q}" for q in request.questions]
        else:
            responses = f"Response to {request.questions}"
        return Mock(responses=responses)
    
    adapter.process_request.side_effect = mock_process_request
    return adapter


class TestBenchmarkBase:
    """Test the base Benchmark class functionality."""
    
    def test_benchmark_initialization(self):
        """Test basic benchmark initialization."""
        benchmark = MockBenchmark()
        assert benchmark.subsets_to_run == ["test_task1", "test_task2"]
        assert benchmark.benchmark_name == "test_benchmark"
        assert benchmark.huggingface_dataset_id == "mock/test_dataset"
    
    def test_benchmark_initialization_with_subsets(self):
        """Test benchmark initialization with custom subsets."""
        benchmark = MockBenchmark(subsets_to_run=["test_task1"])
        assert benchmark.subsets_to_run == ["test_task1"]
    
    def test_benchmark_subset_validation_valid(self):
        """Test that valid subset validation works correctly."""
        benchmark = MockBenchmark(subsets_to_run=["test_task1"])
        assert benchmark.subsets_to_run == ["test_task1"]
        
        benchmark = MockBenchmark(subsets_to_run=["test_task1", "test_task2"])
        assert benchmark.subsets_to_run == ["test_task1", "test_task2"]
    
    def test_benchmark_subset_validation_invalid(self):
        """Test that invalid subset validation raises error."""
        with pytest.raises(ValueError, match="Invalid subsets"):
            MockBenchmark(subsets_to_run=["invalid_task"])
        
        with pytest.raises(ValueError, match="Invalid subsets"):
            MockBenchmark(subsets_to_run=["test_task1", "invalid_task"])
    
    def test_benchmark_initialization_missing_attributes(self):
        """Test that missing required attributes raise errors."""
        class IncompleteBenchmark(Benchmark):
            # Missing all required attributes, but implement abstract method
            def post_run_evaluate(self, results_df):
                return {}
        
        with pytest.raises(ValueError, match="must define all_datasets"):
            IncompleteBenchmark()
    
    def test_get_available_datasets(self):
        """Test getting available datasets."""
        benchmark = MockBenchmark()
        datasets = benchmark.get_available_datasets()
        assert datasets == ["test_task1", "test_task2"]
        
        # Ensure it returns a copy, not the original
        datasets.append("new_task")
        assert benchmark.all_datasets == ["test_task1", "test_task2"]
    
    def test_validate_subsets_method(self):
        """Test the _validate_subsets method directly."""
        benchmark = MockBenchmark()
        
        # Valid subsets should not raise
        benchmark._validate_subsets(["test_task1"])
        benchmark._validate_subsets(["test_task1", "test_task2"])
        
        # Invalid subsets should raise
        with pytest.raises(ValueError, match="Invalid subsets"):
            benchmark._validate_subsets(["invalid_task"])
    
    @patch('benchmark.base.load_dataset')
    def test_load_datasets_with_task_column(self, mock_load_dataset, mock_dataset_df):
        """Test dataset loading functionality when task column exists."""
        # Mock the datasets load_dataset function
        mock_dataset = Mock()
        mock_dataset.to_pandas.return_value = mock_dataset_df
        mock_load_dataset.return_value = mock_dataset
        
        benchmark = MockBenchmark(subsets_to_run=["test_task1"])
        df = benchmark._load_datasets()
        
        # Should filter to only test_task1
        assert len(df) == 2  # Two rows with test_task1
        assert all(df["task"] == "test_task1")
        mock_load_dataset.assert_called_once_with("mock/test_dataset", split="test")
    
    @patch('benchmark.base.load_dataset')
    def test_load_datasets_without_task_column(self, mock_load_dataset):
        """Test dataset loading when task column doesn't exist."""
        # Create dataset without task column
        df_without_task = pd.DataFrame({
            "context": ["Context 1"],
            "question": ["Question 1"],
            "answers": [["Answer 1"]],
            "all_classes": [[]]
        })
        
        mock_dataset = Mock()
        mock_dataset.to_pandas.return_value = df_without_task
        mock_load_dataset.return_value = mock_dataset
        
        benchmark = MockBenchmark()
        df = benchmark._load_datasets()
        
        # Should return the full dataset since no task column to filter on
        assert len(df) == 1
        assert "context" in df.columns
    
    @patch('benchmark.base.load_dataset')
    def test_load_datasets_error_handling(self, mock_load_dataset):
        """Test dataset loading error handling."""
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        benchmark = MockBenchmark()
        with pytest.raises(Exception, match="Failed to load dataset mock/test_dataset"):
            benchmark._load_datasets()
    
    def test_validate_dataset_size_small(self, mock_dataset_df):
        """Test dataset size validation with small dataset."""
        benchmark = MockBenchmark()
        
        # Small dataset should not warn
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            benchmark._validate_dataset_size(mock_dataset_df)
            
            # Filter warnings to only UserWarnings about dataset size
            size_warnings = [warning for warning in w if "Repository not expected to handle large datasets" in str(warning.message)]
        assert len(size_warnings) == 0
    
    def test_validate_dataset_size_large(self, mock_dataset_df):
        """Test dataset size validation with large dataset."""
        benchmark = MockBenchmark()
        
        # Create large dataset
        large_df = pd.concat([mock_dataset_df] * 5000)  # ~15000 rows
        
        with pytest.warns(UserWarning, match="Repository not expected to handle large datasets"):
            benchmark._validate_dataset_size(large_df)
    
    def test_process_all_requests_multiple_questions_per_context(self, mock_adapter, mock_dataset_df):
        """Test processing requests with multiple questions per context."""
        benchmark = MockBenchmark()
        results_df = benchmark._process_all_requests(mock_adapter, mock_dataset_df, {}, {})
        
        # Check that predicted_answer column was added
        assert "predicted_answer" in results_df.columns
        assert len(results_df) == 3
        
        # Check that adapter was called for each unique context
        assert mock_adapter.process_request.called
        
        # Verify responses are assigned correctly
        for answer in results_df["predicted_answer"]:
            assert answer.startswith("Response to")
    
    def test_process_all_requests_single_question_response(self, mock_dataset_df):
        """Test processing when adapter returns single string response."""
        # Create adapter that returns single string for multiple questions
        single_response_adapter = Mock()
        single_response_adapter.process_request.return_value = Mock(responses="Single response")
        
        benchmark = MockBenchmark()
        results_df = benchmark._process_all_requests(single_response_adapter, mock_dataset_df, {}, {})
        
        # Should handle single response for multiple questions
        context1_rows = results_df[results_df["context"] == "Context 1"]
        assert len(context1_rows) == 2
        assert all(context1_rows["predicted_answer"] == "Single response")
    
    def test_process_all_requests_error_handling(self, mock_dataset_df):
        """Test error handling during request processing."""
        # Create adapter that throws errors
        failing_adapter = Mock()
        failing_adapter.process_request.side_effect = Exception("Adapter failed")
        
        benchmark = MockBenchmark()
        results_df = benchmark._process_all_requests(failing_adapter, mock_dataset_df, {}, {})
        
        # Should handle errors gracefully and continue processing
        assert len(results_df) == 3
        assert "predicted_answer" in results_df.columns
        
        # Should have empty responses for failed contexts
        assert all(answer == "" for answer in results_df["predicted_answer"])
    
    def test_process_all_requests_memory_cleanup(self, mock_adapter, mock_dataset_df):
        """Test that memory cleanup is called."""
        with patch('benchmark.base.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            
            benchmark = MockBenchmark()
            benchmark._process_all_requests(mock_adapter, mock_dataset_df, {}, {})
            
            # Should call empty_cache for each context group
            assert mock_torch.cuda.empty_cache.called
