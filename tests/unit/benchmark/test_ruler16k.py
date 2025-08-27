"""Unit tests for Ruler16K benchmark implementation."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from benchmark.ruler16k import Ruler16K


class TestRuler16KUnit:
    """Unit tests for Ruler16K class."""

    def test_ruler16k_initialization(self):
        """Test Ruler16K initialization."""
        ruler16k = Ruler16K()
        assert len(ruler16k.all_datasets) == 13
        assert ruler16k.benchmark_name == "ruler16k"
        assert ruler16k.huggingface_dataset_id == "xAlg-AI/att-hub-ruler-16k"

    def test_ruler16k_initialization_with_subsets(self):
        """Test Ruler16K initialization with custom subsets."""
        subsets = ["niah_single_1", "qa_1"]
        ruler16k = Ruler16K(subsets_to_run=subsets)
        assert ruler16k.subsets_to_run == subsets

    def test_ruler16k_dataset_list_complete(self):
        """Test that Ruler16K has all expected datasets."""
        ruler16k = Ruler16K()

        expected_datasets = [
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
            "vt",
        ]

        for dataset in expected_datasets:
            assert dataset in ruler16k.all_datasets

        # Verify exact count
        assert len(ruler16k.all_datasets) == len(expected_datasets)

    def test_ruler16k_dataset_categories(self):
        """Test that Ruler16K datasets contain expected categories."""
        ruler16k = Ruler16K()

        # Check for different task categories
        niah_tasks = [d for d in ruler16k.all_datasets if d.startswith("niah_")]
        qa_tasks = [d for d in ruler16k.all_datasets if d.startswith("qa_")]
        other_tasks = [
            d for d in ruler16k.all_datasets if not d.startswith(("niah_", "qa_"))
        ]

        assert (
            len(niah_tasks) == 8
        )  # 3 single + 3 multikey + 1 multiquery + 1 multivalue
        assert len(qa_tasks) == 2  # qa_1, qa_2
        assert len(other_tasks) == 3  # cwe, fwe, vt

    def test_ruler16k_subset_selection_valid(self):
        """Test Ruler16K subset selection with valid datasets."""
        # Test with NIAH tasks
        ruler16k = Ruler16K(subsets_to_run=["niah_single_1", "niah_multikey_1"])
        assert ruler16k.subsets_to_run == ["niah_single_1", "niah_multikey_1"]

        # Test with QA tasks
        ruler16k = Ruler16K(subsets_to_run=["qa_1", "qa_2"])
        assert ruler16k.subsets_to_run == ["qa_1", "qa_2"]

        # Test with other tasks
        ruler16k = Ruler16K(subsets_to_run=["cwe", "fwe", "vt"])
        assert ruler16k.subsets_to_run == ["cwe", "fwe", "vt"]

    def test_ruler16k_subset_selection_invalid(self):
        """Test Ruler16K subset selection with invalid datasets."""
        with pytest.raises(ValueError, match="Invalid subsets"):
            Ruler16K(subsets_to_run=["invalid_dataset"])

        with pytest.raises(ValueError, match="Invalid subsets"):
            Ruler16K(subsets_to_run=["niah_single_1", "invalid_dataset"])

        with pytest.raises(ValueError, match="Invalid subsets"):
            Ruler16K(subsets_to_run=["ruler4k_niah_single_1"])  # wrong context length

    @patch("datasets.load_dataset")
    def test_ruler16k_load_datasets_success(self, mock_load_dataset):
        """Test successful dataset loading."""
        # Mock dataset for each subset
        mock_dataset1 = Mock()
        mock_df1 = pd.DataFrame(
            {
                "context": ["Context 1", "Context 2"],
                "question": ["Question 1", "Question 2"],
                "answer": [["Answer 1"], ["Answer 2"]],
                "task": ["niah_single_1", "niah_single_1"],
            }
        )
        mock_dataset1.to_pandas.return_value = mock_df1

        mock_dataset2 = Mock()
        mock_df2 = pd.DataFrame(
            {
                "context": ["Context 3"],
                "question": ["Question 3"],
                "answer": [["Answer 3"]],
                "task": ["qa_1"],
            }
        )
        mock_dataset2.to_pandas.return_value = mock_df2

        mock_load_dataset.side_effect = [mock_dataset1, mock_dataset2]

        ruler16k = Ruler16K(subsets_to_run=["niah_single_1", "qa_1"])
        df = ruler16k._load_datasets()

        # Check that datasets were loaded correctly
        assert len(df) == 3  # 2 + 1 samples
        assert "context_length" in df.columns
        assert all(df["context_length"] == 16384)

        # Check mock calls
        assert mock_load_dataset.call_count == 2
        mock_load_dataset.assert_any_call(
            "xAlg-AI/att-hub-ruler-16k", "niah_single_1", split="niah_single_1"
        )
        mock_load_dataset.assert_any_call(
            "xAlg-AI/att-hub-ruler-16k", "qa_1", split="qa_1"
        )

    @patch("datasets.load_dataset")
    def test_ruler16k_load_datasets_partial_failure(self, mock_load_dataset):
        """Test dataset loading with some failures."""
        # First dataset succeeds, second fails
        mock_dataset = Mock()
        mock_df = pd.DataFrame(
            {
                "context": ["Context 1"],
                "question": ["Question 1"],
                "answer": [["Answer 1"]],
                "task": ["niah_single_1"],
            }
        )
        mock_dataset.to_pandas.return_value = mock_df

        mock_load_dataset.side_effect = [mock_dataset, Exception("Dataset not found")]

        # Use valid subsets, but mock one to fail during loading
        ruler16k = Ruler16K(subsets_to_run=["niah_single_1", "qa_1"])
        df = ruler16k._load_datasets()

        # Should succeed with partial data (only niah_single_1 loaded)
        assert len(df) == 1
        assert "context_length" in df.columns
        assert all(df["context_length"] == 16384)

    @patch("datasets.load_dataset")
    def test_ruler16k_load_datasets_complete_failure(self, mock_load_dataset):
        """Test dataset loading with complete failure."""
        mock_load_dataset.side_effect = Exception("No datasets found")

        ruler16k = Ruler16K(subsets_to_run=["niah_single_1"])

        with pytest.raises(
            Exception, match="No Ruler subsets could be loaded successfully"
        ):
            ruler16k._load_datasets()

    def test_ruler16k_post_run_evaluate_empty_results(self):
        """Test Ruler16K evaluation with empty results."""
        ruler16k = Ruler16K()
        empty_df = pd.DataFrame()

        results = ruler16k.post_run_evaluate(empty_df)
        assert "error" in results
        assert results["error"] == "No results to evaluate"

    def test_ruler16k_post_run_evaluate_with_results(self):
        """Test Ruler16K evaluation with valid results."""
        ruler16k = Ruler16K()

        # Mock results DataFrame
        mock_results = pd.DataFrame(
            {
                "task": ["niah_single_1", "niah_single_1", "qa_1", "qa_1"],
                "predicted_answer": ["Answer 1", "Answer 2", "Answer 3", "Answer 4"],
                "answer": [["Truth 1"], ["Truth 2"], ["Truth 3"], ["Truth 4"]],
                "context_length": [16384, 16384, 16384, 16384],
            }
        )

        with patch("benchmark.ruler16k.ruler16k.calculate_metrics") as mock_calc:
            # Mock different scores for different tasks
            mock_calc.side_effect = [
                {
                    "niah_single_1": {"string_match": 85.0},
                    "qa_1": {"string_match": 90.0},
                },  # first call
                {
                    "niah_single_1": {"string_match": 85.0},
                    "qa_1": {"string_match": 90.0},
                },  # second call for context length
            ]

            results = ruler16k.post_run_evaluate(mock_results)

        # Check structure
        assert "overall_score" in results
        assert "task_scores" in results
        assert "context_length_scores" in results
        assert "summary" in results

        # Check overall score (average of 85.0 and 90.0)
        expected_avg = round((85.0 + 90.0) / 2, 2)
        assert results["overall_score"] == expected_avg

        # Check context length scores
        assert "16384" in results["context_length_scores"]
        assert results["context_length_scores"]["16384"] == expected_avg

        # Check summary
        assert results["summary"]["total_tasks"] == 2
        assert results["summary"]["total_samples"] == 4
        assert results["summary"]["context_lengths"] == ["16384"]

    def test_ruler16k_post_run_evaluate_different_task_types(self):
        """Test Ruler16K evaluation with different task types (QA vs others)."""
        ruler16k = Ruler16K()

        # Mock results with QA and non-QA tasks
        mock_results = pd.DataFrame(
            {
                "task": ["qa_1", "qa_1", "niah_single_1", "cwe"],
                "predicted_answer": [
                    "QA Answer 1",
                    "QA Answer 2",
                    "NIAH Answer",
                    "CWE Answer",
                ],
                "answer": [
                    ["QA Truth 1"],
                    ["QA Truth 2"],
                    ["NIAH Truth"],
                    ["CWE Truth"],
                ],
                "context_length": [16384, 16384, 16384, 16384],
            }
        )

        with patch("benchmark.ruler16k.ruler16k.calculate_metrics") as mock_calc:
            mock_calc.side_effect = [
                {
                    "qa_1": {"string_match": 80.0},
                    "niah_single_1": {"string_match": 75.0},
                    "cwe": {"string_match": 85.0},
                },
                {
                    "qa_1": {"string_match": 80.0},
                    "niah_single_1": {"string_match": 75.0},
                    "cwe": {"string_match": 85.0},
                },
            ]

            results = ruler16k.post_run_evaluate(mock_results)

        # Check that all task types are included
        assert "qa_1" in results["task_scores"]
        assert "niah_single_1" in results["task_scores"]
        assert "cwe" in results["task_scores"]

        # Check overall score includes all tasks
        expected_avg = round((80.0 + 75.0 + 85.0) / 3, 2)
        assert results["overall_score"] == expected_avg

    def test_ruler16k_post_run_evaluate_no_context_length(self):
        """Test evaluation when context_length column is missing."""
        ruler16k = Ruler16K()

        # Mock results without context_length column
        mock_results = pd.DataFrame(
            {
                "task": ["niah_single_1", "qa_1"],
                "predicted_answer": ["Answer 1", "Answer 2"],
                "answer": [["Truth 1"], ["Truth 2"]],
            }
        )

        with patch("benchmark.ruler16k.ruler16k.calculate_metrics") as mock_calc:
            mock_calc.return_value = {
                "niah_single_1": {"string_match": 85.0},
                "qa_1": {"string_match": 90.0},
            }

            results = ruler16k.post_run_evaluate(mock_results)

        # Should still work but without context_length_scores
        assert "overall_score" in results
        assert "task_scores" in results
        assert results["context_length_scores"] == {}
        assert results["summary"]["context_lengths"] == []

    def test_ruler16k_post_run_evaluate_error_handling(self):
        """Test error handling during evaluation."""
        ruler16k = Ruler16K()

        mock_results = pd.DataFrame(
            {
                "task": ["niah_single_1", "qa_1"],
                "predicted_answer": ["Answer 1", "Answer 2"],
                "answer": [["Truth 1"], ["Truth 2"]],
                "context_length": [16384, 16384],
            }
        )

        with patch("benchmark.ruler16k.ruler16k.calculate_metrics") as mock_calc:
            # First call succeeds, second call (for context length) fails
            successful_result = {
                "niah_single_1": {"string_match": 85.0},
                "qa_1": {"string_match": 90.0},
            }
            mock_calc.side_effect = [successful_result, Exception("Evaluation failed")]

            results = ruler16k.post_run_evaluate(mock_results)

        # Should handle errors gracefully
        assert "overall_score" in results
        assert "task_scores" in results

        # Should still compute overall score from successful first call
        expected_avg = round((85.0 + 90.0) / 2, 2)
        assert results["overall_score"] == expected_avg

        # Context length scores should be empty due to the error
        assert results["context_length_scores"] == {}

    def test_ruler16k_post_run_evaluate_missing_string_match(self):
        """Test evaluation when string_match key is missing from some results."""
        ruler16k = Ruler16K()

        mock_results = pd.DataFrame(
            {
                "task": ["niah_single_1", "qa_1"],
                "predicted_answer": ["Answer 1", "Answer 2"],
                "answer": [["Truth 1"], ["Truth 2"]],
                "context_length": [16384, 16384],
            }
        )

        with patch("benchmark.ruler16k.ruler16k.calculate_metrics") as mock_calc:
            # Return results where one task is missing string_match
            mock_calc.side_effect = [
                {
                    "niah_single_1": {"string_match": 85.0},
                    "qa_1": {"other_metric": 90.0},  # Missing string_match
                },
                {
                    "niah_single_1": {"string_match": 85.0},
                    "qa_1": {"other_metric": 90.0},
                },
            ]

            results = ruler16k.post_run_evaluate(mock_results)

        # Should only include tasks with string_match in overall score
        assert results["overall_score"] == 85.0  # Only niah_single_1 score
