"""Unit tests for LongBench benchmark implementation."""

from unittest.mock import patch

import pytest
import pandas as pd

from benchmark import LongBench


class TestLongBenchUnit:
    """Unit tests for LongBench class."""
    
    def test_longbench_initialization(self):
        """Test LongBench initialization."""
        longbench = LongBench()
        assert len(longbench.all_datasets) == 34  # 21 standard + 13 extended
        assert longbench.benchmark_name == "longbench"
        assert longbench.huggingface_dataset_id == "Xnhyacinth/LongBench"
    
    def test_longbench_initialization_with_subsets(self):
        """Test LongBench initialization with custom subsets."""
        subsets = ["narrativeqa", "trec"]
        longbench = LongBench(subsets_to_run=subsets)
        assert longbench.subsets_to_run == subsets
    
    def test_longbench_dataset_list_standard(self):
        """Test that LongBench has correct standard datasets."""
        longbench = LongBench()
        
        # Check some key standard datasets
        standard_datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
            "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", 
            "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", 
            "lsht", "passage_count", "passage_retrieval_en", 
            "passage_retrieval_zh", "lcc", "repobench-p"
        ]
        
        for dataset in standard_datasets:
            assert dataset in longbench.all_datasets
        
        # Verify count of standard datasets
        standard_in_list = [d for d in longbench.all_datasets if not d.endswith("_e")]
        assert len(standard_in_list) == 21
    
    def test_longbench_dataset_list_extended(self):
        """Test that LongBench has correct extended datasets."""
        longbench = LongBench()
        
        # Check some extended datasets
        extended_datasets = [
            "qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e", 
            "gov_report_e", "multi_news_e", "trec_e", "triviaqa_e", 
            "samsum_e", "passage_count_e", "passage_retrieval_en_e", 
            "lcc_e", "repobench-p_e"
        ]
        
        for dataset in extended_datasets:
            assert dataset in longbench.all_datasets
        
        # Verify count of extended datasets
        extended_in_list = [d for d in longbench.all_datasets if d.endswith("_e")]
        assert len(extended_in_list) == 13
    
    def test_longbench_total_dataset_count(self):
        """Test total dataset count is correct."""
        longbench = LongBench()
        assert len(longbench.all_datasets) == 34  # 21 + 13
        
        # Ensure no duplicates
        assert len(set(longbench.all_datasets)) == 34
    
    def test_longbench_subset_selection_valid(self):
        """Test LongBench subset selection with valid datasets."""
        # Test with standard datasets
        longbench = LongBench(subsets_to_run=["narrativeqa", "trec"])
        assert longbench.subsets_to_run == ["narrativeqa", "trec"]
        
        # Test with extended datasets
        longbench = LongBench(subsets_to_run=["qasper_e", "trec_e"])
        assert longbench.subsets_to_run == ["qasper_e", "trec_e"]
        
        # Test with mixed datasets
        longbench = LongBench(subsets_to_run=["narrativeqa", "qasper_e"])
        assert longbench.subsets_to_run == ["narrativeqa", "qasper_e"]
    
    def test_longbench_subset_selection_invalid(self):
        """Test LongBench subset selection with invalid datasets."""
        with pytest.raises(ValueError, match="Invalid subsets"):
            LongBench(subsets_to_run=["invalid_dataset"])
        
        with pytest.raises(ValueError, match="Invalid subsets"):
            LongBench(subsets_to_run=["narrativeqa", "invalid_dataset"])
    
    def test_longbench_post_run_evaluate_empty_results(self):
        """Test LongBench evaluation with empty results."""
        longbench = LongBench()
        empty_df = pd.DataFrame()
        
        results = longbench.post_run_evaluate(empty_df)
        assert "error" in results
        assert results["error"] == "No results to evaluate"
    
    def test_longbench_post_run_evaluate_standard_datasets(self):
        """Test LongBench evaluation for standard datasets."""
        longbench = LongBench()
        
        # Mock data for standard datasets
        mock_results = pd.DataFrame({
            "task": ["narrativeqa", "narrativeqa", "trec", "trec"],
            "predicted_answer": ["Answer 1", "Answer 2", "Answer 3", "Answer 4"],
            "answers": [["Truth 1"], ["Truth 2"], ["Truth 3"], ["Truth 4"]],
            "all_classes": [[], [], ["class1", "class2"], ["class1", "class2"]]
        })
        
        with patch('benchmark.longbench.longbench.calculate_metrics') as mock_calc:
            # Mock different scores for different tasks
            mock_calc.side_effect = [75.5, 82.3]  # narrativeqa, trec
            
            results = longbench.post_run_evaluate(mock_results)
        
        # Check structure
        assert "task_scores" in results
        assert "standard_overall_score" in results
        assert "summary" in results
        
        # Check task scores
        assert "narrativeqa" in results["task_scores"]
        assert "trec" in results["task_scores"]
        assert results["task_scores"]["narrativeqa"] == 75.5
        assert results["task_scores"]["trec"] == 82.3
        
        # Check overall score (average of 75.5 and 82.3)
        expected_avg = round((75.5 + 82.3) / 2, 2)
        assert results["standard_overall_score"] == expected_avg
        assert results["overall_score"] == expected_avg
        
        # Check summary
        assert results["summary"]["total_tasks"] == 2
        assert results["summary"]["standard_tasks"] == 2
        assert results["summary"]["extended_tasks"] == 0
        assert results["summary"]["total_samples"] == 4
    
    def test_longbench_post_run_evaluate_extended_datasets(self):
        """Test LongBench evaluation for extended datasets."""
        longbench = LongBench()
        
        # Mock data for extended datasets
        mock_results = pd.DataFrame({
            "task": ["qasper_e", "qasper_e", "trec_e", "trec_e"],
            "predicted_answer": ["Answer 1", "Answer 2", "Answer 3", "Answer 4"],
            "answers": [["Truth 1"], ["Truth 2"], ["Truth 3"], ["Truth 4"]],
            "all_classes": [[], [], ["class1"], ["class1"]],
            "length": [3000, 6000, 2000, 9000]  # Different lengths for extended dataset
        })
        
        mock_qasper_scores = {"0-4k": 80.0, "4-8k": 75.0, "8k+": 70.0}
        mock_trec_scores = {"0-4k": 85.0, "4-8k": 80.0, "8k+": 75.0}
        
        with patch('benchmark.longbench.longbench.calculate_metrics_e') as mock_calc_e:
            mock_calc_e.side_effect = [mock_qasper_scores, mock_trec_scores]
            
            results = longbench.post_run_evaluate(mock_results)
        
        # Check structure
        assert "task_scores" in results
        assert "extended_overall_scores" in results
        assert "summary" in results
        
        # Check task scores
        assert "qasper_e" in results["task_scores"]
        assert "trec_e" in results["task_scores"]
        assert results["task_scores"]["qasper_e"] == mock_qasper_scores
        assert results["task_scores"]["trec_e"] == mock_trec_scores
        
        # Check extended overall scores (averages)
        assert results["extended_overall_scores"]["0-4k"] == round((80.0 + 85.0) / 2, 2)
        assert results["extended_overall_scores"]["4-8k"] == round((75.0 + 80.0) / 2, 2) 
        assert results["extended_overall_scores"]["8k+"] == round((70.0 + 75.0) / 2, 2)
        
        # Check summary
        assert results["summary"]["total_tasks"] == 2
        assert results["summary"]["standard_tasks"] == 0
        assert results["summary"]["extended_tasks"] == 2
        assert results["summary"]["total_samples"] == 4
    
    def test_longbench_post_run_evaluate_mixed_datasets(self):
        """Test LongBench evaluation with both standard and extended datasets."""
        longbench = LongBench()
        
        # Mock data for mixed datasets
        mock_results = pd.DataFrame({
            "task": ["narrativeqa", "qasper_e", "trec", "trec_e"],
            "predicted_answer": ["Answer 1", "Answer 2", "Answer 3", "Answer 4"],
            "answers": [["Truth 1"], ["Truth 2"], ["Truth 3"], ["Truth 4"]],
            "all_classes": [[], [], ["class1"], ["class1"]],
            "length": [None, 5000, None, 7000]  # length only for extended
        })
        
        mock_extended_scores = {"0-4k": 80.0, "4-8k": 85.0, "8k+": 75.0}
        
        with patch('benchmark.longbench.longbench.calculate_metrics') as mock_calc:
            with patch('benchmark.longbench.longbench.calculate_metrics_e') as mock_calc_e:
                mock_calc.side_effect = [75.0, 82.0]  # narrativeqa, trec
                mock_calc_e.side_effect = [mock_extended_scores, mock_extended_scores]  # qasper_e, trec_e
                
                results = longbench.post_run_evaluate(mock_results)
        
        # Should have both standard and extended scores
        assert "standard_overall_score" in results
        assert "extended_overall_scores" in results
        assert "overall_score" in results
        
        # Check overall score includes all scores
        all_scores = [75.0, 82.0, 80.0, 85.0, 75.0, 80.0, 85.0, 75.0]  # standard + extended values
        expected_overall = round(sum(all_scores) / len(all_scores), 2)
        assert results["overall_score"] == expected_overall
        
        # Check summary counts
        assert results["summary"]["standard_tasks"] == 2
        assert results["summary"]["extended_tasks"] == 2
        assert results["summary"]["total_tasks"] == 4
    
    def test_longbench_post_run_evaluate_error_handling(self):
        """Test error handling during evaluation."""
        longbench = LongBench()
        
        # Create data that might cause evaluation errors
        mock_results = pd.DataFrame({
            "task": ["narrativeqa", "trec"],
            "predicted_answer": ["Answer 1", "Answer 2"],
            "answers": [["Truth 1"], ["Truth 2"]],
            "all_classes": [[], []]
        })
        
        with patch('benchmark.longbench.longbench.calculate_metrics') as mock_calc:
            # Make first call succeed, second call fail
            mock_calc.side_effect = [75.0, Exception("Evaluation failed")]
            
            results = longbench.post_run_evaluate(mock_results)
        
        # Should handle errors gracefully
        assert "task_scores" in results
        assert "narrativeqa" in results["task_scores"]
        assert "trec" in results["task_scores"]
        
        # Successful task should have score
        assert results["task_scores"]["narrativeqa"] == 75.0
        
        # Failed task should have error
        assert "error" in results["task_scores"]["trec"]
        assert "Evaluation failed" in results["task_scores"]["trec"]["error"]
    
    def test_longbench_post_run_evaluate_missing_length_column(self):
        """Test evaluation with extended dataset missing length column."""
        longbench = LongBench()
        
        # Extended dataset without length column (should cause error)
        mock_results = pd.DataFrame({
            "task": ["qasper_e"],
            "predicted_answer": ["Answer 1"],
            "answers": [["Truth 1"]],
            "all_classes": [[]]
            # Missing 'length' column
        })
        
        with patch('benchmark.longbench.longbench.calculate_metrics_e') as mock_calc_e:
            mock_calc_e.side_effect = KeyError("length column missing")
            
            results = longbench.post_run_evaluate(mock_results)
        
        # Should handle missing column error
        assert "task_scores" in results
        assert "qasper_e" in results["task_scores"]
        assert "error" in results["task_scores"]["qasper_e"] 
