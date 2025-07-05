"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-07-03
:summary: Tests for sparse attention. This file is part of the Sparse Attention Hub project.
"""


import pytest
import torch
import numpy as np
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import _compute_masked_exp_attention_weights, get_attention_denominator, get_attention_numerator, get_masked_attention_output
from sparse_attention_hub.sparse_attention.utils.mask import Mask
import mock

@pytest.mark.unit
class TestMaskExpWts:
    """Test class for mask attention utils."""

    def test_compute_masked_attention_weights(self):
        """Test that the masked attention weights are computed correctly."""
        # Test parameters
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        
        # Create test tensors
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        
        # Test case 1: No attention mask, empty sparse mask
        sparse_attention_mask = Mask.create_empty_mask((batch_size, num_heads, seq_len, seq_len))
        
        result = _compute_masked_exp_attention_weights(
            queries=queries,
            keys=keys,
            attention_mask=None,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask
        )
        
        # Verify shape
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Verify that result contains exponential values (should be positive)
        assert torch.all(result > 0)
        
        # Verify numerical correctness: should be exp(QK^T * scaling)
        expected_raw_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        expected_max = torch.max(expected_raw_weights, dim=-1, keepdim=True)[0]
        expected_exp_weights = torch.exp(expected_raw_weights - expected_max)
        
        assert torch.allclose(result, expected_exp_weights, atol=1e-6)


        
    def test_compute_masked_attention_weights_with_attention_mask(self):
        """Test masked attention weights computation with attention mask."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        
        # Create attention mask (lower triangular for causal attention)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float('-inf'))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        
        sparse_attention_mask = Mask.create_empty_mask((batch_size, num_heads, seq_len, seq_len))
        
        result = _compute_masked_exp_attention_weights(
            queries=queries,
            keys=keys,
            attention_mask=attention_mask,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask
        )
        
        # Verify shape
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Verify that upper triangular elements are zero (masked out)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(result[:, :, i, j], torch.zeros_like(result[:, :, i, j]))
        
        # Verify that lower triangular elements are positive
        for i in range(seq_len):
            for j in range(i + 1):
                assert torch.all(result[:, :, i, j] > 0)
    
    def test_compute_masked_attention_weights_with_sparse_mask(self):
        """Test masked attention weights computation with sparse attention mask."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        
        # Create a sparse attention mask (random pattern)
        dense_mask = torch.randint(0, 3, (batch_size, num_heads, seq_len, seq_len)) # assign weights
        dense_mask = dense_mask.float()
        sparse_attention_mask = Mask.create_mask_from_dense_mask(
            (batch_size, num_heads, seq_len, seq_len), 
            dense_mask
        )
        
        result = _compute_masked_exp_attention_weights(
            queries=queries,
            keys=keys,
            attention_mask=None,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask
        )
        
        # Verify shape
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Verify that masked positions (where dense_mask == 0) have zero values
        masked_positions = (dense_mask == 0)
        assert torch.allclose(result[masked_positions], torch.zeros_like(result[masked_positions]))
        
        # Verify that unmasked positions have positive values
        unmasked_positions = (dense_mask > 0)
        assert torch.all(result[unmasked_positions] > 0)

        # verify that the non-zero values are what we expect
        expected_raw_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        expected_max = torch.max(expected_raw_weights, dim=-1, keepdim=True)[0]
        expected_exp_weights = torch.exp(expected_raw_weights - expected_max)
        assert torch.allclose(result[unmasked_positions], expected_exp_weights[unmasked_positions] * dense_mask[unmasked_positions])
    
    def test_compute_masked_attention_weights_with_both_masks(self):
        """Test masked attention weights computation with both attention mask and sparse mask."""

        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        
        # Create causal attention mask
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float('-inf'))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        
        # Create a sparse attention mask (random pattern)
        dense_mask = torch.randint(0, 3, (batch_size, num_heads, seq_len, seq_len)) # assign weights
        dense_mask = dense_mask.float()
        sparse_attention_mask = Mask.create_mask_from_dense_mask(
            (batch_size, num_heads, seq_len, seq_len), 
            dense_mask
        )
        
        result = _compute_masked_exp_attention_weights(
            queries=queries,
            keys=keys,
            attention_mask=attention_mask,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask
        )
        
        # Verify shape
        assert result.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Verify that positions masked by either mask have zero values
        causal_mask = (attention_mask == float('-inf'))
        combined_mask = causal_mask | (dense_mask == 0)
        assert torch.allclose(result[combined_mask], torch.zeros_like(result[combined_mask]))
        
        # Verify that positions not masked by either mask have positive values
        active_positions = ~combined_mask
        assert torch.all(result[active_positions] > 0)

        # verify that the non-zero values are what we expect
        expected_raw_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scaling

        expected_max = torch.max(expected_raw_weights +  attention_mask, dim=-1, keepdim=True)[0]
        expected_exp_weights = torch.exp(expected_raw_weights - expected_max)
        assert torch.allclose(result[active_positions], expected_exp_weights[active_positions] * dense_mask[active_positions])

    
    
    def test_compute_masked_attention_weights_different_scaling(self):
        """Test masked attention weights computation with different scaling factors."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        
        sparse_attention_mask = Mask.create_empty_mask((batch_size, num_heads, seq_len, seq_len))
        
        # Test with different scaling factors
        scaling_factors = [0.1, 0.2, 0.3]
        
        for scaling in scaling_factors:
            result = _compute_masked_exp_attention_weights(
                queries=queries,
                keys=keys,
                attention_mask=None,
                scaling=scaling,
                sparse_attention_mask=sparse_attention_mask
            )
            
            # Verify shape
            assert result.shape == (batch_size, num_heads, seq_len, seq_len)
            
            # Verify that result contains positive values
            assert torch.all(result > 0)
            
            # Verify numerical correctness
            expected_raw_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
            expected_max = torch.max(expected_raw_weights, dim=-1, keepdim=True)[0]
            expected_exp_weights = torch.exp(expected_raw_weights - expected_max)
            
            assert torch.allclose(result, expected_exp_weights, atol=1e-6)
        

@pytest.mark.unit
class TestGetAttentionDenominator:
    """Test class for get attention denominator."""
    # TODO(aditya): test using correct mocking.
    def test_get_attention_denominator(self):
        ''' simple function: replicated from the original implementation to 
        ensure implementation persists'''

        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        
        # Create causal attention mask
        attention_mask = torch.randn(batch_size, num_heads, seq_len, seq_len)
        
        # Create a sparse attention mask (random pattern)
        sparse_attention_mask = Mask.create_empty_mask((batch_size, num_heads, seq_len, seq_len))

        module = torch.nn.Module()
        module.eval()
        module.num_key_value_groups = 1

        with mock.patch('sparse_attention_hub.sparse_attention.utils.mask_attention_utils._compute_masked_exp_attention_weights') as mock_compute_masked_exp_attention_weights:
            mock_compute_masked_exp_attention_weights.return_value = torch.randn(batch_size, num_heads, seq_len, seq_len)
            true_denominator = torch.sum(mock_compute_masked_exp_attention_weights.return_value, dim=-1, keepdim=True)
            denominator = get_attention_denominator(
                module=module,
                queries=queries,
                keys=keys,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=0.0,
                sparse_attention_mask=sparse_attention_mask
            )

        assert torch.allclose(denominator, true_denominator)



@pytest.mark.unit
class TestGetAttentionNumerator:
    """Test class for get attention numerator."""
    # TODO(aditya): test using correct mocking.
    def test_get_attention_numerator(self):
        ''' simple function: replicated from the original implementation to 
        ensure implementation persists'''

        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        values = torch.randn(batch_size, num_heads, seq_len, d_model)

        # Create causal attention mask
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float('-inf'))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        
        # Create a sparse attention mask (random pattern)
        dense_mask = torch.randint(0, 3, (batch_size, num_heads, seq_len, seq_len)) # assign weights
        dense_mask = dense_mask.float()
        sparse_attention_mask = Mask.create_mask_from_dense_mask(
            (batch_size, num_heads, seq_len, seq_len), 
            dense_mask
        )
        
        with mock.patch('sparse_attention_hub.sparse_attention.utils.mask_attention_utils._compute_masked_exp_attention_weights') as mock_compute_masked_exp_attention_weights:
            mock_compute_masked_exp_attention_weights.return_value = torch.randn(batch_size, num_heads, seq_len, seq_len)
            true_numerator = torch.matmul(mock_compute_masked_exp_attention_weights.return_value, values)
            numerator = get_attention_numerator(
                module=None,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=0.0,
                sparse_attention_mask=sparse_attention_mask
            )
        assert torch.allclose(numerator, true_numerator)


@pytest.mark.unit
class TestGetMaskedAttentionOutput:
    """Test class for get masked attention output."""
    
    def test_compare_with_eager_attention_for_no_mask(self):
        """Test that the masked attention output is the same as the eager attention output for no mask."""
        batch_size, num_heads, seq_len, d_model = 2, 4, 8, 16
        scaling = 1.0 / np.sqrt(d_model)
        dropout = 0.1
        
        queries = torch.randn(batch_size, num_heads, seq_len, d_model)
        keys = torch.randn(batch_size, num_heads, seq_len, d_model)
        values = torch.randn(batch_size, num_heads, seq_len, d_model)

        sparse_attention_mask = Mask.create_empty_mask((batch_size, num_heads, seq_len, seq_len))

        # Create attention mask (lower triangular for causal attention)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attention_mask.masked_fill_(attention_mask == 1, float('-inf'))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

        from transformers.models.llama.modeling_llama import eager_attention_forward
        module = torch.nn.Module()
        module.eval()
        print("Is Training: ", module.training)
        module.num_key_value_groups = 1

        eager_attention_output, _ = eager_attention_forward(
            module=module,
            query=queries,
            key=keys,
            value=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout
        )

        result = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask
        )
        print(result.shape, eager_attention_output.shape)
        assert torch.allclose(result, eager_attention_output, atol=1e-6)