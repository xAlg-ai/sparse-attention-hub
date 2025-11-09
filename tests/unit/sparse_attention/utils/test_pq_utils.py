"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for PQ utility functions.
"""

import pytest
import torch


@pytest.mark.unit
class TestPQUtilityFunctions:
    def test_ip2l2_augment(self):
        """Test IP2L2 augmentation for key vectors."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import ip2l2_augment

        n_groups = 2
        n_samples = 5
        d = 4
        
        torch.manual_seed(42)
        xb = torch.randn(n_groups, n_samples, d, dtype=torch.float32)
        
        xb_aug, phi = ip2l2_augment(xb)
        
        assert xb_aug.shape == (n_groups, n_samples, d + 1), (
            f"Expected shape ({n_groups}, {n_samples}, {d + 1}), got {xb_aug.shape}"
        )
        assert phi.shape == (n_groups, 1, 1), (
            f"Expected phi shape ({n_groups}, 1, 1), got {phi.shape}"
        )
        
        assert xb_aug.device == xb.device
        assert phi.device == xb.device
        
        norms_sq = (xb ** 2).sum(dim=2)
        max_norms_sq = norms_sq.max(dim=1)[0]
        
        for g in range(n_groups):
            expected_phi = max_norms_sq[g].item()
            computed_phi = phi[g, 0, 0].item()
            assert torch.isclose(
                torch.tensor(computed_phi),
                torch.tensor(expected_phi),
                rtol=1e-5,
                atol=1e-6
            ), f"Group {g}: phi mismatch, expected {expected_phi:.6f}, got {computed_phi:.6f}"
        
        assert torch.equal(xb_aug[:, :, :d], xb), (
            "Original dimensions should remain unchanged"
        )
        
        for g in range(n_groups):
            for i in range(n_samples):
                norm_sq = norms_sq[g, i].item()
                phi_val = phi[g, 0, 0].item()
                expected_aug = (phi_val - norm_sq) ** 0.5
                computed_aug = xb_aug[g, i, d].item()
                
                assert torch.isclose(
                    torch.tensor(computed_aug),
                    torch.tensor(expected_aug),
                    rtol=1e-5,
                    atol=1e-6
                ), (
                    f"Group {g}, sample {i}: augmented value mismatch, "
                    f"expected {expected_aug:.6f}, got {computed_aug:.6f}"
                )

    def test_ip2l2_augment_queries(self):
        """Test IP2L2 augmentation for query vectors (should add zero column)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            ip2l2_augment_queries,
        )

        n_groups = 3
        n_queries = 4
        d = 8
        
        torch.manual_seed(42)
        xq = torch.randn(n_groups, n_queries, d, dtype=torch.float32)
        phi = torch.ones(n_groups, 1, 1, dtype=torch.float32) * 10.0
        
        xq_aug = ip2l2_augment_queries(xq, phi)
        
        assert xq_aug.shape == (n_groups, n_queries, d + 1), (
            f"Expected shape ({n_groups}, {n_queries}, {d + 1}), got {xq_aug.shape}"
        )
        
        assert xq_aug.device == xq.device
        
        assert torch.equal(xq_aug[:, :, :d], xq), (
            "Original dimensions should remain unchanged"
        )
        
        zero_col = xq_aug[:, :, d]
        assert torch.all(zero_col == 0), (
            "Augmented column should be all zeros for query vectors"
        )
        
        phi_different = torch.ones(n_groups, 1, 1, dtype=torch.float32) * 100.0
        xq_aug_different = ip2l2_augment_queries(xq, phi_different)
        
        assert torch.equal(xq_aug, xq_aug_different), (
            "Query augmentation should be independent of phi value"
        )
        
        x = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
        q = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)
        
        ip = torch.sum(x * q).item()
        
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import ip2l2_augment
        x_aug, phi = ip2l2_augment(x)
        q_aug = ip2l2_augment_queries(q, phi)
        
        l2_sq = torch.sum((x_aug - q_aug) ** 2).item()
        q_norm_sq = torch.sum(q ** 2).item()
        phi_val = phi.item()
        expected_l2_sq = phi_val + q_norm_sq - 2 * ip
        
        assert torch.isclose(
            torch.tensor(l2_sq),
            torch.tensor(expected_l2_sq),
            rtol=1e-5,
            atol=1e-6
        ), (
            f"L2 distance relationship doesn't hold: {l2_sq:.6f} != {expected_l2_sq:.6f}"
        )

