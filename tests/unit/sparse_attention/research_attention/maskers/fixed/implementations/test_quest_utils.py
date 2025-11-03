import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.quest_utils import (
    attention_mask_to_allowed_prob,
    compute_page_min_max,
    pages_to_token_mask,
    pages_valid,
    quest_page_scores,
)


@pytest.fixture
def sample_keys():
    # Create a small test tensor with known values
    # Shape: [B=1, H=2, K=6, D=4]
    return torch.tensor(
        [
            [  # Batch 0
                [  # Head 0
                    [1.0, 2.0, 3.0, 4.0],  # Token 0
                    [2.0, 1.0, 4.0, 3.0],  # Token 1
                    [3.0, 4.0, 1.0, 2.0],  # Token 2
                    [4.0, 3.0, 2.0, 1.0],  # Token 3
                    [5.0, 6.0, 7.0, 8.0],  # Token 4
                    [6.0, 5.0, 8.0, 7.0],  # Token 5
                ],
                [  # Head 1
                    [1.5, 2.5, 3.5, 4.5],
                    [2.5, 1.5, 4.5, 3.5],
                    [3.5, 4.5, 1.5, 2.5],
                    [4.5, 3.5, 2.5, 1.5],
                    [5.5, 6.5, 7.5, 8.5],
                    [6.5, 5.5, 8.5, 7.5],
                ],
            ]
        ]
    ).float()


@pytest.fixture
def sample_queries():
    # Shape: [B=1, H=2, Q=2, D=4]
    return torch.tensor(
        [
            [  # Batch 0
                [  # Head 0
                    [0.5, 1.0, 1.5, 2.0],  # Query 0
                    [2.0, 1.5, 1.0, 0.5],  # Query 1
                ],
                [  # Head 1
                    [1.5, 2.0, 2.5, 3.0],
                    [3.0, 2.5, 2.0, 1.5],
                ],
            ]
        ]
    ).float()


@pytest.fixture
def sample_topk_pages():
    """Fixture for testing page-to-token conversion.
    Returns tensor of shape [B=2, H=2, Q=2, Kp=3] containing page indices.
    """
    return torch.tensor(
        [
            [  # Batch 0
                [  # Head 0
                    [0, 1, 2],  # Query 0: pages 0,1,2
                    [1, 2, 3],  # Query 1: pages 1,2,3
                ],
                [  # Head 1
                    [0, 2, 3],
                    [1, 2, 4],
                ],
            ],
            [  # Batch 1
                [  # Head 0
                    [0, 1, 3],
                    [2, 3, 4],
                ],
                [  # Head 1
                    [1, 3, 4],
                    [0, 2, 4],
                ],
            ],
        ]
    )


def test_compute_page_min_max_with_padding(sample_keys):
    # Test with page_size=4 (requires padding)
    page_size = 4
    num_pages = 2  # Ceil(6/4) = 2

    page_min, page_max = compute_page_min_max(sample_keys, page_size, num_pages)

    # Check shapes
    assert page_min.shape == (1, 2, 2, 4)  # [B=1, H=2, P=2, D=4]
    assert page_max.shape == (1, 2, 2, 4)

    # First page should contain actual mins/maxs
    expected_min_h0_p0 = torch.min(sample_keys[0, 0, 0:4], dim=0)[0]
    expected_max_h0_p0 = torch.max(sample_keys[0, 0, 0:4], dim=0)[0]

    assert torch.allclose(page_min[0, 0, 0], expected_min_h0_p0)
    assert torch.allclose(page_max[0, 0, 0], expected_max_h0_p0)

    # Second page should handle padding correctly
    # Min should be min of remaining tokens (padding is inf)
    expected_min_h0_p1 = torch.min(sample_keys[0, 0, 4:6], dim=0)[0]
    # Max should be max of remaining tokens (padding is -inf)
    expected_max_h0_p1 = torch.max(sample_keys[0, 0, 4:6], dim=0)[0]

    assert torch.allclose(page_min[0, 0, 1], expected_min_h0_p1)
    assert torch.allclose(page_max[0, 0, 1], expected_max_h0_p1)


def test_quest_page_scores_broadcasting(sample_queries):
    # Test with different batch sizes
    B, H, Q, D = sample_queries.shape
    P = 3  # num pages

    # Create page tensors with different batch size
    page_min = torch.randn(2, H, P, D)  # 2 batches
    page_max = torch.randn(2, H, P, D)

    # Query has batch size 1, should broadcast
    scores = quest_page_scores(sample_queries, page_min, page_max)

    assert scores.shape == (2, H, Q, P)  # Output takes larger batch size


def test_compute_page_min_max_no_padding(sample_keys):
    page_size = 2
    num_pages = 3

    page_min, page_max = compute_page_min_max(sample_keys, page_size, num_pages)

    # Check shapes
    assert page_min.shape == (1, 2, 3, 4)
    assert page_max.shape == (1, 2, 3, 4)

    # Check ALL pages for BOTH heads
    for h in range(2):  # Check both heads
        for p in range(3):  # Check all pages
            start_idx = p * page_size
            end_idx = start_idx + page_size
            expected_min = torch.min(sample_keys[0, h, start_idx:end_idx], dim=0)[0]
            expected_max = torch.max(sample_keys[0, h, start_idx:end_idx], dim=0)[0]

            assert torch.allclose(
                page_min[0, h, p], expected_min
            ), f"Min mismatch at head {h}, page {p}"
            assert torch.allclose(
                page_max[0, h, p], expected_max
            ), f"Max mismatch at head {h}, page {p}"


def test_quest_page_scores(sample_queries):
    # Create controlled test data
    B, H, Q, D = sample_queries.shape
    P = 3

    page_min = torch.ones(1, H, P, D)
    page_max = torch.ones(1, H, P, D) * 2

    scores = quest_page_scores(sample_queries, page_min, page_max)

    # Check shape
    assert scores.shape == (1, H, Q, P)

    # Verify scores for ALL queries in ALL heads
    for h in range(H):
        for q in range(Q):
            query = sample_queries[0, h, q]  # [D]
            for p in range(P):
                # Manual computation of score for this (head, query, page)
                prod_min = query * page_min[0, h, p]  # [D]
                prod_max = query * page_max[0, h, p]  # [D]
                expected_score = torch.maximum(prod_min, prod_max).sum()

                assert torch.allclose(
                    scores[0, h, q, p], expected_score, rtol=1e-5
                ), f"Score mismatch at head {h}, query {q}, page {p}"


def test_quest_page_scores_edge_cases():
    # Test more edge cases
    queries = torch.tensor(
        [
            [  # Batch 0
                [  # Head 0
                    [1.0, -1.0, 0.0, float("inf")],  # Query with mixed values
                ]
            ]
        ]
    )  # [B=1, H=1, Q=1, D=4]

    page_min = torch.tensor(
        [
            [  # Batch 0
                [  # Head 0
                    [0.0, 0.0, 1.0, -float("inf")],  # Page with extremes
                ]
            ]
        ]
    )  # [B=1, H=1, P=1, D=4]

    page_max = torch.tensor(
        [
            [  # Batch 0
                [  # Head 0
                    [1.0, 1.0, 2.0, float("inf")],
                ]
            ]
        ]
    )

    scores = quest_page_scores(queries, page_min, page_max)

    # Manual verification for each dimension:
    # dim 0: max(1.0*0.0, 1.0*1.0) = 1.0
    # dim 1: max(-1.0*0.0, -1.0*1.0) = 0.0
    # dim 2: max(0.0*1.0, 0.0*2.0) = 0.0
    # dim 3: inf * (-inf) vs inf * inf -> inf
    expected_sum = 1.0 + 0.0 + 0.0 + float("inf")

    assert torch.allclose(
        scores[0, 0, 0, 0],
        torch.tensor(expected_sum),
        equal_nan=True,  # Handle NaN comparisons
    )


def test_quest_page_scores_numerical_stability():
    # Test numerical stability with very large/small values
    queries = torch.tensor([[[[1e10, 1e-10]]]])  # [B=1, H=1, Q=1, D=2]
    page_min = torch.tensor([[[[1e-10, 1e10]]]])
    page_max = torch.tensor([[[[1e10, 1e20]]]])

    scores = quest_page_scores(queries, page_min, page_max)

    # Manually compute expected results
    expected = torch.tensor([[[[1e20 + 1e10]]]])  # max(1e0, 1e20) + max(1e0, 1e10)

    assert torch.allclose(scores, expected, rtol=1e-5)


@pytest.mark.parametrize("page_size", [1, 2, 3, 6])
def test_compute_page_min_max_param(sample_keys, page_size):
    K = sample_keys.shape[2]
    num_pages = (K + page_size - 1) // page_size
    page_min, page_max = compute_page_min_max(sample_keys, page_size, num_pages)
    assert page_min.shape == (1, 2, num_pages, 4)
    assert page_max.shape == (1, 2, num_pages, 4)


def test_quest_page_scores_broadcasting_heads_queries(sample_queries):
    B, H, Q, D = sample_queries.shape  # 1,2,2,4
    P = 2
    # Broadcast H
    page_min = torch.randn(1, 1, P, D)
    page_max = torch.randn(1, 1, P, D)
    scores = quest_page_scores(sample_queries, page_min, page_max)
    assert scores.shape == (1, H, Q, P)


def test_integration_mask_chain():
    """Test the full chain of mask transformations."""
    # Start with attention mask - reshape to [B, Q, K]
    attn_mask = torch.tensor([[[False, True, False, True]]])  # [B=1, Q=1, K=4]
    K = 4
    page_size = 2
    num_pages = 2

    # Convert to allowed probabilities - should add head dimension
    allowed = attention_mask_to_allowed_prob(attn_mask, K)
    assert allowed.shape == (1, 1, 1, 4)  # [B, H=1, Q, K]
    assert allowed[0, 0, 0, 0] == 1.0  # First token allowed
    assert allowed[0, 0, 0, 1] == 0.0  # Second token forbidden

    # Check page validity
    valid_pages = pages_valid(allowed, page_size, num_pages)
    assert valid_pages.shape == (1, 1, 2)  # [B, Q, P]
    assert valid_pages[0, 0, 0]  # First page should be valid (has allowed token)

    # Create token mask from valid pages
    topk_pages = torch.tensor([[[[0]]]])  # [B=1, H=1, Q=1, Kp=1] Select first page only
    token_mask = pages_to_token_mask(
        topk_pages, K, page_size, device=torch.device("cpu"), dtype=torch.float32
    )

    assert token_mask.shape == (1, 1, 1, 4)  # [B, H, Q, K]
    expected_mask = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])  # First page tokens only
    assert torch.equal(token_mask, expected_mask)


def test_pages_valid_padding():
    """Test page validity with padding."""
    allowed_prob = torch.tensor([[[[1.0, 0.0, 1.0, 0.0, 1.0]]]])  # [B=1,1,Q=1,K=5]
    page_size = 2
    num_pages = 3  # Forces padding of last page

    valid = pages_valid(allowed_prob, page_size, num_pages)

    # Pages: [1,0], [1,0], [1,pad]
    expected = torch.tensor([[[True, True, True]]])

    assert torch.equal(valid, expected)
    assert valid.shape == (1, 1, 3)


def test_pages_valid_basic():
    """Test basic page validity checking."""
    allowed_prob = torch.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # Query 0
                    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                ]
            ]  # Query 1
        ]
    )  # Shape: [B=1, 1, Q=2, K=6]
    page_size = 2
    num_pages = 3

    valid = pages_valid(allowed_prob, page_size, num_pages)

    expected = torch.tensor(
        [
            [
                [True, False, True],  # Q0: Page0 valid, Page1 invalid, Page2 valid
                [False, True, True],
            ]  # Q1: Page0 invalid, Page1 valid, Page2 valid
        ]
    )

    assert torch.equal(valid, expected)
    assert valid.shape == (1, 2, 3)


def test_attention_mask_to_allowed_prob_float():
    """Test conversion of float attention masks."""
    attn_mask = torch.tensor(
        [
            [
                [1.0, -2.0, 0.5],  # Allow, Forbid, Allow
                [-1.0, 3.0, -0.1],
            ],  # Forbid, Allow, Forbid
        ]
    )  # Shape: [B=1, Q=2, K=3]
    K = 3

    allowed = attention_mask_to_allowed_prob(attn_mask, K)

    expected = torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]])

    assert torch.equal(allowed, expected)
    assert allowed.shape == (1, 1, 2, 3)


def test_attention_mask_to_allowed_prob_bool():
    """Test conversion of boolean attention masks."""
    attn_mask = torch.tensor(
        [
            [
                [False, True, False],  # Allow, Forbid, Allow
                [True, False, True],
            ],  # Forbid, Allow, Forbid
            [[False, False, True], [True, True, False]],
        ]
    )  # Shape: [B=2, Q=2, K=3]
    K = 3

    allowed = attention_mask_to_allowed_prob(attn_mask, K)

    expected = torch.tensor(
        [[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]], [[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]
    )

    assert torch.equal(allowed, expected)
    assert allowed.shape == (2, 1, 2, 3)


def test_pages_to_token_mask_empty():
    """Test handling of empty inputs."""
    empty_pages = torch.zeros((2, 2, 2, 0))  # Empty Kp dimension
    K = 10
    page_size = 2
    device = torch.device("cpu")
    dtype = torch.float32

    mask = pages_to_token_mask(empty_pages, K, page_size, device, dtype)
    assert mask.shape == (2, 2, 2, K)
    assert torch.all(mask == 0)


def test_pages_to_token_mask_basic(sample_topk_pages):
    """Test basic token mask generation from page indices."""
    B, H, Q, Kp = sample_topk_pages.shape
    K = 10  # Total tokens
    page_size = 2
    device = torch.device("cpu")
    dtype = torch.float32

    mask = pages_to_token_mask(sample_topk_pages, K, page_size, device, dtype)

    # Check shape and type
    assert mask.shape == (B, H, Q, K)
    assert mask.dtype == dtype

    # Check specific values for first batch, head, query
    # Pages [0,1,2] selected = tokens [0-5] should be 1.0
    expected_first = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=dtype)
    assert torch.equal(mask[0, 0, 0], expected_first)


if __name__ == "__main__":
    pytest.main([__file__])
