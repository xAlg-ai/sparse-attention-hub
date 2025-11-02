"""Double sparsity utility functions."""
import json
import torch.nn.functional as F
from typing import Dict, List, Tuple

import torch    


def compute_page_min_max(keys_rep: torch.Tensor, page_size: int, num_pages: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-page elementwise min/max over keys.
    keys_rep: [B, H, K, D]
    Returns:
        page_min, page_max: [B, H, P, D]
    """
    B, H, K, D = keys_rep.shape
    K_pad = num_pages * page_size
    pad_k = K_pad - K  # >= 0

    if pad_k > 0:
        # Pad along K with constants that won't affect reductions
        pad_min = F.pad(keys_rep, (0, 0, 0, pad_k), value=float("inf"))
        pad_max = F.pad(keys_rep, (0, 0, 0, pad_k), value=float("-inf"))
    else:
        pad_min = keys_rep
        pad_max = keys_rep

    # Reshape to [B,H,P,page_size,D]
    pad_min = pad_min.view(B, H, num_pages, page_size, D)
    pad_max = pad_max.view(B, H, num_pages, page_size, D)

    # Reduce across the page token axis
    page_min = pad_min.amin(dim=3)  # [B,H,P,D]
    page_max = pad_max.amax(dim=3)  # [B,H,P,D]
    return page_min, page_max


def quest_page_scores(queries: torch.Tensor, page_min: torch.Tensor, page_max: torch.Tensor) -> torch.Tensor:
    """
    queries:  [B,H,Q,D]
    page_min: [B,H,P,D]
    page_max: [B,H,P,D]
    Returns:
        scores: [B,H,Q,P] = sum_j max(q_j * min_j, q_j * max_j)
    """
    q = queries.unsqueeze(3)      # [B,H,Q,1,D]
    pmin = page_min.unsqueeze(2)  # [B,H,1,P,D]
    pmax = page_max.unsqueeze(2)  # [B,H,1,P,D]
    prod_min = q * pmin
    prod_max = q * pmax
    return torch.maximum(prod_min, prod_max).sum(dim=-1)  # [B,H,Q,P]