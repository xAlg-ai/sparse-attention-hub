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


def pages_to_token_mask(topk_pages: torch.Tensor, K: int, page_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert selected page indices into a {0,1} token mask (float) of shape [B,H,Q,K].
    Boolean ops are used only locally; the returned mask is float.
    """
    if K == 0 or topk_pages.numel() == 0:
        return torch.zeros((*topk_pages.shape[:3], K), device=device, dtype=dtype)

    B, H, Q, Kp = topk_pages.shape
    P = (K + page_size - 1) // page_size

    token_idx = torch.arange(K, device=device)       # [K]
    page_idx  = torch.arange(P, device=device)       # [P]
    start = page_idx * page_size                     # [P]
    end   = torch.clamp(start + page_size, max=K)    # [P]

    # [P,K] bool: token k is inside page p (local boolean usage)
    page_token_mask_bool = (token_idx.unsqueeze(0) >= start.unsqueeze(1)) & \
                            (token_idx.unsqueeze(0) <  end.unsqueeze(1))

    # Gather masks for the selected pages: [B,H,Q,Kp,K] -> union across pages
    selected = page_token_mask_bool[topk_pages]          # bool
    union_selected = selected.any(dim=3).to(dtype)       # float in {0,1}
    return union_selected

def attention_mask_to_allowed_prob(attention_mask: torch.Tensor, K: int) -> torch.Tensor:
    """
    Convert attention_mask to allowed-probabilities in [0,1], shape [B,1,*,K].
    Heuristics:
        - bool masks: 0 => allow (1.0), 1 => forbid (0.0)
        - additive float masks: >=0 => allow (1.0), negative => forbid (0.0)
    """
    am = attention_mask[..., :K]
    if am.dtype == torch.bool:
        allowed = (am == 0).to(torch.float32)
    else:
        allowed = (am >= 0).to(torch.float32)
    if allowed.dim() == 3:
        allowed = allowed.unsqueeze(1)  # [B,1,*,K]
    return allowed


def pages_valid(allowed_prob: torch.Tensor, page_size: int, num_pages: int) -> torch.Tensor:
    """
    allowed_prob: [B,1,Q,K] or [B,1,1,K], float in [0,1]
    Returns: [B,Q,P] (bool) whether each page has any token with allowed_prob > 0.
    """
    B, _, Q_or_one, K = allowed_prob.shape
    Q = Q_or_one
    K_pad = num_pages * page_size
    pad_k = K_pad - K

    if pad_k > 0:
        ap = F.pad(allowed_prob, (0, pad_k), value=0.0)
    else:
        ap = allowed_prob

    ap = ap.view(B, 1, Q, num_pages, page_size)  # [B,1,Q,P,ps]
    return (ap.max(dim=-1).values > 0).squeeze(1)  # [B,Q,P] bool