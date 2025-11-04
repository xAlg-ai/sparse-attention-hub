"""Quest utility functions."""

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_page_min_max(
    keys_rep: torch.Tensor, page_size: int, num_pages: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    keys_rep: [B, H, K, D]
    returns:
      page_min, page_max: [B, H, P=num_pages, D]
    """
    B, H, K, D = keys_rep.shape

    # Number of full pages and size of the trailing (partial) page
    P_full = K // page_size
    tail = K - P_full * page_size  # 0..page_size-1

    # Fast path: no tail -> just reshape and reduce once
    if tail == 0:
        x = keys_rep.reshape(B, H, P_full, page_size, D)  # safe even if non-contig
        page_min = x.amin(dim=3)  # [B,H,P_full,D]
        page_max = x.amax(dim=3)  # [B,H,P_full,D]
        return page_min, page_max  # here num_pages == P_full

    # General path: full pages + a tiny tail reduction, NO padding
    # Full pages part
    K_main = P_full * page_size
    main = keys_rep[..., :K_main, :].reshape(
        B, H, P_full, page_size, D
    )  # [B,H,P_full,ps,D]
    page_min_main = main.amin(dim=3)  # [B,H,P_full,D]
    page_max_main = main.amax(dim=3)  # [B,H,P_full,D]

    # Tail page (size = tail)
    tail_chunk = keys_rep[..., K_main:, :]  # [B,H,tail,D]
    tail_min = tail_chunk.amin(dim=2).unsqueeze(2)  # [B,H,1,D]
    tail_max = tail_chunk.amax(dim=2).unsqueeze(2)  # [B,H,1,D]

    # Concatenate full pages + tail to reach num_pages
    page_min = torch.cat([page_min_main, tail_min], dim=2)  # [B,H,P_full+1,D]
    page_max = torch.cat([page_max_main, tail_max], dim=2)  # [B,H,P_full+1,D]

    # Sanity: P_full + 1 must equal num_pages
    # (If your caller precomputed num_pages, it should already be ceil(K/page_size))
    return page_min, page_max


def quest_page_scores(
    queries: torch.Tensor, page_min: torch.Tensor, page_max: torch.Tensor
) -> torch.Tensor:
    """
    queries:  [B,H,Q,D]
    page_min: [B,H,P,D]
    page_max: [B,H,P,D]
    Returns:
        scores: [B,H,Q,P] = sum_j max(q_j * min_j, q_j * max_j)
    """
    q = queries.unsqueeze(3)  # [B,H,Q,1,D]
    pmin = page_min.unsqueeze(2)  # [B,H,1,P,D]
    pmax = page_max.unsqueeze(2)  # [B,H,1,P,D]
    prod_min = q * pmin
    prod_max = q * pmax
    return torch.maximum(prod_min, prod_max).sum(dim=-1)  # [B,H,Q,P]


def pages_to_token_mask(
    topk_pages: torch.Tensor,
    K: int,
    page_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return dense {0,1} float mask [B,H,Q,K] without constructing [P,K].
    """
    if K == 0 or topk_pages.numel() == 0:
        return torch.zeros((*topk_pages.shape[:3], K), device=device, dtype=dtype)

    B, H, Q, Kp = topk_pages.shape
    P = (K + page_size - 1) // page_size

    # [B,H,Q,P] page flags
    page_sel = torch.zeros((B, H, Q, P), device=device, dtype=dtype)
    page_sel.scatter_(-1, topk_pages, 1.0)  # in-place, no big temps

    # Expand to tokens (blocks of size page_size), then trim tail
    token_mask = page_sel.repeat_interleave(page_size, dim=-1)[..., :K]
    return token_mask


def attention_mask_to_allowed_prob(
    attention_mask: torch.Tensor, K: int
) -> torch.Tensor:
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


def pages_valid(
    allowed_prob: torch.Tensor, page_size: int, num_pages: int
) -> torch.Tensor:
    """
    allowed_prob: [B,1,Q,K] or [B,1,1,K], float in [0,1]
    Returns: [B,Q,P] (bool) whether each page has any token with allowed_prob > 0.
    """
    B, H, Q_or_one, K = allowed_prob.shape
    Q = Q_or_one
    K_pad = num_pages * page_size
    pad_k = K_pad - K

    if pad_k > 0:
        ap = F.pad(allowed_prob, (0, pad_k), value=0.0)
    else:
        ap = allowed_prob

    if H != 1:
        ap = ap.max(dim=1, keepdim=True).values  # [B,1,Q,K_pad]
    ap = ap.view(B, 1, Q, num_pages, page_size)  # [B,1,Q,P,ps]
    return (ap.max(dim=-1).values > 0).squeeze(1)  # [B,Q,P] bool
