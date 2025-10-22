"""Quest page-Top-K masker implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

import torch
from ray import tune

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import Masker


# ===============================
# Config
# ===============================
@dataclass
class QuestMaskerConfig(MaskerConfig):
    """Configuration for QuestMasker (page-Top-K)."""
    page_size: int = 128          # tokens per page
    top_k_pages: int = 8          # select K pages by Quest score

    # Optional search space for hyperparameter sweeps
    search_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "page_size": tune.grid_search([64, 128, 256]),
            "top_k_pages": tune.grid_search([4, 8, 12, 16]),
        }
    )


# ===============================
# Masker
# ===============================
@MaskerRegistry.register(QuestMaskerConfig)
class QuestMasker(Masker):
    """Quest (page-Top-K) attention masker.

    Stage 1: compute per-page min/max metadata and query-aware upper-bound scores.
    Stage 2: pick Top-K pages, enable all tokens in those pages (sparse attention).
    """

    def __init__(self, config: QuestMaskerConfig) -> None:
        super().__init__(config)
        self.page_size: int = int(config.page_size)
        self.top_k_pages: int = int(config.top_k_pages)
        if self.page_size <= 0 or self.top_k_pages <= 0:
            raise ValueError("page_size and top_k_pages must be positive integers")

    # ---------------------------
    # Public API
    # ---------------------------
    def add_mask(
        self,
        keys: torch.Tensor,             # [B, H_kv, K, D]
        queries: torch.Tensor,          # [B, H,    Q, D]
        values: torch.Tensor,           # (unused for mask) [B, H_kv, K, D]
        attention_mask: torch.Tensor,   # [B, 1, Q, K] or [B, 1, 1, K] or None
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add Quest page-Top-K sparse mask."""
        # Keep existing full mask if any
        if previous_mask.is_full_mask():
            return previous_mask

        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)

        # If too short, use full attention (no sparsity)
        if self._should_use_full_attention(dims):
            return self._create_full_mask(dims, previous_mask.dtype, previous_mask.device)

        quest_mask = self._create_quest_mask(
            dims, keys, queries, attention_mask, previous_mask
        )
        return previous_mask.merge_mask(quest_mask, inplace=False)

    # ---------------------------
    # Core QUEST logic
    # ---------------------------
    def _create_quest_mask(
        self,
        dims: AttentionTensorDimensions,
        keys: torch.Tensor,            # [B, H_kv, K, D]
        queries: torch.Tensor,         # [B, H,    Q, D]
        attention_mask: torch.Tensor,  # [B, 1, Q, K] or None
        previous_mask: Mask,
    ) -> Mask:
        """
        1) Repeat KV heads to match query heads (GQA/MQA -> MHA).
        2) Compute per-page min/max across K dimension (per batch/head/dim).
        3) Score pages with the Quest bound using queries.
        4) Select Top-K pages per (B,H,Q), expand to token indices.
        5) Build dense mask (bool) and wrap into Mask.
        """
        # Repeat KV to match query heads
        ngroups = _get_num_key_value_groups(queries, keys)  # H / H_kv
        keys_rep = repeat_kv(keys, ngroups)                 # [B, H, K, D]

        B, H, K, D = keys_rep.shape
        _, _, Q, _ = queries.shape
        page_size = self.page_size
        num_pages = (K + page_size - 1) // page_size
        # Safety: if somehow K==0, fall back to full mask (consistent shape)
        if num_pages == 0:
            return self._create_full_mask(dims, previous_mask.dtype, previous_mask.device)

        # Compute per-page min/max across tokens for each (B,H,D)
        # Shapes: page_min/max -> [B, H, P, D]
        page_min, page_max = self._compute_page_min_max(keys_rep, page_size, num_pages)

        # Score pages with Quest's upper bound
        # queries: [B,H,Q,D] ; page_{min,max}: [B,H,P,D]
        # score: [B,H,Q,P]
        page_scores = self._quest_page_scores(queries, page_min, page_max)

        # Gate pages by attention_mask validity (optional)
        if attention_mask is not None:
            # valid positions boolean: True where allowed
            # attention_mask is typically 0 for allowed, -inf for disallowed
            valid = (attention_mask[..., :K] == 0)  # [B, 1, Q, K] -> bool
            page_any_valid = self._page_any_valid(valid, page_size, num_pages)  # [B,Q,P]
            # broadcast to heads
            page_any_valid = page_any_valid.unsqueeze(1).expand(B, H, Q, num_pages)
            # set invalid pages to very negative to avoid selection
            page_scores = torch.where(
                page_any_valid, page_scores, torch.finfo(page_scores.dtype).min
            )

        # Select Top-K pages per (B,H,Q)
        k_pages = min(self.top_k_pages, num_pages)
        topk_pages = torch.topk(page_scores, k=k_pages, dim=-1, largest=True).indices  # [B,H,Q,k_pages]

        # Build/merge dense mask
        dense_mask = previous_mask.get_dense_mask()  # [B,H,Q,K] (bool/float)
        if dense_mask.dtype != torch.bool:
            dense_mask = dense_mask != 0

        # Activate all tokens in selected pages
        dense_mask = self._scatter_pages_to_dense_mask(
            dense_mask, topk_pages, K, page_size
        )

        # Respect attention_mask if present: disallow masked tokens
        if attention_mask is not None:
            allowed = (attention_mask[..., :K] == 0)  # [B,1,Q,K] -> bool
            dense_mask = dense_mask & allowed.expand_as(dense_mask)

        # Wrap into framework Mask (keeps dtype of previous)
        mask_shape = (B, H, Q, K)
        return Mask.create_mask_from_dense_mask(mask_shape, dense_mask, dtype=previous_mask.dtype)

    # ---------------------------
    # Utilities
    # ---------------------------
    def _should_use_full_attention(self, dims: AttentionTensorDimensions) -> bool:
        """Use full attention when sequence is shorter than a single Top-K selection."""
        token_budget = self.top_k_pages * self.page_size
        return dims.seq_len_keys <= token_budget

    @staticmethod
    def _compute_page_min_max(
        keys_rep: torch.Tensor, page_size: int, num_pages: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        keys_rep: [B, H, K, D]
        Returns:
            page_min, page_max: [B, H, P, D]
        """
        B, H, K, D = keys_rep.shape
        # Pre-allocate for speed and correctness
        page_min = torch.empty((B, H, num_pages, D), dtype=keys_rep.dtype, device=keys_rep.device)
        page_max = torch.empty((B, H, num_pages, D), dtype=keys_rep.dtype, device=keys_rep.device)

        for p in range(num_pages):
            s = p * page_size
            e = min((p + 1) * page_size, K)
            k_slice = keys_rep[:, :, s:e, :]       # [B,H,L,D] (L>=1 guaranteed if num_pages>0)
            page_min[:, :, p, :] = k_slice.amin(dim=2)
            page_max[:, :, p, :] = k_slice.amax(dim=2)

        return page_min, page_max

    @staticmethod
    def _quest_page_scores(
        queries: torch.Tensor, page_min: torch.Tensor, page_max: torch.Tensor
    ) -> torch.Tensor:
        """
        queries:  [B,H,Q,D]
        page_min: [B,H,P,D]
        page_max: [B,H,P,D]
        Returns:
            scores: [B,H,Q,P] = sum_j max(q_j*min_j, q_j*max_j)
        """
        # Expand to [B,H,Q,P,D]
        q = queries.unsqueeze(3)        # [B,H,Q,1,D]
        pmin = page_min.unsqueeze(2)    # [B,H,1,P,D]
        pmax = page_max.unsqueeze(2)    # [B,H,1,P,D]
        prod_min = q * pmin
        prod_max = q * pmax
        upper = torch.maximum(prod_min, prod_max).sum(dim=-1)  # [B,H,Q,P]
        return upper

    @staticmethod
    def _page_any_valid(valid_mask: torch.Tensor, page_size: int, num_pages: int) -> torch.Tensor:
        """
        valid_mask: [B,1,Q,K] (bool)
        Returns:
            page_has_any_valid: [B,Q,P] (bool)
        """
        B, _, Q, K = valid_mask.shape
        out = []
        for p in range(num_pages):
            s = p * page_size
            e = min((p + 1) * page_size, K)
            out.append(valid_mask[:, 0, :, s:e].any(dim=-1))  # [B,Q]
        return torch.stack(out, dim=-1)  # [B,Q,P]

    @staticmethod
    def _scatter_pages_to_dense_mask(
        dense_mask: torch.Tensor, topk_pages: torch.Tensor, K: int, page_size: int
    ) -> torch.Tensor:
        """
        dense_mask: [B,H,Q,K] (bool)
        topk_pages: [B,H,Q,Kp] (int)
        Mark all tokens in the selected pages as True.
        """
        B, H, Q, _ = dense_mask.shape
        Kp = topk_pages.shape[-1]
        if Kp == 0 or K == 0:
            return dense_mask

        token_idx = torch.arange(K, device=dense_mask.device)  # [K]

        for i in range(Kp):
            pidx = topk_pages[..., i]                # [B,H,Q]
            s = pidx * page_size                     # [B,H,Q]
            e = torch.minimum(s + page_size, torch.as_tensor(K, device=dense_mask.device))
            s_exp = s.unsqueeze(-1)                  # [B,H,Q,1]
            e_exp = e.unsqueeze(-1)                  # [B,H,Q,1]
            in_range = (token_idx >= s_exp) & (token_idx < e_exp)  # [B,H,Q,K]
            dense_mask = dense_mask | in_range

        return dense_mask

    # ---------------------------
    # Registry helpers
    # ---------------------------
    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "QuestMasker":
        if not isinstance(config, QuestMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
