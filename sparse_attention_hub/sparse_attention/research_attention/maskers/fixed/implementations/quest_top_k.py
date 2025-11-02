"""Quest Top-K masker implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn.functional as F
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

from ..base import TopKMasker, TopKMaskerConfig
from .utils.quest_utils import (
    compute_page_min_max,
    quest_page_scores
)

@dataclass
class QuestTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for QuestTopKMasker."""
    page_size: int
    search_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "heavy_size": tune.grid_search(
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
            ),
            "page_size": tune.grid_search([64, 128, 256]),
        }
    )

@MaskerRegistry.register(QuestTopKMaskerConfig)
class QuestTopKMasker(TopKMasker):
    """Quest page-Top-K masker."""

    page_size: int

    def __init__(self, config: QuestTopKMaskerConfig) -> None:
        super().__init__(config)
        if config.page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        self.page_size = int(config.page_size)
        self.heavy_size = config.heavy_size

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add Quest page-Top-K sparse mask."""
        if previous_mask.is_full_mask():
            return previous_mask

        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)
        effective_heavy_size: int = self._calculate_effective_heavy_size(dims.seq_len_keys)

        if self._should_use_full_attention(dims, effective_heavy_size):
            return self._create_full_mask(dims, previous_mask.dtype, previous_mask.device)

        return self._create_quest_page_topk_mask(
            dims,
            effective_heavy_size,
            keys,
            queries,
            attention_mask,
            previous_mask,
        )

    def _create_quest_page_topk_mask(
        self,
        dims: AttentionTensorDimensions,
        heavy_tokens: int,
        keys: torch.Tensor,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
        previous_mask: Mask,
    ) -> Mask:
        """
        Steps:
          1) Repeat KV for GQA/MQA -> MHA.
          2) Page keys; compute per-page min/max: [B,H,P,D].
          3) Score pages with Quest bound: [B,H,Q,P].
          4) Pick Top-K pages where K_pages ~= heavy_tokens / page_size.
          5) Convert chosen pages to token-level {0,1} (float), then probabilistic-OR (max) with previous probabilities.
          6) Gate by external attention_mask probabilities (multiply).
        """
        # 1) Handle GQA/MQA
        ngroups = _get_num_key_value_groups(queries, keys)
        keys_rep = repeat_kv(keys, ngroups)  # [B,H,K,D]

        B, H, K, _ = keys_rep.shape
        _, _, Q, _ = queries.shape

        page_size = self.page_size
        num_pages = (K + page_size - 1) // page_size

        # 2) Per-page min/max (vectorized) via shared utility
        page_min, page_max = compute_page_min_max(keys_rep, page_size, num_pages)  # [B,H,P,D]

        # 3) Quest page scores via shared utility
        page_scores = quest_page_scores(queries, page_min, page_max)  # [B,H,Q,P]

        # Convert external attention mask to allowed probabilities in [0,1]
        # and compute which pages are valid (have any allowed tokens).
        allowed_prob = None
        if attention_mask is not None:
            allowed_prob = self._attention_mask_to_allowed_prob(attention_mask, K)  # [B,1,*,K] float
            page_any_valid = self._page_any_valid(allowed_prob, page_size, num_pages)  # [B,Q,P] bool
            page_any_valid = page_any_valid.unsqueeze(1).expand(B, H, Q, num_pages)    # [B,H,Q,P]
            # Invalidate pages with no allowed tokens by pushing scores to -inf
            page_scores = torch.where(
                page_any_valid,
                page_scores,
                torch.finfo(page_scores.dtype).min,
            )

        # 4) Select top-k pages per (B,H,Q)
        k_pages = min(num_pages, max(3, heavy_tokens // page_size))
        if k_pages <= 0:
            k_pages = 1
        topk_pages = torch.topk(page_scores, k=k_pages, dim=-1, largest=True).indices  # [B,H,Q,Kp]

        # Previous dense mask as probabilities in [0,1] (no boolean casts)
        dense_prev = previous_mask.get_dense_mask()  # [B,H,Q,K]
        if not dense_prev.dtype.is_floating_point:
            dense_prev = dense_prev.to(page_scores.dtype)
        dense_prev = dense_prev.clamp_(0.0, 1.0)

        # 5) Build {0,1} (float) token mask from selected pages, then probabilistic-OR via max
        page_union = self._pages_to_token_mask(
            topk_pages, K, page_size, device=dense_prev.device, dtype=dense_prev.dtype
        )  # [B,H,Q,K], values in {0,1} (float)
        dense_mask = torch.maximum(dense_prev, page_union)

        # 6) Gate by external attention mask via multiplication with allowed probs
        if allowed_prob is not None:
            ap = allowed_prob.to(dense_mask.dtype)
            dense_mask = dense_mask * ap.expand_as(dense_mask)

        mask_shape = (B, H, Q, K)
        return Mask.create_mask_from_dense_mask(mask_shape, dense_mask, dtype=previous_mask.dtype)

    
    def _attention_mask_to_allowed_prob(self, attention_mask: torch.Tensor, K: int) -> torch.Tensor:
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

    def _page_any_valid(self, allowed_prob: torch.Tensor, page_size: int, num_pages: int) -> torch.Tensor:
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

    def _pages_to_token_mask(self, topk_pages: torch.Tensor, K: int, page_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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

    def _calculate_effective_heavy_size(self, seq_len_keys: int) -> int:
        """Token budget based on TopKMaskerConfig.heavy_size (ratio or absolute)."""
        return self._calculate_effective_size(self.heavy_size, seq_len_keys)

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_tokens: int
    ) -> bool:
        """Full attention if the sequence is within the token budget."""
        return dims.seq_len_keys <= max(1, heavy_tokens)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "QuestTopKMasker":
        if not isinstance(config, QuestTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
