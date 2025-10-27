"""Quest Top-K masker implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

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
    """Quest page-Top-K masker (vectorized implementation)."""

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

        effective_heavy_size: int = self._calculate_effective_heavy_size(
            dims.seq_len_keys
        )

        if self._should_use_full_attention(dims, effective_heavy_size):
            return self._create_full_mask(
                dims, previous_mask.dtype, previous_mask.device
            )

        quest_mask = self._create_quest_page_topk_mask(
            dims,
            effective_heavy_size,
            keys,
            queries,
            attention_mask,
            previous_mask,
        )
        return previous_mask.merge_mask(quest_mask, inplace=False)

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
        1) Repeat KV to match query heads (GQA/MQA -> MHA).
        2) Page the keys, compute per-page min/max: [B,H,P,D].
        3) Score pages with Quest bound using queries: [B,H,Q,P].
        4) Select Top-K pages where K_pages = ceil(heavy_tokens / page_size).
        5) Activate all tokens in selected pages, AND with attention_mask.
        """
        ngroups = _get_num_key_value_groups(queries, keys)
        keys_rep = repeat_kv(keys, ngroups)

        B, H, K, D = keys_rep.shape
        _, _, Q, _ = queries.shape

        page_size = self.page_size
        num_pages = (K + page_size - 1) // page_size

        # Step 2.1: per-page min/max (vectorized)
        page_min, page_max = self._compute_page_min_max(
            keys_rep, page_size, num_pages
        )

        # Step 2.2: Quest scores
        page_scores = self._quest_page_scores(queries, page_min, page_max)

        # Respect external attention mask by forbidding pages with no valid tokens
        if attention_mask is not None:
            valid_tok = (attention_mask[..., :K] == 0)  # broadcastable to [B,1,Q,K]
            page_any_valid = self._page_any_valid(valid_tok, page_size, num_pages)  # [B,Q,P]
            page_any_valid = page_any_valid.unsqueeze(1).expand(B, H, Q, num_pages)  # [B,H,Q,P]
            page_scores = torch.where(
                page_any_valid, page_scores, torch.finfo(page_scores.dtype).min
            )

        # Step 3: choose Kp pages per (B,H,Q)
        k_pages = max(1, min(num_pages, (heavy_tokens + page_size - 1) // page_size))
        topk_pages = torch.topk(page_scores, k=k_pages, dim=-1, largest=True).indices  # [B,H,Q,Kp]

        # Start from previous dense mask
        dense_mask = previous_mask.get_dense_mask()
        if dense_mask.dtype != torch.bool:
            dense_mask = dense_mask != 0

        # Step 4: scatter pages to token mask
        dense_mask = self._scatter_pages_to_dense_mask(
            dense_mask, topk_pages, K, page_size
        )

        # Respect token-level attention mask
        if attention_mask is not None:
            allowed = (attention_mask[..., :K] == 0)  # [B,1,Q,K] or [B,1,1,K]
            dense_mask = dense_mask & allowed.expand_as(dense_mask)

        mask_shape = (B, H, Q, K)
        return Mask.create_mask_from_dense_mask(
            mask_shape, dense_mask, dtype=previous_mask.dtype
        )

    def _calculate_effective_heavy_size(self, seq_len_keys: int) -> int:
        """Token budget based on TopKMaskerConfig.heavy_size (ratio or absolute)."""
        return self._calculate_effective_size(self.heavy_size, seq_len_keys)

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_tokens: int
    ) -> bool:
        """Full attention if the sequence is within the token budget."""
        return dims.seq_len_keys <= max(1, heavy_tokens)

    @staticmethod
    def _compute_page_min_max(
        keys_rep: torch.Tensor, page_size: int, num_pages: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def _quest_page_scores(
        queries: torch.Tensor, page_min: torch.Tensor, page_max: torch.Tensor
    ) -> torch.Tensor:
        """
        queries:  [B,H,Q,D]
        page_min: [B,H,P,D]
        page_max: [B,H,P,D]
        Returns:
            scores: [B,H,Q,P] = sum_j max(q_j * min_j, q_j * max_j)
        """
        q = queries.unsqueeze(3)   # [B,H,Q,1,D]
        pmin = page_min.unsqueeze(2)  # [B,H,1,P,D]
        pmax = page_max.unsqueeze(2)  # [B,H,1,P,D]
        prod_min = q * pmin
        prod_max = q * pmax
        return torch.maximum(prod_min, prod_max).sum(dim=-1)  # [B,H,Q,P]

    @staticmethod
    def _page_any_valid(
        valid_mask: torch.Tensor, page_size: int, num_pages: int
    ) -> torch.Tensor:
        """
        valid_mask: [B,1,Q,K] (bool)
        Returns: [B,Q,P] (bool) whether each page has any valid token.
        """
        B, _, Q, K = valid_mask.shape
        K_pad = num_pages * page_size
        pad_k = K_pad - K

        if pad_k > 0:
            vm = F.pad(valid_mask, (0, pad_k), value=False)
        else:
            vm = valid_mask

        # [B,1,Q,P,ps] -> any over page tokens
        vm = vm.view(B, 1, Q, num_pages, page_size)
        return vm.any(dim=-1).squeeze(1)  # [B,Q,P]

    @staticmethod
    def _scatter_pages_to_dense_mask(
        dense_mask: torch.Tensor, topk_pages: torch.Tensor, K: int, page_size: int
    ) -> torch.Tensor:
        """
        Convert selected page indices into token-level mask and OR with dense_mask.
        dense_mask: [B,H,Q,K] (bool)
        topk_pages: [B,H,Q,Kp] (int)
        """
        if K == 0 or topk_pages.numel() == 0:
            return dense_mask

        device = dense_mask.device
        B, H, Q, Kp = topk_pages.shape
        P = (K + page_size - 1) // page_size

        # Build [P,K] lookup: True exactly where token k is inside page p
        token_idx = torch.arange(K, device=device)  # [K]
        page_idx = torch.arange(P, device=device)   # [P]
        start = page_idx * page_size                # [P]
        end = torch.clamp(start + page_size, max=K) # [P]
        page_token_mask = (token_idx.unsqueeze(0) >= start.unsqueeze(1)) & \
                          (token_idx.unsqueeze(0) <  end.unsqueeze(1))    # [P,K]

        # Gather masks for the selected pages: [B,H,Q,Kp,K]
        selected_masks = page_token_mask[topk_pages]  # advanced indexing

        # OR across selected pages: [B,H,Q,K]
        union_selected = selected_masks.any(dim=3)
        return dense_mask | union_selected

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "QuestTopKMasker":
        if not isinstance(config, QuestTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)