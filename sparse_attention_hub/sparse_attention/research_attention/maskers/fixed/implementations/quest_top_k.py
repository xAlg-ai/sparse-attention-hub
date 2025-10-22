"""Quest Top-K masker implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

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

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class QuestTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for QuestTopKMasker."""
    page_size: int = 128
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
    """Quest top-K masker."""

    page_size: int

    def __init__(self, config: QuestTopKMaskerConfig) -> None:
        super().__init__(config)
        if config.page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        self.page_size = int(config.page_size)

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
            return self._create_full_mask(dims, previous_mask.dtype, previous_mask.device)

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

        page_min, page_max = self._compute_page_min_max(keys_rep, page_size, num_pages)


        page_scores = self._quest_page_scores(queries, page_min, page_max)

    
        if attention_mask is not None:
            valid_tok = (attention_mask[..., :K] == 0) 
            page_any_valid = self._page_any_valid(valid_tok, page_size, num_pages) 
            page_any_valid = page_any_valid.unsqueeze(1).expand(B, H, Q, num_pages)
            page_scores = torch.where(
                page_any_valid, page_scores, torch.finfo(page_scores.dtype).min
            )

       
        k_pages = max(1, min(num_pages, (heavy_tokens + page_size - 1) // page_size))
        topk_pages = torch.topk(page_scores, k=k_pages, dim=-1, largest=True).indices

   
        dense_mask = previous_mask.get_dense_mask() 
        if dense_mask.dtype != torch.bool:
            dense_mask = dense_mask != 0

       
        dense_mask = self._scatter_pages_to_dense_mask(dense_mask, topk_pages, K, page_size)

       
        if attention_mask is not None:
            allowed = (attention_mask[..., :K] == 0) 
            dense_mask = dense_mask & allowed.expand_as(dense_mask)

        mask_shape = (B, H, Q, K)
        return Mask.create_mask_from_dense_mask(mask_shape, dense_mask, dtype=previous_mask.dtype)


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
        keys_rep: [B, H, K, D]
        Returns:
            page_min, page_max: [B, H, P, D]
        """
        B, H, K, D = keys_rep.shape
        page_min = torch.empty((B, H, num_pages, D), dtype=keys_rep.dtype, device=keys_rep.device)
        page_max = torch.empty((B, H, num_pages, D), dtype=keys_rep.dtype, device=keys_rep.device)

        for p in range(num_pages):
            s = p * page_size
            e = min((p + 1) * page_size, K)
            k_slice = keys_rep[:, :, s:e, :] 
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
            scores: [B,H,Q,P] = sum_j max(q_j * min_j, q_j * max_j)
        """
        q = queries.unsqueeze(3) 
        pmin = page_min.unsqueeze(2)  
        pmax = page_max.unsqueeze(2)  
        prod_min = q * pmin
        prod_max = q * pmax
        return torch.maximum(prod_min, prod_max).sum(dim=-1)

    @staticmethod
    def _page_any_valid(valid_mask: torch.Tensor, page_size: int, num_pages: int) -> torch.Tensor:
        """
        valid_mask: [B,1,Q,K] (bool)
        Returns: [B,Q,P] (bool) whether each page has any valid token.
        """
        B, _, Q, K = valid_mask.shape
        pages = []
        for p in range(num_pages):
            s = p * page_size
            e = min((p + 1) * page_size, K)
            pages.append(valid_mask[:, 0, :, s:e].any(dim=-1))
        return torch.stack(pages, dim=-1)

    @staticmethod
    def _scatter_pages_to_dense_mask(
        dense_mask: torch.Tensor, topk_pages: torch.Tensor, K: int, page_size: int
    ) -> torch.Tensor:
        """
        dense_mask: [B,H,Q,K] (bool)
        topk_pages: [B,H,Q,Kp] (int)
        """
        if K == 0 or topk_pages.numel() == 0:
            return dense_mask

        token_idx = torch.arange(K, device=dense_mask.device)  
        Kp = topk_pages.shape[-1]

        for i in range(Kp):
            pidx = topk_pages[..., i]          
            s = pidx * page_size               
            e = torch.minimum(s + page_size, torch.as_tensor(K, device=dense_mask.device))
            s_exp = s.unsqueeze(-1)            
            e_exp = e.unsqueeze(-1)            
            in_range = (token_idx >= s_exp) & (token_idx < e_exp)
            dense_mask = dense_mask | in_range

        return dense_mask


    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "QuestTopKMasker":
        if not isinstance(config, QuestTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
