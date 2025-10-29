# """
# :author: Sahil Joshi
# :copyright: 2025 Sparse Attention Hub
# :license: Apache 2.0
# :date: 2025-01-27
# :summary: Correctness tests comparing new QuestTopKMasker with original Quest attention.
# """

# import math
# import os
# from itertools import product
# from typing import Any, Callable, Dict, List, Optional, Tuple

# import copy
# import json
# import pytest
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from transformers.models.llama.configuration_llama import LlamaConfig
# from transformers.models.llama.modeling_llama import (
#     LlamaAttention,
#     LlamaRotaryEmbedding,
#     apply_rotary_pos_emb,
#     repeat_kv,
# )
# from transformers.cache_utils import Cache, DynamicCache

# # === Quest masker ===
# from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.quest_top_k import (  # adjust path if needed
#     QuestTopKMasker,
#     QuestTopKMaskerConfig,
# )
# from sparse_attention_hub.sparse_attention.research_attention import (
#     ResearchAttention,
#     ResearchAttentionConfig,
# )
# from sparse_attention_hub.sparse_attention.base import SparseAttention
# from sparse_attention_hub.sparse_attention.utils.mask import Mask


# # ---------- HF custom attention integration ----------
# def get_custom_attention_function(sparse_attention: SparseAttention) -> Callable:
#     def custom_attention_callable(
#         module: torch.nn.Module,
#         queries: torch.Tensor,
#         keys: torch.Tensor,
#         values: torch.Tensor,
#         attention_mask: Optional[torch.Tensor],
#         scaling: float = 1.0,
#         dropout: float = 0.0,
#         **kwargs: Dict[str, Any],
#     ):
#         if hasattr(module, "layer_idx"):
#             layer_idx = getattr(module, "layer_idx", None)
#             if layer_idx is not None:
#                 kwargs["layer_idx"] = layer_idx

#         if "sparse_meta_data" in kwargs:
#             sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
#             kwargs.pop("sparse_meta_data", None)
#         else:
#             raise ValueError("sparse_meta_data must be provided while calling model.forward()")

#         return sparse_attention.custom_attention(
#             module=module,
#             queries=queries,
#             keys=keys,
#             values=values,
#             attention_mask=attention_mask,
#             scaling=scaling,
#             dropout=dropout,
#             sparse_meta_data=sparse_meta_data,
#             **kwargs,
#         )

#     return custom_attention_callable


# # ---------- Original Quest code ----------
# def _quest_local_heavy_hitter_mask(attn_weights: torch.Tensor, token_budget: int, chunk_size: int) -> torch.Tensor:
#     # attn_weights (B, H, Q, K)
#     seq_length = attn_weights.shape[-1]
#     padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
#     pad_val = torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
#     attn_weights = torch.cat(
#         [
#             attn_weights,
#             torch.ones(
#                 (attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], padding_length),
#                 device=attn_weights.device,
#             )
#             * pad_val,
#         ],
#         dim=-1,
#     )
#     # Chunk to size chunk_size -> max per chunk
#     chunk_attn_weights = attn_weights.reshape(
#         attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2],
#         attn_weights.shape[3] // chunk_size, chunk_size
#     ).amax(dim=-1)

#     k_chunks = min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1))
#     _, topk = chunk_attn_weights.topk(k=k_chunks, dim=-1)

#     # Repeat selection within each chosen chunk
#     topk = topk.unsqueeze(-1).repeat(1, 1, 1, 1, chunk_size) * chunk_size + torch.arange(chunk_size, device=topk.device)
#     topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)

#     mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
#     mask_bottom.scatter_(-1, topk, True)
#     mask_bottom = mask_bottom[:, :, :, :seq_length]
#     return mask_bottom


# def _quest_forward_impl(
#     self: LlamaAttention,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
#     **kwargs,
# ):
#     bsz, q_len, _ = hidden_states.size()

#     # If not single-token decode, fall back to flash/stock forward
#     if q_len > 1 or self.layer_id < 2:
#         return self.flash_forward(
#             hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs
#         )

#     query_states = (self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2))
#     key_states = (self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))
#     value_states = (self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))

#     if isinstance(past_key_value, DynamicCache):
#         kv_seq_len = past_key_value.get_seq_length()
#     else:
#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             assert isinstance(past_key_value, tuple)
#             kv_seq_len += past_key_value[0].shape[-2]

#     cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

#     if isinstance(past_key_value, DynamicCache):
#         if use_cache:
#             key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
#     else:
#         if past_key_value is not None:
#             key_states = torch.cat([past_key_value[0], key_states], dim=2)
#             value_states = torch.cat([past_key_value[1], value_states], dim=2)
#         past_key_value = (key_states, value_states) if use_cache else None

#     kv_seq_len = key_states.shape[2]
#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)

#     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
#     sign = (query_states > 0) + (~(query_states > 0)) * -1
#     max_key = key_states * sign
#     postive_query = query_states * sign

#     seq_length = max_key.shape[-2]
#     padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
#     pad_val = torch.tensor(torch.finfo(max_key.dtype).min, device=max_key.device)
#     max_key = torch.cat(
#         [max_key, torch.ones((max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]), device=max_key.device) * pad_val],
#         dim=-2,
#     )

#     chunk_max_key = max_key.reshape(
#         max_key.shape[0], max_key.shape[1], max_key.shape[2] // self.chunk_size, self.chunk_size, max_key.shape[3]
#     ).amax(dim=-2)
#     chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
#     chunk_max_key = chunk_max_key.reshape(chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1])[:, :, :seq_length, :]

#     quantized_weight = torch.matmul(postive_query.float(), chunk_max_key.transpose(2, 3))

#     if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#         raise ValueError(f"Attention weights should be {(bsz, self.num_heads, q_len, kv_seq_len)}, got {attn_weights.size()}")

#     if attention_mask is not None:
#         if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#             raise ValueError(f"Attention mask should be {(bsz, 1, q_len, kv_seq_len)}, got {attention_mask.size()}")
#         attn_weights = attn_weights + attention_mask
#         attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
#         quantized_weight = quantized_weight + attention_mask
#         quantized_weight = torch.max(quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min))

#     token_budget = min(kv_seq_len, self.token_budget)

#     attn_weights_for_selection = quantized_weight

#     # remove edges from selection (we set edge_budget=0 in tests to align with QuestTopKMasker)
#     if getattr(self, "edge_budget", 0) > 0:
#         eb = self.edge_budget
#         finfo_min = torch.finfo(quantized_weight.dtype).min
#         attn_weights_for_selection[:, :, :, :eb] = finfo_min
#         attn_weights_for_selection[:, :, :, -eb:] = finfo_min

#     if token_budget > 0:
#         mask_bottom = _quest_local_heavy_hitter_mask(attn_weights_for_selection, token_budget, self.chunk_size)
#     else:
#         mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

#     # causal/edge enabling (edge_budget=0 for our tests)
#     diag = position_ids[0][0].item()
#     mask_bottom = torch.tril(mask_bottom, diagonal=diag)
#     if getattr(self, "edge_budget", 0) > 0:
#         eb = self.edge_budget
#         mask_bottom[:, :, :, :eb] = True
#         mask_bottom[:, :, :, -eb:] = True

#     finfo_min = torch.tensor(torch.finfo(attn_weights.dtype).min)
#     attn_weights[~mask_bottom] = finfo_min
#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#     attn_output = torch.matmul(attn_weights, value_states)

#     if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#         raise ValueError(f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}")

#     attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
#     attn_output = self.o_proj(attn_output)
#     if not output_attentions:
#         attn_weights = None
#     return attn_output, attn_weights, past_key_value


# # ---------- Test configuration ----------
# TEST_CONFIG = {
#     "past_sequence_length": int(os.environ.get("PAST_SEQUENCE_LENGTH", 2048)),
#     "page_size": int(os.environ.get("PAGE_SIZE", 128)),
#     "heavy_ratio": float(os.environ.get("HEAVY_RATIO", 0.05)),  # fraction of K
# }

# # Multiple test configurations for stress testing
# TEST_CONFIGS = [
#     {
#         "past_sequence_length": past_seq_len,
#         "page_size": page_sz,
#         "heavy_ratio": heavy_ratio,
#     }
#     for past_seq_len, page_sz, heavy_ratio in product(
#         [8192, 16384, 32768],          # past_sequence_length
#         [16, 32, 64, 128],                # page_size
#         [0.02, 0.05, 0.10],            # heavy ratio of keys
#     )
# ]


# # ---------- Fixtures ----------
# @pytest.fixture
# def test_config() -> LlamaConfig:
#     """Llama-3.1-8B-Instruct-like dims (matches your other test)."""
#     return LlamaConfig(
#         vocab_size=128256,
#         hidden_size=4096,
#         intermediate_size=14336,
#         num_hidden_layers=32,
#         num_attention_heads=32,
#         num_key_value_heads=8,
#         max_position_embeddings=131072,
#         rope_theta=500000.0,
#         attention_dropout=0.0,
#         attention_bias=False,
#     )


# @pytest.fixture(params=TEST_CONFIGS)
# def test_params(request) -> Dict[str, Any]:
#     return request.param


# @pytest.fixture
# def batch_size() -> int:
#     return 1


# @pytest.fixture
# def small_sequence_length() -> int:
#     # single-token decode path
#     return 1


# @pytest.fixture
# def past_sequence_length(test_params: Dict[str, Any]) -> int:
#     return test_params["past_sequence_length"]


# @pytest.fixture
# def tolerance() -> float:
#     return 1e-2


# @pytest.fixture
# def hidden_states(test_config: LlamaConfig, batch_size: int, small_sequence_length: int) -> torch.Tensor:
#     g = torch.Generator().manual_seed(1)
#     return torch.randn(batch_size, small_sequence_length, test_config.hidden_size, generator=g)


# @pytest.fixture
# def past_key_value(test_config: LlamaConfig, batch_size: int, past_sequence_length: int) -> Optional[Cache]:
#     cache = DynamicCache()
#     head_dim = test_config.hidden_size // test_config.num_attention_heads
#     gk = torch.Generator().manual_seed(2)
#     gv = torch.Generator().manual_seed(3)
#     past_k = torch.randn(batch_size, test_config.num_key_value_heads, past_sequence_length, head_dim, generator=gk)
#     past_v = torch.randn(batch_size, test_config.num_key_value_heads, past_sequence_length, head_dim, generator=gv)
#     cache_kwargs = {"cache_position": torch.arange(past_sequence_length)}
#     cache.update(past_k, past_v, layer_idx=4, cache_kwargs=cache_kwargs)
#     return cache


# @pytest.fixture
# def attention_mask() -> Optional[torch.Tensor]:
#     # Let algorithms decide sparsity; keep None for simplicity/consistency with your other test.
#     return None


# @pytest.fixture
# def position_ids(small_sequence_length: int, past_sequence_length: int) -> torch.LongTensor:
#     start_pos = past_sequence_length
#     end_pos = start_pos + small_sequence_length
#     return torch.arange(start_pos, end_pos).unsqueeze(0)


# @pytest.fixture
# def position_embeddings(test_config: LlamaConfig, position_ids: torch.LongTensor):
#     from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
#     rotary_emb = LlamaRotaryEmbedding(config=test_config)
#     head_dim = test_config.hidden_size // test_config.num_attention_heads
#     dummy = torch.randn(1, test_config.num_attention_heads, position_ids.shape[1], head_dim)
#     cos, sin = rotary_emb(dummy, position_ids)
#     return cos, sin

# def _ensure_rotary(attn: LlamaAttention, cfg: LlamaConfig):
#     # Some transformers versions don’t put rotary_emb on the attention module.
#     if not hasattr(attn, "rotary_emb"):
#         attn.rotary_emb = LlamaRotaryEmbedding(cfg)

# def _ensure_llama_attrs(attn: LlamaAttention):
#     # derive attributes from config if missing
#     if not hasattr(attn, "num_heads"):
#         attn.num_heads = attn.config.num_attention_heads
#     if not hasattr(attn, "num_key_value_heads"):
#         attn.num_key_value_heads = attn.config.num_key_value_heads
#     if not hasattr(attn, "hidden_size"):
#         attn.hidden_size = attn.config.hidden_size
#     if not hasattr(attn, "head_dim"):
#         attn.head_dim = attn.hidden_size // attn.num_heads
#     if not hasattr(attn, "num_key_value_groups"):
#         # GQA/MQA grouping
#         attn.num_key_value_groups = max(1, attn.num_heads // attn.num_key_value_heads)


# @pytest.fixture
# def original_attention(test_config: LlamaConfig, test_params: Dict[str, Any], past_sequence_length: int) -> nn.Module:
#     """
#     Build a LlamaAttention and monkeypatch its forward with Quest's original implementation.
#     Align parameters (chunk_size/page_size; token_budget computed as ratio * K; edge_budget=0).
#     """
#     import types

#     attn = LlamaAttention(config=test_config, layer_idx=4)
#     _ensure_llama_attrs(attn)
#     _ensure_rotary(attn, test_config) 

#     # Store default (flash) forward so Quest can fall back if needed.
#     attn.flash_forward = attn.forward
#     attn.forward = types.MethodType(_quest_forward_impl, attn)

#     # Quest-specific attributes expected by the original code
#     attn.layer_id = 32  # large enough so q_len==1 path goes through Quest route
#     attn.chunk_size = int(test_params["page_size"])
#     attn.edge_budget = 0  # to match QuestTopKMasker semantics

#     # token_budget as absolute count: ceil(ratio * kv_seq_len)
#     kv_len = past_sequence_length
#     attn.token_budget = max(1, int(math.ceil(test_params["heavy_ratio"] * kv_len)))
#     return attn

# def _effective_ratio(past_len: int, page_size: int, ratio: float) -> float:
#     # Mirror original: pick at least 3 chunks worth of tokens
#     min_tokens = 3 * page_size
#     want_tokens = max(min_tokens, math.ceil(ratio * past_len))
#     return want_tokens / past_len

# @pytest.fixture
# def new_attention(test_config: LlamaConfig, test_params: Dict[str, Any]) -> nn.Module:
#     """
#     Build LlamaAttention using our ResearchAttention + QuestTopKMasker.
#     """
#     from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
#     from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
#     from transformers.masking_utils import eager_mask

#     past_len  = int(test_params["past_sequence_length"])
#     page_size = int(test_params["page_size"])
#     ratio     = float(test_params["heavy_ratio"])
#     eff_ratio = _effective_ratio(past_len, page_size, ratio)

#     cfg = QuestTopKMaskerConfig(
#         heavy_size=eff_ratio,   # ratio of keys
#         page_size=int(test_params["page_size"]),
#     )
#     masker = QuestTopKMasker.create_from_config(cfg)
#     research_attention = ResearchAttention(sparse_attention_config=cfg, maskers=[masker])
#     custom_fn = get_custom_attention_function(research_attention)

#     custom_name = f"quest_top_k_masker_{test_params['page_size']}_{test_params['heavy_ratio']}"
#     ALL_ATTENTION_FUNCTIONS.register(custom_name, custom_fn)
#     ALL_MASK_ATTENTION_FUNCTIONS.register(custom_name, eager_mask)

#     llama_attention = LlamaAttention(config=test_config, layer_idx=4)
#     llama_attention.config._attn_implementation = custom_name
#     return llama_attention


# # ---------- Tests ----------
# @pytest.mark.unit
# class TestQuestSparsityCorrectness:
#     def test_compare_outputs(
#         self,
#         original_attention: nn.Module,
#         new_attention: nn.Module,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor],
#         position_ids: torch.LongTensor,
#         position_embeddings,
#         past_key_value: Optional[Cache],
#         tolerance: float,
#     ) -> bool:
#         """
#         Compare original Quest attention with our QuestTopKMasker-backed attention
#         under identical weights and cache, for single-token decode.
#         """
#         original_attention.eval()
#         new_attention.eval()

#         # synchronize parameters
#         param_dict = original_attention.state_dict()
#         new_attention.load_state_dict(param_dict)

#         # Sanity: state dicts match
#         new_dict = new_attention.state_dict()
#         for k, v in original_attention.state_dict().items():
#             assert k in new_dict, f"Missing key in new attention: {k}"
#             assert torch.allclose(v, new_dict[k], atol=tolerance), f"Param mismatch at {k}"

#         with torch.no_grad():
#             # ORIGINAL (uses position_ids, past_key_value)
#             pkv1 = copy.deepcopy(past_key_value)
#             out_orig, attn_orig, _ = original_attention(
#                 hidden_states=hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=pkv1,
#                 output_attentions=True,
#                 use_cache=True,
#             )

#             # NEW (uses position_embeddings, past_key_values, sparse_meta_data)
#             pkv2 = copy.deepcopy(past_key_value)
#             out_new, attn_new = new_attention(
#                 hidden_states=hidden_states,
#                 attention_mask=attention_mask,
#                 position_embeddings=position_embeddings,
#                 past_key_values=pkv2,
#                 output_attentions=True,
#                 use_cache=True,
#                 sparse_meta_data={},  # required
#             )

#         # Compare
#         assert torch.allclose(out_orig, out_new, atol=tolerance), "Attention outputs differ"
#         assert torch.allclose(attn_orig, attn_new, atol=tolerance), "Attention weights differ"


# tests/test_quest_topk_usecache_false.py

import math
import os
import sys
import types
import tempfile
import subprocess
from itertools import product
from typing import Any, Callable, Dict, Optional, Tuple

import copy
import pytest
import torch
import torch.nn as nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.cache_utils import Cache, DynamicCache

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.quest_top_k import (
    QuestTopKMasker,
    QuestTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttention
from sparse_attention_hub.sparse_attention.base import SparseAttention


# ------------------------- Repo bootstrap -------------------------

def _get_quest_path() -> str:
    """
    Clone https://github.com/mit-han-lab/Quest into a temp dir if not present.
    Return the local path.
    """
    tmp = tempfile.gettempdir()
    quest_path = os.path.join(tmp, "Quest")
    if os.path.exists(quest_path):
        return quest_path
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/mit-han-lab/Quest.git", quest_path],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone Quest: {e.stderr}") from e
    return quest_path


def _load_quest_forward():
    """
    Import evaluation/quest_attention.py and return the forward function
    to monkey-patch onto HF LlamaAttention.
    """
    quest_path = _get_quest_path()
    if quest_path not in sys.path:
        sys.path.append(quest_path)

    import importlib
    mod = importlib.import_module("evaluation.quest_attention")

    candidates = [
        "quest_forward_impl",
        "_quest_forward_impl",
        "quest_attention_forward",
        "quest_forward",
        "forward",
    ]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn

    raise ImportError(
        "Could not find a quest attention forward function in evaluation/quest_attention.py. "
        "Checked: " + ", ".join(candidates)
    )


def get_custom_attention_function(sparse_attention: SparseAttention) -> Callable:
    def custom_attention_callable(
        module: torch.nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float = 1.0,
        dropout: float = 0.0,
        **kwargs: Dict[str, Any],
    ):
        if hasattr(module, "layer_idx"):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None:
                kwargs["layer_idx"] = layer_idx

        if "sparse_meta_data" in kwargs:
            sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
            kwargs.pop("sparse_meta_data", None)
        else:
            raise ValueError("sparse_meta_data must be provided while calling model.forward()")

        return sparse_attention.custom_attention(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
            **kwargs,
        )

    return custom_attention_callable


# ------------------------- Test params -------------------------

TEST_CONFIG = {
    "past_sequence_length": int(os.environ.get("PAST_SEQUENCE_LENGTH", 2048)),
    "page_size": int(os.environ.get("PAGE_SIZE", 128)),
    "heavy_ratio": float(os.environ.get("HEAVY_RATIO", 0.05)),
}

TEST_CONFIGS = [
    {
        "past_sequence_length": past_seq_len,
        "page_size": page_sz,
        "heavy_ratio": heavy_ratio,
    }
    for past_seq_len, page_sz, heavy_ratio in product(
        [8192, 16384, 32768],
        [16, 32, 64, 128],
        [0.02, 0.05, 0.10],
    )
]


# ------------------------- Fixtures -------------------------

@pytest.fixture
def test_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        attention_dropout=0.0,
        attention_bias=False,
    )


@pytest.fixture(params=TEST_CONFIGS)
def test_params(request) -> Dict[str, Any]:
    return request.param


@pytest.fixture
def batch_size() -> int:
    return 1


@pytest.fixture
def small_sequence_length() -> int:
    # single-token decode path
    return 1


@pytest.fixture
def past_sequence_length(test_params: Dict[str, Any]) -> int:
    return test_params["past_sequence_length"]


@pytest.fixture
def tolerance() -> float:
    return 1e-2


@pytest.fixture
def hidden_states(test_config: LlamaConfig, batch_size: int, small_sequence_length: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randn(batch_size, small_sequence_length, test_config.hidden_size, generator=g)


@pytest.fixture
def past_key_value(test_config: LlamaConfig, batch_size: int, past_sequence_length: int) -> Optional[Cache]:
    """
    Make a DynamicCache and ALSO stash the generated tensors on it so we
    don't depend on private HF internals in different versions.
    """
    cache = DynamicCache()
    head_dim = test_config.hidden_size // test_config.num_attention_heads
    gk = torch.Generator().manual_seed(2)
    gv = torch.Generator().manual_seed(3)
    past_k = torch.randn(batch_size, test_config.num_key_value_heads, past_sequence_length, head_dim, generator=gk)
    past_v = torch.randn(batch_size, test_config.num_key_value_heads, past_sequence_length, head_dim, generator=gv)
    cache_kwargs = {"cache_position": torch.arange(past_sequence_length)}
    layer_idx = 32
    cache.update(past_k, past_v, layer_idx=layer_idx, cache_kwargs=cache_kwargs)

    # Stash for version-agnostic extraction
    cache._test_layer_idx = layer_idx
    cache._test_past_k = past_k
    cache._test_past_v = past_v
    return cache


@pytest.fixture
def attention_mask() -> Optional[torch.Tensor]:
    # Keep None; let sparsity machinery decide
    return None


@pytest.fixture
def position_ids(small_sequence_length: int, past_sequence_length: int) -> torch.LongTensor:
    start_pos = past_sequence_length
    end_pos = start_pos + small_sequence_length
    return torch.arange(start_pos, end_pos).unsqueeze(0)


@pytest.fixture
def position_embeddings(test_config: LlamaConfig, position_ids: torch.LongTensor):
    rotary = LlamaRotaryEmbedding(config=test_config)
    head_dim = test_config.hidden_size // test_config.num_attention_heads
    dummy = torch.randn(1, test_config.num_attention_heads, position_ids.shape[1], head_dim)
    cos, sin = rotary(dummy, position_ids)
    return cos, sin


def _effective_ratio(past_len: int, page_size: int, ratio: float) -> float:
    # Match Quest: pick at least 3 chunks worth of tokens
    min_tokens = 3 * page_size
    want_tokens = max(min_tokens, math.ceil(ratio * past_len))
    return want_tokens / max(1, past_len)


@pytest.fixture
def original_attention(test_config: LlamaConfig, test_params: Dict[str, Any]) -> nn.Module:
    """
    Build a LlamaAttention and monkeypatch its forward with the upstream Quest implementation.
    """
    quest_forward = _load_quest_forward()

    attn = LlamaAttention(config=test_config, layer_idx=32)

    # Preserve original forward as "flash" fallback if Quest code expects it
    attn.flash_forward = attn.forward
    attn.forward = types.MethodType(quest_forward, attn)

    # Attributes expected by Quest code in evaluation/quest_attention.py
    attn.layer_id = 32
    attn.chunk_size = int(test_params["page_size"])
    attn.edge_budget = 0
    # token_budget = ceil(ratio * K), but ensure >= 3 chunks like Quest
    past_len = int(test_params["past_sequence_length"])
    ratio = float(test_params["heavy_ratio"])
    want = max(3 * attn.chunk_size, math.ceil(ratio * past_len))
    attn.token_budget = max(1, want)

    # Ensure HF attrs exist
    if not hasattr(attn, "num_heads"):
        attn.num_heads = attn.config.num_attention_heads
    if not hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = attn.config.num_key_value_heads
    if not hasattr(attn, "hidden_size"):
        attn.hidden_size = attn.config.hidden_size
    if not hasattr(attn, "head_dim"):
        attn.head_dim = attn.hidden_size // attn.num_heads
    if not hasattr(attn, "num_key_value_groups"):
        attn.num_key_value_groups = max(1, attn.num_heads // attn.num_key_value_heads)
    if not hasattr(attn, "rotary_emb"):
        attn.rotary_emb = LlamaRotaryEmbedding(test_config)

    return attn


@pytest.fixture
def new_attention(test_config: LlamaConfig, test_params: Dict[str, Any]) -> nn.Module:
    """
    Build LlamaAttention using ResearchAttention + QuestTopKMasker.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import eager_mask

    past_len  = int(test_params["past_sequence_length"])
    page_size = int(test_params["page_size"])
    ratio     = float(test_params["heavy_ratio"])
    eff_ratio = _effective_ratio(past_len, page_size, ratio)

    cfg = QuestTopKMaskerConfig(
        heavy_size=eff_ratio,
        page_size=page_size,
    )
    masker = QuestTopKMasker.create_from_config(cfg)
    research_attention = ResearchAttention(sparse_attention_config=cfg, maskers=[masker])
    custom_fn = get_custom_attention_function(research_attention)

    custom_name = f"quest_top_k_masker_{page_size}_{ratio}"
    ALL_ATTENTION_FUNCTIONS.register(custom_name, custom_fn)
    ALL_MASK_ATTENTION_FUNCTIONS.register(custom_name, eager_mask)

    llama_attention = LlamaAttention(config=test_config, layer_idx=32)
    llama_attention.config._attn_implementation = custom_name
    return llama_attention


# ------------------------- Helpers -------------------------

def _extract_layer_kv(dc: DynamicCache, layer_idx: int):
    """
    Robustly extract (past_k, past_v) from a DynamicCache across HF versions.
    Prefer the tensors we stashed in the fixture; otherwise try common layouts.
    """
    # Preferred: version-agnostic stashed tensors from our fixture
    if hasattr(dc, "_test_past_k") and hasattr(dc, "_test_past_v"):
        return dc._test_past_k, dc._test_past_v

    # Common newer layout: top-level lists
    if hasattr(dc, "key_cache") and hasattr(dc, "value_cache"):
        if layer_idx < len(dc.key_cache) and dc.key_cache[layer_idx] is not None:
            return dc.key_cache[layer_idx], dc.value_cache[layer_idx]

    # Some builds: dc.layers is a list of per-layer caches/tuples
    if hasattr(dc, "layers"):
        if layer_idx < len(dc.layers) and dc.layers[layer_idx] is not None:
            lyr = dc.layers[layer_idx]
            if isinstance(lyr, tuple) and len(lyr) >= 2:
                return lyr[0], lyr[1]
            if hasattr(lyr, "key_cache") and hasattr(lyr, "value_cache"):
                return lyr.key_cache, lyr.value_cache

    # Some builds: dc.caches
    if hasattr(dc, "caches"):
        if layer_idx < len(dc.caches) and dc.caches[layer_idx] is not None:
            lyr = dc.caches[layer_idx]
            if isinstance(lyr, tuple) and len(lyr) >= 2:
                return lyr[0], lyr[1]
            if hasattr(lyr, "key_cache") and hasattr(lyr, "value_cache"):
                return lyr.key_cache, lyr.value_cache

    raise RuntimeError("DynamicCache missing key/value for extraction on this HF build")


# ------------------------- Test -------------------------

@pytest.mark.unit
class TestQuestSparsityCorrectness_Upstream:
    def test_compare_outputs(
        self,
        original_attention: nn.Module,
        new_attention: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        position_embeddings,
        past_key_value: Optional[Cache],
        tolerance: float,
    ) -> bool:
        original_attention.eval()
        new_attention.eval()

        # sync parameters
        param_dict = original_attention.state_dict()
        new_attention.load_state_dict(param_dict)

        # sanity: params match
        new_dict = new_attention.state_dict()
        for k, v in original_attention.state_dict().items():
            assert k in new_dict, f"Missing key in new attention: {k}"
            assert torch.allclose(v, new_dict[k], atol=tolerance), f"Param mismatch at {k}"

        with torch.no_grad():
            # ORIGINAL (upstream Quest) — pass a tuple so it concatenates past even with use_cache=False
            # IMPORTANT: do NOT deepcopy here; some HF builds drop ad-hoc attrs on deepcopy.
            past_k, past_v = _extract_layer_kv(past_key_value, layer_idx=32)
            out_orig, attn_orig, _ = original_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(past_k, past_v),
                output_attentions=True,
                use_cache=False,
            )

            # NEW (ResearchAttention + QuestTopKMasker) — pass the cache object directly
            out_new, attn_new = new_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_value,
                output_attentions=True,
                use_cache=False,
                sparse_meta_data={},
            )

        assert torch.allclose(out_orig, out_new, atol=tolerance), "Attention outputs differ"
        assert torch.allclose(attn_orig, attn_new, atol=tolerance), "Attention weights differ"
