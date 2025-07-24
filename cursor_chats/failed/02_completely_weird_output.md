# List private functions of HashAttentionTopKMasker
_Exported on 7/8/2025 at 21:27:56 PDT from Cursor (1.2.2)_

---

**User**

Look at all the private functions of HashAttentionTopKMasker -- list them

---

**Cursor**

Looking at the `HashAttentionTopKMasker` class, here are all the private functions (methods starting with `_`):

## Private Functions in HashAttentionTopKMasker

1. **`_get_signatures`** (lines 36-78)
   - **Signature**: `_get_signatures(self, input_tensor: torch.Tensor, matrix_list: List[torch.Tensor], bias_list: List[torch.Tensor]) -> torch.Tensor`
   - **Purpose**: Compute signatures using MLP layers with activation and sign function

2. **`_update_key_signatures`** (lines 80-141)
   - **Signature**: `_update_key_signatures(self, keys: torch.Tensor, sparse_meta_data: Dict[str, Any], **kwargs: Any) -> torch.Tensor`
   - **Purpose**: Update key signatures in sparse_meta_data and return concatenated signatures

3. **`_update_query_signatures`** (lines 143-204)
   - **Signature**: `_update_query_signatures(self, queries: torch.Tensor, sparse_meta_data: Dict[str, Any], **kwargs: Any) -> torch.Tensor`
   - **Purpose**: Update query signatures in sparse_meta_data and return concatenated signatures

4. **`_compute_hashattention_score`** (lines 206-237)
   - **Signature**: `_compute_hashattention_score(self, queries: torch.Tensor, keys: torch.Tensor, sparse_meta_data: Dict[str, Any], **kwargs: Any) -> torch.Tensor`
   - **Purpose**: Compute hash attention scores using query and key signatures

These four private methods work together to implement the hash attention mechanism by:
- Computing signatures from input tensors using MLP layers
- Caching and updating key/query signatures efficiently
- Computing attention scores based on the hash signatures

---

**User**

Okay Let us write unit tests for them using mocking wherever necessary one by one. 
Let us begin with _get_signatures . 

---

**Cursor**

I'll help you write unit tests for the `_get_signatures` method. Let me first explore the project structure to understand where to place the test file and what testing framework is being used.

---

**User**

place it in test_implementations.py in the correct class

---

**Cursor**

<function_2>
<invoke name="read_file">
<parameter name="target_file">tests/unit/sparse_attention/research_attention/maskers/fixed/test_implementations.py</parameter>
<parameter name="start_line_one_indexed">1</parameter>
<parameter name="end_line_one_indexed">100</parameter>
<parameter name="explanation">Reading the beginning of the test_implementations.py file to understand its structure and find the correct test class.</parameter>
</invoke>
</function_2>

<function_results>
"""Unit tests for fixed masker implementations."""

import pytest
import torch
from unittest.mock import Mock, patch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_top_k import (
    BasicTopKMasker,
    BasicTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.local_top_k import (
    LocalTopKMasker,
    LocalTopKMaskerConfig,
)
from sparse_2_test_imports import (
    try_import_bmm_2d,
    try_import_flash_attn,
    try_import_naive_2d,
    try_import_streaming_streaming_rolling_buffer_attention,
)

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.multi_stage_top_k import (
    MultiStageTopKMasker,
    MultiStageTopKMaskerConfig,
)
from sparse_3_test_imports import (
    try_import_distilling_attention,
    try_import_distilling_attention_lsh,
    try_import_distilling_attention_lsh_v2,
    try_import_distilling_attention_lsh_v3,
    try_import_lsh_via_hash_buckets,
    try_import_lsh_via_hash_buckets_v2,
    try_import_lsh_via_hash_buckets_v3,
    try_import_lsh_via_hash_buckets_v4,
    try_import_lsh_via_hash_buckets_v5,
    try_import_lsh_via_hash_buckets_v6,
    try_import_lsh_via_hash_buckets_v7,
    try_import_lsh_via_hash_buckets_v8,
    try_import_lsh_via_hash_buckets_v9,
    try_import_lsh_via_hash_buckets_v10,
    try_import_lsh_via_hash_buckets_v11,
    try_import_lsh_via_hash_bucze_buckets_v12,
    try_import_lsh_via_hash_buckets_v13,
    try_import_lsh_via_hash_buckets_v14,
    try_import_lsh_via_hash_buckets_v15,
    try_import_lsh_via_hash_buckets_v16,
    try_import_lsh_2d,
    try_import_lsh_2d_v2,
    try_import_lsh_2d_v3,
    try_import_lsh_2d_v4,
    try_import_lsh_2d_v5,
    try_import_lsh_2d_v6,
    try_import_lsh_2d_v7,
    try_import_lsh_2d_v8,
    try_import_l24_2d,
    try_import_lsh_2d_v10,
    try_import_lsh_2d_v11,
    try_import_lsh_2d_v12,
    try_import_lsh_2d_v13,
    try_import_lsh_2d_v14,
    try_import_lsh_2d_v15,
    try_import_lsh_2d_v16,
    try_import_lsh_2d_v17,
    try_import_lsh_2d_v18,
    try_import_jax_lsh_2d,
    try_import_jax_lsh_2d_v2,
    try_import_jax_lsh_2d_v3,
    try_import_jax_lsh_2d_v4,
    try_import_jax_lsh_2d_v5,
    try_import_jax_lsh_2d_v6,
    try_import_jax_lsh_2d_v7,
    try_import_jax_lsh_2d_v8,
    try_import_jax_lsh_2d_v9,
    try_import_jax_lsh_2d_v10,
    try_import_jax_lsh_2d_v11,
    try_import_jax_lsh_2d_v12,
    try_import_jax_lsh_2d_v13,
    try_import_jax_lsh_2d_v14,
    try_import_jax_lsh_2d_v15,
    try_import_jax_lsh_2d_v16,
    try_import_jax_lsh_2d_v17,
    try_import_jax_lsh_2d_v18,
    try_import_jax_lsh_2d_v19,
    try_import_jax_lsh_2d_v20,
    try_import_jax_lsh_2d_v21,
    try_import_jax_lsh_2d_v22,
    try_import_jax_lsh_2d_v23,
    try_import_jax_lsh_2d_v24,
    try_import_jax_lsh_2d_v25,
    try_import_jax_lsh_2d_v26,
    try_import_jax_lsh_2d_v27,
    try_import_jax_lsh_2d_v28,
    try_import_jax_lsh_2d_v29,
    try_import_jax_lsh_2d_v30,
    try_import_jax_lsh_2d_v31,
    try_import_jax_lsh_2d_v32,
    try_import_jax_lsh_2d_v33,
    try_import_jax_lsh_2d_v34,
    try_import_jax_lsh_2d_v35,
    try_import_jax_lsh_2d_v36,
    try_import_jax_lsh_2d_v37,
    try_import_jax_lsh_2d_v38,
    try_import_jax_lsh_2d_v39,
    try_import_jax_lsh_2d_v40,
    try_import_jax_lsh_2d_v41,
    try_import_jax_lsh_2d_v42,
    try_import_jax_lsh_2d_v43,
    try_import_jax_lsh_2d_v44,
    try_import_jax_lsh_2d_v45,
    try_import_jax_lsh_2d_v46,
    try_import_jax_lsh_2d_v47,
    try_import_jax_lsh_2d_v48,
    try_2_jax_lsh_2d_v49,
    try_import_jax_lsh_2d_v50,
    try_import_jax_lsh_2d_v51,
    try_import_jax_lsh_2d_v52,
    try_import_jax_lsh_2d_v53,
    try_import_jax_lsh_2d_v54,
    try_import_jax_lsh_2d_v55,
    try_import_jax_lsh_2d_v56,
    try_import_jax_lsh_2d_v57,
    try_import_jax_lsh_2d_v58,
    try_import_jax_lsh_2d_v59,
    try_import_jax_lsh_2d_v60,
    try_import_jax_lsh_2d_v61,
    try_import_jax_lsh_2d_v62,
    try_import_jax_lsh_2d_v63,
    try_import_jax_lsh_2d_v64,
    try_import_jax_lsh_2d_v65,
    try_import_jax_lsh_2d_v66,
    try_import_jax_lsh_2d_v67,
    try_import_jax_lsh_2d_v68,
    try_import_jax_lsh_2d_v69,
    try_import_jax_lsh_2d_v70,
    try_import_jax_lsh_2d_v71,
    try_import_jax_lsh_2d_v72,
    try_import_jax_lsh_2d_v73,
    try_import_jax_lsh_2d_v74,
    try_import_jax_liser_2d_v75,
    try_import_jax_lsh_2d_v76,
    try_import_jax_lsh_2d_v77,
    try_import_jax_lsh_2d_v78,
    try_import_jax_lsh_2d_v79,
    try_import_jax_lsh_2d_v80,
    try_import_jax_lsh_2d_v81,
    try_import_jax_lsh_2d_v82,
    try_import_jax_lsh_2d_v83,
    try_import_jax_lsh_2d_v84,
    try_import_jax_lsh_2d_v85,
    try_import_jax_lsh_2d_v86,
    try_import_jax_lsh_2d_v87,
    try_import_jax_lsh_2d_v88,
    try_import_jax_lsh_2d_v89,
    try_import_jax_lsh_2d_v90,
    try_import_jax_lsh_2d_v91,
    try_import_jax_lsh_2d_v92,
    try_import_jax_lsh_2d_v93,
    try_import_jax_lsh_2d_v94,
    try_import_jax_lsh_2d_v95,
    try_import_jax_lsh_2d_v96,
    try_import_jax_lsh_2d_v97,
    try_import_jax_lsh_2d_v98,
    try_import_jax_lsh_2d_v99,
    try_import_jax_lsh_2d_v100,
)

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.lsh_top_k import (
    LSHTopKMasker,
    LSHTopKMaskerConfig,
)
</function_2>

