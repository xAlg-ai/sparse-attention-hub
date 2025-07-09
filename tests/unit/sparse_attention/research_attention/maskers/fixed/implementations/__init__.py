"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for masker implementations - organized by implementation type.
"""

# This package contains organized tests for each masker implementation type:
# - test_basic_fixed.py: LocalMasker, CausalMasker, SinkMasker
# - test_oracle_top_k.py: OracleTopK
# - test_pq_top_k.py: PQCache
# - test_hashattention_top_k.py: HashAttentionTopKMasker
# - test_double_sparsity_top_k.py: DoubleSparsityTopKMasker
# - test_imports.py: Import tests for all implementations 