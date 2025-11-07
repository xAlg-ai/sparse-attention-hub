"""Run configuration for hyperparameter search.

All configuration parameters for the hyperparameter search are defined here.
Modify this file to change search behavior without editing the main script.
"""

import os
from typing import Dict, List, Optional

# Model configurations
# Weight files are loaded from SPARSE_ATTENTION_WEIGHTS_DIR environment variable
# Set it to the directory containing your HashAttention weight files
hashattention_dir: str = os.environ.get("HASHATTENTION_WEIGHTS_DIR", "./")
doublesparsity_config_dir: str = os.environ.get("DOUBLE_SPARSITY_CONFIG_DIR", "./")


MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "llama3.1-8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "llama3.1-8b-patch.64K.v1.hat_weights.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "meta-llama/Llama-3.1-8B-Instruct.json"),
    },
    "llama3.2-1b": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DNE.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "meta-llama/Llama-3.2-1B-Instruct.json"),
    },
    "llama3.2-3b": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DNE.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "meta-llama/Llama-3.2-3B-Instruct.json"),
    },
    "deepseek": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DeepSeek-R1-Distill-Llama-8B-patch-layers2-dim64-max-context-24K_hat_weights.pkl"),
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "Mistral-7B-Instruct-v0.3.24K.20.500.hat_weights.pkl"),
    },
    "qwen3-30b-moe": {
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DNE.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "Qwen/Qwen3-30B-A3B-Instruct-2507.json"),
    },
    "qwen3-4b": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DNE.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "Qwen/Qwen3-4B-Instruct-2507.json"),
    },
}

DEFAULT_MODEL: str = "llama"

# Task configurations
DEBUG_TASKS: List[str] = ["loogle/shortdep_qa"]

RUN_TASKS: List[str] = [
    "ruler32k/vt",
    "ruler32k/qa_1",
    "ruler32k/qa_2",
    "ruler32k/fwe",
    "ruler32k/niah_multikey_2",
    "ruler32k/niah_multikey_3",
]

# Hyperparameter search configuration
OBJECTIVE: str = "default"  # Objective function to use for optimization
NUM_SAMPLES: int = 100  # Number of samples per hyperparameter search
SEARCH_MAX_NEW_TOKENS: int = 100  # Max new tokens for search trials
SEARCH_MAX_CONTEXT_LENGTH: int = 2048  # Max context length for search trials
SEARCH_MAX_REQUESTS: int = 10  # Max requests per search trial
DEBUG: bool = False  # Debug mode with minimal configs
FORCE_SEARCH: bool = False  # Force re-run of search even if configs exist
OPTIMAL_CONFIGS_DIR: str = "./debug"  # Directory for storing optimal configurations
RAY_RESULTS_DIR: str = "./ray_results"  # Directory for Ray Tune results
SEARCH_TIMEOUT: int = 900  # Timeout per search trial in seconds
ACTORS_PER_GPU: int = 1  # Number of actors per GPU for resource allocation
MEMORY_OBJECTIVE: Optional[str] = None  # Memory objective parameter (e.g., "memory_32") for configs that need it

# Config builder configuration
BUILDER_NAMES: List[str] = ["dense", "oracle_topk"]  # Specify which builders to use (e.g., ["magicpig"], ["dense"], ["double_sparsity"])

