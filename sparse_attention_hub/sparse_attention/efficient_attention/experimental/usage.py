from sparse_attention_hub.sparse_attention.research_attention import ResearchAttention, ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import SinkMaskerConfig, LocalMaskerConfig, HashAttentionTopKMaskerConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import AdaptiveSamplingMaskerConfig
import torch
import torch.nn as nn
from typing import Optional
import math
import time
from ref_vAttention import *
import pickle

def convert_to_device_dtype(list_of_tensors, device, dtype):
    return [tensor.to(device, dtype) for tensor in list_of_tensors]

def extract_layer_weights(hat_weight_file, layer_idx, dtype, device):
    with open(hat_weight_file, "rb") as f:
        hat_weights = pickle.load(f)
    wts =  hat_weights[layer_idx]
    return (
        convert_to_device_dtype(wts["query_matrix"], device, dtype), 
        convert_to_device_dtype(wts["query_bias"], device, dtype), 
        convert_to_device_dtype(wts["key_matrix"], device, dtype), 
        convert_to_device_dtype(wts["key_bias"], device, dtype)
    )



def run_mha_example():
    batch_size = 1
    query_heads = 32
    kv_heads = 32
    query_len = 1
    kv_len = 10240
    d_model = 128

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # attention parameters
    queries = torch.randn(batch_size, query_heads, query_len, d_model, dtype=torch.float32).to("cuda") # [1,32,1,128]
    keys = torch.randn(batch_size, kv_heads, kv_len, d_model, dtype=torch.float32).to("cuda") # [1,32,1024,128]
    values = torch.randn(batch_size, kv_heads, kv_len, d_model, dtype=torch.float32).to("cuda") # [1,32,1024,128]
    attention_mask = None # = torch.randn(batch_size, query_heads, query_len, kv_len, dtype=torch.float32).to("cuda") # [1,32,1,1024]
    scaling = 1.0 / math.sqrt(d_model)
    dropout = 0.0

    # sparse attention parameters
    sparse_meta_data = {}
    layer_idx = 0 

    # sink masker config parameters
    sink_size = 0.05

    # local masker config parameters
    window_size = 0.05

    # hash attention top k masker config parameters
    heavy_size = 0.05
    hat_bits = 32
    hat_mlp_layers = 3
    hat_mlp_hidden_size = 128
    hat_mlp_activation = "silu"
    # present in HashAttention-1.0 repo in xAlg-ai
    # hat_weight_file = "/home/apd10/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"
    hat_weight_file = "/FirstIntelligence/home/shuo/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"

    # adaptive sampling masker config parameters
    base_rate_sampling = 0.05
    epsilon = 0.25
    delta = 0.25
    init_offset = 0.05
    local_offset = 0.05
    module = nn.Module().to("cuda") # sample module
    module.layer_idx = layer_idx

    from scipy.stats import norm
    hat_weights_query_matrix, hat_weights_query_bias, hat_weights_key_matrix, hat_weights_key_bias = extract_layer_weights(hat_weight_file, layer_idx, dtype=torch.float32, device="cuda")

    ref_pytorch_answer = ref_vAttention_fwd(
        queries = queries.squeeze(2), # [1, 32, 128]
        keys = keys, # [1, 32, 1024, 128]
        values = values, # [1, 32, 1024, 128]
        cached_key_signatures=None, # [1, 32, 1, 1024]
        sink_size = init_offset,
        window_size = local_offset,
        heavy_size= heavy_size,
        hat_bits=hat_bits,
        hat_mlp_layers=hat_mlp_layers,
        hat_mlp_hidden_size = hat_mlp_hidden_size,
        hat_mlp_activation = hat_mlp_activation, # this can be removed if we use standard silu
        hat_weights_query_matrix = hat_weights_query_matrix,
        hat_weights_query_bias = hat_weights_query_bias,
        hat_weights_key_matrix = hat_weights_key_matrix,
        hat_weights_key_bias = hat_weights_key_bias,
        base_rate_sampling = base_rate_sampling,
        epsilon = epsilon,
        delta_ppf = norm.ppf(1 - delta),
    )


def benchmark_vattention(num_runs: int = 20, warmup_runs: int = 5):
    """
    Simple micro-benchmark for ref_vAttention_fwd to measure average runtime (milliseconds) on GPU.
    """
    import time
    from scipy.stats import norm

    batch_size = 1
    query_heads = 32
    kv_heads = 32
    query_len = 1
    kv_len = 10240
    d_model = 128

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Attention tensors
    queries = torch.randn(batch_size, query_heads, query_len, d_model, dtype=torch.float32).to("cuda")
    keys = torch.randn(batch_size, kv_heads, kv_len, d_model, dtype=torch.float32).to("cuda")
    values = torch.randn(batch_size, kv_heads, kv_len, d_model, dtype=torch.float32).to("cuda")

    # vAttention configuration
    sink_size = 0.05
    window_size = 0.05
    heavy_size = 0.05
    hat_bits = 32
    hat_mlp_layers = 3
    hat_mlp_hidden_size = 128
    hat_mlp_activation = "silu"
    hat_weight_file = "/FirstIntelligence/home/shuo/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"

    base_rate_sampling = 0.05
    epsilon = 0.25
    delta = 0.25

    hat_weights_query_matrix, hat_weights_query_bias, hat_weights_key_matrix, hat_weights_key_bias = extract_layer_weights(
        hat_weight_file, 0, dtype=torch.float32, device="cuda"
    )

    # Warm-up runs to stabilize kernels
    for _ in range(warmup_runs):
        _ = ref_vAttention_fwd(
            queries=queries.squeeze(2),
            keys=keys,
            values=values,
            cached_key_signatures=None,
            sink_size=sink_size,
            window_size=window_size,
            heavy_size=heavy_size,
            hat_bits=hat_bits,
            hat_mlp_layers=hat_mlp_layers,
            hat_mlp_hidden_size=hat_mlp_hidden_size,
            hat_mlp_activation=hat_mlp_activation,
            hat_weights_query_matrix=hat_weights_query_matrix,
            hat_weights_query_bias=hat_weights_query_bias,
            hat_weights_key_matrix=hat_weights_key_matrix,
            hat_weights_key_bias=hat_weights_key_bias,
            base_rate_sampling=base_rate_sampling,
            epsilon=epsilon,
            delta_ppf=norm.ppf(1 - delta),
        )
        torch.cuda.synchronize()

    # Benchmark timed runs
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        _ = ref_vAttention_fwd(
            queries=queries.squeeze(2),
            keys=keys,
            values=values,
            cached_key_signatures=None,
            sink_size=sink_size,
            window_size=window_size,
            heavy_size=heavy_size,
            hat_bits=hat_bits,
            hat_mlp_layers=hat_mlp_layers,
            hat_mlp_hidden_size=hat_mlp_hidden_size,
            hat_mlp_activation=hat_mlp_activation,
            hat_weights_query_matrix=hat_weights_query_matrix,
            hat_weights_query_bias=hat_weights_query_bias,
            hat_weights_key_matrix=hat_weights_key_matrix,
            hat_weights_key_bias=hat_weights_key_bias,
            base_rate_sampling=base_rate_sampling,
            epsilon=epsilon,
            delta_ppf=norm.ppf(1 - delta),
        )
        torch.cuda.synchronize()
    avg_ms = (time.time() - start_time) * 1000 / num_runs
    print(f"ref_vAttention_fwd average runtime over {num_runs} runs: {avg_ms:.2f} ms")


if __name__ == "__main__":
    run_mha_example()
    benchmark_vattention()
    
