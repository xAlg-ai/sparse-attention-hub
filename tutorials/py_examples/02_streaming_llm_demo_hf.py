#!/usr/bin/env python3
"""
You can also choose to have more control instead calling process_request directly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.adapters.huggingface import ModelAdapterHF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# D model_name = "microsoft/DialoGPT-medium"
# D model_name = "google/gemma-3n-E4B-it"
model_name = "microsoft/Phi-4-mini-instruct"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

import os

os.chdir("/data/apdesai/code/sparse-attention-hub")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# Create StreamingLLM configuration: Sink (4 tokens) + Local (64 tokens)
sink_config = SinkMaskerConfig(sink_size=4)
local_config = LocalMaskerConfig(window_size=4)
research_config = ResearchAttentionConfig(masker_configs=[sink_config, local_config])

print(f"✅ StreamingLLM config: Sink(4) + Local(64)")

adapter = ModelAdapterHF(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    sparse_attention_config=research_config,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda:0",
)


test_content = """ 
        The concept of attention mechanisms has revolutionized natural language processing and machine learning.
        StreamingLLM addresses efficiency challenges by implementing sparse attention patterns that combine:
        1. Sink tokens: The first few tokens contain crucial global information
        2. Local attention: Recent tokens are most relevant for next token prediction
        This approach maintains performance while reducing computational costs for long sequences.
        "Summarize the above in a single title with less than 10 words. Given only the title."
        """

print("Running Full Attention")

messages = [{"role": "user", "content": test_content}]
test_content = adapter.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_ids = adapter.tokenizer.encode(
    test_content, return_tensors="pt", add_special_tokens=False
).to(device)


output_ids = adapter.model.generate(input_ids, max_new_tokens=50, do_sample=False)
output_text = adapter.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Full Attention Response: ", output_text)

print("Running Sparse Attention")


with adapter.enable_sparse_mode():
    output_ids = adapter.model.generate(
        input_ids, max_new_tokens=50, do_sample=False, sparse_meta_data={}
    )
    # model.generate() validates kwargs and will not let sparse_meta_data to be passed in which is required
    # Locally, I removed the validation for this to work. TODO(we can write our own generate function for utility)
    output_text = adapter.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Streaming Attention Response: ", output_text)
