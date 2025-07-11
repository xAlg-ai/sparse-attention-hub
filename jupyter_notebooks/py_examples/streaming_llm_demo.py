#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)
from sparse_attention_hub.sparse_attention.integrations.hugging_face import SparseAttentionHF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "meta-llama/Llama-3.1-8B-Instruct"
#D model_name = "microsoft/DialoGPT-medium"
#D model_name = "google/gemma-3n-E4B-it"
model_name = "microsoft/Phi-4-mini-instruct"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print(f"Loaded {model_name}")
    
except Exception as e:
    raise e
sink_config = SinkMaskerConfig(sink_size=4)
local_config = LocalMaskerConfig(window_size=64)
research_config = ResearchAttentionConfig(masker_configs=[sink_config, local_config])

sparse_attention_hf = SparseAttentionHF.create_from_config(research_config)

test_text = """
The concept of attention mechanisms has revolutionized natural language processing and machine learning. 
StreamingLLM addresses efficiency challenges by implementing sparse attention patterns that combine:
1. Sink tokens: The first few tokens contain crucial global information
2. Local attention: Recent tokens are most relevant for next token prediction

This approach maintains performance while reducing computational costs for long sequences.
Summarize the above in a single title with less than 10 words.
"""

inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

model.eval()
max_new_tokens = 50

start_time = time.time()
with torch.no_grad():
    full_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
full_time = time.time() - start_time

full_generated_ids = full_outputs[0]
full_generated_text = tokenizer.decode(full_generated_ids, skip_special_tokens=True)

# attention mask is set to None
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

start_time = time.time()
with torch.no_grad():
    with sparse_attention_hf(model) as sparse_model:
        sparse_outputs = sparse_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            sparse_meta_data={}
        )
sparse_time = time.time() - start_time

sparse_generated_ids = sparse_outputs[0]
sparse_generated_text = tokenizer.decode(sparse_generated_ids, skip_special_tokens=True)

speedup = full_time / sparse_time if sparse_time > 0 else 0
print(f"Full attention: {full_time:.2f}s")
print(f"Sparse attention: {sparse_time:.2f}s")
print(f"Speedup: {speedup:.2f}x")

full_input_length = len(input_ids[0])
full_new_text = tokenizer.decode(full_generated_ids[full_input_length:], skip_special_tokens=True)
sparse_new_text = tokenizer.decode(sparse_generated_ids[full_input_length:], skip_special_tokens=True)

print(f"Full attention output: {full_new_text}")
print(f"Sparse attention output: {sparse_new_text}")
print(f"StreamingLLM successfully applied with {speedup:.2f}x speedup!") 
