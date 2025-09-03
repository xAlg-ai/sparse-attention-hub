import os

os.chdir("/data/apdesai/code/sparse-attention-hub")

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
print(f"Device: {device}")

sink_config = SinkMaskerConfig(sink_size=4)
local_config = LocalMaskerConfig(window_size=16)
research_config = ResearchAttentionConfig(masker_configs=[sink_config, local_config])

print(f"✅ StreamingLLM config: Sink(4) + Local(16)")

# model_name = "microsoft/Phi-4-mini-instruct"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

adapter = ModelAdapterHF(
    model_name=model_name,
    sparse_attention_config=research_config,
    model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "cuda"},
    device="cuda",
)

from sparse_attention_hub.adapters import Request


test_context = """ 
        The concept of attention mechanisms has revolutionized natural language processing and machine learning.
        StreamingLLM addresses efficiency challenges by implementing sparse attention patterns that combine:
        1. Sink tokens: The first few tokens contain crucial global information
        2. Local attention: Recent tokens are most relevant for next token prediction
        This approach maintains performance while reducing computational costs for long sequences.
        """
test_questions = [
    "Summarize the above in a single title with less than 10 words. Given only the title.",
    "What are other attention mechanisms that are used in the field of LLMs?",
]

request = Request(
    context=test_context,
    questions=test_questions,
)

print("Running Streaming Attention on Question")
response = adapter.process_request(request)
response_text = response.responses
print("Streaming Attention Response: ", response_text)
