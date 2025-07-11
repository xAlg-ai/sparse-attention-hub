"""Integration tests for end-to-end sparse attention functionality."""

import pytest


@pytest.mark.integration
class TestEndToEndSparseAttention:
    """Test end-to-end sparse attention pipeline."""

    def test_full_attention_pipeline(self, medium_sequence_length):
        pass

    def test_benchmark_integration(self, small_sequence_length):
        pass

    def test_model_hub_integration(self):
        pass

    def test_metrics_integration(self):
        pass


# @pytest.mark.integration
# class TestUserStory:
#     """Test end-to-end sparse attention pipeline."""

#     def test_user_story_1(self):
#         from sparse_attention_hub.sparse_attention.generator import SparseAttentionHF
#         sparse_attention_config = OracleTopKConfig(heavy_size=0.25)
#         sparse_attention = SparseAttentionHF(sparse_attention_config)
#         custom_attention = sparse_attention.get_custom_attention()
#         from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
#         ALL_ATTENTION_FUNCTIONS.register("custom_attention", custom_attention)
#         model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, _attn_implementation="custom_attention")
#         sparse_attention_meta_data = {}
#         prompt = "Hello, world!" *100
#         inputs = tokenizer(prompt, return_tensors="pt")
#         outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, sparse_attention_meta_data=sparse_attention_meta_data)
#         print(tokenizer.decode(outputs[0], skip_special_tokens=True))
