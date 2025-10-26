# Double Sparsity Sparse Attention Baseline

## 1. Double Algorithm Description
Double Sparsity[1] is a top-k based sparse attention method. in ordder to detect the top-k, it proposes to use fewer channels than using all the channels ( e.g. using 16 instead of 128 in llama model). The importance of channels are identified using offline profiling and we use the channel config stored in the author repository. During inference, only a few channels (set using group_factor ) are used for inner product computation. The channels are same across different heads and sparsity is also common across different heads. DS uses grouped query and key representation ( sum across heads) to compute the top-k tokens per layer.

    masker_config = 
    masker = DoubleSparsityTopKMasker.create_from_config(masker_config)
    research_attention = ResearchAttention(sparse_attention_config=masker_config, maskers=[masker])

### Example config in sparse-attention-hub
```
    config = ResearchAttentionConfig(masker_configs=[
        DoubleSparsityTopKMaskerConfig(
        heavy_size=0.1, # also can use integer e.g. 128
        group_factor=8,
        label_bits=8,
        sorted_channel_file="<json file from double sparsity repository>",
        )
    ])
```

## 2. Matching the author's code behavior exactly in sparse-attention-hub
We were able to exactly match the behavior of our code with double sparsity (core logic) here : [test_file](../../tests/unit/sparse_attention/research_attention/maskers/fixed/implementations/test_double_sparsity_correctness.py)

```
(sparse_attention_hub) ubuntu@192-222-56-211:~/sparse-attention-hub$ pytest tests/unit/sparse_attention/research_attention/maskers/fixed/implementations/test_double_sparsity_correctness.py

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================================================== 81 passed, 1 warning in 63.71s (0:01:03) ==================================================================================
```


### Experimental Setup
We use multifieldqa and hotpotqa from Longbench to compare our and author's code since author's code supports Longbench. 

|      | Our Code Base |          |      | Their Code Base |          |
|------|---------------|----------|------|-----------------|----------|
|      | MULTIFIED_QA  | HOTPOTQA |      | MULTIFIED_QA    | HOTPOTQA |
|  256 |         53.29 |     51.2 |  256 |           52.07 |    50.76 |
| 1024 |         52.04 |    53.05 | 1024 |           53.27 |    55.31 |
| 4096 |         53.73 |    51.38 | 4096 |           53.55 |    56.22 |


## 4. Explanations of Differences
There are two known differences in experimental setup. 
1. Author's code uses dense attention for layers 0 and 1. We on the other hand use all layers sparse
2. Author's code uses dense attention for processing of question and context. We on the other hand use dense attention for context and sparse attention for question and susbsequent generations. The setup is explained in [evaluation_setup](../general/evaluation_setup.md)

Since, we obtain the exact behavior of Attention using author's and our code, and we have known differences in evaluation, we do not further investigate the differences. 

## References

[1] Yang, S., Sheng, Y., Gonzalez, J.E., Stoica, I. and Zheng, L., 2024. Post-training sparse attention with double sparsity. arXiv preprint arXiv:2408.07092.