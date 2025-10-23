# MagicPig Sparse Attention Baseline

## 1. MagicPig Algorithm Description
To the best of our knowledge, MagicPig[1] was the first work to highlight the issues associated with top-k-based sparse attention. The method leverages Locality Sensitive Hashing (LSH) to select which tokens participate in attention computation. While LSH is generally considered suboptimal for approximate nearest neighbor (ANN) search due to its data-agnostic projections, its use here offers a principled and novel mechanism for approximating attention. LSH-based retrieval can be viewed as a sampler. Thus, the tokens retrieved from LSH have probabilities associated with them, under which they were sampled in the randomized construction of the LSH table. We can estimate the numerator and denominator of attention using the importance sampling formulation. While MagicPig was designed for CPU GPU hybrid management of KV Cache, in the context of this repository, we extract the core sparsity algorithm of MagicPig.

### Core Algorithm

MagicPig uses sink and local window tokens as is the case with most other methods. For token selection among rest of the tokens, it uses LSH tables. The keys are hashed into $L$ tables of range $2^K$ (i.e. K bit address space.) The query is hashed into the same set of hash tables to retrieve a set of tokens. (standard LSH retreival). Probability of retreival under randomized construction of LSH Table is computed for retrieved tokens. The inverse of these probabilities are used to weight the attention computation. 

### Example config in sparse-attention-hub
```
    config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        MagicPigConfig(
            lsh_l=75,  
            lsh_k=8
        )
    ])
```

## 2. Experiment Chosen for Reproducing

### Experimental Setup
Some datasets from the RULER benchmark

## 3. Comparative Results

|              | setup                                            | magicpig (K=8,L=75)                                                    | niah_single_1 | niah_single_2 | niah_single_3 | niah_multikey_2 | niah_multikey_3 | niah_multivalue |
|--------------|--------------------------------------------------|------------------------------------------------------------------------|---------------|---------------|---------------|-----------------|-----------------|-----------------|
| Authors code | A = B + questions processed via  dense attention | A =B (paper description) + dense layers(0,16) + no simpleLSH transform | 100           | 100           | 100           | 98              | 98              | 98              |
| Our code     | B                                                | B                                                                      | 100           | 96            | 76            | 46              | 12              | 81.5            |
|              | B                                                | B  + dense layers(0,16)                                                | 100           | 96            | 96            | 74              | 60              | 84.5            |
|              | A                                                | B  + dense layers(0,16)                                                | 100           | 98            | 98            | 94              | 90              | 88              |
|              | A                                                | A                                                                      | 100           | 100           | 100           | 98              | 98              | 95.5            |


**Evaluation setups:**

A: The entire prompt  (context +  question) is first processed with full attention, and sparse attention is applied only during the decoding phase. In this setup, the first token generated already benefits from full attention.

B: Split-prompt processing (context vs. question)}: The prompt is divided into two parts:
1. Context is processed with full attention
2. Question + subsequent generations are processed with sparse attention

**Method Setups**
A: dense layers used in (0,16) no simpleLSH transformation
B: All layers are considered sparse layers and simpleLSH asymmetric transformation is applied. 



## 4. Explanations of Differences
Matching the settings for method and evaluation closely in our code (i.e. A setup for both), we were able to reproduce the results shown by author's code. The biggest drops come when we remove the dense layers and use the setup B of evaluation where question is not processed via dense attention. The rationale behind evaluation setup can be found here [evaluation setup choices](../general/evaluation_setup.md)
There are some differences by using the simpleLSH transformation. But using the transformation is generally theoretically sound, and we do not expect it to significantly affect the quality at least in the negative sense. Also, it gives MagicPig a better chance at MIPS. So we choose to keep it in our implementation.


## References

[1] Chen, Z., Sadhukhan, R., Ye, Z., Zhou, Y., Zhang, J., Nolte, N., Tian, Y., Douze, M., Bottou, L., Jia, Z. and Chen, B., 2024. Magicpig: Lsh sampling for efficient llm generation. arXiv preprint arXiv:2410.16179.