# Quest Sparse Attention Baseline

## 1. Quest Algorithm Description
The **Quest** algorithm is a *sparse attention selection mechanism* that approximates full attention by selecting a small subset of key positions that are most likely to contribute high attention scores for each query.

### Page-level Vectorization
- Instead of scoring every query–key pair $O(QKD)$, the algorithm scores only query–page pairs ($O(QPD)$, with $P \ll K$).
- Each page is represented by its per-dimension min/max bounds, allowing fully vectorized GPU computation.

### Quest Bound (Efficient Upper-bound Scoring)
- Provides a **tight upper bound** on potential dot-products without explicit key–query multiplication.
- Geometry-aware and monotonic: larger bound ⇒ potentially larger true score.

### Deterministic Top-K Masking
- Unlike sampling-based sparse attention (MagicPig), Quest produces a **deterministic mask** from pure tensor operations — no randomness or collisions.

### Core Algorithm

1. **Paging Keys**  
   The key sequence of length $K$ is divided into contiguous *pages* of fixed size $P_s$.  
   Number of pages: $P = \left\lceil \frac{K}{P_s} \right\rceil$

2. **Per-page Bounds**  
   For each page $p$, compute the **elementwise min and max** of all key vectors inside that page:

   $ k_{\text{min}}^{(p)}[d] = \min_{k \in p} K[k, d], \quad k_{\text{max}}^{(p)}[d] = \max_{k \in p} K[k, d] $
   
   These bounds define a hyper-rectangle in $D$-dimensional space containing all keys of that page.

4. **Query–Page Scoring via the Quest Bound**  
   For each query vector $q$ and page $p$, compute a fast *upper bound* on the possible dot-product between $q$ and any key inside page $p$:

   $ S(q,p) = \sum_{d=1}^{D} \max\!\big(q_d \, k_{\text{min}}^{(p)}[d],\; q_d \, k_{\text{max}}^{(p)}[d]\big) $

   This **Quest bound** overestimates the true maximum similarity but preserves the ranking of highly relevant pages.

5. **Selecting Top-K Pages**  
   Each query selects the top

   $ K_p = \left\lceil \frac{K_{\text{heavy}}}{P_s} \right\rceil $

   pages by $S(q,p)$.  
   All tokens within these pages are activated (unmasked).  
   The union of their indices defines the **sparse attention mask**.

6. **Final Mask Formation**  
   The page-selected mask is merged with any previous or external attention mask, ensuring that only valid tokens remain active.

### Theoretical Foundation

The Quest algorithm relies on a **bounding property of linear forms over intervals**:

For a query $q \in \mathbb{R}^D$ and any key $x \in [k_{\min}, k_{\max}]$:

$ \max_{x \in [k_{\min}, k_{\max}]} q \cdot x = \sum_{d=1}^{D} \max(q_d k_{\min}[d],\, q_d k_{\max}[d]) $

Thus, $S(q,p)$ gives a guaranteed upper bound:

$ q \cdot k \le S(q,p), \quad \forall k \in \text{page } p $

Selecting top pages by $S(q,p)$ ensures that pages containing the top-scoring keys are highly likely to be selected — similar to *branch-and-bound search*.


### Example config in sparse-attention-hub
```
    config = ResearchAttentionConfig(masker_configs=[
        QuestTopKMaskerConfig(
            heavy_size=0.128,
            page_size=16
        )
    ])
```

## 2. Experiment Chosen for Reproducing
### Experimental Setup
Some datasets from the LongBench benchmark

## 3. Comparative Results

| Dataset | Token Budget 256 | Token Budget 512 | Token Budget 1024 | Token Budget 2048 | Token Budget 4096 |
|:--------|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| **Qasper** | 2.9 | 4.08 | 6.33 | 16.34 | 24.27 |
| **HotpotQA** | 1.97 | 15.19 | 24.06 | 27.96 | 40.06 |
| **MultifieldQA** | 2.48 | 5.1 | 18.15 | 30.66 | 43.84 |
| **GovReport** |  |  |  |  | |

## 4. Explanations of Differences
The original paper uses dense layers for first 2 transformer blocks. In our setting we are using all sparse layers.

## References
[1] Tang, Jiaming; Zhao, Yilong; Zhu, Kan; Xiao, Guangxuan; Kasikci, Baris; Han, Song (2024). Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference. arXiv preprint arXiv:2406.10774.
