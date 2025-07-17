# Benchmarks

We currently support the following datasets:
- [Loogle](loogle/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/loogle))
- [RULER](ruler/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/ruler))
- [Zero Scrolls](zero_scrolls/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/zero_scrolls))
- [Infinitebench](infinite_bench/README.md) ([hf link](https://huggingface.co/datasets/MaxJeblick/InfiniteBench))
- [longbench](longbench/README.md)([hf link](https://huggingface.co/datasets/Xnhyacinth/LongBench))
- [longbench-v2](longbenchv2/README.md)([hf link](https://huggingface.co/datasets/Xnhyacinth/LongBench-v2))

Please refer to the README of each dataset for more information on how the Hugging Face dataset was generated.

<details><summary> 

### RULER
</summary>

Average performance the 13 tasks of the RULER dataset with 4k context length (per task results [here](../evaluation/assets/)):

![RULER](../evaluation/assets/ruler_4096_average%20score.png) 

Observations: 
- snapkv w/ question consistently outperforms other methods. However this method can't be use for use cases such as prompt caching as it requires the question to be known beforehand.
- All presses show degradation in performance even for small compression ratios.
- llama3.1-8b-instruct is more robust to compression than other models and expected attention performs better than others.
- mistral-nemo-instruct-2407 is more robust to random pruning than other models.
- For phi-3.5-mini and mistral-nemo-instruct-2407, all presses perform poorly compared to baseline presses such as random (remove KV pairs randomly) or streaming llm (remove the middle KV pairs). This is especially true for the subtask [niah_single_3](assets/ruler_4096_niah_single_3.png) where most presses fail to perform a proper copy-paste of a long needle in a haystack. This might be related to [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- For phi-3.5-mini, we ran an additional experiment with a different compression ratio per layer (as in [this notebook](../notebooks/per_layer_compression_demo.ipynb)) which  largely outperformed it's uniform compression counterpart (see purple cross on 2nd plot). The ratios where determined by grid search on 20/6500 samples from RULER (so results can be questionable).

</details>

<details><summary> 

### Loogle
</summary>

Shortdep_qa
![shortdep_qa](../evaluation/assets/loogle_shortdep_qa.png)
Shortdep_cloze
![shortdep_cloze](../evaluation/assets/loogle_shortdep_cloze.png)
Longdep_qa
![longdep_qa](../evaluation/assets/loogle_longdep_qa.png) 

Observations: 
- Metrics are adapted from loogle benchmark, see [here](../evaluation/loogle/calculate_metrics.py). The plot show the average score (mean over all submetrics) for each task.
- The metrics are not always correlated with the quality of the answer, especially for longdep_qa task. LLM-as-a-judge may better suited for a more refined evaluation.
- Again, snapkv w/ question consistently outperforms other methods.
- In longdep_qa, the model looses track on counting (e.g. answer to "How many times is person x mentioned?" gets lower with increased compression ratio). This is not necessarily reflected in the metrics.
- Llama3.1-8b-instruct seems to be more robust to compression.
- Observed attention context had to be truncated at 10 000 tokens to prevent OOM issues, as the attention matrix needs to be materialized.
- For shortdep_cloze task, the output formatting is often ignored leading to performance degradation even for low compression ratios. Interestingly, the model may still be able to answer the question correctly.
- mistral-nemo-instruct-2407 fails to perform well on the shortdep_cloze task, even without compression, and is thus not reported.
- shortdep_cloze task runs OOM for phi-3.5-mini at compression ratio 0.0 and is thus missing.

</details>

<details><summary> 

### Infinitebench
</summary>

kv_retrieval
![kv_retrieval](../evaluation/assets/infinitebench_kv_retrieval.png)
longbook_choice_eng
![longbook_choice_eng](../evaluation/assets/infinitebench_longbook_choice_eng.png)
longbook_qa_eng
![longbook_qa_eng](../evaluation/assets/infinitebench_longbook_qa_eng.png)
longdialogue_qa_eng
![longdialogue_qa_eng](../evaluation/assets/infinitebench_longdialogue_qa_eng.png)


Observations: 
- All task where run with max_len=70_000 tokens, except for observed attention which used 10_000 tokens.
- For kv-retrieval subtask, streaming LLM (keep last N tokens) performs better than other methods. While this may be surprising at first, respecting the format of the task `(Extract the value corresponding to the specified key in the JSON object below. JSON data: {"7de93460-b65f-404e-9a7d-af2da2c8abb5": "2d9ab7c8-394a-4062-9928-310e39201a2f", ...}. Key: "70d1b207-d1e8-4591-95b8-9c85aceb8956"`
helps to understand this behavior. The information is homogeneously distributed in the context, and any token could potentially be relevant for answering the question. Streaming LLM will have access to all last tokens, while other methods will potentially create "holes".
- Mistral-nemo-instruct-2407 performs poorly on kv-retrieval subtask compared to other models and is thus excluded from the plots.
- For longbook-choice-eng, many compression methods are able to obtain good compression ratios. Thus, longbook-choice-eng is an example of a task that can be compressed effectively.
- For longbook-qa-eng, expected attention and snapkv perform better than other methods (note the performance difference of llama3.1-8b-instruct and phi3.5/mistral-nemo).
- For longdialogue-qa-eng, there's an interesting crossover between different compression methods. For higher compression, snapkv performs relatively well across models.

</details>


## How to add a dataset

Each dataset directory is structured as follows:

```bash
$dataset
├── README.md
├── calculate_metrics.py
├── create_huggingface_dataset.py
```

Where:
- `create_huggingface_dataset.py` is a script that generates the Hugging Face dataset from the original dataset. Each dataset is associated with a set of parquet files with the following structure:
  - `context`: ... 
  - `question`: ...
  - `answer_prefix`: ...
  - `answer`:  ...
  - `max_new_tokens`:  ...
- `calculate_metrics.py` is a script that calculates the metrics based on the output of `evaluate.py`
