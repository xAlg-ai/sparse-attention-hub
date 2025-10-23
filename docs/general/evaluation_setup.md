While focusing on the problem of using sparse attention to solve the decode latency of full attention. We have potentitally two evauation setups that we can use.

A: The entire prompt  (context +  question) is first processed with full attention, and sparse attention is applied only during the decoding phase. In this setup, the first token generated already benefits from full attention.

B: Split-prompt processing (context vs. question)}: The prompt is divided into two parts:
1. Context is processed with full attention
2. Question + subsequent generations are processed with sparse attention


Some earlier works, such as MagicPig, adopt the first setup. In contrast, more recent approaches—including HashAttention and SqueezeAttention—follow the second. A methodology similar to the second is also used in NVIDIA’s KVPress– a framework to compare KV cache compression methods, where the KV cache is compressed after the context is processed but before the question is introduced. We argue that the second setup is the more meaningful choice. The reasoning is as follows. Sparse attention for long-context evaluation is usually tested on datasets with relatively short generations (e.g., RULER, LongBench, etc). Suppose the entire context is first processed by full attention (setup 1). In that case, all the necessary information to answer the question has already been extracted by the time the first token is predicted. Applying sparse attention only after this point, especially with a fixed local attention window, does not truly test its ability to retrieve and utilize information from the long context. 