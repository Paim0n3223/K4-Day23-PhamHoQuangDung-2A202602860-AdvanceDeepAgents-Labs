# Efficient Inference and Small Language Models: Methods, Systems, and Open Problems
## TL;DR
- Efficient inference is now a central design goal because serving language models is constrained by latency, throughput, memory, power, and storage, especially in edge and mobile settings [1][2].
- Compression remains the most established path to smaller and faster models: quantization, pruning, distillation, and low-rank methods reduce memory and compute, but often require calibration or retraining and can be sensitive to data choice [3][4][5].
- Architecture and decoding tricks can speed up generation without changing the base model size: early-exit and self-speculative decoding, speculative decoding, and KV-cache compression all trade extra control logic for lower per-token cost [6][7][8][9][10].
- Recent small-model results show that carefully trained compact models can stay competitive on benchmark tasks and even run efficiently on smartphones, while evaluation work is also shifting toward more robust judge models and benchmark design [11][12][13].
- Open problems remain around benchmark realism, heterogeneous deployment, cache and privacy risks, and the fact that gains in speed often come with hidden overheads or quality instability [14][15][16][17].

## Background
Small language models and efficient inference address the gap between model capability and deployment constraints. Surveys of efficient inference emphasize that large model size, quadratic attention, and autoregressive decoding make latency, throughput, memory footprint, and power consumption central bottlenecks [1]. Small-language-model surveys frame the same problem in mobile, on-device, and edge settings, where compute, speed, memory, and latency must all be optimized together [2].

Foundational compression work shows why this matters. DistilBERT reported a 40% size reduction, 97% retained language-understanding capability, and 60% faster inference relative to BERT, while also introducing a distillation-based training recipe [18]. MobileBERT and TinyBERT similarly show that compact students can preserve much of the teacher’s quality while reducing latency and model size substantially [19][20]. These early results established the core tradeoff that still organizes the field: reducing parameters or precision tends to improve serving efficiency, but the challenge is preserving task quality and robustness [18][19][20].

## Compression and Parameter Reduction
Compression is the most direct family of efficiency methods. Surveys group it into quantization, pruning, distillation, low-rank factorization, and related compact-architecture designs [3][4][1]. Quantization maps values into a smaller finite set and is attractive because it reduces memory cost and can improve speed, especially when low-bit hardware support exists [3]. The same surveys distinguish post-training quantization from quantization-aware training, noting that the latter adds retraining cost but can reduce quantization error [4].

Pruning removes unimportant neurons, heads, layers, or other components to lower inference cost, but its effectiveness is often less robust than quantization or distillation for large language models because compression usually depends on fine-tuning or retraining [3][4]. Distillation remains one of the most reliable ways to compress a model while retaining behavior, as the student learns from a teacher rather than from task labels alone [18][3]. Low-rank factorization and hybrid compression methods further reduce parameter count by decomposing or re-parameterizing weight matrices, but they sit on a spectrum from cheap post-training transformations to more invasive training-time redesigns [4].

A recurring finding is that compression quality depends strongly on calibration data and the exact pipeline. One Hugging Face paper reports significant performance variability for post-training pruning and quantization depending on calibration data, underscoring that compression is not purely a model-intrinsic property [5]. More broadly, the compression literature separates high-cost methods that retrain on large datasets from low-cost or tuning-free methods that are easier to deploy but may be less stable [4].

## Architecture and Decoding Co-Design
A second family improves efficiency by changing how inference is executed rather than only shrinking the model. Early-exit methods attach intermediate prediction points so that easy tokens can stop before the final layer stack, and LayerSkip shows that training with layer dropout and a shared early-exit loss can improve early-exit accuracy without auxiliary modules [6]. Self-speculative decoding pushes the same idea further: the model drafts tokens using a partial forward pass and then verifies them with the full model, allowing speedups without a separate draft network [6][7][10].

Speculative decoding more generally is a major inference-time acceleration strategy because it can reduce inter-token latency under memory-bound workloads [9]. In this family, a smaller draft model or a partial computation path proposes candidate tokens, and the larger model verifies them in fewer expensive steps [7][9]. The Hugging Face coverage of SpecExec and Kangaroo shows continued interest in this direction, especially for consumer hardware and interactive settings where keeping latency low matters more than raw batch throughput [9][10].

Sparse mixture-of-experts and router-based designs offer another architectural route to efficiency. Instead of activating every parameter for every token, sparse MoE routes each input to only a few experts, and router designs such as HyperRouter or task-level MoE aim to make this cheaper or more deployment-friendly [21][22]. The appeal is that the model can preserve capacity while only paying for a small activated subset at inference time [21][22].

## Cache and Memory-Efficient Inference
A major practical bottleneck in autoregressive generation is the KV cache. Surveys of efficient inference explicitly call out KV-cache size as a core memory issue because cache growth scales with input length [1]. Methods such as LESS, LCKV-style designs, and KV cache compression frameworks target this bottleneck directly, often combining recurrence, selective eviction, or cache summarization with the goal of keeping memory bounded while minimizing quality loss [8][15].

These methods are conceptually different from parameter compression. Instead of shrinking the weights, they reduce the runtime state needed to generate long outputs, which is especially important for long-context or high-throughput serving [8][15]. The tradeoff is that cache management introduces its own scheduling and approximation choices, and systems such as KVPress emphasize that there are many competing policies for what to keep, merge, or evict [15].

## Recent Results and Benchmarks
Recent work suggests that compact models can still be highly capable when training and evaluation are handled carefully. H2O-Danube3 is presented as a series of small language models with high benchmark performance and efficient local inference on smartphones [11]. Orca 2 shows that smaller models can be taught richer reasoning strategies and can outperform larger models on complex reasoning tasks in some settings [12]. Multi-token prediction is another recent line showing that predicting multiple future tokens can improve sample efficiency, generative benchmark performance, and inference time [23].

The evaluation side is also changing. One Hugging Face daily paper argues that replacing a single large judge model with a panel of diverse models can reduce cost and intramodel bias in evaluation [13]. That matters for small-model research because benchmark ranking can vary depending on how outputs are judged, especially when quality differences are subtle [13]. Taken together, these results suggest that the field is moving from isolated compression tricks toward full stack evaluation: model size, decoding policy, hardware, and judge methodology all affect the apparent efficiency-quality frontier [11][23][13].

## Trends and open problems
The strongest trend is toward hybrid systems that combine several efficiency mechanisms at once. Many recent methods mix compact models, early exit, speculative decoding, cache compression, and smarter routing, rather than relying on a single trick [6][8][22][9][10][15]. This reflects a practical reality: compute savings at one stage can be offset by overhead elsewhere, so end-to-end serving performance matters more than isolated benchmark numbers [1][15].

Several open problems remain unresolved. First, evaluation is still fragmented: surveys point out the need for metrics that capture latency, memory, privacy, and energy together, not just accuracy [2][14][16][17]. Second, deployment settings are heterogeneous, and IID NLP benchmarks do not fully reflect device constraints, scheduling issues, or collaborative edge-cloud execution [15][17]. Third, compression and cache-based methods can be sensitive to calibration data and may incur hidden accuracy losses or overheads [5][15][16]. Finally, safety and privacy concerns are increasingly visible: surveys highlight hallucination, bias, energy use, cache leaks, and the risk that leaked data could reconstruct user conversations [14][15][17].

## References
[1] A Survey on Efficient Inference for Large Language Models. web. https://arxiv.org/html/2404.14294v2 (n.d.)
[2] A Survey of Small Language Models. hf-search. https://huggingface.co/papers/2410.20011 (2024-10-25)
[3] Model Compression and Efficient Inference for Large Language Models: A Survey. hf-search. https://huggingface.co/papers/2402.09748 (2024-02-15)
[4] A Comprehensive Survey of Compression Algorithms for Language Models. hf-search. https://huggingface.co/papers/2401.15347 (2024-01-27)
[5] How Does Calibration Data Affect the Post-training Pruning and Quantization of Large Language Models?. hf-search. https://huggingface.co/papers/2311.09755 (2023-11-16)
[6] LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. web. https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/ (2024-06-14)
[7] Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding. web. https://aclanthology.org/2024.acl-long.607/ (n.d.)
[8] Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference. hf-search. https://huggingface.co/papers/2402.09398 (2024-02-14)
[9] SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices. hf-search. https://huggingface.co/papers/2406.02532 (2024-06-04)
[10] Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting. hf-daily. https://huggingface.co/papers/2404.18911 (2024-04-29)
[11] H2O-Danube3 Technical Report. hf-search. https://huggingface.co/papers/2407.09276 (2024-07-12)
[12] Orca 2: Teaching Small Language Models How to Reason. hf-search. https://huggingface.co/papers/2311.11045 (2023-11-18)
[13] Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models. hf-daily. https://huggingface.co/papers/2404.18796 (2024-04-29)
[14] Small Language Models:Survey, Measurements, and Insights. arxiv. https://arxiv.org/abs/2409.15790 (2025-02-26)
[15] Taming the Titans: A Survey of Efficient LLM Inference Serving. web. https://aclanthology.org/2025.inlg-main.32.pdf (n.d.)
[16] Optimization Methods, Challenges, and Opportunities for Edge Inference: A Comprehensive Survey. web. https://www.mdpi.com/2079-9292/14/7/1345 (2025-03-02)
[17] Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey of Algorithms, Execution, and Open Challenges. web. https://tianweiz07.github.io/Papers/25-csur-2.pdf (n.d.)
[18] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arxiv. https://arxiv.org/abs/1910.01108 (2019-10-02)
[19] MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. web. https://aclanthology.org/2020.acl-main.195.pdf (n.d.)
[20] TinyBERT: Distilling BERT for Natural Language Understanding. web. https://aclanthology.org/2020.findings-emnlp.372.pdf (n.d.)
[21] HyperRouter: Towards Efficient Training and Inference of Sparse Mixture of Experts via HyperNetwork. web. https://aclanthology.org/2023.emnlp-main.351.pdf (n.d.)
[22] Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference. web. https://aclanthology.org/2021.findings-emnlp.304.pdf (n.d.)
[23] Better & Faster Large Language Models via Multi-token Prediction. hf-search. https://huggingface.co/papers/2404.19737 (2024-04-30)
