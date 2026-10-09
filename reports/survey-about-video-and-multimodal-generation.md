# Survey of video and multimodal generation
## TL;DR
- Video and multimodal generation moved from CLIP-conditioned two-stage image systems to latent diffusion and then to large-scale text-to-video models, with cross-attention, classifier-free guidance, and latent representations becoming recurring design patterns [1][2][3][4].
- Diffusion remains the dominant quality-oriented paradigm for video generation, but recent surveys note that autoregressive and token-based models are competitive when efficiency, controllability, or unified multimodal modeling matter [4][5][6][7][8].
- Recent unified systems increasingly treat images, video, audio, language, and even action as tokenized signals in one model, enabling instruction following, free-form composition, and cross-modal transfer [9][10][11][12][13][14].
- Evaluation is shifting from single scores to benchmark suites that break quality into temporal consistency, motion, condition alignment, and safety, because standard metrics often disagree with human judgment [15][16][17][18][19][20][21].

## Background
Modern video and multimodal generation builds on two related ideas: first, learn a semantic latent space from paired data; second, generate in that latent space rather than directly in pixels. A CLIP-based two-stage design illustrates this transition: a prior maps text to an image embedding and a diffusion decoder turns that embedding into an image, improving diversity while preserving photorealism and caption alignment [1][2]. Surveys of text-to-image diffusion describe the same broader shift from pixel-space systems to latent-space systems, with cross-attention and classifier-free guidance as especially influential conditioning mechanisms [3].

These ideas matter for video because video models inherit both the representational challenge of images and the temporal challenge of coherent dynamics. A recent survey of video diffusion frames GANs, autoregressive models, and diffusion models as the major paradigms, and emphasizes that diffusion has become the leading approach for high-quality and temporally consistent video generation [4]. In practice, that means modern systems usually combine a learned latent representation with some form of text or image conditioning, then add temporal modules or schedules to maintain coherence across frames [1][3][4].

## Diffusion-based video generation
Diffusion-based methods remain the clearest path to high visual fidelity. The survey evidence is consistent: video diffusion is the dominant quality-oriented paradigm, and recent implementations increasingly mix pixel-based and latent-based modeling, optical flow, adaptive noise scheduling, and agent-driven frameworks [4]. The common strength is visual realism and temporal smoothness; the common weakness is computational cost, especially when the model must denoise many frames jointly [4].

The literature also suggests that video diffusion is broadening beyond simple text-to-video. The benchmark and safety papers treat image-to-video, time-lapse generation, and other conditional settings as distinct subproblems, which implies that “video diffusion” is now a family of tasks rather than a single formulation [16][17][20]. In this setting, the core question is less whether diffusion works, and more how to improve motion control, temporal structure, and safe generation without losing fidelity [4][16][17][20].

## Autoregressive, token-based, and masked modeling approaches
A parallel family models video as a sequence of tokens. The retrieved Hugging Face papers emphasize learned tokenizers, continuous tokens, discrete tokenization, masked autoregressive generation, and decoder-only architectures [5][6][7][8]. For example, LARP focuses on learned tokenization for video autoregression, VideoMAR uses continuous tokens in a decoder-only setup, and MAGI combines masked and causal modeling to address exposure bias [5][6][7].

This family is attractive because it targets controllability and efficiency. The notes repeatedly say that adaptive tokenization, semantic-aware permutation, and sparse attention can reduce compute while maintaining or improving generation quality [7][8]. Compared with diffusion, these approaches often claim stronger resource efficiency and more flexible decoding, while still aiming for long coherent sequences [5][6][7][8]. The trade-off is that the evidence in the notes is mostly paper-summary evidence rather than a single universally accepted benchmark victory, so the family should be viewed as a strong alternative rather than a settled replacement for diffusion [5][6][7][8].

## Unified multimodal generation across image, video, audio, and action
Recent systems increasingly merge generation and understanding across modalities. Unified-IO 2 tokenizes images, text, audio, actions, and bounding boxes into a shared semantic space and trains a single encoder-decoder Transformer for both understanding and generation [10]. Uni-MoE extends this idea with shared self-attention and sparse token-level routing across audio, speech, images, text, and video [22]. UniVLA goes further by treating vision, language, and action as discrete token sequences, using video for world modeling and downstream policy transfer [9].

For generation specifically, the trend is toward free-form composition and instruction-tuned control. OmniWeaving frames video generation as composition over interleaved text, multi-image, and video inputs, while UniAVGen and UniForm focus on joint audio-video generation with aligned outputs and modality-specific control signals [11][12][13]. JavisGPT similarly targets sounding-video comprehension and generation in one system [14]. Across these systems, the common design choice is to condition generation on richer multimodal context, rather than only on text prompts [9][10][11][12][13][14].

## Trends and open problems
The evaluation literature shows that progress is no longer judged well by a single number. VBench argues that IS, FID, FVD, and CLIPSIM can diverge from human judgment, so it decomposes evaluation into many dimensions including motion smoothness, temporal flickering, subject consistency, and spatial relationships [15]. AIGCBench, VBench++, Video-Bench, and ChronoMagic-Bench extend that logic with task-specific metrics and human-aligned scoring [16][17][18][19].

Safety and trustworthiness are now central open problems rather than side issues. T2VSafetyBench enumerates 12 safety aspects, including misinformation, copyright and trademark infringement, and temporal risk, and reports a trade-off between usability and safety [20]. A broader survey of video evaluation argues that trustworthy assessment should combine quality, safety, and provenance [21]. Taken together, the field’s main open problems are long-horizon coherence, faithful condition alignment, controllability, evaluation that matches human judgment, and safety-aware generation at scale [4][15][16][17][18][19][20][21].

## References
[1] Hierarchical text-conditional image generation with CLIP latents. arxiv. https://arxiv.org/abs/2204.06125 (2022-04-13)
[2] DALL-E 2 paper. web. https://cdn.openai.com/papers/dall-e-2.pdf (n.d.)
[3] Text-to-image Diffusion Models in Generative AI: A Survey. web. https://arxiv.org/html/2303.07909v3 (n.d.)
[4] Survey of Video Diffusion Models: Foundations, Implementations, and Applications. web. https://arxiv.org/abs/2504.16081 (n.d.)
[5] Taming Teacher Forcing for Masked Autoregressive Video Generation. hf-search. https://huggingface.co/papers/2501.12389 (2025-01-21)
[6] VideoMAR: Autoregressive Video Generatio with Continuous Tokens. hf-search. https://huggingface.co/papers/2506.14168 (2025-06-17)
[7] LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior. hf-search. https://huggingface.co/papers/2410.21264 (2024-10-28)
[8] Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation. hf-search. https://huggingface.co/papers/2505.18875 (2025-05-24)
[9] Unified Vision-Language-Action Model. web. https://arxiv.org/abs/2506.19850 (2025-06-00)
[10] Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action. web. https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_Unified-IO_2_Scaling_Autoregressive_Multimodal_Models_with_Vision_Language_Audio_CVPR_2024_paper.pdf (2024-00-00)
[11] OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning. web. https://huggingface.co/tencent/HY-OmniWeaving (n.d.)
[12] UniAVGen - Unified Audio and Video Generation. web. https://mcg-nju.github.io/UniAVGen/ (n.d.)
[13] UniForm. web. https://uniform-t2av.github.io/ (n.d.)
[14] JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation. web. https://github.com/LJungang/JavisGPT (n.d.)
[15] VBench: Comprehensive Benchmark Suite for Video Generative Models. web. https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf (2024-06-01)
[16] AIGCBench: Comprehensive Evaluation of Image-to-Video Content Generated by AI. arxiv. https://arxiv.org/abs/2401.01651 (n.d.)
[17] ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation. hf-search. https://huggingface.co/papers/2406.18522 (2024-06-26)
[18] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2411.13503 (2024-11-20)
[19] Video-Bench: Human-Aligned Video Generation Benchmark. hf-search. https://huggingface.co/papers/2504.04907 (2025-04-07)
[20] T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models. web. https://arxiv.org/html/2407.05965v2 (n.d.)
[21] Generative AI Video Evaluation: Survey of Metrics, Benchmarks, and Trustworthiness. web. https://openaccess.thecvf.com/content/CVPR2026W/VGBE/papers/Safavigerdini_Generative_AI_Video_Evaluation_Survey_of_Metrics_Benchmarks_and_Trustworthiness_CVPRW_2026_paper.pdf (n.d.)
[22] Uni-MoE. web. https://uni-moe.github.io/ (n.d.)
