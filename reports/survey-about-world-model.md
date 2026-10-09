# World Models in Machine Learning and Robotics: A Survey
## TL;DR
- World models are internal predictive simulators used to compress experience, imagine futures, and support planning or policy learning; the idea was revived in 2018 and is now central to debates about multimodal AI and action-controllable generation [1][2][3].
- Latent world-model methods such as predictive state representations, Dreamer-style latent imagination, and newer latent-dynamics improvements emphasize efficient rollouts in compact state spaces rather than pixel-level generation [4][5][6][7][8][9].
- Generative video world models focus on future visual synthesis and increasingly on action-conditioned simulation, but plausible video alone is not enough unless the future is sensitive to candidate actions [10][11][12][13][14][15][16].
- In embodied settings, world models are used as planners, simulators, and memory systems for navigation and robotics, often coupled with policies in either separate-module or one-model architectures [17][18][19][20][21][22].
- Benchmarking is shifting from visual realism toward decision usefulness, with stronger attention to action fidelity, planning success, physics disentanglement, and uncertainty calibration; open problems include long-horizon stability, compositional reuse, sim-to-real transfer, and scalable evaluation [23][24][25][26][27][28].

## Background
World models are usually described as internal models or internal representations of an environment, and surveys frame them as systems for understanding the present state of the world or predicting future dynamics [1][2][3]. In the modern RL and AI literature, they are often expected to do more than simply generate plausible observations: they should support prediction, counterfactual reasoning, planning, and policy improvement [3][12][16]. The idea became especially visible after the 2018 World Models paper, which framed learning a compressed spatiotemporal representation and training a small controller inside a dreamed environment as a path to sample-efficient control [1][2].

The recent surge of interest comes from two directions. First, model-based RL and embodied robotics keep pushing for better simulators of action consequences, especially when real interaction is expensive or risky [16][17][29]. Second, foundation-model-era systems have made video generation and multimodal prediction much more capable, which renewed the question of whether a generator can also function as a world model [3][30][10][11].

## Latent dynamics and dreaming-based learning
A classic family of world models represents the environment in a compact latent space and learns dynamics there rather than in pixel space [4][5][6][16]. Predictive State Representations (PSRs) are an early conceptual predecessor: instead of storing hidden state as an unobserved variable, they model state as a set of predictions about observable outcomes, and the survey notes that PSRs were studied together with planning techniques and research challenges [4]. This line of work is important because it clarifies a central design choice: state should be defined by what matters for future prediction and control, not just by reconstruction fidelity.

Dreamer-style methods pushed latent imagination into mainstream model-based RL. The notes describe Dreamer as learning a world model that predicts outcomes of potential actions, then training an actor and critic on imagined trajectories in the abstract latent space [5][6]. The attraction is computational: latent rollouts are cheaper than pixel rollouts, and many imagined trajectories can be processed in parallel [5][6]. Recent papers in this family continue to focus on sample efficiency, smooth latent dynamics, and scaling the imagination process [7][8][9].

The evidence in the notes suggests a steady trend: newer latent methods keep the control advantages of imagination while trying to make the latent transition model more stable or more efficient [7][8][9]. Sparse token processing, JEPA-style scaling, and gradient-penalized latent dynamics all point to the same basic bottleneck: latent imagination is powerful, but rollouts can become inaccurate or expensive as horizons grow [7][8][9].

## Generative video and action-conditioned world models
A second family models the future more directly in visual space. These methods predict future frames or video sequences from past observations, sometimes with task text or action sequences as additional inputs [10][11][12][14][15]. They are attractive because they make the future interpretable and can be used for planning, but they are also more demanding: visually pleasing predictions are not enough unless the generated future changes appropriately with different actions [15][16].

The notes highlight a shift from plain video prediction toward action-conditioned generation. UniPi treats sequential decision making as text-guided video generation and then uses an inverse-dynamics model to recover actions from the synthesized video [14][31]. AVID adapts pretrained video diffusion models to become action-conditioned world models by learning an adapter on action-labeled videos [15]. The broader survey notes that recent work spans language-guided video generation, multimodal inputs such as images and actions, and dynamic environment generation for embodied tasks [12][13].

This family differs from latent dynamics methods in where fidelity is spent. Latent methods spend capacity on compact predictive state and control-friendly rollouts, whereas video methods spend capacity on pixel realism and human-interpretable futures [10][11][12][16]. The notes also caution that a model should not be called a world model merely because it generates plausible videos; the predicted future must be action-sensitive and useful for closed-loop decision making [16][23].

## World models for embodied agents and navigation
Embodied AI adds additional requirements: the model must support long-horizon planning, memory, and feedback under physical constraints [17][18][19][22]. The notes describe embodied world models as internal simulators that help robots understand, interact with, and make decisions in real environments, and they classify them into video generation-based models, 3D reconstruction-enhanced models, and latent world models [17]. A repeated architectural distinction is whether the world model is separate from the policy or whether the policy itself is built around the model as the planner [17][22].

Navigation and web-agent examples show how world models can be coupled with memory. WMNav predicts possible outcomes, maintains a curiosity value map, and feeds that memory back into a policy module for object-goal navigation [18]. Navigation World Models simulate trajectories and rank them against goals, while WebDreamer uses LLMs as world models to simulate action outcomes in web environments [19][29]. These systems emphasize that world models are not only predictors; they can also act as structured memory and deliberation substrates [18][29][20].

The notes also show a move toward richer embodied stacks. HoloAgent-0 couples language-to-skill translation, 3D spatial memory, and closed-loop execution, while TANGO uses LLMs to compose navigation and memory-based policies [20][21]. The common pattern is iterative: imagine a candidate future, compare it with the goal, update memory, and re-plan [17][18][20][22].

## Evaluation and benchmarks
Benchmarking has become one of the most active parts of the field because visual quality alone does not tell us whether a model is a useful world model [23][24][26][27]. A decision-making-centric position argues that evaluation should progress from visual plausibility to interventional tests such as counterfactual action fidelity, closed-loop rollout validity, reward or value prediction, policy-ranking agreement, and actual optimization lift [23]. This is a meaningful shift because a model can produce realistic-looking futures while still being unusable for control [23][26].

Several recent benchmarks reflect this broader view. WorldModelBench evaluates video generation models as world models using instruction-following and physics-adherence criteria, with a learned judger trained on human labels [24][32]. Text2World evaluates symbolic world model generation with execution-based metrics over many PDDL domains [25]. WorldBench isolates physics concepts so that one test targets one law or constant at a time, reducing entanglement in evaluation [26]. Together, these benchmarks show a move toward task-grounded and concept-specific measurement rather than generic video quality [23][24][25][26].

## Trends and open problems
The notes converge on several open problems. First, long-horizon stability remains difficult: both latent and video world models can drift or lose action fidelity over extended rollouts [7][9][15][16][27]. Second, uncertainty calibration is still underdeveloped, even though planning systems need to know when their imagined futures are unreliable [23][27]. Third, compositional reuse and continual learning remain open, with benchmark results suggesting that modularity helps but does not fully solve forgetting or reuse across task combinations [28].

A broader challenge is that the field still lacks a stable universal definition of what counts as a world model [3][16][27]. Some papers stress prediction and action sensitivity, others emphasize internal understanding, and still others focus on video realism or symbolic state generation [3][16][23][25]. This diversity is not merely terminological: it means different model families should be judged by different criteria, and a single leaderboard would be misleading [23][27].

The most plausible near-term direction is therefore pluralistic. Latent dynamics methods are likely to remain strong where efficient control matters [5][6][7][8][9]. Video and action-conditioned generators will matter where interpretability, robotics, and multimodal supervision matter [10][11][14][15][16]. Benchmarks will need to keep moving toward decision-useful, physics-aware, and uncertainty-aware tests if the field wants to know whether a world model is useful beyond producing attractive predictions [23][24][26][27].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-05-09)
[2] World Models. web. https://worldmodels.github.io/ (2018-03-27)
[3] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://arxiv.org/html/2411.14499v2 (n.d.)
[4] Survey of predictive state representations. web. https://exa.ai/library/publication/bv6ds61hnnm (2010-01-01)
[5] Learning Behaviors by Latent Imagination. web. https://arxiv.org/pdf/1912.01603 (n.d.)
[6] Dreamer. web. https://arxiv.org/pdf/2301.04104 (n.d.)
[7] Sparse Imagination for Efficient Visual World Model Planning. hf-search. https://huggingface.co/papers/2506.01392 (2025-06-02)
[8] RoboJEPA: Scaling Robotic Latent World Models. hf-search. https://huggingface.co/papers/2610.10515 (2026-10-07)
[9] Dreaming Smoothly and Sample Efficiently with Gradient Penalized Latent Dynamics. hf-search. https://huggingface.co/papers/2605.23089 (2026-05-21)
[10] Towards General World Model with Natural Language Actions and Video States. web. https://arxiv.org/html/2406.09455 (2024-06-12)
[11] Pre-Trained Video Generative Models as World Simulators. web. https://ojs.aaai.org/index.php/AAAI/article/view/42465/46426 (n.d.)
[12] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[13] World Action Models: A Survey. web. https://arxiv.org/html/2606.20781 (2026-06-18)
[14] Learning Universal Policies via Text-Guided Video Generation. web. https://proceedings.neurips.cc/paper_files/paper/2023/file/1d5b9233ad716a43be5c0d3023cb82d0-Paper-Conference.pdf (n.d.)
[15] Adapting Video Diffusion Models to World Models - AVID - arXiv. web. https://arxiv.org/abs/2410.12822 (2024-11-24)
[16] World Model for Robot Learning: A Comprehensive Survey. web. https://arxiv.org/html/2605.00080 (n.d.)
[17] A Survey of Embodied World Models. web. https://fi.ee.tsinghua.edu.cn/public/publications/0940dda4-af15-11f0-9d60-0242ac120002.pdf (n.d.)
[18] WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation. arxiv. https://arxiv.org/abs/2503.02247 (n.d.)
[19] Navigation World Models. web. https://arxiv.org/pdf/2412.03572v2.pdf (n.d.)
[20] HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory. hf-search. https://huggingface.co/papers/2606.23565 (2026-06-22)
[21] TANGO: Training-free Embodied AI Agents for Open-world Tasks. hf-search. https://huggingface.co/papers/2412.10402 (2024-12-05)
[22] A Comprehensive Survey on World Models for Embodied AI. web. https://arxiv.org/html/2510.16732v3 (n.d.)
[23] How Should World Models Be Evaluated? A Decision-Making-Centric Position. web. https://arxiv.org/html/2606.15032v1 (n.d.)
[24] WorldModelBench: Judging Video Generation Models As World Models. hf-search. https://huggingface.co/papers/2502.20694 (2025-02-28)
[25] Text2World: Benchmarking Large Language Models for Symbolic World Model Generation. hf-search. https://huggingface.co/papers/2502.13092 (2025-02-18)
[26] WorldBench: Benchmarking Physical Understanding of World Models by Isolating Physics Concepts. web. https://world-bench.github.io/ (n.d.)
[27] State of World Models 2026: Taxonomy, Benchmarks and Open Challenges. web. https://world-models.io/reports/state-of-world-models-2026/state-of-world-models-2026-v1.0.pdf (n.d.)
[28] Benchmarking World Models for Continual Learning on Compositional Tasks. web. https://arxiv.org/html/2609.22055v1 (n.d.)
[29] Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents. hf-search. https://huggingface.co/papers/2411.06559 (2024-11-10)
[30] LeCun's big bet for building intelligent machines. web. https://irving-piano.technologyreview.com/2022/06/24/1054817/yann-lecun-bold-new-vision-future-ai-deep-learning-meta/ (2022-06-24)
[31] UniPi: Learning universal policies via text-guided video generation. web. https://research.google/blog/unipi-learning-universal-policies-via-text-guided-video-generation/ (2023-04-12)
[32] WorldModelBench: Judging Video Generation Models As World Models. web. https://worldmodelbench-team.github.io/ (n.d.)
