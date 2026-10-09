# Survey of Reinforcement Learning for LLM Reasoning
## TL;DR
- RL has become a core recipe for turning LLMs into large reasoning models, with recent milestones showing that verifiable rewards can induce planning, reflection, and self-correction [1][2][3].
- In RLHF-style pipelines, reward models plus PPO remain important, but direct preference optimization and related methods aim to simplify training while process reward models move supervision from final answers to intermediate steps [4][5][6][7][8].
- Verifiable-reward RL is the most convincing family for reasoning tasks with checkable answers: it uses deterministic evaluators for math, code, logic, or theorem proving and reports gains on reasoning benchmarks and transfer settings [9][10][11][12].
- For multi-step agentic behavior, RL is increasingly used for planning, tool use, search, and long-horizon interaction, often with reward designs that make tool selection and search policies more robust [13][14][15][16][17][18][19][20].
- The field is still moving fast, but benchmark volatility, contamination, overthinking, safety risks, and unclear scaling laws remain major open problems [21][22][23][24][25][26][27][28][29].

## Background
Reinforcement learning for LLM reasoning refers to training language models with rewards that favor better multi-step problem solving rather than only next-token prediction or static imitation. In the recent reasoning literature, this often means optimizing models so they can decompose problems, verify intermediate steps, backtrack, and improve answer quality under verifiable or preference-based feedback [1][2][3][30].

This became especially important once frontier models began showing that pure RL or RL-style post-training can unlock behaviors that supervised fine-tuning alone does not reliably produce, including self-reflection, searching alternative solutions, and longer internal reasoning traces [2][3]. Surveys now describe RL as a foundational methodology for large reasoning models, while also noting that the field still lacks a settled answer on how much of the gain comes from RL itself versus data, priors, and training recipes [1][23][24].

A useful way to organize the area is by the kind of feedback signal. Some systems learn from human or model preferences, some from process-level supervision, and some from verifiable task outcomes such as correct math answers, passing unit tests, or proof checker validation [30][7][8][9][11]. These families overlap, but they differ in where they place the credit assignment signal and how directly they connect rewards to actual reasoning quality.

## RLHF, preference optimization, and process rewards
A first major family is RLHF-style optimization, where a reward model scores candidate responses and a policy optimizer such as PPO updates the LLM. The HF survey on PPO emphasizes that successful RLHF depends on an accurate reward model, careful hyperparameter tuning, and a stable policy-constraint strategy [4]. This family is still central when the target behavior is subjective or preference-shaped, but it is less direct for reasoning than for style or helpfulness.

Recent work tries to reduce RLHF complexity. DPO replaces explicit RL with a direct preference objective, arguing that a language model can be optimized from pairwise preferences without training a separate reward model first [5]. For reasoning tasks, follow-up work such as EPO argues that pairwise preferences may be too noisy and instead uses groups of samples to estimate preferences more reliably, reporting better zero-shot reasoning accuracy across benchmarks [6]. The main trend here is to preserve preference optimization while making it more suitable for structured reasoning targets.

Process reward models represent a different shift. Instead of judging only the final answer, they score steps or trajectories, which improves credit assignment and makes debugging easier [7][8]. Survey material on PRMs explicitly frames them as useful for test-time scaling and PRM-guided RL, and as a remedy for the weaknesses of outcome-only reward models [7][8]. For reasoning, this matters because many failures happen in the middle of a solution, not only at the end.

## Verifiable-reward RL and outcome-based reasoning training
The strongest empirical story for reasoning is verifiable-reward RL, often abbreviated RLVR. In this setup, the model receives reward from deterministic checks rather than human preference labels, so the signal can come from answer correctness, executable code tests, proof validation, or other automatic verifiers [30][9][11]. This is attractive because it avoids a learned reward model and aligns optimization with task-level correctness.

The RLVR paper argues that using answer correctness can still incentivize correct reasoning, not just lucky outputs, and reports that it extends the reasoning boundary for math and coding tasks [9]. It also introduces CoT-Pass@K to require both a correct answer and correct intermediate reasoning, highlighting that outcome success alone can hide flawed reasoning [9]. Related work on REASONING GYM broadens the idea by generating many automatically verifiable environments across algebra, logic, graph theory, and games, and reports transfer gains on benchmarks such as MATH, GSM8K, Big-Bench Hard, and MMLU-Pro [10].

DeepSeek-R1 is a major milestone in this family because it shows that pure RL on pre-trained models can elicit reasoning behaviors with only correctness-based rewards, including self-reflection and exploring alternative approaches [2]. OpenAI’s o1 page similarly describes reinforcement learning that teaches models to think before answering, refine their chain of thought, and spend more time reasoning at test time [3]. In theorem proving, process-verified RL through Lean shows that fine-grained tactic-level rewards can outperform outcome-only baselines in most settings and improve benchmarks such as MiniF2F and ProofNet [11]. JudgeRLVR extends the same logic by training a judge first and then generating with RLVR, reporting accuracy gains and shorter generations [12].

## Agentic RL for planning, tools, and search
A third family treats RL as a control problem over multi-step agent behavior. Surveys describe this as moving from single-step MDPs to temporally extended agentic settings, where the model must plan, call tools, use memory, and self-improve over longer horizons [13][14]. In this framing, RL is not just for answering questions but for learning when to search, when to invoke tools, and how to coordinate across multiple reasoning turns.

ToolRL argues that supervised fine-tuning struggles to generalize to unfamiliar tool-use scenarios, and that carefully designed rewards for tool selection and application can improve both generalization and metacognitive behavior [15]. Search-R1 similarly shows that prompting alone is often suboptimal for using search engines; it trains models to generate multiple search queries during step-by-step reasoning and reports improvements on seven QA datasets over retrieval baselines [16]. The common pattern is to reward the process of interaction, not only the final response.

Recent agentic work in Hugging Face papers pushes the same idea into more complex settings. HYDRA combines a planner, an RL agent, and a reasoner for compositional visual reasoning [20]. Other recent daily papers describe closed-loop planning for mobile agents, process-supervised interactive multimodal tool use, and multi-small-agent reinforcement learning that decouples decomposition from tool use [17][18][19]. Together, these works suggest that RL for reasoning is increasingly becoming RL for action-conditioned reasoning in real environments.

## Trends and open problems
The biggest recent trend is toward stronger evaluation and more realistic benchmarks. LiveBench is built to reduce contamination by releasing new questions regularly and using objective answers, while MR-BEN focuses on meta-reasoning by asking models to detect mistakes in their own steps [25][28]. At the same time, benchmark papers on mathematical reasoning warn that short-answer tests can be saturated, contaminated, or overly sensitive to prompts and decoding settings [22][29].

Scaling work suggests that RL performance often follows sigmoidal compute curves, meaning that much of the engineering effort improves compute efficiency more than the final ceiling [21]. But the same literature also warns that gains can be unstable: models may overfit small benchmarks, show volatility across seeds, or exhibit underthinking and overthinking [22][23][24]. In other words, progress exists, but it is not yet uniformly robust.

Safety and reliability are another open frontier. Surveys of large reasoning models highlight risks from chained execution, retrieval and agentic attack surfaces, and backdoor-style vulnerabilities in reasoning traces [23][26]. HF daily work also shows that recent models can regress into recitation under subtle changes, underscoring how fragile reasoning behavior can be [27].

The open research agenda is therefore fairly clear. The field still needs better credit assignment, better process supervision, stronger contamination-resistant evaluation, and a clearer theory of when RL improves genuine reasoning rather than benchmark-specific behavior [1][7][8][22][23][24]. The most promising direction is probably not a single algorithm, but a combination of verifiable rewards, process rewards, and agentic training setups that can be evaluated on live, difficult, and safety-aware tasks [9][10][11][13][15][16][25][26].

## References
[1] A Survey of Reinforcement Learning for Large Reasoning Models. arxiv. https://arxiv.org/abs/2509.08827 (n.d.)
[2] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. web. https://arxiv.org/abs/2501.12948 (2025-01-20)
[3] Learning to reason with LLMs - OpenAI. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[4] Secrets of RLHF in Large Language Models Part I: PPO. hf-search. https://huggingface.co/papers/2307.04964 (2023-07-11)
[5] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arxiv. https://arxiv.org/abs/2305.18290 (2024-07-29)
[6] Expectation Preference Optimization: Reliable Preference Estimation for Improving the Reasoning .... web. https://aclanthology.org/2025.emnlp-main.1532/ (n.d.)
[7] A Comprehensive Survey of Process Reward Models: Data Generation, Model Construction, and Usage. web. https://aclanthology.org/2026.acl-long.163/ (n.d.)
[8] A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models. web. https://aclanthology.org/2026.acl-long.163.pdf (n.d.)
[9] Reinforcement Learning with Verifiable Rewards. arxiv. https://arxiv.org/abs/2506.14245 (n.d.)
[10] REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards. arxiv. https://arxiv.org/abs/2505.24760 (n.d.)
[11] Process-Verified Reinforcement Learning for Theorem Proving via Lean. web. https://neurips.cc/virtual/2025/131058 (n.d.)
[12] JudgeRLVR: Judge First, Generate Second for Efficient .... web. https://arxiv.org/abs/2601.08468 (n.d.)
[13] The Landscape of Agentic Reinforcement Learning for LLMs. arxiv. https://arxiv.org/abs/2509.02547 (n.d.)
[14] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. arxiv. https://arxiv.org/abs/2406.05804 (n.d.)
[15] ToolRL: Reward is All Tool Learning Needs. arxiv. https://arxiv.org/abs/2504.13958 (2025-04-16)
[16] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. arxiv. https://arxiv.org/abs/2503.09516 (2025-08-05)
[17] Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents. hf-daily. https://huggingface.co/papers/2609.29892 (2026-09-24)
[18] Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents. hf-daily. https://huggingface.co/papers/2509.14480 (2025-09-17)
[19] MSARL: Decoupling Reasoning and Tool Use with Multi-Small-Agent Reinforcement Learning. hf-daily. https://huggingface.co/papers/2508.08882 (2025-08-12)
[20] HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning. hf-search. https://huggingface.co/papers/2403.12884 (2024-03-19)
[21] The Art of Scaling Reinforcement Learning Compute for LLMs. web. https://arxiv.org/abs/2510.13786 (n.d.)
[22] A Sober Look at Progress in Language Model Reasoning. web. https://arxiv.org/abs/2504.07086 (2025-04-09)
[23] Reasoning Beyond Limits: Advances and Open Problems for LLMs. web. https://arxiv.org/abs/2503.22732 (2025-03-26)
[24] Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle. web. https://arxiv.org/abs/2509.16679 (2025-09-20)
[25] LiveBench. web. https://livebench.github.io/ (n.d.)
[26] Safety in Large Reasoning Models: A Survey. web. https://aclanthology.org/2025.findings-emnlp.185/ (n.d.)
[27] Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems?. hf-daily. https://huggingface.co/papers/2504.00509 (2025-04-01)
[28] MR-BEN: A Comprehensive Meta-Reasoning Benchmark for Large Language Models. hf-search. https://huggingface.co/papers/2406.13975 (2024-06-20)
[29] Benchmarking LLMs on Advanced Mathematical Reasoning. web. https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-121.pdf (n.d.)
[30] Reinforcement Learning with Verifiable Rewards (RLVR) & The Reasoning Recipe — The LLM Stack. web. https://prakashkagitha.github.io/llm-stack-book/05-posttraining-alignment/09-rlvr-reasoning.html (n.d.)
