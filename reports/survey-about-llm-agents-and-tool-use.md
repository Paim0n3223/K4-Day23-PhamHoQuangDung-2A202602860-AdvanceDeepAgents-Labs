# LLM Agents and Tool Use: Methods, Systems, and Evaluation
## TL;DR
- LLM agents became important because plain text-only models struggle with up-to-date information, arithmetic, factual lookup, and real-world action; early systems responded by adding browsing, API calls, and environment interaction [1][2][3].
- A central design pattern is interleaving reasoning and action, as in ReAct, while other methods make planning more explicit through decomposition, reflection, memory, or structured prompts [4][5][6][7].
- Function calling and tool-use systems increasingly rely on schema-constrained outputs, tool selection policies, and training or retrieval over large API sets to reduce hallucinated or invalid calls [8][9][10][11][12].
- Frameworks for agentic tool use now span browser automation, desktop/GUI stacks, and software-engineering harnesses, reflecting the move from toy action spaces to practical multi-step workflows [13][14][15][16][17][18].
- Evaluation is shifting from single-turn call accuracy toward stateful, multi-turn, and safety-aware benchmarks, but stability, robustness, and long-horizon reasoning remain open problems [19][20][21][22][23][24][25][26].

## Background
LLM agents combine a language model with actions in external environments, so the system can plan, search, call tools, and observe results rather than only generate text. This matters because plain LLMs have fixed knowledge and cannot directly act on software or web services, which limits usefulness for dynamic tasks [1][2][3][27]. Early browser-assisted and tool-augmented systems showed that models can browse, retrieve, and collect evidence while producing answers, and that interleaving reasoning with action helps the model update plans and recover from mistakes [4][1].

The foundational pattern is therefore not just “better prompting,” but a change in the interface between model and world. WebGPT used a browsing environment plus human feedback to ground answers in sources [1]. Toolformer showed that models can learn when to call tools and how to incorporate results into next-token prediction [2]. Gorilla extended the idea to real APIs, emphasizing that correct argument formation and API choice are hard and that retrieval can reduce hallucination when documentation changes [3].

## Reasoning-Action Prompting
The most influential prompting-centric approach is ReAct, which interleaves thoughts, actions, and observations. Its strength is that reasoning can induce and update a plan while actions gather external information; this is especially helpful in tasks such as HotpotQA, FEVER, ALFWorld, and WebShop [4][1]. The tradeoff is that complex action spaces can require many demonstrations, and prompts can become long or brittle when too many steps are needed [4].

A related line of work makes the plan explicit before acting. Plan-and-Solve first constructs subtasks and then executes them, aiming to reduce missing-step errors in zero-shot chain-of-thought [6]. Surveys of planning for LLM agents break the problem into task decomposition, plan selection, external modules, reflection, and memory-augmented planning [5][28]. In that view, planning improves control and long-horizon performance, but adds workflow complexity and introduces another failure surface if plans or retrieved memories are misgrounded [5][28][7].

Recent variants push the reasoning loop toward more grounded state tracking. ReflAct argues that ordinary ReAct can generate incoherent reasoning steps, and replaces the core cycle with goal-state reflection; its reported gains over ReAct suggest that state awareness can matter as much as generic chain-of-thought [7]. STEP similarly treats agents as a Planner/Executor/Evaluator/Memory system, where memory stores experience for later retrieval and evaluation checks action quality [29].

## Function Calling and Action-Space Design
A different family of methods focuses on how tools are represented and selected. The function-calling guide frames tools as schema-defined interfaces, with controls such as auto, required, or forced tool choice, and recommends structured outputs to improve reliability [8]. This design makes tool invocation easier to parse and validate, but it also constrains the model to produce syntactically valid arguments and correct tool names [8][9].

Structured reasoning methods try to make that process more explicit. Guided-Structured Templates decompose tool use into identifying the tool, judging relevance, reading documentation, extracting parameters, drafting, and revalidating; the paper argues that free-form chain-of-thought is insufficient for structured calling tasks [9]. The broader survey of tool learning likewise separates task planning, tool selection, execution, and response generation, and notes that selection can be retriever-based or LLM-based [10].

Training and dataset design matter because tool-use quality depends on both selection and formatting. ToolLLM targets 16,000+ APIs, API Pack expands multilingual API-call generation, and Octopus focuses on on-device software APIs [10][11][12]. ToolChain* treats action-space navigation as a search problem and uses A* search to prune incorrect API calls, while CodeAct changes the action space itself by letting the agent emit executable Python code [30][31]. Together these methods show two main strategies: narrow the output format, or widen the action space but add search, retrieval, or training to keep it tractable [10][30][31].

## Frameworks and Environments
Frameworks and environments operationalize these ideas in real tasks. SWE-agent provides an agent-computer interface for software engineering tasks such as repository navigation, file editing, and test execution; it is evaluated on SWE-bench and HumanEvalFix [16]. OpenHands, Aider, and OpenDevin extend this direction with software-agent platforms, terminal workflows, and browser/shell/code-editor combinations, showing that code-centric agents often need a full workspace rather than just an API wrapper [32][33][34][35][18].

Browser-centric systems target website workflows directly. Browser Use exposes open-source browser automation for reading pages, clicking, typing, and filling forms, while Browser Harness emphasizes a self-healing CDP-based browser harness that can scale with many browsers in parallel [13][36][17]. Skyvern combines LLMs, computer vision, and Playwright-compatible automation with no-code workflows and multi-step task execution, which makes it more of a website-operations platform than a single prompt template [14].

Desktop and general computer-use systems broaden the environment further. UI-TARS Desktop and Agent TARS span terminal, browser, and product workflows, while Agent Zero and OpenComputer emphasize Dockerized desktops, delegated subtasks, and verifiable state over broad application coverage [37][38][15]. These systems suggest a common pattern: the harder the task, the more the agent needs a persistent environment, a verifier, and a way to recover from partial failures [38][15].

## Trends and open problems
Benchmarks are moving from static, single-turn tool calls toward multi-turn, stateful, and realistic interaction settings. BFCL evolved from simple function-calling tests to single-turn, crowd-sourced, multi-turn, and agentic evaluation, while AgentBench and StableToolBench highlight the need for diverse environments and stable execution when online APIs change [39][40][41][20][42][43][20]. The reported shift is from “did the model emit the right string” to “did the agent choose, sequence, and execute actions correctly over time” [19][39][41][42].

Safety has become a first-class benchmark dimension. ToolEmu uses an LM-emulated sandbox to test risky behavior at scale, SafeToolBench studies prospective tool-use security, R-Judge measures safety-risk awareness from interaction records, and Agent-SafetyBench reports broad safety weaknesses across many agents [21][22][23][24]. Recent HF daily papers on reward hacking and MCP attacks reinforce the idea that tool-using agents can exploit environments or protocol surfaces in ways not captured by ordinary success metrics [25][26].

The main open problems are therefore reliability, stability, and measurement. Tool benchmarks can break when APIs change, automatic evaluators may miss state changes or non-state-changing tool calls, and long-horizon reasoning remains weak even when single-turn function calling looks strong [19][39][41][20][42][20]. Future progress likely depends on better environment design, stronger verification, and evaluations that jointly measure usefulness, robustness, and safety under realistic tool use [19][20][21][22][23][24].

## References
[1] WebGPT: Browser-assisted question-answering with human feedback. arxiv. https://arxiv.org/abs/2112.09332 (2022-01-10)
[2] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[3] Gorilla: Large Language Model Connected with Massive APIs. arxiv. https://arxiv.org/abs/2305.15334 (2023-05-24)
[4] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[5] Understanding the planning of LLM agents: A survey - arXiv. arxiv. https://arxiv.org/abs/2402.02716 (2024-02-05)
[6] Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. hf-search. https://huggingface.co/papers/2305.04091 (2023-05-06)
[7] ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection. web. https://aclanthology.org/2025.emnlp-main.1697.pdf (n.d.)
[8] Function calling. web. https://developers.openai.com/api/docs/guides/function-calling (n.d.)
[9] Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates. web. https://aclanthology.org/2025.emnlp-main.1242.pdf (n.d.)
[10] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. hf-search. https://huggingface.co/papers/2307.16789 (2023-07-31)
[11] API Pack: A Massive Multilingual Dataset for API Call Generation. hf-search. https://huggingface.co/papers/2402.09615 (2024-02-14)
[12] Octopus: On-device language model for function calling of software APIs. hf-search. https://huggingface.co/papers/2404.01549 (2024-04-02)
[13] Browser Use. web. https://github.com/browser-use/browser-use/tree/main (2024-10-31)
[14] Skyvern-AI/skyvern. web. https://github.com/skyvern-ai/skyvern (2024-02-28)
[15] OpenComputer. web. https://github.com/echo0715/opencomputer (n.d.)
[16] SWE-agent: Agent-Computer Interfaces Enable Language Models to Autonomously Solve Software Engineering Tasks. arxiv. https://arxiv.org/abs/2405.15793 (2024-05-24)
[17] Browser Harness | Self-healing harness that enables LLMs to complete any task.. web. https://github.com/browser-use/browser-harness (n.d.)
[18] OpenDevin: Code Less, Make More - GitHub. web. https://github.com/AI-App/OpenDevin.OpenDevin (n.d.)
[19] Evaluation and Benchmarking of LLM Agents: A Survey. web. https://dl.acm.org/doi/10.1145/3711896.3736570 (2025-08-03)
[20] StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models. web. https://aclanthology.org/anthology-files/pdf/findings/2024.findings-acl.664.pdf (n.d.)
[21] ToolEmu: Identifying the Risks of Agents with an LM-Emulated Sandbox. web. https://proceedings.iclr.cc/paper_files/paper/2024/file/7274ed909a312d4d869cc328ad1c5f04-Paper-Conference.pdf (n.d.)
[22] Agent-SafetyBench: Evaluating the Safety of LLM Agents. web. https://arxiv.org/abs/2412.14470 (n.d.)
[23] SafeToolBench: Pioneering a Prospective Benchmark to Evaluating Tool Utilization Safety in LLMs. web. https://aclanthology.org/2025.findings-emnlp.958.pdf (n.d.)
[24] R-Judge: Benchmarking Safety Risk Awareness for LLM Agents. web. https://aclanthology.org/2024.findings-emnlp.79/ (n.d.)
[25] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use. hf-daily. https://huggingface.co/papers/2605.02964 (2026-05-03)
[26] MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents. hf-daily. https://huggingface.co/papers/2510.15994 (2025-10-14)
[27] A Survey on Evaluation of LLM-based Agents. web. https://aclanthology.org/2026.findings-acl.1330.pdf (2026-07-02)
[28] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. arxiv. https://arxiv.org/abs/2406.05804 (n.d.)
[29] STEP: Stepwise Planning for Language Agents. hf-search. https://huggingface.co/papers/2411.08432 (2024-11-11)
[30] Executable Code Actions Elicit Better LLM Agents. hf-search. https://huggingface.co/papers/2402.01030 (2024-02-01)
[31] ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search. hf-search. https://huggingface.co/papers/2310.13227 (2023-10-20)
[32] OpenHands | Open Source AI Coding Agent Platform. web. https://www.openhands.dev/ (n.d.)
[33] OpenHands Docs: Introduction. web. https://docs.openhands.dev/overview/introduction (n.d.)
[34] SWE-agent: Getting Started. web. https://swe-agent.com/latest/ (n.d.)
[35] Aider - AI Pair Programming in Your Terminal. web. https://aider.chat/ (n.d.)
[36] Browser Use Agents & Browser Infrastructure | Browser Use. web. https://browser-use.com/ (n.d.)
[37] bytedance/UI-TARS-desktop. web. https://github.com/bytedance/agent-tars (2025-01-19)
[38] agent0ai/agent-zero. web. https://github.com/agent0ai/agent-zero (n.d.)
[39] The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. hf-daily. https://huggingface.co/papers/2511.22659 (2025-11-27)
[40] The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. web. https://raw.githubusercontent.com/mlresearch/v267/main/assets/patil25a/patil25a.pdf (n.d.)
[41] BFCL V3 • Multi-Turn & Multi-Step Function Calling - Gorilla. web. https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html (n.d.)
[42] AgentBench: Evaluating LLMs as Agents. web. https://arxiv.org/abs/2308.03688 (2025-10-04)
[43] THUDM/AgentBench. web. https://github.com/thudm/agentbench (n.d.)
