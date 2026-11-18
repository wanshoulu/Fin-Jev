# 用 Jev 实现 LLM 语义路由：调研与设计方案

> 版本 v1.0 · 2026-11-18 · 状态：调研完成，待评审实施
>
> 目标：将 Jev（System One 判定模型）从现有的意图识别用途，扩展为**语义路由层**——根据用户输入自动分类任务（简单问答 / 深度推理 / AI coding 等），并映射到不同档位的模型（小模型 / 大模型 / 专用模型），在保证质量的前提下降低成本与延迟。

---

## 1. 结论速览

1. **可行，且是 Jev 的官方推荐用法**。TypeSafe 官方将"路由到正确的模型"列为核心用例，LangChain 已内置 `ModelRouterMiddleware`，社区已有 68+ 个路由类项目（含 597★ 的 Claude Code 路由器）。
2. **推荐形态**：一个 `choice` 问题（最多 255 个候选类）一次往返完成分类，返回每类概率；无需训练、无需 embedding 库，用自然语言 criteria 定义类别即可扩展。
3. **成本与延迟**：Jev 输入 $0.042–0.42/1M tokens（输出免费），实测单请求 ~0.2s / ~131 tokens（6 题并行）。对生成式调用而言路由开销可忽略（<1% token 增量）。
4. **决策策略**：argmax + 双阈值（高置信直选 / 中置信保守 / 低置信升级 System Two 小 LLM 复判）+ 路由失败 fallback 默认模型。参考 OpenRouter auto 的降级设计。
5. **预期收益**：级联/路由类方案的公开数据为成本降 40–85%（质量不降）；我们目标设定为综合成本降 40–70%，先评测后上线。

---

## 2. 问题定义：什么是语义路由

语义路由（semantic routing / model routing）是在请求进入生成式 LLM 之前的**决策层**：读取请求语义 → 判定任务类型与难度 → 把请求分派到"够用且最便宜"的模型。与网关（只管转发/限流/密钥）不同，路由器的核心是**每请求的模型选择决策**。

动机数据（公开基准）：

| 数据点 | 数值 | 来源 |
|---|---|---|
| RouterBench 数据规模 | 405k 推理结果，覆盖常识/数学/代码/RAG 等 | [1] |
| 顶级路由方法 vs 最优单模型 | 同质量下成本降至 31.7%，或准确率提升至 +4% | [2] |
| RouteLLM（LMSYS） | MT Bench 成本降 85%+、MMLU 45%、GSM8K 35% | [3] |
| FrugalGPT（级联） | GPT-4 级质量、成本降至 98% | [4] |
| 关键 caveat | 并非所有路由器都能跑赢"最优单模型"，需按场景评测 | [2] |

三类主流路由架构：

1. **前置分类路由**（predictive routing）：先判任务，再选模型。RouteLLM 的 BERT/MF 路由器、OpenRouter auto、NotDiamond 属此类。
2. **级联路由**（cascading）：先调便宜模型，用判别器判断是否接受，不行再升级。FrugalGPT 属此类。缺点是"便宜模型先跑"带来尾部延迟。
3. **规则/向量路由**（static/encoder routing）：按关键词或 embedding 相似度匹配。aurelio semantic-router 属此类。快但语义泛化弱。

Jev 属第 1 类的变体：**System One 判定模型**——不生成文本，单次往返对一组问题返回类型化、已校准的概率值。

---

## 3. 现有方案对比

| 方案 | 机制 | 需要训练 | 分类能力 | 延迟 | 成本 | 适用性 |
|---|---|---|---|---|---|---|
| RouteLLM（LMSYS，开源） | Chatbot Arena 偏好数据训练 4 种路由器（相似度加权 Elo / 矩阵分解 / BERT / causal LLM） | 是，需偏好数据 | 二元（强/弱模型） | <50ms | 本地推理，近零 | 有 ML 训练链路、双模型场景 |
| aurelio semantic-router（开源） | encoder + utterance 向量库相似度匹配，带置信度 | 否，但需写示例 utterance | 任意 route 数 | 10–50ms | embedding 调用 | 边界清晰的高频分类 |
| OpenRouter auto（商用） | 快速决策模型单次调用评 11 个维度（任务类型/难度/速度要求等）→ 映射 cost_tier；聚合全网用户行为 | 否（官方托管） | 任意模型池 | +一次模型调用 | 计入用量 | 直接用 OpenRouter 的团队 |
| NotDiamond（商用） | `/v2/modelRouter/modelSelect`，按复杂度/能力/成本/延迟偏好路由，可训练自定义路由 | 可选 | 任意模型池 | +一次模型调用 | SaaS 订阅 | 多模型池商用场景 |
| 级联（FrugalGPT 类） | 便宜模型先行 + 正确性判别器决定升级 | 需训练判别器 | 任意（逐级升级） | 尾部延迟高 | 最省 | 质量可验证的场景 |
| **Jev（本方案）** | System One 判定模型：choice/noul/score 三原语，单次往返返回校准概率 | 否，自然语言 criteria | **单题最多 255 类**，多题并行 | **实测 ~0.2s**（官方 70–500ms） | $0.042–0.42/1M 输入 tokens，输出免费 | 不想维护训练链路、类别自然语言可描述 |

**Jev 的差异化优势**：

- **零训练**：RouteLLM 需要偏好数据与训练管线；semantic-router 需要为每类手写几十条 utterance；Jev 只需一段 criteria 描述。
- **校准概率**：返回的每类概率已校准（官方声明），可直接做阈值决策；embedding 相似度分数通常未校准。
- **单请求多题并行**：所有问题共享同一份 state、独立评估、一次往返（博查/火山文档均确认）。
- **类型安全**：构造上不会输出格式错误（choice 返回 key + 每选项概率，noul 返回 0–1 概率）。
- **官方生态背书**：LangChain `ModelRouterMiddleware`（experimental）、Vercel AI Gateway / OpenRouter / Cloudflare 网关均接入，社区路由器见 §5。

**劣势与约束**：

- 依赖第三方 API（网络往返、可用性），比本地 BERT 路由慢 1–2 个数量级。
- 分类边界靠自然语言描述，模糊边界的稳定性需实测校准。
- 博查版本处于早期，字段校验严格（多传字段会被拒，PoC 已验证）。

---

## 4. Jev API 能力评估（实测 + 文档）

### 4.1 三原语

| 原语 | 返回 | 路由用途 |
|---|---|---|
| `noul` | 校准的 0–1 是/否概率；接近 0.5 表示模型不确定 | 二元闸门（如"是否需要深度推理""是否含代码"） |
| `choice` | 胜出 key + 每选项概率 + 置信度；**每题 1–255 个候选选项** | 主分类器（任务类型一次判完） |
| `score` | 2–10 级有序分数 + 完整分布 | 难度/复杂度打分（如 1–5 复杂度分档） |

### 4.2 关键规格（火山方舟 System One 文档，博查版本行为一致）

- 上下文：128K（state + questions 共用）；state 支持文本、JSON 对象、多模态（图/视频/音频）。
- questions：至少 1 题，各题独立评估、响应顺序不保证（按问题 ID 取结果）。
- choice 的 criteria 支持 **1–255 个选项** → 一次请求即可承载完整分类体系，无需拆多请求。
- 计费按输入 tokens：state 计一次，每个问题的 instructions/criteria 各自计入；输出免费。

### 4.3 内部实测（Fin-Jev PoC 环境）

| 指标 | 实测值 |
|---|---|
| 单请求（6 个 noul 并行）延迟 | ~0.2s |
| 输入 tokens | ~131 |
| 多传字段 | 被拒（extra_forbidden），body 必须严格 `model`/`state`/`questions` 三键 |
| 输出格式 | `{answers: {qid: {noul: 0.99}}}`，稳定 |

### 4.4 生态先例（证明该模式已跑通）

| 项目 | 做法 | 热度 |
|---|---|---|
| `gargpratyush/jev-router` | 把每个 Claude Code 任务路由到"最便宜能胜任的模型" | 597★ |
| `pi-jev-model-router` | Jev 回答 4 个窄问题 → 组合成能力档位 + 预算策略 → 切换模型（规划→推理模型、实现→coding 专用、闲聊→便宜快模型） | pi 官方包 |
| `BillionsBobby/JevRouter` | 一个 choice 问题选 model / tool / subagent | 470★ |
| LangChain 官方 | `ModelRouterMiddleware`：`ModelChoice(fast/powerful)` 各带 criteria，Jev 选档 | 官方 experimental |
| madewithjev.com "Routing and models" 类目 | 68 个路由相关 builds | 社区 |

结论：**"Jev 打分 → 代码分档 → 切模型"是被广泛验证的成熟模式**，我们要做的是把它工程化到 Fin-Jev / Jev-Agent 的链路里。

---

## 5. 推荐方案设计

### 5.1 任务分类体系（v1）

主判用**一个 choice 问题**，v1 五类（可平滑扩到 255 上限内）：

| 类别 key | 判定标准（criteria 描述） | 映射模型档位 |
|---|---|---|
| `chat_simple` | 日常闲聊、单点事实问答、格式改写、短文翻译等单步任务 | 小模型（如 glm-5.3-flash / 同档 7B 级） |
| `reasoning_deep` | 多步推理、数学、因果分析、长链规划、需要论证的任务 | 大模型（如 glm-5.3 / 同档旗舰） |
| `ai_coding` | 写代码、改代码、调试、代码审查、技术实现讨论 | coding 专用模型 |
| `long_context` | 超长文档理解、整仓分析、大量上下文引用 | 长上下文模型 |
| `default` | 以上都不确定/混合 | 保守默认档（中档模型） |

辅助 noul 问题（可选、与主判同请求并行）：

- `needs_deep_reasoning`：是/否，用于 `chat_simple` 与 `reasoning_deep` 边界的二次确认；
- `complexity`（score 1–5）：为未来按难度细分子档位预留数据。

> 设计原则：**choice 为主判（一次定类），noul/score 为旁路信号**。旁路信号先记录不参与决策，积累数据后再决定是否用于二次仲裁——避免一开始就把规则写死。

### 5.2 决策策略（置信分层 + 兜底）

```
Jev 返回 {category_key, prob_max, probs}
│
├─ prob_max ≥ τ_high (0.8)     → 直选该类模型（90%+ 流量）
├─ τ_low (0.55) ≤ prob < τ_high → 保守策略：升一档选模型（宁可选大不选小）
└─ prob_max < τ_low             → 升级 System Two：小 LLM 复判一次
                                   ├─ 复判结果 → 采用
                                   └─ 复判失败/超时 → default 档模型
异常（Jev 超时/报错）            → default 档模型（永不因路由失败而失败）
```

与 OpenRouter auto 的降级哲学一致：**分类不可用时 fallback 默认模型，请求永不失败**。阈值 τ 通过评测集校准（见 §6）。

### 5.3 服务端集成架构

两条路径，推荐 A：

**方案 A（推荐）：扩展 jev_proxy.py 为路由网关**

```
客户端 → jev_proxy.py (8787)
           ├─ POST /v1/router/chat    ← 新增：路由入口
           │    1. 读 messages（OpenAI 兼容格式）
           │    2. 拼路由 state（取最后 1–2 条用户消息，截断至 ~2K tokens）
           │    3. 调 Jev systemone → 决策 → 选模型
           │    4. 转发至 llm-router(8117)/GLM 通道或直连目标模型 API
           │    5. 响应头回传 X-Route-{Category,Model,Prob,JevMs} 便于观测
           ├─ 现有 /v1/systemone 同源代理（保持不变）
           └─ 现有静态页/探针服务（保持不变）
```

- 单进程、零新增部署面，复用现有 systemd 服务与密钥管理（`jev_key.local.json`）。
- `routes.json` 配置化：类别 ↔ 模型 ↔ 阈值，改配置不重启（或重启成本低）。

**方案 B：独立 router 服务**——与 jev_proxy 解耦，适合后续多前端共用；初期不必要。

模型通道说明：本机 llm-router（8117，GLM 系）作为 chat 模型池；服务器侧无 GLM key，走直连各模型 API 或后续统一网关。

### 5.4 关键实现细节

1. **路由 state 构造**：只送任务语义所需的最近用户消息（截断 ~2K tokens），不送全对话历史——省 tokens 且避免历史偏置分类。
2. **多轮上下文**：对话中途的追问（"再优化一下"）语义不足时，附带最近一轮对话摘要进 state。
3. **缓存**：相似前缀/相同 query 的路由结果 LRU 缓存（命中即免 Jev 调用）。
4. **观测**：每次路由记录 `{category, probs, chosen_model, prob_max, jev_ms, total_ms}`，落 JSONL，供评测与阈值校准。
5. **配置热载**：`routes.json` 含分类体系、模型映射、阈值、fallback 链，agent 侧可调。

---

## 6. 评测方案

参照 RouterBench 方法论（405k 推理结果、成本-质量前沿），但规模务实化：

| 项 | 内容 |
|---|---|
| 标注集 | 300–500 条真实 query，按 5 类人工标注（含 50 条混合/模糊样本专测边界） |
| 分类准确率 | Jev choice vs 人工标注，目标 ≥90%；混合类单独报告 |
| 阈值校准 | 在标注集上扫 τ_high/τ_low，画（成本节省 × 质量保持率）曲线取 Pareto 点 |
| 路由开销 | p50/p95 路由延迟增量，目标 p95 < 500ms（实测 ~200ms 起步） |
| 成本收益 | 路由后混合成本 vs 全大模型基线，目标降 40–70% |
| 质量对比 | 抽样路由任务 vs 全大模型回答，人工盲评（RouterBench 的 cost-matched 思路简化版） |
| 回归机制 | 每周用线上 JSONL 记录抽检，概率分布漂移（如某类 prob_max 均值下降 >0.1）触发 criteria 重校 |

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 混合意图 query（一条既问代码又要求推理） | 分类抖动 | criteria 中明确"复合任务归入更重的类"；评测集专测；必要时主判 + 双 noul 仲裁 |
| 概率校准漂移（模型/提示词更新后） | 阈值失效 | 每周回归抽检；漂移超阈值重校 τ 或重写 criteria |
| Jev API 单点（网络/服务不可用） | 路由失败 | fallback default 档；路由结果缓存；代理层重试一次 |
| 类别数增长 → criteria 变长 → tokens 上涨 | 成本上升 | 255 类上限远超需求；每类描述控制在 ~50 tokens；state 截断 |
| 博查版本早期，行为可能与火山版有差 | 迁移成本 | 关键调用封装在 provider 层，博查/火山可切换 |
| 分类延迟叠加在首字延迟上 | 体感变慢 | ~200ms 可接受；缓存 + 复用连接；后续可对流式请求并行预路由 |

---

## 8. 实施路线图

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 PoC | jev_proxy.py 加 `/v1/router/chat`：choice 主判 + 阈值决策 + fallback；routes.json | 手工跑通 5 类路由，curl 观测头正确 |
| P1 评测 | 300+ 条标注集；分类准确率、阈值曲线、延迟、成本报告 | 准确率 ≥90%，成本节省 ≥40% |
| P2 上线 | duel_chat / 生产前端接入；观测 JSONL；每周回归 | 线上无路由故障，成本下降可复算 |
| P3 迭代 | score 难度分档、多通道（coding 专用模型接入）、阈值自动化校准 | 细分档位带来增量收益 |

---

## 参考资料

1. RouterBench: A Benchmark for Multi-LLM Routing System — https://arxiv.org/abs/2403.12031
2. LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing — https://arxiv.org/html/2601.07206
3. RouteLLM: Learning to Route LLMs with Preference Data (ICLR 2025) 及 LMSYS 博客 — https://www.lmsys.org/blog/2024-07-01-routellm/ · https://proceedings.iclr.cc/paper_files/paper/2025/file/5503a7c69d48a2f86fc00b3dc09de686-Paper-Conference.pdf
4. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance — https://arxiv.org/pdf/2305.05176
5. Aurelio AI Semantic Router 文档 — https://www.aurelio.ai/semantic-router · https://docs.aurelio.ai/docs/semantic-router/v0/user-guide/components/routers
6. OpenRouter Auto Router — https://openrouter.ai/docs/guides/routing/routers/auto-router · https://openrouter.ai/blog/announcements/introducing-the-new-auto-router/
7. NotDiamond modelSelect API — https://docs.notdiamond.ai/reference/token_model_select_v2_modelrouter_modelselect_post
8. Jev by TypeSafe AI（官方：三原语、70–500ms、定价、模型路由用例）— https://jevtypesafeai.com/zh
9. 火山方舟 System One API（beta）文档（128K、choice 1–255 选项、score 1–10 档、多模态 state）— https://docs.volcengine.com/docs/ark/system-one-api
10. eu/jev（EU 托管版，€37/B input tokens）— https://jev.bevel.software/docs
11. Abacus.ai SystemOne API（$0.042/1M input）— https://abacus.ai/help/route-llm/systemone
12. 生态路由器：gargpratyush/jev-router、pi-jev-model-router、BillionsBobby/JevRouter — https://madewithjev.com/categories/routing-and-models · https://pi.dev/packages/pi-jev-model-router
13. LangChain Jev 路由中间件 — https://www.langchain.com/blog/building-a-harness-with-jev
14. 级联路由综述（cascade 结构与判别器变体）— https://arxiv.org/html/2606.13241v1

> 注：标注 [4][5] 等定价与规格以官方文档为准；内部实测数据来自本仓库 PoC 环境（博查 jev.bocha.cn，2026-11），随环境变化需重测。
