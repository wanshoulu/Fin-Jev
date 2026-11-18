# 用 Jev 实现 LLM 语义路由：调研与设计方案

> 版本 v1.0 · 2026-11-18 · 状态：调研完成，待评审实施
>
> 目标：将 Jev（System One 判定模型）从现有的意图识别用途，扩展为**语义路由层**——根据用户输入自动分类任务（简单问答 / 深度推理 / AI coding 等），并映射到不同档位的模型（小模型 / 大模型 / 专用模型），在保证质量的前提下降低成本与延迟。

---

## 1. 结论速览

1. **可行，且是 Jev 的官方推荐用法**。TypeSafe 官方将"路由到正确的模型"列为核心用例，LangChain 已内置 `ModelRouterMiddleware`，社区已有 68+ 个路由类项目（含 597★ 的 Claude Code 路由器）。
2. **推荐形态**：一个 `choice` 问题（最多 255 个候选类）一次往返完成分类，返回每类概率；无需训练、无需 embedding 库，用自然语言 criteria 定义类别即可扩展。分类体系以**金融企业员工典型应用**为蓝本（§5.1，14 类 + 兜底），并叠加**监管高风险闸门**（§5.2，金监总局高风险应用定义）。
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

### 5.1 任务分类体系：金融企业员工典型应用

分类体系不从抽象的"任务类型"出发，而是以金融企业员工的**真实角色与日常应用**为蓝本。行业调研显示员工侧 AI 应用已高度分层（依据见右列/文末参考）：

- **全员办公**：制度问答、公文纪要、翻译润色——北京银行 AIB 平台 50 款运营效率工具 + 10 项创作助手 [17]；中小银行"OA 办公搭子"（席位订阅、OA 内嵌智能体）[27]；建行 AI 助手作为一线"超级导师"，网点问题响应覆盖率 99.42%、日均 10 万+ 次访问 [16]。
- **客户经营**：客户经理话术、客户画像、营销文案、普惠核查——建行"帮得"使核查报告 2–3 天缩至 3 小时 [18]；百融"硅基员工"覆盖营销报告/贷后监测 [19]。
- **投研投行**：财报分析、财务建模底稿（DCF/LBO/三表联动）、研报生成——券商研报场景 [20]、Anthropic 投行智能体（Earnings Reviewer/Model Builder）[21]、Goldman 财报电话会实时分析 [22]。
- **合规风控**：监管政策解读、合同审查、反洗钱 STR、可疑交易——反洗钱 GenAI 全流程（CDD/EDD、PEP 筛查、STR 叙述准备）[23]、AI 异常检测广泛用于 AML/CFT 与欺诈识别 [24]。
- **科技研发**：代码开发、PR 审查、监控根因分析——JPMorgan Claude Code 全员推广（超出工程师范围）[25]。
- **监管红线**：金监总局《银行业保险业人工智能安全开发应用指导意见》明确**高风险应用**范围——涉及资金交易、资产评估、信贷审批、承保理赔、风险管理及与客户利益直接相关的场景，须风险管理委员会批准并建立人工监督干预机制 [15]。**路由系统必须感知该维度**，这是与通用路由器的本质差异。

主判用**一个 choice 问题**承载下表 14 类（可平滑扩到 255 上限内）：

| 类别 key | 场景（criteria 要点） | 典型使用者 | 模型档位 | 监管风险 | 行业依据 |
|---|---|---|---|---|---|
| `office_chat` | 闲聊、单点事实问答、翻译、文本润色改写 | 全员 | 小模型 | 低 | [16][27] |
| `kb_qa` | 制度/产品/操作规程/内部知识库问答（RAG 检索为主） | 全员 | 小模型 + RAG | 低 | [16] |
| `office_writing` | 公文、邮件、通知、会议纪要、PPT 大纲等常规写作 | 全员 | 中模型 | 低 | [16][26] |
| `meeting_assist` | 会议转写整理、要点/待办提取（输入为长转写文本） | 全员 | 中模型（长上下文） | 低 | [26] |
| `data_query` | 取数、SQL 生成、报表解读、数据核对 | 运营/数据 | 中模型 | 低 | [19] |
| `customer_mgr` | 客户经理话术、客户画像、营销文案、拓客线索 | 客户经理 | 中模型 | 中（涉客户利益） | [18][19] |
| `doc_heavy` | 超长文档摘要/比对：合同包、招股书、流水、舆情包 | 前中后台 | 长上下文专用 | 低 | [19] |
| `reasoning_deep` | 复杂多步推理、数学、长链规划（不专属某业务域） | 全员 | 大模型（旗舰） | 低 | 用户显式诉求 |
| `research` | 投研分析、财报解读、财务建模底稿、业绩预测、研报草拟 | 研究员/投行 | 大模型（旗舰） | 中 | [20][21][22] |
| `credit_dd` | 对公信贷尽调、授信材料、信贷材料校验、普惠核查报告 | 对公/普惠客户经理 | 大模型（旗舰） | **高（信贷审批相关）** | [18][19] |
| `compliance_qa` | 监管政策解读、合规要求问答、监管报送口径 | 合规/全员 | 大模型 | 中 | [15] |
| `compliance_review` | 合同条款审查、合规一致性检查、制度冲突排查 | 法务/合规 | 大模型（旗舰） | **高** | [15] |
| `risk_aml` | 可疑交易分析、反洗钱 STR、欺诈识别、贷后风险监测 | 风控 | 大模型（旗舰） | **高（风险管理）** | [23][24] |
| `ai_coding` | 写代码、调试、代码审查、技术方案 | 科技部门 | coding 专用模型 | 低 | [25] |
| `default` | 以上均不确定或混合 | — | 中模型（保守档） | — | 兜底 |

辅助信号（与主判**同一请求并行**返回，不增加往返）：

| 信号 | 类型 | 决策用途 |
|---|---|---|
| `high_risk` | noul | 是否命中监管高风险定义（资金交易/资产评估/信贷审批/承保理赔/风险管理/客户利益直接相关）→ 触发强模型强制升档 + 人工复核标记 |
| `needs_long_context` | noul | 输入超长（>32K tokens 的摘要/比对任务）→ 切长上下文档位 |
| `complexity` | score 1–5 | 难度分布记录，为后续按难度细分子档位积累数据 |

> 设计原则：**choice 为主判（一次定类），noul/score 为旁路信号**。`high_risk` 与 `needs_long_context` 两个信号直接参与决策（有明确监管/工程依据）；`complexity` 先记录不参与决策，积累数据后再启用——避免一开始就把规则写死。

### 5.2 决策策略（高风险闸门 + 置信分层 + 兜底）

```
Jev 返回 {category_key, prob_max, probs, high_risk, needs_long_context}
│
├─ 0) 风险闸门（凌驾于分类档位之上）
│     high_risk ≥ 0.8 → 强制大模型档 + 响应标记"高风险需人工复核"
│                        + 写审计日志 + 禁用路由缓存（金监总局人工监督要求 [15]）
├─ 1) 上下文闸门
│     needs_long_context ≥ 0.8 → 切长上下文档位（与类别档位取更高者）
│
├─ 2) 置信分层
│     prob_max ≥ τ_high (0.8)     → 直选该类模型（90%+ 流量）
│     τ_low (0.55) ≤ prob < τ_high → 保守策略：升一档选模型（宁可选大不选小）
│     prob_max < τ_low             → 升级 System Two：小 LLM 复判一次
│                                      ├─ 复判结果 → 采用
│                                      └─ 复判失败/超时 → default 档模型
├─ 3) 交叉仲裁：主判 prob_max ∈ [τ_low, τ_high] 且主判与 high_risk/complexity
│     信号矛盾（如主判 office_chat 但 complexity ≥ 4）→ 升档处理
└─ 异常（Jev 超时/报错）            → default 档模型（永不因路由失败而失败）
```

与 OpenRouter auto 的降级哲学一致：**分类不可用时 fallback 默认模型，请求永不失败**。阈值 τ 通过评测集校准（见 §6）。风险闸门的判定输出与最终"是否人工复核"一起写入审计日志，满足监管对高风险应用可追溯的要求 [15]。

### 5.3 分层路由管道（Gate → Cache → Rules → Jev → Fallback）

路由不是一个 Jev 调用，而是一条分层管道。原则：**确定性越高的判断放越靠前（越便宜、越可审计），语义模糊的判断才交给模型**；上层可以直接放行或拦截，但规则层只做"捷径"，不做对下层结果的否决。

```
请求 → L0 门禁 → L1 缓存 → L2 规则 → L3 Jev 语义 → L4 复判/兜底 → 生成模型
```

| 层 | 职责 | 判定依据 | 延迟/成本 | 预期流量占比 |
|---|---|---|---|---|
| **L0 门禁 Gate** | 硬策略：鉴权、限流、请求体校验；**显式 override**（接入方在 header/参数里直接指定模型或路由，如 code 平台固定走 coding 模型——直接放行跳过 L2/L3）；**硬红线**（请求体命中卡号/身份证等敏感数据 pattern → 强制私有化通道 + 审计；明确违规内容 → 直接拒绝） | 配置化硬规则，100% 确定性 | <1ms | 全量；拦截 + override 约 5–10% |
| **L1 缓存 Cache** | 归一化后相同 query 复用上次路由决策；高风险类（credit_dd/compliance_review/risk_aml）及 high_risk 命中的请求**不入缓存** | LRU + 归一化 key | ~1ms | 5–15% 命中 |
| **L2 规则 Rules** | 强信号捷径：显式指令（`/sql`、`/code` 前缀）、来源应用标记（OA 入口默认 office_writing）、输入长度超阈值直接切长上下文档位 | 关键词/正则/元数据，配置化于 routes.json 的 `rules` 节 | ~1ms | 10–20% 命中 |
| **L3 Jev 语义** | 主路径：choice 14 类主判 + `high_risk`/`needs_long_context` noul 闸门同请求并行；按 §5.2 阈值分层决策 | 语义判定，校准概率 | ~200ms | ~70–85% |
| **L4 复判/兜底** | Jev 低置信（prob_max < τ_low）→ System Two 小 LLM 复判一次；Jev 异常/超时 → default 档模型。**请求永不失败** | 二次判定 / 静态兜底 | +200ms～2s | <5%，<1% |

设计约束：

1. **L0/L1/L2 不可否决 L3**：规则只能"提前命中"高确定性 case，不能拦截或改写 Jev 的语义判定——规则误命中时最多损失一次 Jev 调用机会，不会锁死路由；反过来 Jev 判定结果永远以语义为准。
2. **override 是显式的**：只有 L0 的显式指定可以绕过语义层（调用方自己负责），规则层不允许配置 override。
3. **每层独立可观测**：管道输出 `{layer_hit: gate|cache|rule|jev|fallback, category, prob_max, chosen_model, ...}` 落 JSONL——分层命中率是后续调优（扩规则集 vs 调阈值）的直接依据。
4. **全部配置化**：门禁规则、缓存策略、规则捷径、类别映射、阈值都在 routes.json，改配置不改代码。

### 5.4 总体架构与部署拓扑

**方案 A（推荐，P0–P2）：扩展 jev_proxy.py 为路由网关**——单进程、零新增部署面，复用现有 systemd 服务（fin-jev）与密钥管理（`jev_key.local.json`），现有 `/v1/systemone` 同源代理与静态页保持不变。

```
接入方（duel_chat / OA 助手 / 代码平台 / 数据问答客户端）
   │  POST /v1/router/chat —— OpenAI 兼容，可带 X-Route-Override
   ▼
┌─────────────────  jev_proxy.py : 8787（单进程）  ─────────────────┐
│                                                                  │
│  Router Pipeline（server/router/）                               │
│   L0 GateGuard ──── 鉴权/限流/敏感红线 → 拦截或强制私有化；        │
│   │                 override 命中 → 生成显式决策跳过 L1–L3        │
│   L1 RouteCache ── 归一化 LRU 命中 → 直接复用上次决策             │
│   L2 RuleEngine ── 强信号捷径命中 → 直接出决策（仅 shortcut）     │
│   L3 JevClassifier ── choice 主判 + high_risk /                  │
│   │                 needs_long_context 两个 noul 同请求并行       │
│   L4 DecisionEngine ── 风险/上下文闸门 + 置信分层；               │
│         │              低置信 → System Two 复判                   │
│         ▼  决策 {tier, model, review_required, use_private}      │
│  ModelDispatcher —— 流式透传 + 通道级 fallback 链                 │
│   ├─ llm-router : 8117（GLM 池）   ← small / mid 档              │
│   ├─ 直连各模型 API                ← large / coding / longctx 档 │
│   └─ 私有化通道                    ← 敏感数据强制（L0/L4 命中）   │
│                                                                  │
│  AuditLogger（异步队列 → JSONL）   ConfigCenter（routes.json 热载）│
│                                                                  │
│  （原有）POST /v1/systemone 同源代理 · 静态页/探针 —— 保持不变     │
└──────────────────────────────────────────────────────────────────┘
```

**方案 B（演进，P3）：Pipeline 拆为独立 router 服务**——多前端共用或需独立扩缩容时，把 `server/router/` 整体迁出为独立进程；接入方只改 base_url，`/v1/router/chat` 契约不变。初期不必要。

模型通道：本机 llm-router（8117，GLM 系）作为 small/mid 档模型池；large/coding/长上下文档走直连各模型 API（服务器侧无 GLM key），或后续统一收敛至自建网关——差异由 Dispatcher 层屏蔽，接入方无感。

### 5.5 核心组件设计

```
server/
├── jev_proxy.py          # 现有进程入口：systemone 同源代理 + 静态页（保持不变）
├── router/
│   ├── pipeline.py       # 管道编排：L0→L4 顺序执行、异常边界、耗时埋点
│   ├── gate.py           # L0 GateGuard
│   ├── cache.py          # L1 RouteCache
│   ├── rules.py          # L2 RuleEngine
│   ├── classifier.py     # L3 JevClassifier（含 Jev provider 抽象）
│   ├── decision.py       # L4 DecisionEngine（含 System Two 复判）
│   ├── dispatcher.py     # ModelDispatcher
│   ├── audit.py          # AuditLogger
│   └── config.py         # ConfigCenter
└── routes.json           # 全部路由配置（schema 见 §5.6）
```

| 组件 | 层 | 职责 | 失败行为 |
|---|---|---|---|
| **GateGuard** | L0 | Bearer 鉴权、限流、body 校验；敏感数据 pattern 命中 → 强制私有化通道 + 同步审计；违规内容 → 403；读 `X-Route-Override` → 产出显式决策并跳过 L1–L3 | 配置缺失/正则编译失败 → 拒绝启动（fail-fast） |
| **RouteCache** | L1 | 归一化（小写、去多余空白/标点）query → 上次决策；TTL + 容量双限；高风险类与 high_risk 命中的请求不写入 | 关闭/异常 → 直接透传下一层 |
| **RuleEngine** | L2 | 把 `routes.json.rules` 编译为有序 matcher，首个命中即产出决策；只做 shortcut、不否决语义层 | 编译期失败 → 启动报错；运行期单条规则异常 → 记日志并跳过该条 |
| **JevClassifier** | L3 | state 构造（最近 1–2 条用户消息截断 ~2K tokens，多轮追问附上一轮摘要）；一个 choice（14 类）+ `high_risk`/`needs_long_context` 两个 noul 同请求并行；provider 抽象（博查/火山可切换）；body 严格 `model/state/questions` 三键 | 超时 800ms 重试 1 次，仍失败 → 转 L4 fallback 分支 |
| **DecisionEngine** | L4 | 风险闸门 → 上下文闸门 → 置信分层（τ_high 直选 / 区间升档 / τ_low 以下复判）→ 交叉仲裁；低置信调 System Two 小 LLM 复判一次；产出 `{tier, model, review_required, use_private}` | 复判失败/超时 → default 档；永不向上游抛异常 |
| **ModelDispatcher** | 出口 | tier → 通道/模型映射；流式与非流式透传；通道级 fallback 链（主通道超时/5xx → 备通道）；写 `X-Route-*` 观测头 | 全链失败 → 502 + audit 记录 |
| **AuditLogger** | 横切 | 内存队列 + 后台线程批量写 JSONL（schema 见 §5.6）；高风险事件（review_required / 敏感命中）同步落盘不丢 | 队列满 → 丢低风险记录、高风险降级同步写 |
| **ConfigCenter** | 横切 | routes.json mtime 检测热载；读临时文件 → schema 校验 → 原子替换；类别 ↔ 模型 ↔ 阈值全部配置化，改配置不改代码 | 校验失败 → 保留旧配置并告警 |

### 5.6 数据契约

**路由入口 `POST /v1/router/chat`（OpenAI 兼容）**

```http
POST /v1/router/chat
Authorization: Bearer <token>              # GateGuard 鉴权
X-Route-Override: coding                   # 可选：显式类别/档位/模型，命中即跳过 L1–L3
Content-Type: application/json

{ "model": "auto", "messages": [ ... ], "stream": true }   # 与 OpenAI chat/completions 完全兼容
```

响应为目标模型原样响应（流式透传），附加观测头：

```http
X-Route-Layer: jev             # gate | cache | rule | jev | fallback
X-Route-Category: data_query   # 14 类之一；override 时回显 override 值
X-Route-Prob: 0.91             # prob_max（非 L3 路径为 1.0）
X-Route-Model: glm-4.7         # 最终选定模型
X-Route-Risk: 0.03             # high_risk noul 输出（非 L3 路径为 0）
X-Route-Jev-Ms: 187            # Jev 耗时；未走 L3 为 0
X-Route-Review: false          # 高风险人工复核标记
```

**routes.json schema**（ConfigCenter 校验此结构）

```json
{
  "categories": {
    "office_chat": { "tier": "small", "risk": "low", "cacheable": true,
                     "criteria": "日常寒暄、翻译、润色改写、单点事实问答；与业务无关的短请求" },
    "credit_dd":   { "tier": "large", "risk": "high", "cacheable": false,
                     "criteria": "对公信贷尽调、授信材料、普惠核查报告撰写与校验" }
    // ……共 14 类 + default，criteria 要点见 §5.1
  },
  "tiers": {
    "small":   { "channel": "llm-router", "model": "<GLM 小档>",   "fallback": "mid" },
    "mid":     { "channel": "llm-router", "model": "<GLM 中档>",   "fallback": "large" },
    "large":   { "channel": "direct",     "model": "<旗舰模型>",   "fallback": "mid" },
    "coding":  { "channel": "direct",     "model": "<coding 专用>", "fallback": "large" },
    "longctx": { "channel": "direct",     "model": "<128K 长上下文>", "fallback": "large" },
    "default": { "tier": "mid" }
  },
  "gates": {
    "sensitive_patterns": ["\\b\\d{16,19}\\b", "<身份证/证件号正则>"],
    "private_channel": { "channel": "on-prem", "model": "<私有化模型>" }
  },
  "thresholds": {
    "tau_high": 0.8, "tau_low": 0.55,
    "high_risk": 0.8, "long_ctx": 0.8,
    "state_max_tokens": 2000, "jev_timeout_ms": 800
  },
  "rules": [
    { "type": "prefix",  "value": "/code",               "category": "ai_coding" },
    { "type": "keyword", "value": ["招股书", "审计报告"], "category": "doc_heavy" },
    { "type": "source",  "value": "oa",                  "category": "office_writing" },
    { "type": "len_gt",  "value": 30000,                 "tier": "longctx" }
  ],
  "cache": { "max_size": 4096, "ttl_s": 3600 },
  "jev": {
    "provider": "bocha",
    "endpoint": "<systemone endpoint>",
    "questions": {
      "main":               { "type": "choice", "options": "<14 类 + default 的 key>" },
      "high_risk":          { "type": "noul",   "statement": "<是否命中高风险应用定义（§5.1）>" },
      "needs_long_context": { "type": "noul",   "statement": "<输入是否为超长文档处理任务>" }
    }
  }
}
```

**观测 JSONL（AuditLogger 每请求一条）**

```json
{"ts":"2026-11-18T10:00:00+08:00","req_id":"r-3f9a","tenant":"duel_chat",
 "layer_hit":"jev","category":"data_query","prob_max":0.91,
 "probs":{"data_query":0.91,"reasoning_deep":0.05,"office_chat":0.04},
 "high_risk":0.03,"needs_long_context":0.11,"tier":"mid","chosen_model":"<GLM 中档>",
 "jev_ms":187,"total_route_ms":212,"override":null,"cache_hit":false,
 "state_tokens":312,"review_required":false,"use_private":false}
```

字段用途：`layer_hit` 分布决定调优方向（cache/rule 命中高 → 收益主要来自省钱；fallback 高 → Jev 链路不稳）；`probs` 全量分布用于漂移检测（§6 回归）；`review_required=true` 的事件即监管审计线索。

### 5.7 关键实现细节

1. **路由 state 构造**：只送任务语义所需的最近用户消息（截断 ~2K tokens），不送全对话历史——省 tokens 且避免历史偏置分类。
2. **多轮上下文**：对话中途的追问（"再优化一下"）语义不足时，附带最近一轮对话摘要进 state。
3. **缓存**：归一化 query 级 LRU（见 §5.3 L1），高风险类不入缓存（合规要求，见 §5.2）。
4. **观测**：每次路由按 §5.6 JSONL schema 落一条记录，供评测与阈值校准。
5. **配置热载**：`routes.json` 含分类体系、模型映射、阈值、fallback 链，agent 侧可调。

---

## 6. 评测方案

参照 RouterBench 方法论（405k 推理结果、成本-质量前沿），但规模务实化：

| 项 | 内容 |
|---|---|
| 标注集 | 300–500 条真实 query，按 §5.1 场景配额覆盖 14 类（每类 ≥15 条），另含 50 条混合/模糊样本专测边界、30 条高风险样本专测 `high_risk` 闸门召回 |
| 分类准确率 | Jev choice vs 人工标注，目标 ≥90%；混合类单独报告 |
| 阈值校准 | 在标注集上扫 τ_high/τ_low，画（成本节省 × 质量保持率）曲线取 Pareto 点 |
| 路由开销 | p50/p95 路由延迟增量，目标 p95 < 500ms（实测 ~200ms 起步） |
| 成本收益 | 路由后混合成本 vs 全大模型基线，目标降 40–70% |
| 质量对比 | 抽样路由任务 vs 全大模型回答，人工盲评（RouterBench 的 cost-matched 思路简化版） |
| 高风险闸门召回 | 30 条高风险样本的 `high_risk` 召回率/误报率，目标召回 ≥95%——漏判属合规风险，误报仅造成升档成本，阈值偏召回 |
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
| P0 PoC | jev_proxy.py 加 `/v1/router/chat`：choice 主判 + 风险/上下文闸门 + 阈值决策 + fallback；routes.json 含 14 类映射 | 手工跑通 14 类路由与高风险升档，curl 观测头正确 |
| P1 评测 | 300+ 条标注集（14 类配额 + 高风险样本）；分类准确率、闸门召回、阈值曲线、延迟、成本报告 | 准确率 ≥90%、高风险召回 ≥95%、成本节省 ≥40% |
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
15. 国家金融监督管理总局《关于银行业保险业人工智能安全开发应用的指导意见》（高风险应用定义：资金交易/资产评估/信贷审批/承保理赔/风险管理；高风险准入与人工监督要求）— https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1261784&itemId=928
16. 新华网：1800 亿元加码金融科技，13 家上市银行大模型应用爆发式增长（工行 500+ 场景、招行 856 场景、建行 AI 助手网点覆盖率 99.42%）— https://www2.xinhuanet.com/20260415/d9222a7625f640fa9cb5fda34f12e6b9/c.html
17. 新华网：北京银行 AIB 平台（80 项大模型服务、50 款运营效率工具、10 项 GPT 创作助手）— http://bj.news.cn/20250806/b7c7242db2ff43379b0a565d5a2038ca/c.html
18. 新华网：建行"帮得"客户经理助手（普惠核查报告 2–3 天 → 3 小时，日均 Token 消耗破百亿）— https://app.xinhuanet.com/news/article.html?articleId=20260917f46225ee6cf948c1b8084b86b9f2dc98
19. 羊城晚报：百融智能"硅基员工"（营销报告、财务报表与流水分析、信贷材料校验、贷后监测）— https://ep.ycwb.com/epaper/ycwb/html/2026-10/30/content_2987_770422.htm
20. 中国证券业协会：券商加码布局 AI 智能体（研报信息提取 → 财务模型底稿、产品分析研报）— https://www.sac.net.cn/hydt/202608/t20260811_82123.html
21. 财联社：Anthropic 10 个智能体重构投行、基金工作流（Earnings Reviewer、Model Builder：DCF/LBO/三表联动）— https://www.cls.cn/detail/2373585
22. Kellton：Generative AI in Banking 2026 Expert Guide（Goldman 财报电话会实时情绪/指引提取、算法策略文档）— https://www.kellton.com/kellton-tech-blog/generative-ai-in-banking
23. LTIMindtree：Conversational Intelligence for Anti-Money Laundering using GenAI（CDD/EDD、PEP 筛查、UBO 识别、STR 叙述准备）— https://www.ltim.com/content/dam/ltimcorporatewebsite/uploads/2026/02/v3-aml-360-index-linked.pdf
24. OECD：Generative Artificial Intelligence in Finance（AI 异常检测在 AML/CFT 与欺诈检测的应用）— https://www.oecd-ilibrary.org/deliver/ac7149cc-en.pdf
25. Anthropic：JPMorganChase expands Claude Code（推广至工程师之外，PR 审查、监控告警根因分析）— https://claude.com/resources/articles/jpmorganchase-expands-claude-code
26. 金融电子化："对话即服务"重构智能办公流程（会议预订、全角色转写、纪要与待办生成）— http://m.toutiao.com/group/7576467223280108047/
27. 南方财经：中小银行拥抱 AI"办公搭子"（席位订阅、Token 采购、OA 内嵌智能体）— https://m.sfccn.com/2026/11-13/0NMDE0MDZfMjMwNDA0NQ.html
28. 上海财经大学 FinEval 6.0：金融大模型评测基准（金融学术知识/行业理解/安全认知/智能体应用/多模态/严谨性六大模块，26000+ 题）— https://m.thepaper.cn/newsDetail_forward_30906783

> 注：标注 [4][5] 等定价与规格以官方文档为准；内部实测数据来自本仓库 PoC 环境（博查 jev.bocha.cn，2026-11），随环境变化需重测。
