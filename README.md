# Fin-Jev

不训练、零样本的银行意图路由 PoC：**6 个并行是非题 + 三层门控 + 图谱/向量双视图选题**，直通流量 21.7ms 落定，模糊流量升级 GLM 精判兜底。

![双引擎对决：同一条消息，两个引擎各自到达](docs/duel_chat_hero.png)

输入任意客户消息：左边两张引擎卡片先后亮起（Jev 秒回、GLM ~3s 后到，延迟差肉眼可见），右边是每一次调用的完整原始报文。

## 项目简介

**先说清楚名字里的每个词**：

| 名词 | 是什么 |
| --- | --- |
| **Jev** | **"System One（系统一）模型"**——美国旧金山 TypeSafe AI（创始人 Diogo Almeida 为前 OpenAI RLHF 研究员、InstructGPT 论文作者之一）2026 年 9 月发布的首个该类模型：**不生成任何文本**，给它一份状态和若干道带判据的类型化问题，它直接返回带校准概率的结构化决策（官方口径见 [TypeSafe 发布博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)：端到端 70~500ms，宣称同类任务比前沿 LLM 快 40~200 倍）。名字取自经济学家杰文斯（Jevons）。原版 API 未对中国大陆开放，本项目用**博查的中国版**（模型名 `bocha-jev-v1`，[open.bocha.cn](https://open.bocha.cn)） |
| **Fin-Jev** | **Fin**（银行金融场景）+ **Jev**（决策引擎）——验证"用 Jev 做银行客服意图路由"这条技术路线的 PoC |
| **GLM** | 智谱的生成式大模型（本项目用 `glm-5.3-flash`），负责模糊流量的语义精判 |
| noul | Jev 三种问题类型之一（另两种 Choice 候选选择、Score 等级评分，本项目未用）：布尔命题判定，每道是非题独立返回 0~1 连续概率（如 0.87 = 87% 认为命中）而非硬标签——连续概率是门控分诊的原料 |

**要解决的问题**：银行手机 App 的客服消息首先要路由进业务队列（转账 / 理财 / 账户 / 挂失止付 / 投诉 / 生活缴费）。三条传统路线各有硬伤：微调小模型要几千条标注、加意图重新训练，知识烧进参数无法审计；LLM 直接分类 2~4 秒一条还按 token 计费，多分类概率挤在中间地带；图谱 + LLM 先建图才有路由，检索链路自带失败点。

**Fin-Jev 的做法**：一份手工维护的意图图谱（`bank_graph.json`，六域 199 词面）机械编译成 6 道是非题 → Jev 一次请求并行打分（21.7ms）→ 本地三层门控分诊（top1 置信 / 前两名 gap / 涉诈词）→ 三门全过直接路由，任一命中才升级 GLM 候选集精判（2.7~4.4s）。选题层用图谱词面锚定 + 向量语义召回双视图共同决定"这次出哪几道题"——向量分数未校准不进判分，只进选题。

**核心发现**：同一个模型，多分类直问准确率 **25%**，拆成是非题并行 **83%**——问题形态比模型选择更重要。状态：PoC（样本 23~49 条，百分比不外推生产指标）。意图图谱的工程化建模在姊妹仓库 [graph-builder](https://gitcode.com/wangkun_123/graph-builder)。

## Quick start

```bash
git clone https://gitcode.com/wangkun_123/Fin-Jev.git
cd Fin-Jev/server
python jev_proxy.py        # Python 3.8+，零依赖
```

打开 http://127.0.0.1:8787 ，点快捷 chips 里的「我的银行卡丢了！赶紧帮我挂失冻结」：
Jev 卡片秒回（挂失止付 87%，gap 0.05 触发升级）→ GLM 卡片约 3 秒后到达，复核 95%。

**Key 配置**（源码零 Key，真实密钥永不入库）：

| 引擎 | 必需？ | 配置方式 |
| --- | --- | --- |
| Jev（Bocha） | 必需 | 页面 ⚙ 填 [open.bocha.cn](https://open.bocha.cn) Key（存本机 localStorage）；或 `jev_key.local.json` 填 `{"apiKey": "..."}` |
| GLM（智谱，可选） | 可选 | `server/glm_key.local.json` 填 Key（Coding Plan 走 `/api/coding/paas/v4`）。不填则 GLM 卡片降级为本地基线（明确标注），门控演示不受影响 |
| 向量库（OntoMind，可选） | 可选 | 本地 17386 端口。不在线时自动降级纯图谱选题 |

安全模型：前端只发 `Bearer BUILTIN_*` 占位符，`jev_proxy.py` 从 `*.local.json`（gitignore 排除）读取真实 Key 替换。

也可以直接双击 `demos/duel_chat.html`——页面自动探测本地转发器并切换完整地址通道。

## How it works

![架构：一条消息的完整路径](docs/architecture_flow.png)

**一次真实调用走查**（截图就是它）：

| 环节 | 结果 |
| --- | --- |
| Jev 阵列 | 挂失止付 **87%** · 账户查询 **82%** · 其余 ≤49% |
| 门控 | top1 ✓ 但 gap=0.05 < 0.2 → 升级（"挂失"与"查账户"语义相邻，argmax 不可信） |
| GLM 复核 | 挂失止付 **95%**——3.6s 的钱花在了刀刃上 |

右侧报文面板看每次调用的完整请求/响应：

![原始报文追踪面板](docs/trace_panel.png)

## 与三条老路的差别

| 路线 | 硬伤 | Fin-Jev 的回答 |
| --- | --- | --- |
| 微调小模型 | 几千条标注起步；加意图重新训练；知识烧进参数无法审计 | 加意图 = 图谱加一域 + 点一次重编译（零训练） |
| LLM 直接分类 | 2~4s 一条；多分类概率挤中间地带；按 token 计费 | 直通流量 0.1s 内落定，模糊流量才花 LLM 的钱 |
| 图谱 + LLM | 先建图才有路由；检索链路自带失败点 | 图谱同时服务选题与判据定型，向量只补选题盲区 |

## 四个带数据的发现

1. **问题形态 > 模型选择。** 同一个 Jev，"六类里选哪类"直问准确率 **25%**（瞎猜水平）；拆成独立是非题并行（noul 化）后 **83%**。多分类选择题让一个判定头同时评估 6 个互斥假设，概率挤在 0.3~0.5；是非题让每个头只面对一个假设。方法与数据：[experiments.md §E2](docs/experiments.md#e2--多分类直问翻车与-noul-化修复)。
2. **criteria 写作质量 = 模型上限。** `true:"和理财有关"` 给出 0.6x 模糊概率；`true:"涉及理财/基金购买，或净值、收益、到期、风险等级咨询"` 给出 0.9+ 干脆判定。它是 Jev 模式下唯一的知识载体，要当代码管（编译自校验 + 回归集保护）。见 [E1](docs/experiments.md#e1--criteria-写法对照实验)。
3. **门控不能建在 LLM 自报概率上。** Jev 输出"一超多弱"（87% vs 30%）或"双高"（87% vs 82%），分布自带置信信号；GLM 永远输出陡峭分布（95% vs 0~5%），gap 门控在它身上失效。见 [E4](docs/experiments.md#e4--双引擎延迟与概率形态实测)。
4. **词面召回有结构性盲区，需要第二个视图。** "我要转300元"这类单动词+金额的中文高频表达，在词面表（≥2 字过滤）里 8 条同构消息 **0 命中**——泛化锚定（动词+紧邻金额的结构正则）补了一半，另一半（"帮我把这个月费用交一下"，图谱完全无词面）只有向量语义召回救得活：**向量不进判分（dense 分数未校准，撑不起 gap 门控），但进选题（与图谱锚定取并集）**。"向量 or 图谱"是假问题，"向量选什么题 + 图谱怎么打分"才是真结构。见 [architecture.md §5.5b](docs/architecture.md)。

## 实测速览

全部本机真实调用，测量方法见 [experiments.md](docs/experiments.md)：

| 指标 | Jev noul 阵列 | GLM-5.3-Flash |
| --- | --- | --- |
| 服务端推理 | **21.7ms**（inference_ms） | —（无此字段） |
| 端到端（含网络） | 103~168ms | 2686~4416ms |
| 6 类意图 noul 化准确率 | 83%（直问仅 25%） | 高，但慢 28 倍 |
| 选题层开销 | 词面遍历 0~1ms + 向量检索 40~80ms | — |
| 新增意图成本 | 图谱加一域 + 点一次重编译按钮 | 改 prompt |
| 已知失败模式 | 相邻意图抢票（gap 小）；选题裁剪后分数漂移 ±pp | 过度自信，gap 门控失效 |

样本量为 PoC 级（23~49 条），百分比不外推为生产指标。

## 改知识 = 改图谱

![知识闭环](docs/knowledge_loop.png)

`bank_graph.json` 是唯一人工维护的事实源（6 域 · 199 词面 · 9 FAQ · 4 混淆边）。存盘后页面上点「⟳ 重新编译图谱」，`graph_to_probes.py` 机械编译（ct/cf 自校验：与 R40 实测定稿逐字比对，漂移即拒绝产出），`probes.compiled.json` 热加载——判据改一处、两引擎同时生效，无手工同步点。活体实验验证过：加词面"异地汇款"→ 一键重编译 → 线上立即锚定 → 回滚还原。

## Demos：按演进顺序

每个 demo 是探索一个阶段留下的可运行产物，互不依赖，均可双击直开（file:// 已适配）：

| # | 文件 | 回答的问题 |
| --- | --- | --- |
| 0 | `jev_finance_poc.py` | Jev API 支持多问并行吗？返回概率还是标签？ |
| 1 | `demos/kb_lab.html` | criteria 写法对判定质量影响多大？ |
| 2 | `demos/jev_vs_llm_demo.html` | 多分类直问 vs noul 化差多少？（25%→83% 现场复现） |
| 3 | `demos/jev_pipeline_demo.html` | 聚合 + 门控流水线怎么走？8 条典型消息全可复跑 |
| 4 | `demos/duel_demo.html` | 双引擎同题对决（静态版） |
| 5 | `demos/duel_chat.html` | **最终形态**：对话式测试台（推荐入口，截图即此页） |

## 仓库结构

```
Fin-Jev/
├── README.md                  # 本文件
├── bank_graph.json            # 意图图谱唯一事实源（六域）
├── graph_to_probes.py         # 图谱→判据/索引编译器（含自校验）
├── probes.compiled.json       # 编译产物（6 题 + graphIndex 20 节点）
├── r45_lab.js / r46d_lab.js   # 回归断言（node 直跑：36 + 17 条）
├── docs/
│   ├── architecture.md        # 架构与设计决策（含代价）+ 已知限制（15 条）
│   ├── experiments.md         # E0~E5 实验记录：方法、数据、结论
│   ├── architecture_flow.png  # 架构图（一条消息的完整路径）
│   ├── knowledge_loop.png     # 知识闭环图（源文件 .svg 同目录）
│   ├── duel_chat_hero.png     # 工具截图（本页头图）
│   └── trace_panel.png        # 报文面板截图
├── server/jev_proxy.py        # 本地一体化服务器：静态页 + Jev/GLM/向量库转发 + 一键重编译
├── jev_finance_poc.py         # E0：API 能力验证脚本
└── demos/                     # 上表 6 个演进 demo
```

## 常见问题

**没有 GLM Key 能用吗？** 能。Jev 侧完整真实调用，GLM 卡片降级为本地基线（标注"本地基线·演示"）。

**没有向量库能用吗？** 能。OntoMind 不在线时自动降级纯图谱选题，页面流水线面板有明确标注。

**浏览器直开 Jev 接口为什么失败？** Bocha 不返回 CORS 头（实测 OPTIONS 403）。`jev_proxy.py` 同源转发解决。

**GLM 调用 429？** Coding Plan Key 必须走 `/api/coding/paas/v4`，改 `server/glm_key.local.json` 的 url。

**双击 HTML 打不开？** 需转发器在运行；页面会自动探测 `127.0.0.1:8787` 并切换通道（见 [architecture.md §5](docs/architecture.md#5-通道与安全设计)）。

**延迟数据可复现吗？** 21.7ms 是服务端口径，量级可跨环境复现；103~168ms 往返含本机转发，公网部署数字会变。

## 文档

- [架构与设计决策](docs/architecture.md) — 术语、三视图召回机制、门控阈值依据、通道/Key 安全模型、设计决策（每条含代价）、已知限制（15 条诚实清单）
- [实验记录](docs/experiments.md) — E0~E5 六组实验唯一权威，失败模式分析
- [graph-builder（姊妹仓库）](https://gitcode.com/wangkun_123/graph-builder) — 意图图谱建模与工程化，本仓库路线的上游知识层

## License

[MIT](LICENSE)
