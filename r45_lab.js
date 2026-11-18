/* R45 端到端离线验证：候选集请求拼装 + FAQ/混淆注入（不发真实请求） */
const fs = require("fs");
const html = fs.readFileSync(__dirname + "/duel_chat.html", "utf8");
const s = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1])[0];

const mConst = s.match(/var PROBES = [\s\S]*?^var GRAPH = null;/m);
const mFraud = s.match(/function fraudHit\(text\)\{[\s\S]*?\n\}/);
const mAgg = s.match(/function aggregate\(probs, text\)\{[\s\S]*?\n\}/);
const mReq = s.match(/function buildGlmRequest\(text, mode, candidates, vecHits, gr\)\{[\s\S]*?\n\}/);
const mQ = s.match(/function buildQuestions\(gr\)\{[\s\S]*?\n\}/);
if (!mConst || !mFraud || !mAgg || !mReq || !mQ){ console.log("extract fail", !!mConst, !!mFraud, !!mAgg, !!mReq, !!mQ); process.exit(1); }

const GRAPH = {
  confusables: [
    { fromDomain: "transfer", toDomain: "loss_fraud", evidence: "E1" },
    { fromDomain: "wealth", toDomain: "complaint", evidence: "E2" },
    { fromDomain: "account", toDomain: "transfer", evidence: "E3" }
  ],
  faqs: [
    { domain: "wealth", question: "QW1", answer: "AW1" },
    { domain: "transfer", question: "QT1", answer: "AT1" },
    { domain: "complaint", question: "QC1", answer: "AC1" }
  ]
};

const code = ("var window = typeof window === 'undefined' ? { __vecSelDomains: [] } : window;\n" + mConst[0] + "\n" + mFraud[0] + "\n" + mAgg[0] + "\n" + mReq[0] + "\n" + mQ[0] + "\nreturn { PROBES, aggregate, buildGlmRequest, buildQuestions };")
  .replace(/^var GRAPH = null;/m, "var GRAPH = " + JSON.stringify(GRAPH) + ";");

const factory = new Function("CFG", "GAP_GATE", "FRAUD_WORDS", code);
const ctx = factory({ glmModel: "glm-5.3-flash" }, 0.2, []);

let fail = 0;
function assert(name, ok){
  console.log((ok ? "OK   " : "FAIL ") + name);
  if (!ok) fail++;
}

/* 场景1：升级路径 —— wealth 0.42 / complaint 0.31，gap=0.11 < 0.2 触发升级 */
console.log("=== 场景1 升级路径：推荐理财（候选集精判）===");
const text1 = "推荐理财吧";
const probs1 = { transfer: 0.05, wealth: 0.42, account: 0.18, loss_fraud: 0.04, complaint: 0.31 };
const agg1 = ctx.aggregate(probs1, text1);
assert("门控触发升级（gap<0.2）", agg1.escalate === true);
const topCands = agg1.arr.slice(0, 3).map(a => ({ key: a.key, v: a.v }));
console.log("top3: " + topCands.map(c => c.key + ":" + c.v).join(", "));
assert("top3 组成正确（wealth/complaint/account）",
  JSON.stringify(topCands.map(c => c.key)) === JSON.stringify(["wealth", "complaint", "account"]));
const req = ctx.buildGlmRequest(text1, "candidates", topCands);
assert("请求带 _meta.mode=candidates", req._meta && req._meta.mode === "candidates");
assert("_meta.candidates=top3 keys", JSON.stringify(req._meta.candidates) === JSON.stringify(["wealth", "complaint", "account"]));
const sys1 = req.messages[0].content;
assert("prompt 只含 3 条候选判据", sys1.split("\n").filter(l => l.startsWith("· ")).length >= 3);
assert("含 Jev 概率标注", sys1.includes("[Jev 概率 42%]"));
assert("含图谱混淆提示（wealth↔complaint）", sys1.includes("注意区分 wealth 与 complaint"));
assert("含候选域 FAQ（wealth）", sys1.includes("[wealth] QW1"));
assert("不含候选集外 FAQ（transfer/account 泄漏检查）", !sys1.includes("QT1"));
assert("含否决权规则（top=none）", sys1.includes('"top":"wealth|complaint|account|none"'));

/* 场景2：直通路径 —— gap 充足走全量影子对照 */
console.log("=== 场景2 直通路径：查余额（全量影子对照）===");
const req2 = ctx.buildGlmRequest("查下我卡里余额", "full", null);
assert("全量模式无 _meta", req2._meta === undefined);
assert("system 含全部判据（bank6 起 6 类）", req2.messages[0].content.split("\n").filter(l => l.startsWith("· ")).length === 6);
assert("不含否决权（全量无需）", !req2.messages[0].content.includes("none"));

/* 场景3：涉诈词强制升级 —— 候选集含 loss_fraud */
console.log("=== 场景3 涉诈强制升级：安全账户 ===");
const probs3 = { transfer: 0.35, wealth: 0.02, account: 0.10, loss_fraud: 0.48, complaint: 0.05 };
const agg3 = ctx.aggregate(probs3, "把钱转到安全账户");
assert("涉诈词触发升级", agg3.escalate === true);
const top3b = agg3.arr.slice(0, 3).map(a => a.key);
const req3 = ctx.buildGlmRequest("把钱转到安全账户", "candidates", agg3.arr.slice(0, 3).map(a => ({ key: a.key, v: a.v })));
assert("候选集含 loss_fraud", req3._meta.candidates.includes("loss_fraud"));
console.log("top3: " + top3b.join(", "));

/* 场景4：向量召回片段注入（R46 链路） */
console.log("=== 场景4 向量召回片段注入 ===");
const vecHits = [
  { score: 0.776, content: "标问：银行卡挂失\n相似问：卡片丢了怎么办；挂失我的卡", source: "ccb.credit.card.loss-report__卡片挂失" },
  { score: 0.702, content: "银行卡管理常见问题", source: "ccb.me.cards.faq__银行卡管理·常见问题" }
];
const req4 = ctx.buildGlmRequest("我的银行卡丢了要挂失", "candidates",
  [{ key: "loss_fraud", v: 0.48 }, { key: "transfer", v: 0.35 }, { key: "account", v: 0.10 }], vecHits);
const sys4 = req4.messages[0].content;
assert("prompt 含向量召回片段标注", sys4.includes("(相似度 0.78)"));
assert("含片段来源内容", sys4.includes("银行卡挂失"));
assert("_meta 记录 vecHits 数量", req4._meta.vecHits === 2);
const req4b = ctx.buildGlmRequest("测试", "candidates", [{ key: "wealth", v: 0.5 }], []);
assert("空召回不注入该区块", !req4b.messages[0].content.includes("向量知识库召回"));

/* 场景5：图遍历证据注入（R46d 链路）——图谱结构在运行期被消费的直接输出 */
console.log("=== 场景5 图遍历证据注入 ===");
const grHit = {
  hits: [
    { id: "demo.complaint.emotion#DO", type: "scene", parent: "demo.complaint.emotion", title: "情绪投诉", terms: ["你们什么态度", "我要投诉"] },
    { id: "demo.complaint", type: "domain", parent: "demo.root", title: "投诉", terms: ["投诉"] }
  ],
  domains: [{ domain: "complaint", score: 6 }, { domain: "wealth", score: 1 }],
  expanded: ["wealth"],
  faqs: [],
  ms: 0
};
const req5 = ctx.buildGlmRequest("你们什么态度我要投诉", "candidates",
  [{ key: "complaint", v: 0.42 }, { key: "wealth", v: 0.31 }, { key: "account", v: 0.18 }], [], grHit);
const sys5 = req5.messages[0].content;
assert("prompt 含图谱遍历证据区块", sys5.includes("图谱遍历召回证据"));
assert("含锚定节点+命中词面", sys5.includes("demo.complaint.emotion#DO 命中词面 [你们什么态度、我要投诉]"));
assert("含域得分", sys5.includes("complaint=6"));
assert("含混淆边扩展提示", sys5.includes("混淆边扩展域：wealth"));
assert("_meta 记录 grHits", req5._meta.grHits === 2);
const req5b = ctx.buildGlmRequest("这钱怎么莫名其妙少了", "candidates", [{ key: "account", v: 0.4 }], [], { hits: [], domains: [], expanded: [], faqs: [], ms: 0 });
assert("空遍历不注入该区块", !req5b.messages[0].content.includes("图谱遍历召回证据"));
const req5c = ctx.buildGlmRequest("查下余额", "full", null, null, grHit);
assert("全量模式不注入图遍历（保持影子对照精简）", !req5c.messages[0].content.includes("图谱遍历召回证据"));

/* 场景6：Jev 图谱选题（R46e）——根据任务需要召回合适的题给模型 */
console.log("=== 场景6 Jev 图谱选题 ===");
const selHit = ctx.buildQuestions({ domains: [{ domain: "transfer", score: 3 }], expanded: ["loss_fraud"] });
assert("选题：保留 top1 域题", selHit.kept.includes("transfer"));
assert("选题：保留混淆边扩展域题", selHit.kept.includes("loss_fraud"));
assert("选题：裁剪无关域题", selHit.dropped.includes("wealth") && selHit.dropped.includes("complaint") && selHit.dropped.includes("account"));
assert("选题：qs 只含保留域", Object.keys(selHit.qs).length === 2 && selHit.qs.transfer && selHit.qs.loss_fraud);
assert("选题：full 标记为 false", selHit.full === false);
const selNo = ctx.buildQuestions(null);
assert("无锚定回退全量（bank6 起 6 题）", selNo.full === true && Object.keys(selNo.qs).length === 6 && selNo.dropped.length === 0);
const selEmpty = ctx.buildQuestions({ domains: [], expanded: [] });
assert("空域得分同样回退全量", selEmpty.full === true && Object.keys(selEmpty.qs).length === 6);
const selG = ctx.buildQuestions({ domains: [{ domain: "complaint", score: 6 }], expanded: ["wealth"] });
assert("选题判据文本与全量一致（编译产物同源，未改写）", selG.qs.complaint.criteria.true === selNo.qs.complaint.criteria.true);
const sel1 = ctx.buildQuestions({ domains: [{ domain: "account", score: 2 }], expanded: [] });
assert("单域无扩展时保底第二名域（≥2题 gap 才可计算）", sel1.kept.length === 2 && sel1.kept.includes("account"));
assert("单域选题已不产生 full=true", sel1.full === false);

console.log(fail === 0 ? "\n全部通过 ✓" : "\n有 " + fail + " 项失败 ✗");
process.exit(fail === 0 ? 0 : 1);
