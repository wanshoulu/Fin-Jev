/* R46d 运行期验证：从 duel_chat.html 提取 graphRecall，注入真实 graphIndex，跑真实 query */
var fs = require("fs");
var html = fs.readFileSync("duel_chat.html", "utf8");

/* 提取 graphRecall + fmtGraphRecallDetail 两个函数源码 */
function extractFn(name){
  var re = new RegExp("function " + name + "\\(text\\)\\{|function " + name + "\\(d\\)\\{");
  var m = html.match(re);
  if (!m) throw new Error("找不到函数 " + name);
  var start = m.index, depth = 0, i = html.indexOf("{", m.index);
  for (var j = i; j < html.length; j++){
    if (html[j] === "{") depth++;
    else if (html[j] === "}"){ depth--; if (!depth){ return html.slice(start, j + 1); } }
  }
  throw new Error("括号配平失败 " + name);
}
var code = extractFn("graphRecall") + "\n" + extractFn("fmtGraphRecallDetail");

/* 沙箱：注入真实编译产物 + performance stub */
var GRAPH = JSON.parse(fs.readFileSync("probes.compiled.json", "utf8"));
var sandbox = new Function("GRAPH", "performance", "esc",
  "var GRECALL_TRACE = null;\n" + code + "\nreturn { graphRecall: graphRecall, fmt: fmtGraphRecallDetail, trace: function(){ return GRECALL_TRACE; } };");
var api = sandbox(GRAPH, { now: function(){ return Date.now(); } }, function(s){ return String(s); });
/* C_COLOR 是页面全局色板，沙箱补 stub */
var C_COLOR = { transfer:"#4098fc", wealth:"#7c5cff", account:"#00b578", loss_fraud:"#fa5151", complaint:"#c288fb" };
api.fmt = new Function("GRAPH", "performance", "esc", "C_COLOR",
  "var GRECALL_TRACE = null;\n" + extractFn("graphRecall") + "\n" + extractFn("fmtGraphRecallDetail") +
  "\nreturn function(d){ graphRecall(d && d.domains && d.domains.length ? \"\" : \"\"); return fmtGraphRecallDetail(d); };")(GRAPH, { now: function(){ return Date.now(); } },
  function(s){ return String(s); }, C_COLOR);
/* 上面包装会重置 GRECALL_TRACE，改用更直接的方式：直接借 fmt 源码构造独立实例 */
api.fmt = new Function("C_COLOR", "esc", "GRAPH",
  "return " + extractFn("fmtGraphRecallDetail").replace("function fmtGraphRecallDetail", "function"))(C_COLOR, function(s){ return String(s); }, GRAPH);

/* 断言工具 */
var pass = 0, fail = 0;
function assert(name, cond, detail){
  if (cond){ pass++; console.log("  通过  " + name); }
  else { fail++; console.log("  失败  " + name + (detail ? "  → " + detail : "")); }
}

/* ---------- 场景 1：投诉（词面锚定 + 上溯 + 挂载 FAQ 全链路） ---------- */
console.log("\n[场景1] 「你们什么态度我要投诉」");
var r1 = api.graphRecall("你们什么态度我要投诉");
console.log("  锚定节点:", r1.hits.map(function(h){ return h.id + "[" + h.terms.join("/") + "]"; }).join("  "));
console.log("  域得分:", JSON.stringify(r1.domains));
console.log("  扩展:", r1.expanded.join(",") || "无", " FAQ:", r1.faqs.length, "条 耗时:", r1.ms + "ms");
assert("S1 锚定 complaint.emotion", r1.hits.some(function(h){ return h.id === "demo.complaint.emotion"; }));
assert("S1 top1 域 = complaint", r1.domains.length && r1.domains[0].domain === "complaint", JSON.stringify(r1.domains));
assert("S1 挂载 FAQ ≥1", r1.faqs.length >= 1);
assert("S1 无转账误命中", !r1.domains.some(function(d){ return d.domain === "transfer"; }));

/* ---------- 场景 2：转账（scene terms 命中） ---------- */
console.log("\n[场景2] 「帮我给张三转五千块」");
var r2 = api.graphRecall("帮我给张三转五千块");
console.log("  锚定节点:", r2.hits.map(function(h){ return h.id + "[" + h.terms.join("/") + "]"; }).join("  "));
console.log("  域得分:", JSON.stringify(r2.domains), " 扩展:", r2.expanded.join(",") || "无");
assert("S2 锚定 transfer.peer#DO（长词面命中）",
  r2.hits.some(function(h){ return h.id === "demo.transfer.peer#DO"; }), JSON.stringify(r2.hits.map(function(h){return h.id;})));
assert("S2 top1 域 = transfer", r2.domains.length && r2.domains[0].domain === "transfer", JSON.stringify(r2.domains));
assert("S2 沿混淆边扩展 loss_fraud", r2.expanded.indexOf("loss_fraud") > -1, JSON.stringify(r2.expanded));

/* ---------- 场景 3：查余额 ---------- */
console.log("\n[场景3] 「帮我查下卡里余额」");
var r3 = api.graphRecall("帮我查下卡里余额");
console.log("  锚定节点:", r3.hits.map(function(h){ return h.id + "[" + h.terms.join("/") + "]"; }).join("  "));
console.log("  域得分:", JSON.stringify(r3.domains));
assert("S3 top1 域 = account", r3.domains.length && r3.domains[0].domain === "account", JSON.stringify(r3.domains));

/* ---------- 场景 4：边界——"手机号"误命中检查 ---------- */
console.log("\n[场景4-边界] 「我的手机号收不到验证码」");
var r4 = api.graphRecall("我的手机号收不到验证码");
console.log("  锚定节点:", r4.hits.map(function(h){ return h.id + "[" + h.terms.join("/") + "]"; }).join("  "));
console.log("  域得分:", JSON.stringify(r4.domains));
console.log("  → transfer.terms 含『手机号转账』不匹配『手机号』本身（精确包含 whole-word 需子串完整出现）");
assert("S4 无域得分（『手机号转账』≠『手机号』子串包含不成立）",
  r4.domains.length === 0, JSON.stringify(r4.domains));

/* ---------- 场景 5：纯语义无词面 ---------- */
console.log("\n[场景5] 「这钱怎么莫名其妙少了」");
var r5 = api.graphRecall("这钱怎么莫名其妙少了");
console.log("  锚定:", r5.hits.length, "个 域得分:", JSON.stringify(r5.domains));
assert("S5 空结果不抛异常且结构完整", Array.isArray(r5.hits) && Array.isArray(r5.domains) && typeof r5.ms === "number");

/* ---------- 场景 6：证据链完整性（GRECALL_TRACE 四步齐全） ---------- */
console.log("\n[场景6] 证据链完整性（重放场景2）");
var r6 = api.graphRecall("帮我给张三转五千块");
var tr = api.trace();
assert("S6 steps.anchor 非空", tr.steps.anchor.length > 0);
assert("S6 steps.up 非空（上溯记录）", tr.steps.up.length > 0);
assert("S6 steps.expand 非空（混淆边）", tr.steps.expand.length > 0);
assert("S6 hitNodes 带词面证据", tr.hitNodes.every(function(h){ return h.terms && h.terms.length; }));

/* ---------- 渲染器冒烟 ---------- */
console.log("\n[渲染器] fmtGraphRecallDetail 冒烟");
var htmlOut = api.fmt(r6);
assert("渲染含四段标题", htmlOut.indexOf("词面锚定") > -1 && htmlOut.indexOf("域得分") > -1);
assert("渲染含遍历耗时", htmlOut.indexOf("遍历耗时") > -1);
assert("渲染无 undefined/null 泄漏", htmlOut.indexOf("undefined") === -1 && htmlOut.indexOf("null") === -1);

console.log("\n========== 结果: " + pass + " 通过 / " + fail + " 失败 ==========");
process.exit(fail ? 1 : 0);
