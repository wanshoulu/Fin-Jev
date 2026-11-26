'use strict';
/* serve_nav.js — 模型驾驶连续导航服务（第 6 应用「多模态截图路由」行动环后端 · v2）
 *
 * 决策链全部交给模型，服务端没有关键词表 / 状态机指纹 / 策略表 / 写死选择器：
 *   navstart：GLM（glm-5.3-flash）解析一句话 → 意图 + 槽位 + 到达目标描述（自然语言）
 *   navstep 循环：截图 → d3 三题（:8788，视觉证据 + 页面异常硬保护）
 *     → 提取当前屏真实可交互元素（DOM 清单，带序号）→ GLM 决定下一步动作
 *     → 执行 click / fill（槽位驱动填参）/ enter / scroll → 再观察，直到模型判定 reached（完成）
 *
 * 运行： node server/serve_nav.js   （依赖本机 :3000 mock 银行 + :8788 serve_d3 + ROOT/glm_key.local.json）
 * 端口： 127.0.0.1:8789（jev_proxy 将 /v1/nav* 转发至此；服务器无 GPU/内存受限，仅本地跑）
 * 接口：
 *   GET  /v1/navhealthz   {ok, sessions, glm, model, bank, d3}
 *   POST /v1/navstart     {sentence} → {ok, session_id, target{intent,label,slots,goal_desc,done_hint}, max_steps, glm_ms}
 *   POST /v1/navstep      {session_id} → {ok, step, done, match, stop_reason, observe, action, model, final}
 *   POST /v1/navclose     {session_id} → {ok}
 * 会话：内存 Map，上限 2 个，15 分钟 TTL 自动回收；每会话一个 headless Edge 实例。
 * 环境变量：NAV_PORT / NAV_D3_BASE / NAV_BANK_URL / NAV_EDGE / NAV_PW_MODULE / NAV_GLM_BASE / NAV_GLM_MODEL
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const HOST = '127.0.0.1';
const PORT = Number(process.env.NAV_PORT || 8789);
const D3_BASE = process.env.NAV_D3_BASE || 'http://127.0.0.1:8788';
const BANK_URL = process.env.NAV_BANK_URL || 'http://localhost:3000/tools/ai-native-bank.html';
const EDGE_PATH = process.env.NAV_EDGE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const PW_MODULE = process.env.NAV_PW_MODULE || 'D:/llm-training/multimodal-jev/node_modules/playwright-core';
const GLM_BASE = process.env.NAV_GLM_BASE || 'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions';
const GLM_MODEL = process.env.NAV_GLM_MODEL || 'glm-5.3-flash';
const GLM_TIMEOUT = 45000;
const MAX_STEPS = 10;
const MAX_SESSIONS = 2;
const SESSION_TTL = 15 * 60 * 1000;
const D3_TIMEOUT = 150000;
const MAX_ELEMS = 40;

/* GLM Key：ROOT/glm_key.local.json（{"apiKey":...}），缺失则导航服务显式不可用 */
let GLM_KEY = null;
try { GLM_KEY = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'glm_key.local.json'), 'utf8')).apiKey || null; }
catch (e) { /* 无文件 */ }

let chromium = null;
let pwErr = null;
try { ({ chromium } = require(PW_MODULE)); }
catch (e) { pwErr = String((e && e.message) || e); }

/* 去壳 CSS：只留纯手机 UI（与 reshoot_mobile.js 同款） */
const CLEAN_CSS = [
  'body{background:#fff!important;}',
  '.phone{border:0!important;border-radius:0!important;box-shadow:none!important;}',
  '.phone::before{display:none!important;}',
].join('\n');

/* ---------- GLM 文本模型客户端（OpenAI 兼容 chat/completions） ---------- */

function glmChat(messages) {
  return new Promise(resolve => {
    const t0 = Date.now();
    const u = new URL(GLM_BASE);
    // thinking 禁用：glm-5.3-flash 为思考型模型，思考内容会挤占 max_tokens 导致 content 为空
    const data = JSON.stringify({ model: GLM_MODEL, messages, temperature: 0.2, max_tokens: 1024, thinking: { type: 'disabled' } });
    const req = https.request({
      hostname: u.hostname, port: 443, path: u.pathname + (u.search || ''), method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data), Authorization: 'Bearer ' + GLM_KEY },
      timeout: GLM_TIMEOUT,
    }, res => {
      let raw = '';
      res.on('data', c => { raw += c; });
      res.on('end', () => {
        try {
          const d = JSON.parse(raw);
          const content = d && d.choices && d.choices[0] && d.choices[0].message && d.choices[0].message.content;
          if (typeof content === 'string' && content.trim()) resolve({ ok: true, text: content, ms: Date.now() - t0 });
          else resolve({ ok: false, err: 'GLM 响应无内容（HTTP ' + res.statusCode + '）', ms: Date.now() - t0 });
        } catch (e) { resolve({ ok: false, err: 'GLM 响应非 JSON（HTTP ' + res.statusCode + '）', ms: Date.now() - t0 }); }
      });
    });
    req.on('timeout', () => req.destroy(new Error('GLM 请求超时')));
    req.on('error', e => resolve({ ok: false, err: String(e.message || e), ms: Date.now() - t0 }));
    req.end(data);
  });
}

/* 宽容 JSON 提取：剥 markdown 围栏、截取首个 {…} */
function looseJson(text) {
  let t = String(text || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  const a = t.indexOf('{');
  const b = t.lastIndexOf('}');
  if (a >= 0 && b > a) t = t.slice(a, b + 1);
  return JSON.parse(t);
}

async function glmJson(messages) {
  const r = await glmChat(messages);
  if (r.ok) {
    try { return { ok: true, json: looseJson(r.text), ms: r.ms }; }
    catch (e) { /* 首次解析失败 → 纠偏重问一次 */ }
  } else return r;
  const r2 = await glmChat(messages.concat([{ role: 'user', content: '上面的回答不是合法 JSON。只输出一个合法 JSON 对象，不要任何解释或代码围栏。' }]));
  if (!r2.ok) return r2;
  try { return { ok: true, json: looseJson(r2.text), ms: r2.ms }; }
  catch (e) { return { ok: false, err: 'GLM 输出无法解析为 JSON', ms: r2.ms }; }
}

/* ---------- d3 三题判定（内部直连 :8788，视觉证据 + 页面异常硬保护） ---------- */

function d3route(image) {
  return new Promise(resolve => {
    const t0 = Date.now();
    const data = JSON.stringify({ image });
    const u = new URL(D3_BASE + '/v1/d3route');
    const req = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
      timeout: D3_TIMEOUT,
    }, res => {
      let raw = '';
      res.on('data', c => { raw += c; });
      res.on('end', () => {
        try {
          const d = JSON.parse(raw);
          if (res.statusCode === 200) resolve({ ok: true, data: d, ms: Date.now() - t0 });
          else resolve({ ok: false, err: (d && d.detail) || ('HTTP ' + res.statusCode), ms: Date.now() - t0 });
        } catch (e) { resolve({ ok: false, err: 'd3 响应非 JSON', ms: Date.now() - t0 }); }
      });
    });
    req.on('timeout', () => req.destroy(new Error('d3 推理超时')));
    req.on('error', e => resolve({ ok: false, err: String(e.message || e), ms: Date.now() - t0 }));
    req.end(data);
  });
}

/* 从 serve_d3 的 routing 里取前端需要的字段子集 */
function pickVerdict(routing) {
  if (!routing) return null;
  return {
    domain: routing.domain, scene: routing.scene, page_state: routing.page_state,
    action: routing.action, label: routing.label, intent_id: routing.intent_id, conf: routing.conf,
  };
}

/* ---------- 会话管理 ---------- */

const sessions = new Map();

function touchTTL(id) {
  const sess = sessions.get(id);
  if (!sess) return;
  if (sess.timer) clearTimeout(sess.timer);
  sess.timer = setTimeout(() => closeSession(id), SESSION_TTL);
}

async function closeSession(id) {
  const sess = sessions.get(id);
  if (!sess) return;
  sessions.delete(id);
  if (sess.timer) clearTimeout(sess.timer);
  try { await sess.browser.close(); } catch (e) { /* 已退出则忽略 */ }
  console.log(`[nav] close ${id}（steps=${sess.step}）`);
}

/* ---------- 页面操作 ---------- */

async function openBankPage() {
  const browser = await chromium.launch({ executablePath: EDGE_PATH, headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 960 }, deviceScaleFactor: 2 });
  const page = await context.newPage();
  await page.goto(BANK_URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.addStyleTag({ content: CLEAN_CSS });
  await page.locator('#classicHome').waitFor({ state: 'visible', timeout: 30000 });
  await page.locator('#classicHome').getByText('总资产').first().waitFor({ state: 'visible', timeout: 20000 });
  await page.waitForTimeout(400);
  return { browser, page };
}

/* ---------- 可交互元素提取（真实 DOM 交给模型选；SEL 匹配序 n 与 page.locator(SEL).nth(n) 一致） ---------- */

const SEL = 'button,a,input,textarea,select,[onclick],[role="button"],[data-task-query],[data-open-feature],[data-catalog-action],[data-feature-action],[data-bottom-nav],[data-app]';

async function listElements(page) {
  return page.evaluate(sel => {
    /* 只收当前激活手机屏内的可见元素（与截图所见一致）；内部滚动区外的内容由 scroll 动作处理 */
    const act = document.querySelector('.phone-screen.active');
    const ar = act ? act.getBoundingClientRect() : { left: 0, top: 0, right: innerWidth, bottom: innerHeight };
    const out = [];
    const nodes = document.querySelectorAll(sel);
    for (let n = 0; n < nodes.length && out.length < 40; n++) {
      const el = nodes[n];
      const cs = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      if (cs.display === 'none' || cs.visibility === 'hidden' || cs.pointerEvents === 'none') continue;
      if (r.width < 6 || r.height < 6) continue;
      if (r.bottom <= ar.top + 2 || r.top >= ar.bottom - 2 || r.right <= ar.left + 2 || r.left >= ar.right - 2) continue;
      const tag = el.tagName.toLowerCase();
      const it = { n, tag, y: Math.round(r.top) };
      if (tag === 'input' || tag === 'textarea') {
        it.text = (el.placeholder || el.getAttribute('aria-label') || '输入框').trim().slice(0, 24) || '输入框';
        it.type = el.type || 'text';
      } else {
        let t = (el.innerText || el.textContent || el.getAttribute('aria-label') || '').trim().replace(/\s+/g, ' ').slice(0, 36);
        if (!t) {
          const img = el.querySelector('img[alt]');
          t = img ? '[图标]' + String(img.alt).slice(0, 20) : '';
        }
        if (!t) continue;
        it.text = t;
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') it.disabled = true;
      }
      out.push(it);
    }
    return out;
  }, SEL);
}

/* ---------- 模型决策提示词（不含任何业务导航知识：页面内容全靠真实元素清单） ---------- */

const DECIDE_SYS = '你是手机银行 App 的导航智能体（UI Agent）。用户给了一句办理请求，你要根据当前页面截图的多模态判定与真实可交互元素清单，逐步操作页面直到帮用户办成。\n'
  + '严格只输出一个 JSON 对象（不要 markdown、不要解释、不要额外文字）：\n'
  + '{"thought":"下一步理由（20字内）","action":"click|fill|enter|scroll|done|abort","element":元素序号n或null,"value":"fill的输入值/scroll的up或down/其余null","reached":true或false,"final":"done或abort时给用户的答复（60字内），其余为null"}\n'
  + '输出格式示例（仅示意）：\n'
  + '{"thought":"先进入转账功能","action":"click","element":5,"value":null,"reached":false,"final":null}\n'
  + '{"thought":"往下查看更多账单","action":"scroll","element":null,"value":"down","reached":false,"final":null}\n'
  + '{"thought":"账单已展示","action":"done","element":null,"value":null,"reached":true,"final":"已为您打开待缴账单"}\n'
  + '规则：\n'
  + '- element 必须是元素清单里的 n 序号（scroll/done/abort 可为 null）；fill/enter 只能作用于输入框。\n'
  + '- 需要查看屏幕下方更多内容时用 scroll（value=down/up），不要假设清单之外的内容。\n'
  + '- 资金类最终确认按钮（确认办理/确认还款/提交/确定等）只有在页面展示的内容与用户请求的参数完全一致时才可点击；不一致时不要确认，用 abort 在 final 里说明差异。\n'
  + '- reached=true 表示页面已展示用户请求要的结果（办理计划/待缴账单/还款回执/明细/产品页等），此时 action=done；不确定一律 false。\n'
  + '- 页面不支持该业务或无法完成时 abort 并解释。不要重复失败过且原因相同的动作。';

function decideUser(sess, elements, verdict, d3ok) {
  const L = [];
  L.push('【用户请求】' + sess.sentence);
  L.push('【意图理解】业务=' + sess.target.label + '(' + sess.target.intent + ') 槽位=' + JSON.stringify(sess.target.slots || {})
    + ' 目标=' + sess.target.goal_desc + ' 完成提示=' + sess.target.done_hint);
  L.push('【d3 截图判定】' + (d3ok && verdict
    ? '域=' + (verdict.domain || 'none') + ' 场景=' + (verdict.scene || 'none') + ' 页面' + (verdict.page_state ? '正常'
      : '异常（d3 视觉判定仅供参考，可能是内容稀疏的合法页面；以下方真实元素清单为准，若页面确实异常无法办理则 abort）')
    : '不可用（仅凭元素清单决策）'));
  L.push('【当前屏幕可交互元素】(n · 标签 · 文本)');
  elements.forEach(it => {
    L.push(it.n + '. ' + it.tag + (it.type ? '[' + it.type + ']' : '') + (it.disabled ? '[禁用]' : '') + ' 「' + it.text + '」');
  });
  if (sess.history.length) {
    L.push('【历史动作】');
    sess.history.slice(-4).forEach(h => L.push(h));
  }
  return L.join('\n');
}

/* ---------- 业务逻辑 ---------- */

async function navstart(body) {
  if (pwErr) return [503, { ok: false, detail: `playwright 不可用：${pwErr}` }];
  if (!GLM_KEY) return [503, { ok: false, detail: 'GLM 未配置（ROOT/glm_key.local.json 缺失），导航服务不可用' }];
  const sentence = String((body && body.sentence) || '').trim();
  if (!sentence) return [400, { ok: false, detail: 'sentence 为空' }];
  if (sessions.size >= MAX_SESSIONS) {
    return [429, { ok: false, detail: '导航会话已达上限（2），请 navclose 释放或等待 TTL 回收' }];
  }

  /* 1. 模型意图理解：一句话 → 意图 + 槽位 + 到达目标（替代关键词表，支持任意表述） */
  const parseSys = '你是手机银行 App 的意图理解模块。分析用户的一句话请求，严格只输出一个 JSON 对象（不要 markdown、不要解释）：\n'
    + '{"intent":"业务域标识","label":"业务名(中文2-6字)","slots":{...},"goal_desc":"到达什么页面/状态算完成(中文一句话)","done_hint":"页面出现什么内容算达成(中文一句话)"}\n'
    + '- intent 取值：demo.transfer(转账汇款) demo.utility(生活缴费) demo.account(账户查询) demo.wealth(理财) demo.loss_fraud(挂失止付) demo.complaint(投诉咨询)，都不像则 other。\n'
    + '- slots 抽取请求关键参数（收款人/金额/账单类型/产品名/卡号等），没有就 {}；金额统一写成「数字+元」。\n'
    + '- goal_desc/done_hint 必须贴合请求参数（如「给张三转100元」的目标是给张三转100元的转账办理计划，不是泛泛的转账页）。';
  const pr = await glmJson([
    { role: 'system', content: parseSys },
    { role: 'user', content: sentence },
  ]);
  if (!pr.ok) return [502, { ok: false, detail: 'GLM 意图理解失败：' + pr.err }];
  const j = pr.json || {};
  const target = {
    intent: String(j.intent || 'other'),
    label: String(j.label || '银行业务'),
    slots: (j.slots && typeof j.slots === 'object' && !Array.isArray(j.slots)) ? j.slots : {},
    goal_desc: String(j.goal_desc || ''),
    done_hint: String(j.done_hint || j.goal_desc || ''),
  };

  /* 2. 打开 mock 手机银行 */
  let browser, page;
  try { ({ browser, page } = await openBankPage()); }
  catch (e) {
    try { if (browser) await browser.close(); } catch (_) { /* noop */ }
    return [502, { ok: false, detail: `打开 mock 手机银行失败：${String((e && e.message) || e).slice(0, 200)}` }];
  }
  const id = crypto.randomBytes(8).toString('hex');
  const sess = { id, sentence, target, browser, page, step: 0, lastImg: null, sameCount: 0, failCount: 0, history: [], busy: false, timer: null };
  sessions.set(id, sess);
  touchTTL(id);
  console.log(`[nav] start ${id} → ${target.intent}「${sentence}」slots=${JSON.stringify(target.slots)} glm=${pr.ms}ms`);
  return [200, { ok: true, session_id: id, target, max_steps: MAX_STEPS, glm_ms: pr.ms, model: GLM_MODEL }];
}

async function navstep(body) {
  const id = String((body && body.session_id) || '');
  const sess = sessions.get(id);
  if (!sess) return [404, { ok: false, detail: '会话不存在或已过期（15 分钟 TTL）' }];
  if (sess.busy) return [409, { ok: false, detail: '上一步仍在处理中，请等待返回后重试' }];
  sess.busy = true;
  touchTTL(id);
  const t0 = Date.now();
  try {
    /* 1. 观察：截图 + d3 三题 + 元素提取（跳转过渡态自愈：零元素时等 1.2s 重观察一次） */
    const observe = async () => {
      const buf = await sess.page.locator('.phone').screenshot({ type: 'jpeg', quality: 70 });
      const img = 'data:image/jpeg;base64,' + buf.toString('base64');
      const d3 = await d3route(img);
      const verdict = d3.ok ? pickVerdict(d3.data && d3.data.routing) : null;
      const abnormal = !!(d3.ok && verdict && verdict.page_state === false);
      const elements = await listElements(sess.page);   /* d3 判异常也提取 DOM：异常可能只是视觉误判（如内容稀疏的合法页） */
      return { img, d3, verdict, abnormal, elements };
    };
    let obs = await observe();
    if (obs.elements.length === 0) {
      await sess.page.waitForTimeout(1200);   /* 可能只是页面跳转过渡态，等一拍再确认 */
      obs = await observe();
    }
    const img = obs.img;
    const same = sess.lastImg === img;
    sess.sameCount = same ? sess.sameCount + 1 : 0;
    sess.lastImg = img;
    const d3 = obs.d3, verdict = obs.verdict, abnormal = obs.abnormal, elements = obs.elements;
    /* 硬保护收窄：仅当 d3 判异常且 DOM 零元素（真白屏）才跳过决策；d3 异常但 DOM 有元素视为视觉误判，交模型判断 */

    /* 3. 模型决策：真实元素清单 → GLM 决定下一步（真白屏时跳过决策直接硬停） */
    const hardStop = abnormal && elements.length === 0;
    let glm = { ok: false, err: hardStop ? '页面异常跳过' : '未执行', ms: 0 };
    let dec = null;
    if (!hardStop) {
      const decideMsgs = [
        { role: 'system', content: DECIDE_SYS },
        { role: 'user', content: decideUser(sess, elements, verdict, d3.ok) },
      ];
      glm = await glmJson(decideMsgs);
      if (!glm.ok) glm = await glmJson(decideMsgs);   /* 整轮再试一次（glmJson 内部已含一次格式纠偏） */
      if (glm.ok) dec = glm.json || {};
    }

    /* 4. 执行模型动作（点击 / 槽位填参 / 回车 / 滚动） */
    const act = dec ? String(dec.action || '').toLowerCase() : '';
    const ACTS = ['click', 'fill', 'enter', 'scroll'];
    let execErr = null, execDesc = null;
    if (glm.ok && ACTS.includes(act)) {
      if (act === 'scroll') {
        const dir = String(dec.value || 'down') === 'up' ? -1 : 1;
        try {
          const box = await sess.page.locator('.phone-screen.active').boundingBox();
          if (box) await sess.page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
          await sess.page.mouse.wheel(0, dir * 400);
          execDesc = dir > 0 ? '向下滚动' : '向上滚动';
          await sess.page.waitForTimeout(600);
        } catch (e) { execErr = String((e && e.message) || e).slice(0, 160); }
      } else {
        const el = elements.find(x => x.n === Number(dec.element));
        if (!el) execErr = '元素序号 ' + dec.element + ' 不在清单内';
        else if ((act === 'fill' || act === 'enter') && el.tag !== 'input' && el.tag !== 'textarea') execErr = '目标不是输入框';
        else if (el.disabled) execErr = '元素已禁用';
        else {
          try {
            if (act === 'click') {
              await sess.page.locator(SEL).nth(el.n).click({ timeout: 5000 });
              execDesc = '点击「' + el.text + '」';
            } else if (act === 'fill') {
              const v = String(dec.value == null ? '' : dec.value).slice(0, 60);
              await sess.page.locator(SEL).nth(el.n).fill(v, { timeout: 5000 });
              execDesc = '填写「' + el.text + '」= ' + (v || '（清空）');
            } else {
              await sess.page.locator(SEL).nth(el.n).press('Enter', { timeout: 5000 });
              execDesc = '在「' + el.text + '」回车';
            }
            await sess.page.waitForTimeout(1000);
          } catch (e) { execErr = String((e && e.message) || e).slice(0, 160); }
        }
      }
    }

    /* 5. 终止判定（模型 reached 为准 + 安全网） */
    sess.step += 1;
    const step = sess.step;
    const reached = !!(dec && dec.reached === true);
    let done = false, stop = null;
    if (hardStop) { done = true; stop = 'abnormal'; }   /* d3 判异常且 DOM 零元素：真白屏，硬停优先于模型失败，语义不混淆 */
    else if (!glm.ok) { done = true; stop = 'model_error'; }
    else if (act === 'done') { done = true; stop = reached ? 'arrived' : 'gave_up'; }
    else if (act === 'abort') { done = true; stop = 'model_abort'; }
    else if (!ACTS.includes(act)) { done = true; stop = 'model_error'; }
    else if (execErr) {
      sess.failCount += 1;
      if (sess.failCount >= 3) { done = true; stop = 'act_failed'; }
    } else sess.failCount = 0;
    if (!done && sess.sameCount >= 3) { done = true; stop = 'no_progress'; }
    if (!done && step >= MAX_STEPS) { done = true; stop = 'max_steps'; }
    const match = done && stop === 'arrived';

    /* 6. 轨迹记录（喂给下一步模型） */
    const hLine = !glm.ok
      ? ('S' + step + ' 模型输出无效（' + (glm.err || 'GLM 失败') + '）')
      : ('S' + step + ' ' + (execDesc || (act ? act + '（' + ((dec && dec.thought) || '') + '）' : '无动作'))
        + (execErr ? ' → 失败：' + execErr : ' → 成功'));
    sess.history.push(hLine);

    console.log(`[nav] step ${step} ${sess.id} d3=${d3.ok ? ((verdict && verdict.label) || 'ok') : `降级(${d3.err})`}`
      + ` elems=${elements.length} glm=${glm.ok ? glm.ms + 'ms' : '失败(' + glm.err + ')'}`
      + ` → ${execDesc || stop || act || '?'}${done ? ' · ' + stop : ''}`);

    return [200, {
      ok: true, step, max_steps: MAX_STEPS,
      done, match, stop_reason: stop,
      state_name: (verdict && verdict.scene && verdict.scene !== 'none') ? verdict.scene : (d3.ok ? 'unrecognized' : '页面'),
      target: sess.target,
      observe: {
        image: img, verdict,
        degraded: !d3.ok, d3_ms: d3.ms, d3_error: d3.ok ? null : d3.err,
        elements: elements.length,
      },
      action: (execDesc || execErr) ? { desc: execDesc || ('执行失败'), error: execErr || undefined } : null,
      model: { glm_ms: glm.ms, thought: (dec && dec.thought) || null, act: act || null, glm_error: glm.ok ? null : glm.err },
      final: (match || stop === 'gave_up' || stop === 'model_abort') && dec && dec.final ? dec.final : undefined,
      elapsed_ms: Date.now() - t0,
    }];
  } finally {
    sess.busy = false;
  }
}

async function navclose(body) {
  const id = String((body && body.session_id) || '');
  const existed = sessions.has(id);
  await closeSession(id);
  return [200, { ok: true, closed: existed }];
}

/* ---------- HTTP 服务 ---------- */

function setCORS(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-JEV-Token');
}

function json(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', c => {
      raw += c;
      if (raw.length > 2 * 1024 * 1024) { reject(new Error('请求体超限')); req.destroy(); }
    });
    req.on('end', () => resolve(raw));
    req.on('error', reject);
  });
}

async function handle(req, res) {
  const path = req.url.split('?')[0];
  setCORS(res);
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  if (req.method === 'GET' && path === '/v1/navhealthz') {
    return json(res, 200, {
      ok: !pwErr && !!GLM_KEY,
      error: pwErr || (!GLM_KEY ? 'GLM 未配置（ROOT/glm_key.local.json 缺失）' : null),
      sessions: sessions.size, max_sessions: MAX_SESSIONS,
      bank: BANK_URL, d3: D3_BASE, edge: EDGE_PATH, max_steps: MAX_STEPS,
      glm: !!GLM_KEY, model: GLM_MODEL,
    });
  }

  if (req.method === 'POST') {
    const raw = await readBody(req);
    let body = {};
    try { body = raw ? JSON.parse(raw) : {}; } catch (e) { return json(res, 400, { ok: false, detail: '请求体不是合法 JSON' }); }
    if (path === '/v1/navstart') { const [s, o] = await navstart(body); return json(res, s, o); }
    if (path === '/v1/navstep') { const [s, o] = await navstep(body); return json(res, s, o); }
    if (path === '/v1/navclose') { const [s, o] = await navclose(body); return json(res, s, o); }
  }
  json(res, 404, { ok: false, detail: '未知路径，可用：GET /v1/navhealthz；POST /v1/navstart|/v1/navstep|/v1/navclose' });
}

const server = http.createServer((req, res) => {
  handle(req, res).catch(e => {
    console.error('[nav] 500:', e && (e.stack || e.message || e));
    try { json(res, 500, { ok: false, detail: `服务内部错误：${String((e && e.message) || e).slice(0, 300)}` }); }
    catch (_) { /* 响应已发出 */ }
  });
});

server.listen(PORT, HOST, () => {
  console.log(`[nav] serve_nav v2（模型驾驶）已启动 http://${HOST}:${PORT}`);
  console.log(`[nav]   bank=${BANK_URL}`);
  console.log(`[nav]   d3=${D3_BASE}  edge=${EDGE_PATH}`);
  console.log(`[nav]   glm=${GLM_MODEL} key=${GLM_KEY ? '已加载' : '缺失!'}`);
  if (pwErr) console.log(`[nav]   !! playwright 加载失败：${pwErr}`);
  console.log(`[nav]   会话上限 ${MAX_SESSIONS} · TTL ${SESSION_TTL / 60000} 分钟 · max_steps ${MAX_STEPS}`);
});
