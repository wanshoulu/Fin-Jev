'use strict';
/* serve_nav.js — 连续导航服务（第 6 应用「多模态截图路由」行动环后端）
 *
 * 接通行动环：一句话 → 目标锚点 → observe·act 循环
 *   截图（headless Edge 打开 mock 手机银行）→ d3 三题判定（:8788）
 *   → 路由裁决 → 按「当前页指纹 × 目标」策略表点击 → 再截图 → 到达目标页
 *
 * 运行： node server/serve_nav.js          （依赖本机 :3000 mock 银行 + :8788 serve_d3）
 * 端口： 127.0.0.1:8789（jev_proxy 将 /v1/nav* 转发至此；服务器无 GPU/内存受限，仅本地跑）
 * 接口：
 *   GET  /v1/navhealthz   就绪探测 {ok, sessions, intents}
 *   POST /v1/navstart     {sentence} → {ok, session_id, target, max_steps}
 *   POST /v1/navstep      {session_id} → {ok, step, done, match, state, observe:{image, verdict}, action, stop_reason}
 *   POST /v1/navclose     {session_id} → {ok}
 * 会话：内存 Map，上限 2 个，15 分钟 TTL 自动回收；每会话一个 headless Edge 实例。
 * 环境变量：NAV_PORT / NAV_D3_BASE / NAV_BANK_URL / NAV_EDGE / NAV_PW_MODULE
 */

const http = require('http');
const crypto = require('crypto');

const HOST = '127.0.0.1';
const PORT = Number(process.env.NAV_PORT || 8789);
const D3_BASE = process.env.NAV_D3_BASE || 'http://127.0.0.1:8788';
const BANK_URL = process.env.NAV_BANK_URL || 'http://localhost:3000/tools/ai-native-bank.html';
const EDGE_PATH = process.env.NAV_EDGE || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const PW_MODULE = process.env.NAV_PW_MODULE || 'D:/llm-training/multimodal-jev/node_modules/playwright-core';
const MAX_STEPS = 8;
const MAX_SESSIONS = 2;
const SESSION_TTL = 15 * 60 * 1000;
const D3_TIMEOUT = 150000;

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

/* ---------- 目标锚点：一句话 → 演示意（5 条确定性关键词映射） ---------- */

const INTENTS = [
  {
    key: 'transfer', kw: ['转账', '张伟', '汇款'], goal: 'transfer.plan',
    label: '转账汇款', desc: '给张伟转账 1000 元（生成任务计划待本人确认）', leaf: 'demo.transfer.peer',
  },
  {
    key: 'utility', kw: ['缴费', '燃气', '话费', '水电', '交费'], goal: 'utility.bills',
    label: '生活缴费', desc: '查看待缴账单（燃气费/话费）', leaf: 'demo.utility.pay',
  },
  {
    key: 'account', kw: ['余额', '账户', '查账', '明细'], goal: 'account',
    label: '账户查询', desc: '查看工资卡余额与账户明细', leaf: 'demo.account.query',
  },
  {
    key: 'wealth', kw: ['理财', '龙钱宝', '财富', '收益'], goal: 'wealth',
    label: '理财', desc: '进入财富频道查看龙钱宝', leaf: 'demo.wealth.buy',
  },
  {
    key: 'loss', kw: ['挂失', '丢卡', '信用卡'], goal: 'card.manage',
    label: '挂失止付', desc: '进入卡片管理挂失入口（域级判定）', leaf: 'demo.loss_fraud',
  },
];

function parseIntent(sentence) {
  const text = String(sentence || '');
  const hit = INTENTS.find(t => t.kw.some(k => text.includes(k)));
  if (!hit) return null;
  return {
    key: hit.key, label: hit.label, desc: hit.desc, leaf: hit.leaf,
    goal: hit.goal, goal_name: STATE_NAMES[hit.goal] || hit.goal,
  };
}

/* ---------- 页面状态机：DOM 指纹（确定性导航依据） ---------- */

const STATE_NAMES = {
  'home': 'App 首页宫格',
  'transfer.entry': '转账汇款入口页',
  'transfer.plan': '转账任务计划（智能服务屏）',
  'utility.entry': '生活缴费服务页',
  'utility.bills': '待缴账单列表',
  'wealth': '财富频道（龙钱宝）',
  'credit.channel': '信用卡频道',
  'card.manage': '卡片管理（挂失入口）',
  'account': '账户查询（工资卡总览）',
  'unknown': '未识别页面',
};

function classify(fp) {
  if (fp.screen === 'assistant') return 'transfer.plan';
  if (fp.screen === 'classic-home') return 'home';
  if (fp.screen === 'feature') {
    const f = fp.feat;
    if (f.includes('Mock 初始数据')) return 'utility.bills';
    if (f.includes('最近收款人')) return 'transfer.entry';
    if (f.includes('服务内容') && f.includes('卡片管理')) return 'card.manage';
    if (f.includes('服务内容') && f.includes('生活缴费')) return 'utility.entry';
    if (f.includes('活钱管理')) return 'wealth';
    if (f.includes('龙卡消费享积分')) return 'credit.channel';
    if (f.includes('工资卡 · 尾号')) return 'account';
  }
  return 'unknown';
}

/* ---------- 点击策略表：[当前页指纹][目标意] → 动作（选择器已实测） ---------- */

const ACTIONS = {
  'home': {
    transfer: { sel: '#classicHome [data-open-feature="transfer"]', desc: '点击首页「转账汇款」宫格入口', wait: 800 },
    utility: { sel: '#classicHome [data-catalog-action="home:utility"]', desc: '点击首页「生活缴费」宫格入口', wait: 800 },
    account: { sel: '#classicHome [data-open-feature="account"]', desc: '点击首页「账户查询」入口', wait: 800 },
    wealth: { sel: '#classicHome [data-bottom-nav="wealth"]', desc: '点击底部导航「财富」频道', wait: 900 },
    loss: { sel: '#classicHome [data-bottom-nav="credit"]', desc: '点击底部导航「信用卡」频道', wait: 900 },
  },
  'credit.channel': {
    loss: { sel: '[data-catalog-action="credit:card-manage"]', desc: '点击「卡片管理」服务', wait: 1000 },
  },
  'transfer.entry': {
    transfer: { sel: '#featureContent [data-task-query="给张伟转账1000元"]', desc: '点击「向张伟转账 ¥1,000」进入任务计划', wait: 2600 },
  },
  'utility.entry': {
    utility: { sel: '#featureContent [data-feature-action="home:utility:overview"]', desc: '点击「待缴账单」二级服务', wait: 900 },
  },
};

function planAction(state, intentKey) {
  const tbl = ACTIONS[state];
  return (tbl && tbl[intentKey]) || null;
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

/* ---------- d3 三题判定（内部直连 :8788，不经 jev_proxy） ---------- */

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

/* ---------- 页面操作 ---------- */

async function fingerprint(page) {
  return page.evaluate(() => {
    const act = document.querySelector('.phone-screen.active');
    const tx = s => { const el = document.querySelector(s); return el ? (el.textContent || '') : ''; };
    return {
      screen: act ? (act.dataset.phoneScreen || '') : '',
      title: tx('#featureTitle'),
      feat: tx('#featureContent').slice(0, 3000),
    };
  });
}

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

/* ---------- 业务逻辑 ---------- */

async function navstart(body) {
  if (pwErr) return [503, { ok: false, detail: `playwright 不可用：${pwErr}` }];
  const sentence = String((body && body.sentence) || '').trim();
  if (!sentence) return [400, { ok: false, detail: 'sentence 为空' }];
  const target = parseIntent(sentence);
  if (!target) {
    return [400, { ok: false, detail: '未识别意图。演示支持：' + INTENTS.map(t => `${t.label}（含「${t.kw.join('/')}/…」）`).join('；') }];
  }
  if (sessions.size >= MAX_SESSIONS) {
    return [429, { ok: false, detail: '导航会话已达上限（2），请 navclose 释放或等待 TTL 回收' }];
  }
  let browser, page;
  try { ({ browser, page } = await openBankPage()); }
  catch (e) {
    try { if (browser) await browser.close(); } catch (_) { /* noop */ }
    return [502, { ok: false, detail: `打开 mock 手机银行失败：${String((e && e.message) || e).slice(0, 200)}` }];
  }
  const id = crypto.randomBytes(8).toString('hex');
  const sess = { id, target, browser, page, step: 0, lastImg: null, sameCount: 0, busy: false, timer: null };
  sessions.set(id, sess);
  touchTTL(id);
  console.log(`[nav] start ${id} → ${target.key}「${sentence}」`);
  return [200, { ok: true, session_id: id, target, max_steps: MAX_STEPS }];
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
    /* 1. 观察：截图 */
    const buf = await sess.page.locator('.phone').screenshot({ type: 'jpeg', quality: 90 });
    const img = 'data:image/jpeg;base64,' + buf.toString('base64');
    const same = sess.lastImg === img;
    sess.sameCount = same ? sess.sameCount + 1 : 0;
    sess.lastImg = img;

    /* 2. 多模态感知：d3 三题判定（不可用则降级继续导航） */
    const d3 = await d3route(img);
    const verdict = d3.ok ? pickVerdict(d3.data && d3.data.routing) : null;

    /* 3. 状态识别：DOM 指纹（确定性） */
    const fp = await fingerprint(sess.page);
    const state = classify(fp);

    /* 4. 到达判定（确定性：指纹 == 目标页） */
    const reached = state === sess.target.goal;
    const abnormal = !!(d3.ok && verdict && verdict.page_state === false);

    /* 5. 行动规划：策略表 [state][intent] */
    let plan = reached || abnormal ? null : planAction(state, sess.target.key);
    let actErr = null;
    if (plan) {
      try {
        await sess.page.locator(plan.sel).first().click({ timeout: 10000 });
        await sess.page.waitForTimeout(plan.wait || 700);
      } catch (e) {
        actErr = String((e && e.message) || e).slice(0, 200);
      }
    }
    sess.step += 1;
    const step = sess.step;

    /* 6. 终止判定 */
    let done = false;
    let stop_reason = null;
    if (reached) { done = true; stop_reason = 'arrived'; }
    else if (abnormal) { done = true; stop_reason = 'abnormal'; }
    else if (actErr) { done = true; stop_reason = 'act_failed'; }
    else if (!plan) { done = true; stop_reason = 'no_route'; }
    else if (sess.sameCount >= 2) { done = true; stop_reason = 'no_progress'; }
    else if (step >= MAX_STEPS) { done = true; stop_reason = 'max_steps'; }

    console.log(`[nav] step ${step} ${sess.id} state=${state} d3=${d3.ok ? `${verdict ? verdict.label : 'ok'}` : `降级(${d3.err})`} → ${plan ? plan.desc : '停'}${done ? ` · ${stop_reason}` : ''}`);
    return [200, {
      ok: true, step, max_steps: MAX_STEPS,
      state, state_name: STATE_NAMES[state] || state,
      done, match: reached, stop_reason,
      target: sess.target,
      observe: {
        image: img, verdict,
        degraded: !d3.ok, d3_ms: d3.ms, d3_error: d3.ok ? null : d3.err,
      },
      action: plan ? { desc: plan.desc, error: actErr || undefined } : null,
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
      ok: !pwErr, error: pwErr, sessions: sessions.size, max_sessions: MAX_SESSIONS,
      bank: BANK_URL, d3: D3_BASE, edge: EDGE_PATH, max_steps: MAX_STEPS,
      intents: INTENTS.map(t => ({ key: t.key, label: t.label })),
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
  console.log(`[nav] serve_nav 已启动 http://${HOST}:${PORT}`);
  console.log(`[nav]   bank=${BANK_URL}`);
  console.log(`[nav]   d3=${D3_BASE}  edge=${EDGE_PATH}`);
  if (pwErr) console.log(`[nav]   !! playwright 加载失败：${pwErr}`);
  console.log(`[nav]   会话上限 ${MAX_SESSIONS} · TTL ${SESSION_TTL / 60000} 分钟 · max_steps ${MAX_STEPS}`);
});
