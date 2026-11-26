# -*- coding: utf-8 -*-
r"""
serve_d3.py — 多模态 JEV 推理服务（第 6 应用「截图路由测试台」后端）

接收手机银行 App 截图（内置截图名或上传 base64），按 mmjev schema 出三题
（业务域 choice / 办理场景 choice / 页面状态 noul），d3-lite 多模态
SystemOne 并行判定，映射 bank_graph intent id 并给出操作指引。

运行:  python server/serve_d3.py     （先启动本服务，再启动 jev_proxy.py）
端口:  127.0.0.1:8788（jev_proxy 把 /v1/d3* 转发到这里）
接口:  GET  /v1/d3healthz   就绪探测 {ok, device, ready, adapter, load_ms}
       POST /v1/d3route     {shot|image, name?} → {answers, routing, ms}
模型:  D3_MODEL_DIR 环境变量，默认 D:\llm-training\models\d3-lite
适配器:D3_ADAPTER_DIR 可选挂载 LoRA（默认不挂；adapter-v0 文本训练已废弃）

出题模板与 multimodal-jev/build_dataset.py 逐字一致（改一处须同步另一处）。
"""
import base64
import binascii
import json
import os
import pathlib
import re
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = int(os.environ.get("D3_PORT", "8788"))
MODEL_DIR = os.environ.get("D3_MODEL_DIR", r"D:\llm-training\models\d3-lite")
ADAPTER_DIR = os.environ.get("D3_ADAPTER_DIR", "")
SHOTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "demos" / "screenshots"
MAX_BODY = 12 * 1024 * 1024

# ---- 出题模板（与 build_dataset.py QUESTION_TEMPLATES 保持一致）----
DOMAIN_NAMES = {
    "demo.transfer": "转账汇款",
    "demo.utility": "生活缴费",
    "demo.wealth": "理财",
    "demo.account": "账户查询",
    "demo.loss_fraud": "挂失止付",
    "demo.complaint": "投诉纠纷",
}
LEAF_NAMES = {
    "demo.transfer.peer": "转账汇款",
    "demo.utility.pay": "生活缴费",
    "demo.wealth.buy": "理财申购",
    "demo.account.query": "账户查询",
    "demo.loss_fraud.protect": "挂失止付",
    "demo.complaint.emotion": "投诉纠纷",
}
QUESTIONS = {
    "domain": {"type": "choice", "instructions": "这张截图属于手机银行App的哪个业务域？",
               "criteria": dict(DOMAIN_NAMES)},
    "scene": {"type": "choice", "instructions": "该截图对应的办理场景是？",
              "criteria": {"none": "不是具体办理场景（如首页或入口页）", **LEAF_NAMES}},
    "page_state": {"type": "noul",
                   "instructions": "该页面是否处于已加载出业务内容的状态（而非错误提示或空白占位）？"},
}
STATE = "客户在手机银行App客服消息里发送了一张截图。请判断该业务属于哪个域、哪个办理场景，并核验页面当前状态。"

# ---- 路由表：intent id → 操作指引（客服引导口径）----
GUIDES = {
    "demo.transfer.peer": ["首页 → 转账/汇款 → 选择收款人（或录入新收款人）",
                           "输入金额 → 选付款账户 → 核对费率与到账时效",
                           "确认付款 → 短信验证码 → 完成后留存电子回单"],
    "demo.utility.pay": ["首页 → 生活缴费 → 选择缴费项目（燃气/话费/水电）",
                         "确认户号 → 核对待缴账单与金额",
                         "确认缴费 → 完成支付并留存凭证"],
    "demo.wealth.buy": ["首页 → 投资理财 → 选择产品（如龙钱宝）",
                        "阅读产品说明书与风险提示 → 完成风险测评",
                        "输入申购金额 → 确认买入"],
    "demo.account.query": ["首页 → 我的账户 → 账户总览",
                           "查看余额与最近交易明细（可导出流水）"],
    "demo.loss_fraud.protect": ["先安抚客户，确认卡片是否本人持有、资金有无异动",
                                "我的 → 银行卡管理 → 挂失/止付（紧急可先电话口头挂失）",
                                "提示可疑交易申诉流程与防诈注意事项"],
    "demo.complaint.emotion": ["先安抚情绪 → 记录诉求工单",
                               "明确处理时限 → 转人工坐席跟进"],
}
DOMAIN_GUIDES = {
    "demo.transfer": ["首页 → 转账汇款 入口", "选择转账类型（同行/跨行/手机号转账）"],
    "demo.utility": ["首页 → 生活缴费 入口", "选择缴费城市与项目"],
    "demo.wealth": ["首页 → 投资理财 入口", "浏览产品列表或我的持仓"],
    "demo.account": ["首页 → 我的账户 入口", "选择要查看的卡/账户"],
    "demo.loss_fraud": ["我的 → 银行卡管理 → 挂失与安全"],
    "demo.complaint": ["我的 → 意见反馈，或在客服会话中直接表达诉求"],
}
HOME_GUIDE = ["识别为 App 首页/门户：客户尚未进入具体业务",
              "询问客户要办理的业务，按关键词引导点击对应宫格入口"]
ABNORMAL_GUIDE = ["页面加载异常（报错卡片/空白占位）：先请客户退出页面重新进入",
                  "重进仍失败 → 建议稍后再试或转人工受理，避免客户反复重试"]

CONF_MIN = 0.45  # 路由置信度下限，低于则建议人工确认

# ---- d3 模型（后台线程加载，加载完成前置 READY）----
_D3 = None
_D3_ERR = None
_READY = threading.Event()
_T0 = time.time()
_gpu_lock = threading.Lock()


def _load_model():
    global _D3, _D3_ERR
    try:
        sys.path.insert(0, MODEL_DIR)
        import torch  # noqa: E402
        from d3_runtime import D3  # noqa: E402
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[serve_d3] 加载 {MODEL_DIR} → {device} ...", flush=True)
        _D3 = D3(MODEL_DIR, device=device)
        if ADAPTER_DIR and (pathlib.Path(ADAPTER_DIR) / "adapter_config.json").exists():
            from peft import PeftModel
            _D3.backbone = PeftModel.from_pretrained(_D3.backbone, ADAPTER_DIR)
            _D3.backbone.eval()
            print(f"[serve_d3] 已挂载 LoRA adapter: {ADAPTER_DIR}", flush=True)
        print(f"[serve_d3] 模型就绪（{_D3.loaded_seconds:.1f}s, device={_D3.device}）", flush=True)
        _READY.set()
    except Exception as exc:  # noqa: BLE001
        _D3_ERR = f"{type(exc).__name__}: {exc}"
        print(f"[serve_d3] 模型加载失败: {_D3_ERR}")


threading.Thread(target=_load_model, daemon=True).start()


def _fit_image(path):
    """降采样到最长边 960 后另存临时文件（与训练同分布、省显存提速）。

    返回新路径；无需缩放时返回 None（沿用原图）。调用方负责删除返回的临时文件。
    """
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 960 / max(w, h)
    if s >= 1:
        return None
    im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    fd, out = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    im.save(out, "JPEG", quality=90)
    return out


def _decode_image(payload):
    """dataURL 或裸 base64 → 临时文件路径。"""
    if not isinstance(payload, str) or not payload:
        raise ValueError("image 为空")
    data = re.sub(r"^data:image/\w+;base64,", "", payload.strip())
    raw = base64.b64decode(data, validate=True)
    if len(raw) < 64:
        raise ValueError("image 内容过短")
    suffix = ".png" if payload.startswith("data:image/png") else ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="mmjev-")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def _argmax(probs):
    key = max(probs, key=probs.get)
    return key, float(probs[key])


def _route(answers):
    """三题判定 → 路由裁决（action / intent_id / guides / conf）。"""
    dom, dom_p = _argmax(answers["domain"]["probabilities"])
    scene, scene_p = _argmax(answers["scene"]["probabilities"])
    ps = float(answers["page_state"]["noul"])
    page_ok = ps > 0.5
    leaf = scene if scene != "none" else None
    intent_id = leaf or dom if dom != "none" else None

    if not page_ok:
        action, label = "abnormal", "页面异常 · 转异常处理"
        guides = ABNORMAL_GUIDE
        conf = ps  # noul 越低越异常，用异常侧概率做置信
    elif leaf:
        action, label = "route", "已路由到办理指引"
        guides = GUIDES.get(leaf, [])
        conf = scene_p
    elif dom != "none":
        action, label = "route", "域级入口指引"
        guides = DOMAIN_GUIDES.get(dom, [])
        conf = dom_p
    else:
        action, label = "home", "首页导航"
        guides = HOME_GUIDE
        conf = scene_p

    low_conf = page_ok and conf < CONF_MIN
    if low_conf:
        action, label = "confirm", "低置信 · 建议人工确认"
    return {
        "action": action, "label": label,
        "domain": None if dom == "none" else dom,
        "scene": leaf, "intent_id": intent_id,
        "page_state": page_ok,
        "conf": round(min(conf, 1.0), 4),
        "low_conf": low_conf,
        "guides": guides,
        "picks": {"domain": [dom, dom_p], "scene": [scene, scene_p], "page_state": ps},
    }


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/d3healthz":
            self._json(200, {
                "ok": _READY.is_set() and _D3_ERR is None,
                "ready": _READY.is_set(), "error": _D3_ERR,
                "model": MODEL_DIR, "adapter": ADAPTER_DIR or None,
                "device": str(_D3.device) if _D3 else None,
                "load_ms": int((time.time() - _T0) * 1000),
                "shots_dir": str(SHOTS_DIR),
            })
            return
        self._json(404, {"detail": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/v1/d3route":
            self._json(404, {"detail": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json(413, {"detail": "payload too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:  # noqa: BLE001
            self._json(400, {"detail": "bad json"})
            return

        if not _READY.is_set():
            if _D3_ERR:
                self._json(500, {"detail": f"model load failed: {_D3_ERR}"})
            else:
                self._json(503, {"detail": "model loading, retry later"})
            return

        tmp_path = None
        fit_tmp = None
        try:
            if body.get("shot"):
                name = pathlib.Path(body["shot"]).name  # 防目录穿越，只取文件名
                fpath = SHOTS_DIR / name
                if not fpath.is_file():
                    self._json(400, {"detail": f"unknown shot: {name}"})
                    return
                img_path = str(fpath)
            elif body.get("image"):
                tmp_path = img_path = _decode_image(body["image"])
            else:
                self._json(400, {"detail": "需要 shot（内置截图名）或 image（base64）"})
                return

            fit_tmp = _fit_image(img_path)
            if fit_tmp:
                img_path = fit_tmp

            t0 = time.time()
            with _gpu_lock:
                result = _D3.system_one(state=STATE, questions=QUESTIONS, images=[img_path])
            ms = int((time.time() - t0) * 1000)
            answers = result.get("answers", {})
            if not all(k in answers for k in QUESTIONS):
                self._json(502, {"detail": f"模型返回缺题: {sorted(answers)}", "raw": result})
                return
            routing = _route(answers)
            print(f"[serve_d3] route {body.get('shot') or 'image:' + str(len(body.get('image') or '')) + 'B'} "
                  f"ms={ms} → {routing.get('label')}（domain={routing.get('domain')} "
                  f"scene={routing.get('scene')} ps={routing.get('page_state')} conf={routing.get('conf')}）",
                  flush=True)
            self._json(200, {
                "ok": True, "ms": ms, "model": pathlib.Path(MODEL_DIR).name,
                "adapter": pathlib.Path(ADAPTER_DIR).name if ADAPTER_DIR else None,
                "state": STATE, "questions": QUESTIONS,
                "answers": answers, "routing": routing,
            })
        except (binascii.Error, ValueError) as exc:
            self._json(400, {"detail": f"bad image: {exc}"})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"detail": f"{type(exc).__name__}: {exc}"})
        finally:
            for extra in (fit_tmp,):
                if extra:
                    try:
                        os.remove(extra)
                    except OSError:
                        pass
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def log_message(self, *a):  # 安静模式
        pass


if __name__ == "__main__":
    print(f"[serve_d3] listening on http://{HOST}:{PORT}  (POST /v1/d3route, GET /v1/d3healthz)")
    print(f"[serve_d3] screenshots dir: {SHOTS_DIR}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
