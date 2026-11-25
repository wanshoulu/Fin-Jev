# -*- coding: utf-8 -*-
"""
jev_proxy.py — 本地一体化服务器（静态页面 + Jev API 同源转发）
解决浏览器直调 https://jev.bocha.cn 遇到的 CORS / 系统代理拦截问题。

运行:  python server/jev_proxy.py   （仓库根目录执行；平面布局直接 python jev_proxy.py）
访问:  http://127.0.0.1:8787
页面与接口同源（无 CORS），转发由 Python 直连完成（绕过系统代理）。

R52/R53 安全加固（2026-11-21，部署版：保留本机现网全部机制，叠加 token 闸门）：
- API 转发通道（/v1/systemone、/v1/glm、/v1/recompile、/v1/vkb/*）在既有 host白名单
  + 限流之上，新增 X-Workbench-Token 校验（不符 403）。
- token 首次启动自动生成存 ROOT/token.local.json，经同源 GET /token.txt 下发
  （每 IP 2次/60秒限流 + Sec-Fetch-Site: cross-site 拒绝）。
- 前端页面已注入补丁：自动 fetch /token.txt 并给同源 /v1/ 请求附带 token 头。
- 环境变量 JEV_TOKEN 可显式指定 token；token 文件不可写时跳过校验（行为同旧版）。
"""
import json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT = os.environ.get("JEV_HOST", "127.0.0.1"), int(os.environ.get("JEV_PORT", "8787"))
API_URL = "https://jev.bocha.cn/v1/systemone"
# R55：Jev 通道上游可切换——JEV_UPSTREAM 环境变量指向自建 kev serve（Tailscale 内网），
# 如 JEV_UPSTREAM=http://100.79.155.126:8011/v1/systemone；留空=官方博查端点（默认）
JEV_UPSTREAM = os.environ.get("JEV_UPSTREAM", "").strip() or API_URL
# 布局自适应：仓库布局（本文件在 server/，demos/ 与图谱编译链在上级目录）
# 或平面布局（页面/图谱/脚本与本文件同目录）均可运行——同一份代码两种落位
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE) if os.path.isdir(os.path.join(_HERE, "..", "demos")) else _HERE

# 图谱编译链（R46d 一键重编译）：改 bank_graph.json 存盘后页面点按钮即生效
GRAPH_COMPILER = os.path.join(ROOT, "graph_to_probes.py")

# OntoMind 向量知识库（local_rag）：意图语义索引库，供页面做运行期向量召回
VKB_PREFIX = "/v1/vkb/"
VKB_BASE = "http://127.0.0.1:17386/ontomind/api/v1/vector-kb/"

# 多模态 JEV 推理服务（serve_d3.py）：第 6 应用「截图路由测试台」后端，/v1/d3* 前缀转发
D3_PREFIX = "/v1/d3"
D3_BASE = os.environ.get("D3_BASE", "http://127.0.0.1:8788")

# GLM 配置：优先读 glm_key.local.json（已验证可用的 coding 端点 + Key）
_glm_cfg_path = os.path.join(ROOT, "glm_key.local.json")
GLM_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
GLM_KEY = ""
if os.path.isfile(_glm_cfg_path):
    try:
        with open(_glm_cfg_path, encoding="utf-8") as _f:
            _cfg = json.load(_f)
        GLM_URL = _cfg.get("url", GLM_URL)
        GLM_KEY = _cfg.get("apiKey", "")
    except Exception:
        pass

# Jev Key 可选注入：把 Key 放 ROOT/jev_key.local.json（gitignore 已排除），
# 前端无 Key 发出的请求（Bearer BUILTIN_JEV）由服务器替换——本机开发免配置
_jev_cfg_path = os.path.join(ROOT, "jev_key.local.json")
JEV_KEY = ""
if os.path.isfile(_jev_cfg_path):
    try:
        with open(_jev_cfg_path, encoding="utf-8") as _f:
            JEV_KEY = json.load(_f).get("apiKey", "")
    except Exception:
        pass

# 强制直连，绕过系统代理（与 jev_finance_poc.py 同款处理，已实测可用）
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# GLM 转发走系统默认网络（智谱无代理问题时的直连 opener）
glm_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

ALLOWED_EXT = {".html", ".js", ".css", ".svg", ".png", ".ico", ".jpg", ".jpeg", ".webp"}
# .json 只放行图谱编译产物（防止公网直接下载 jev_key.local.json 等敏感文件）
ALLOWED_JSON = "probes.compiled.json"

# BUILTIN Key 注入开关：前端无 Key 时发 Bearer BUILTIN_JEV / Bearer BUILTIN 占位符，
# 服务器替换为本地 Key（公网部署不想开内置 Key 时设 JEV_BUILTIN=0 / GLM_BUILTIN=0）
BUILTIN_JEV_ENABLED = os.environ.get("JEV_BUILTIN", "1") != "0"
BUILTIN_GLM_ENABLED = os.environ.get("GLM_BUILTIN", "1") != "0"

# ---- R52：访问令牌 ----
# 优先级：环境变量 JEV_TOKEN > ROOT/token.local.json > 首次启动自动生成
#（token.local.json 属 *.local.json 家族，仓库 .gitignore 已排除）。
# 校验仅针对 API 转发通道；静态页面与 /token.txt 自身不拦。
ACCESS_TOKEN = os.environ.get("JEV_TOKEN", "")
_token_cfg_path = os.path.join(ROOT, "token.local.json")
if not ACCESS_TOKEN and os.path.isfile(_token_cfg_path):
    try:
        with open(_token_cfg_path, encoding="utf-8") as _f:
            ACCESS_TOKEN = json.load(_f).get("token", "")
    except Exception:
        pass
if not ACCESS_TOKEN:
    ACCESS_TOKEN = secrets.token_urlsafe(24)
    try:
        with open(_token_cfg_path, "w", encoding="utf-8") as _f:
            json.dump({"token": ACCESS_TOKEN}, _f, ensure_ascii=False, indent=2)
        print("R52: 生成访问令牌 → %s（重启不变；JEV_TOKEN 环境变量可覆盖）" % _token_cfg_path)
    except Exception:
        print("R52: 令牌文件不可写，本次运行跳过令牌校验（行为同旧版）")
        ACCESS_TOKEN = ""

# R53：/token.txt 下发端点专用限流（每 IP 2 次/60 秒）——
# 页面启动只拉 1 次，刷新不误伤；防脚本高频拉 token 配合限流窗口打满上游配额
TOKEN_FETCH_N = 20
_TOKEN_FETCH_LOCK = threading.Lock()
_TOKEN_FETCH_HITS = {}


def _token_fetch_ok(ip):
    now = time.time()
    ts = _TOKEN_FETCH_HITS.get(ip, [])
    ts = [t for t in ts if now - t < 60]
    if len(ts) >= TOKEN_FETCH_N:
        _TOKEN_FETCH_HITS[ip] = ts
        return False
    ts.append(now)
    _TOKEN_FETCH_HITS[ip] = ts
    if len(_TOKEN_FETCH_HITS) > 5000:
        for k in [k for k, v in _TOKEN_FETCH_HITS.items() if not v or now - v[-1] > 60]:
            _TOKEN_FETCH_HITS.pop(k, None)
    return True


def _token_ok(handler):
    """R52：校验 X-Workbench-Token；ACCESS_TOKEN 为空 = 不启用（保持旧版兼容）。"""
    if not ACCESS_TOKEN:
        return True
    supplied = handler.headers.get("X-Workbench-Token", "")
    return secrets.compare_digest(supplied, ACCESS_TOKEN)

# POST body 上限（字节）；12MB 以容纳 mm_routing 上传截图的 base64
MAX_BODY = 12 * 1024 * 1024

# 固定窗口限流：{路径桶: (窗口内最大次数, 窗口秒)}；/v1/vkb 为前缀桶
RATE_LIMITS = {
    "/v1/systemone": (60, 60),
    "/v1/glm": (30, 60),
    "/v1/recompile": (2, 60),
    "/v1/vkb": (60, 60),
    "/v1/d3route": (30, 60),
}
_rate_bucket = {}


def _host_only(h):
    """取 Host 的纯主机名（去端口；IPv6 形如 [::1]:8787 → ::1）"""
    h = (h or "").strip().lower()
    if h.startswith("["):
        return h[1:h.index("]")] if "]" in h else h
    return h.split(":")[0]


def _rate_ok(bucket, ip):
    """固定窗口限流；条目数超限时清理过期键防内存膨胀"""
    now = time.time()
    n, win = RATE_LIMITS.get(bucket, (60, 60))
    key = (bucket, ip)
    ts = [t for t in _rate_bucket.get(key, []) if now - t < win]
    if len(ts) >= n:
        _rate_bucket[key] = ts
        return False
    ts.append(now)
    _rate_bucket[key] = ts
    if len(_rate_bucket) > 5000:
        for k in [k for k, v in _rate_bucket.items() if not v or now - v[-1] > win]:
            _rate_bucket.pop(k, None)
    return True


# Host 白名单：环境变量 JEV_ALLOWED_HOSTS 可扩展（逗号分隔主机名，精确匹配），部署到服务器时加服务器 IP
ALLOWED_HOSTS = tuple(
    h.strip() for h in os.environ.get("JEV_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if h.strip()
)


class Handler(BaseHTTPRequestHandler):
    def _check_origin(self):
        """安全加固：默认只允许本地访问；部署服务器时用 JEV_ALLOWED_HOSTS 放行。
        精确匹配纯主机名（不计端口）——前缀匹配可被 127.0.0.1.evil.com 这类 Host 绕过。"""
        host = _host_only(self.headers.get("Host"))
        if host and host not in {_host_only(h) for h in ALLOWED_HOSTS}:
            self.send_response(403)
            self._cors()
            self.end_headers()
            self.wfile.write(b"forbidden: local access only")
            return False
        return True

    def _cors(self):
        # R52：Allow-Headers 增补 X-Workbench-Token；Origin 仍为 *（前端已由 token 闸门保护）
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Workbench-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("X-Content-Type-Options", "nosniff")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if not self._check_origin(): return
        path = self.path.split("?")[0]
        # R52：同源下发访问令牌（前端注入补丁自动 fetch；页面业务代码零改动）。
        # R53：端点自身限流（2次/60秒）+ cross-site 拒绝（现代浏览器自动带 Sec-Fetch-Site）。
        if path == "/token.txt":
            site = self.headers.get("Sec-Fetch-Site", "")
            if site in ("cross-site", "same-protocol-cross-site"):
                self.send_response(403)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "cross-site token fetch denied"}).encode())
                return
            ip = self.client_address[0] if self.client_address else "?"
            if not _token_fetch_ok(ip):
                self.send_response(429)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "token fetch rate limit: 2 / 60s per IP"}).encode())
                return
            body = (ACCESS_TOKEN or "").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        # R57：通道清单（下拉数据源）——同源可读，不含敏感信息
        if path == "/v1/channels":
            up = JEV_UPSTREAM if JEV_UPSTREAM != API_URL else "bocha"
            payload = json.dumps({
                "jev": {
                    "current": "kev" if up != "bocha" else "bocha",
                    "options": [
                        {"id": "bocha", "name": "博查官方 bocha-jev-v1（云，快）"},
                        {"id": "kev", "name": "本机 kev 0.8B（自训，经 Tailscale）"},
                    ],
                }
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self._cors()
            self.end_headers()
            self.wfile.write(payload)
            return
        # 向量知识库 GET 转发（stats / chunks / documents / export）
        if path.startswith(VKB_PREFIX):
            if not _rate_ok(VKB_PREFIX.rstrip("/"), self.client_address[0]):
                self._rate_limited()
                return
            if not _token_ok(self):
                self._token_denied()
                return
            self._forward_vkb("GET", path[len(VKB_PREFIX):], None)
            return
        if path.startswith(D3_PREFIX):
            if not _rate_ok("/v1/d3route", self.client_address[0]):
                self._rate_limited()
                return
            self._forward_d3("GET", path, None)
            return
        if path in ("/", "/index.html"):
            # 仓库布局：页面在 demos/ 子目录；平面布局：与本文件同目录
            # 根路径 = Fin-Jev 工作台（左栏切换 意图识别 / 语义路由）
            fname = "demos/workbench.html" if ROOT != _HERE else "workbench.html"
        else:
            fname = path.lstrip("/")
            # 仓库布局兼容：/demos/xxx.html 直映 demos/ 子目录
            if fname.startswith("demos/"):
                fname = os.path.join("demos", fname[len("demos/"):])
            if path == "/favicon.ico":
                fname = os.path.join("demos", "favicon.ico")
            elif fname.endswith("probes.compiled.json") and not os.path.isfile(os.path.join(ROOT, fname)):
                # 编译产物唯一事实源在仓库根（graph_to_probes.py 的 OUT）；
                # duel_chat 从 /demos/ 相对加载时回退根文件，避免两份副本漂移
                fname = "probes.compiled.json"
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.normpath(os.path.join(ROOT, fname))
        # .json 只放行编译产物（防止公网直接下载 jev_key.local.json 等敏感文件）；
        # ROOT 前缀比较带 os.sep 收尾，避免 fin-jev-review2 这类同名兄弟目录误放行
        if ext == ".json" and os.path.basename(fpath) != ALLOWED_JSON:
            ext = ""  # .json 只放行编译产物，其余 json 视为拒绝
        inside = fpath == ROOT or fpath.startswith(ROOT + os.sep)
        ok_ext = ext in ALLOWED_EXT or ext == ".json"  # 走到这里 ".json" 必为编译产物
        if not ok_ext or not inside or not os.path.isfile(fpath):
            self.send_response(404)
            self._cors()
            self.end_headers()
            self.wfile.write(b"not found")
            return
        with open(fpath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type",
                         mimetypes.guess_type(fname)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        # R58：迭代期禁缓存——html/js 每次拉最新，避免改版后浏览器还跑旧脚本
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _too_large(self):
        self.send_response(413)
        self._cors()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"payload too large")

    def _rate_limited(self):
        self.send_response(429)
        self._cors()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"rate limited, slow down")

    def _token_denied(self):
        """R52：token 校验失败统一响应。"""
        self.send_response(403)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"detail": "invalid or missing X-Workbench-Token"}).encode())

    def _read_body(self):
        """读取请求体；超过 MAX_BODY 直接 413 并返回 None。"""
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._too_large()
            return None
        return self.rfile.read(length) if length else b"{}"

    def do_POST(self):
        if not self._check_origin(): return
        path = self.path.split("?")[0]
        bucket = path if path in RATE_LIMITS else (VKB_PREFIX.rstrip("/") if path.startswith(VKB_PREFIX) else None)
        if bucket and not _rate_ok(bucket, self.client_address[0]):
            self._rate_limited()
            return
        # R52：API 转发通道 token 闸门（限流之后、业务处理之前）
        if bucket and not _token_ok(self):
            self._token_denied()
            return
        if path == "/v1/channels/switch":
            # R57 修正：switch 不在 RATE_LIMITS → bucket=None → 通用闸门被跳过；
            # 必须显式校验 token，否则任何人可重启服务/切换通道
            if not _token_ok(self):
                self._token_denied()
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                mode = json.loads(body.decode("utf-8")).get("jev", "")
            except Exception:
                mode = ""
            if mode not in ("bocha", "kev"):
                payload = json.dumps({"ok": False, "detail": "jev must be bocha|kev"}).encode("utf-8")
                self.send_response(400); self.send_header("Content-Type", "application/json"); self._cors(); self.end_headers(); self.wfile.write(payload)
                return
            conf = "/etc/systemd/system/fin-jev.service.d/kev-upstream.conf"
            content = ("[Service]\nEnvironment=JEV_UPSTREAM=http://100.79.155.126:8011/v1/systemone\n"
                       if mode == "kev" else "[Service]\n# bocha official\n")
            try:
                with open(conf, "w") as _f:
                    _f.write(content)
                import subprocess as _sp
                _sp.run(["sudo", "-n", "systemctl", "daemon-reload"], timeout=10, stdin=subprocess.DEVNULL)
                _sp.run(["sudo", "-n", "systemctl", "restart", "fin-jev"], timeout=30, stdin=subprocess.DEVNULL)
                time.sleep(2)  # restart 完成后再返回，保证后续 /v1/channels 读到新状态
                payload = json.dumps({"ok": True, "jev": mode}, ensure_ascii=False).encode("utf-8")
                status = 200
            except Exception as e:
                payload = json.dumps({"ok": False, "detail": str(e)}).encode("utf-8")
                status = 500
            self.send_response(status); self.send_header("Content-Type", "application/json"); self._cors(); self.end_headers(); self.wfile.write(payload)
            return
        if path == "/v1/recompile":
            self._recompile()
            return
        if path.startswith(VKB_PREFIX):
            body = self._read_body()
            if body is None: return
            self._forward_vkb("POST", path[len(VKB_PREFIX):], body)
            return
        if path.startswith(D3_PREFIX):
            body = self._read_body()
            if body is None: return
            self._forward_d3("POST", path, body)
            return
        if path.startswith("/v1/systemone"):
            target, fwd = JEV_UPSTREAM, opener
        elif path.startswith("/v1/glm"):
            target, fwd = GLM_URL, glm_opener
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        body = self._read_body()
        if body is None: return
        auth = self.headers.get("Authorization", "")
        # GLM 通道：Bearer BUILTIN 表示用服务器内置 Key；其余原样透传（GLM_BUILTIN=0 可关闭注入）
        if path.startswith("/v1/glm") and GLM_KEY and BUILTIN_GLM_ENABLED and "BUILTIN" in auth:
            auth = "Bearer " + GLM_KEY
        # Jev 通道：前端无 Key 时发 BUILTIN_JEV 占位符，服务器有本地 Key 则替换（JEV_BUILTIN=0 可关闭注入）
        if path.startswith("/v1/systemone") and JEV_KEY and BUILTIN_JEV_ENABLED and "BUILTIN_JEV" in auth:
            auth = "Bearer " + JEV_KEY
        req = urllib.request.Request(
            target, data=body, method="POST",
            headers={
                "Authorization": auth,
                "Content-Type": "application/json",
            })
        try:
            with fwd.open(req, timeout=60) as resp:
                payload, status = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
        except Exception as e:  # noqa: BLE001
            payload = json.dumps({"detail": "local forwarder error: %s" % e}).encode("utf-8")
            status = 502
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def _recompile(self):
        """运行图谱编译器 graph_to_probes.py，返回编译结果 JSON。
        自校验（EXPECTED_CT_CF）失败时返回 exit code + stderr，编译产物不被污染。"""
        try:
            proc = subprocess.run(
                [sys.executable, GRAPH_COMPILER],
                cwd=ROOT, capture_output=True, timeout=30,
                stdin=subprocess.DEVNULL)  # 服务器后台化（nohup/&）时坏 stdin 会被子进程继承导致静默 exit 1
            ok = proc.returncode == 0
            out = (proc.stdout or b"").decode("utf-8", "replace").strip()
            err = (proc.stderr or b"").decode("utf-8", "replace").strip()
            payload = json.dumps({
                "ok": ok, "exitCode": proc.returncode,
                "stdout": out[-2000:], "stderr": err[-2000:],
            }, ensure_ascii=False).encode("utf-8")
            status = 200 if ok else 500
        except Exception as e:  # noqa: BLE001
            payload = json.dumps({"ok": False, "exitCode": -1,
                                  "stdout": "", "stderr": "recompile error: %s" % e},
                                 ensure_ascii=False).encode("utf-8")
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def _forward_vkb(self, method, sub, body):
        """转发 OntoMind 向量库请求：/v1/vkb/<kbId>/<action> → ontomind/api/v1/vector-kb/<kbId>/<action>"""
        target = VKB_BASE + sub
        req = urllib.request.Request(
            target, data=body if method == "POST" else None, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=30) as resp:
                payload, status = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
        except Exception as e:  # noqa: BLE001
            payload = json.dumps({"detail": "vkb forwarder error: %s" % e}).encode("utf-8")
            status = 502
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def _forward_d3(self, method, path, body):
        """转发多模态 JEV 推理服务：/v1/d3* → serve_d3.py（d3-lite 加载/推理较慢，超时放宽到 180s）"""
        target = D3_BASE + path
        req = urllib.request.Request(
            target, data=body if method == "POST" else None, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=180) as resp:
                payload, status = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
        except Exception as e:  # noqa: BLE001
            payload = json.dumps({"detail": "d3 forwarder error: %s" % e}).encode("utf-8")
            status = 502
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # 安静模式
        pass


if __name__ == "__main__":
    print("Serving duel_chat.html + Jev forwarder at http://127.0.0.1:%d" % PORT)
    if ACCESS_TOKEN:
        print("R52: API 转发通道已启用令牌校验（/token.txt 同源下发，2次/60秒）")
    else:
        print("R52: 未启用令牌校验（token 文件不可写），行为同旧版")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
