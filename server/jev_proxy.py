# -*- coding: utf-8 -*-
"""
jev_proxy.py — 本地一体化服务器（静态页面 + Jev API 同源转发）
解决浏览器直调 https://jev.bocha.cn 遇到的 CORS / 系统代理拦截问题。

运行:  python jev_proxy.py
访问:  http://127.0.0.1:8787
页面与接口同源（无 CORS），转发由 Python 直连完成（绕过系统代理）。
"""
import json
import mimetypes
import os
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT = os.environ.get("JEV_HOST", "127.0.0.1"), int(os.environ.get("JEV_PORT", "8787"))
API_URL = "https://jev.bocha.cn/v1/systemone"
# 布局自适应：仓库布局（本文件在 server/，demos/ 与图谱编译链在上级目录）
# 或平面布局（页面/图谱/脚本与本文件同目录）均可运行——同一份代码两种落位
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE) if os.path.isdir(os.path.join(_HERE, "..", "demos")) else _HERE

# 图谱编译链（R46d 一键重编译）：改 bank_graph.json 存盘后页面点按钮即生效
GRAPH_COMPILER = os.path.join(ROOT, "graph_to_probes.py")

# OntoMind 向量知识库（local_rag）：意图语义索引库，供页面做运行期向量召回
VKB_PREFIX = "/v1/vkb/"
VKB_BASE = "http://127.0.0.1:17386/ontomind/api/v1/vector-kb/"

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

ALLOWED_EXT = {".html", ".js", ".css", ".svg", ".png", ".ico", ".json"}


# Host 白名单：环境变量 JEV_ALLOWED_HOSTS 可扩展（逗号分隔前缀），部署到服务器时加服务器 IP
ALLOWED_HOSTS = tuple(
    h.strip() for h in os.environ.get("JEV_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if h.strip()
)


class Handler(BaseHTTPRequestHandler):
    def _check_origin(self):
        """安全加固：默认只允许本地访问，拒绝同网段远程请求；部署服务器时用 JEV_ALLOWED_HOSTS 放行。"""
        host = self.headers.get("Host", "")
        if host and not any(host.startswith(h) for h in ALLOWED_HOSTS):
            self.send_response(403)
            self._cors()
            self.end_headers()
            self.wfile.write(b"forbidden: local access only")
            return False
        return True

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if not self._check_origin(): return
        path = self.path.split("?")[0]
        # 向量知识库 GET 转发（stats / chunks / documents / export）
        if path.startswith(VKB_PREFIX):
            self._forward_vkb("GET", path[len(VKB_PREFIX):], None)
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
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.normpath(os.path.join(ROOT, fname))
        if ext not in ALLOWED_EXT or not fpath.startswith(ROOT) or not os.path.isfile(fpath):
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
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self._check_origin(): return
        if self.path.split("?")[0] == "/v1/recompile":
            self._recompile()
            return
        if self.path.startswith(VKB_PREFIX):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            self._forward_vkb("POST", self.path[len(VKB_PREFIX):].split("?")[0], body)
            return
        if self.path.startswith("/v1/systemone"):
            target, fwd = API_URL, opener
        elif self.path.startswith("/v1/glm"):
            target, fwd = GLM_URL, glm_opener
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        auth = self.headers.get("Authorization", "")
        # GLM 通道：Bearer BUILTIN 表示用服务器内置 Key；其余原样透传
        if self.path.startswith("/v1/glm") and GLM_KEY and "BUILTIN" in auth:
            auth = "Bearer " + GLM_KEY
        # Jev 通道：前端无 Key 时发 BUILTIN_JEV 占位符，服务器有本地 Key 则替换
        if self.path.startswith("/v1/systemone") and JEV_KEY and "BUILTIN_JEV" in auth:
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

    def log_message(self, *a):  # 安静模式
        pass


if __name__ == "__main__":
    print("Serving duel_chat.html + Jev forwarder at http://127.0.0.1:%d" % PORT)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
