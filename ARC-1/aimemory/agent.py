import json
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Any


def _read_json(path: str) -> Dict[str, Any]:
    if not path or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"error": "failed_to_read_metrics"}


def _health(metrics: Dict[str, Any], max_staleness_s: float) -> Dict[str, Any]:
    now = time.time()
    ts = float(metrics.get("ts", 0.0) or 0.0)
    stale_s = (now - ts) if ts > 0 else 1e9
    safe_mode = bool(metrics.get("safe_mode", False))
    ok = (stale_s <= max_staleness_s) and (not safe_mode)
    return {
        "ok": bool(ok),
        "safe_mode": bool(safe_mode),
        "stale_s": float(stale_s),
        "max_staleness_s": float(max_staleness_s),
        "disable_reason": str(metrics.get("disable_reason", "")),
    }


def run_agent(bind: str, port: int, metrics_path: str = "", heartbeat_path: str = "", heartbeat_interval_s: float = 5.0):
    hb_stop = threading.Event()

    def _heartbeat_loop():
        if not heartbeat_path:
            return
        os.makedirs(os.path.dirname(heartbeat_path) or ".", exist_ok=True)
        while not hb_stop.is_set():
            data = {
                "ts": float(time.time()),
                "metrics_path": str(metrics_path),
            }
            try:
                with open(heartbeat_path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass
            hb_stop.wait(timeout=max(0.25, float(heartbeat_interval_s)))

    t = threading.Thread(target=_heartbeat_loop, daemon=True)
    t.start()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/healthz", "/health"):
                data = _read_json(metrics_path)
                body_obj = _health(data, max_staleness_s=max(2.0, 2.5 * float(heartbeat_interval_s)))
                body = json.dumps(body_obj).encode()
                self.send_response(200 if body_obj.get("ok", False) else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path in ("/metrics", "/status"):
                data = _read_json(metrics_path)
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path in ("/heartbeat",):
                hb = _read_json(heartbeat_path) if heartbeat_path else {"ok": False, "reason": "heartbeat_disabled"}
                body = json.dumps(hb).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path in ("/diag", "/diagnostics"):
                metrics = _read_json(metrics_path)
                body_obj = {
                    "health": _health(metrics, max_staleness_s=max(2.0, 2.5 * float(heartbeat_interval_s))),
                    "metrics_summary": {
                        "backend": str(metrics.get("backend", "")),
                        "step_p99_ms": float(metrics.get("step_p99_ms", 0.0)),
                        "prefetch_hit_rate": float(metrics.get("prefetch_hit_rate", 0.0)),
                        "spills": int(metrics.get("spills", 0)),
                        "restores": int(metrics.get("restores", 0)),
                        "rank": int(metrics.get("rank", 0)),
                    },
                }
                body = json.dumps(body_obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    srv = ThreadingHTTPServer((bind, int(port)), _Handler)
    print(f"[aimemory-agent] listening on http://{bind}:{port}")
    try:
        srv.serve_forever()
    finally:
        hb_stop.set()
