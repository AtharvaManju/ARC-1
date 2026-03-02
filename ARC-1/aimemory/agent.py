import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def run_agent(bind: str, port: int, metrics_path: str = ""):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/healthz", "/health"):
                body = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path in ("/metrics", "/status"):
                data = {}
                p = metrics_path
                if p and os.path.exists(p):
                    try:
                        with open(p, "r") as f:
                            data = json.load(f)
                    except Exception:
                        data = {"error": "failed_to_read_metrics"}
                body = json.dumps(data).encode()
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
    srv.serve_forever()
