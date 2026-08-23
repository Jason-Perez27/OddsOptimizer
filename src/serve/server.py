"""
Zero-dependency HTTP server for the pre-game decision dashboard.

Uses only the Python standard library (http.server) so it runs in VS Code with
no `pip install` beyond what the project already pins -- launch via the repo-root
`run_dashboard.py` or `python -m src.serve.server`. It's a thin shell over
src/serve/data.py (which does the real work and is unit-tested independently):

Routes:
    GET /                      -> the dashboard page (static/index.html)
    GET /static/<file>         -> static assets (app.js, styles.css)
    GET /api/dates             -> {"dates": [...], "latest": "..."}
    GET /api/slate?date=YYYY-MM-DD (default: latest) -> the slate dict
    GET /healthz               -> {"status": "ok"}

Read-only over the data/processed partitions. Never triggers a refresh. Binds
127.0.0.1 by default (personal research tool) -- pass --host 0.0.0.0 to expose it
deliberately.
"""

import argparse
import json
import os
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.serve import data as slate_data

DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class DashboardHandler(BaseHTTPRequestHandler):
    processed_dir = DEFAULT_PROCESSED_DIR

    def log_message(self, fmt, *args):  # quieter console; keep one-line access logs
        print(f"[dashboard] {self.address_string()} {fmt % args}")

    # -- helpers -------------------------------------------------------------
    def _send_json(self, payload, status=200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path):
        safe = os.path.normpath(rel_path).lstrip(os.sep)
        full = os.path.join(STATIC_DIR, safe)
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            self._send_json({"error": "not found"}, status=404)
            return
        ext = os.path.splitext(full)[1]
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routing -------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("/", "/index.html"):
                self._send_static("index.html")
            elif route.startswith("/static/"):
                self._send_static(route[len("/static/"):])
            elif route == "/healthz":
                self._send_json({"status": "ok"})
            elif route == "/api/dates":
                dates = slate_data.list_game_dates(self.processed_dir)
                self._send_json({"dates": dates, "latest": dates[-1] if dates else None})
            elif route == "/api/slate":
                qs = parse_qs(parsed.query)
                date = (qs.get("date") or [None])[0] or slate_data.latest_game_date(self.processed_dir)
                prop = (qs.get("prop") or [None])[0]
                if not date:
                    self._send_json(
                        {"error": "no partitions found -- run `python -m src.pipeline.refresh` first"},
                        status=404,
                    )
                    return
                try:
                    self._send_json(slate_data.load_slate(self.processed_dir, date, prop=prop))
                except FileNotFoundError:
                    self._send_json({"error": f"no data for {date}"}, status=404)
            else:
                self._send_json({"error": "not found"}, status=404)
        except BrokenPipeError:
            pass
        except Exception as exc:  # never crash the server on one bad request
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)


def build_server(host, port, processed_dir):
    handler = partial(DashboardHandler)
    DashboardHandler.processed_dir = processed_dir
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description="Serve the pitcher strikeout decision dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (use 0.0.0.0 to expose)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()

    server = build_server(args.host, args.port, args.processed_dir)
    url = f"http://{args.host}:{args.port}/"
    print(f"Dashboard serving {os.path.abspath(args.processed_dir)} at {url}")
    print("Read-only. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
