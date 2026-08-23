"""
Launch the pitcher strikeout decision dashboard (local web app).

Run it straight from VS Code:
    - open this file and press the Run/▶ button (or F5 with the bundled
      .vscode/launch.json config "Dashboard"), or
    - from the integrated terminal:  python run_dashboard.py

It serves the most recent slate written by the refresh pipeline
(data/processed/predictions/) at http://127.0.0.1:8000 and opens your browser.
Read-only: it never triggers a refresh or writes anything. Stop with Ctrl+C.

If you see "no partitions found", run the pipeline first:
    python -m src.pipeline.refresh
"""

import argparse
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.serve.server import DEFAULT_PROCESSED_DIR, build_server


def main():
    parser = argparse.ArgumentParser(description="Run the OddsOptimizer decision dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (use 0.0.0.0 to expose on your network)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args()

    server = build_server(args.host, args.port, args.processed_dir)
    url = f"http://{args.host}:{args.port}/"
    print(f"Dashboard serving {os.path.abspath(args.processed_dir)}")
    print(f"  -> {url}   (read-only; Ctrl+C to stop)")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
