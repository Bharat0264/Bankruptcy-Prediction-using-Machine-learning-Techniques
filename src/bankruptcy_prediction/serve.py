from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .config import ROOT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the dashboard and generated reports.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Dashboard: http://{args.host}:{args.port}/app/dashboard.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
