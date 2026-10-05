"""Local demo server for a configurable Decider checkpoint.

Serves the page on http://127.0.0.1:8787. The active model is demo/models.json.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
CATALOG_PATH = Path(__file__).resolve().parent / "models.json"
HOST = "127.0.0.1"
PORT = 8787
DTYPES = {
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float16": "float16",
}


def load_selection() -> tuple[str, dict]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    model_id = catalog["active"]
    try:
        spec = catalog["models"][model_id]
    except KeyError as exc:
        known = ", ".join(catalog.get("models", {}))
        raise SystemExit(f"demo/models.json active id {model_id!r} is not in the catalog ({known}).") from exc
    dtype = spec.get("dtype", "float32")
    if dtype not in DTYPES:
        raise SystemExit(f"demo/models.json dtype {dtype!r} for {model_id} must be one of {', '.join(DTYPES)}.")
    return model_id, spec


MODEL_ID, SPEC = load_selection()
MODEL_DIR = ROOT / "models" / SPEC["directory"]

_lock = threading.Lock()
_state = {
    "status": "loading",
    "error": None,
    "id": MODEL_ID,
    "name": SPEC["name"],
    "repo": SPEC["repo"],
    "revision": SPEC["revision"],
    "device": SPEC.get("device", "cpu"),
    "dtype": SPEC.get("dtype", "float32"),
}


def _load() -> None:
    if not (MODEL_DIR / "model.safetensors").is_file():
        _state["status"] = "error"
        _state["error"] = (
            "Weights are not downloaded yet. Run Start-Demo.ps1 so it can fetch "
            f"{SPEC['repo']} at {SPEC['revision'][:12]}."
        )
        return
    try:
        import torch
        from decider.infer import Decider

        torch.set_num_threads(6)
        dtype = getattr(torch, SPEC.get("dtype", "float32"))
        decider = Decider(
            str(MODEL_DIR),
            device=SPEC.get("device", "cpu"),
            dtype=dtype,
            use_graphs=False,
        )
        _state["decider"] = decider
        _state["status"] = "ready"
        _state["name"] = SPEC["name"]
        _state["runtime_name"] = getattr(decider, "name", SPEC["name"])
        _state["device"] = str(decider.dev)
        _state["dtype"] = SPEC.get("dtype", "float32")
        _state["temperature"] = decider.T
    except Exception:
        _state["status"] = "error"
        _state["error"] = traceback.format_exc()


def _public_state() -> dict:
    return {k: v for k, v in _state.items() if k != "decider"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[demo] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(200, _public_state())
            return
        if path == "/":
            path = "/index.html"
        file = (STATIC / path.lstrip("/")).resolve()
        if not str(file).startswith(str(STATIC.resolve())) or not file.is_file():
            self._json(404, {"error": "Not found"})
            return
        kind = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }.get(file.suffix, "application/octet-stream")
        self._send(200, file.read_bytes(), kind)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/systemone":
            self._json(404, {"error": "Not found"})
            return
        if _state["status"] != "ready":
            self._json(503, {"error": _state["error"] or "Decider is still loading.", "status": _state["status"]})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            state = body["state"]
            questions = body["questions"]
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must be a non-empty object")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self._json(422, {"error": str(exc)})
            return
        started = time.perf_counter()
        try:
            with _lock:
                result = _state["decider"].system_one(state, questions)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        result["request_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self._json(200, result)


def main() -> None:
    threading.Thread(target=_load, name="decider-load", daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[demo] http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
