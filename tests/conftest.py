"""Shared stubs: a programmable control server and a protocol-3 packet builder.

These tests exercise the client alone. The wire contract against a real engine
is covered in the server repository (tests/test_remote_*.py).
"""

import json
import os
import re
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

DROP = object()  # responder return value: close the socket without replying

ENV = {
    "instance_id": "inst",
    "control_interval": 0.05,
    "action_dim": 4,
    "active_control_mode": "ee_pos_delta",
    "observation_keys": ["robot0_eef_pos"],
    "observation_schema": {"robot0_eef_pos": {"shape": [3], "dtype": "float32"}},
    "control_mode_schemas": {
        "ee_pos_delta": {"action_dim": 4, "bounds": [-1.0, 1.0]},
        "osc_pose_delta": {"action_dim": 7, "bounds": [-1.0, 1.0]},
        "osc_pose_abs": {"action_dim": 7, "bounds": None},
    },
    "available_cameras": {},
    "cameras": {},
}
OBS = {"robot0_eef_pos": [0.1, 0.2, 0.3], "object_ids": ["cube"]}


class ControlStub:
    """Records every request; ``responders[path]`` overrides the default reply.

    GET: ``files[path]`` serves bytes (with single byte ranges, like the
    engine and the platform); ``gets[path](handler)`` answers by itself."""

    def __init__(self):
        self.calls = []
        self.responders = {}
        self.step_id = 0
        self.episode = 0
        self.files = {}  # GET path -> bytes (the /render/ prefix)
        self.gets = {}  # GET path -> callable(handler) that writes the whole response

    def get_calls(self, path):
        """The headers of every GET of ``path``, in order."""
        return [headers for p, body, headers in self.calls if p == path and body is None]

    def default(self, path, body):
        path = path[path.index("/control/") :]  # tolerate a gateway prefix
        info = {"episode_id": f"e{self.episode}", "step_id": self.step_id}
        if path == "/control/connect":
            return {
                "ok": True,
                "session_id": "sess",
                "proto": body.get("proto", 3),
                "instance_id": "inst",
                "episode_id": "e0",
                "env": ENV,
            }
        if path == "/control/reset":
            self.episode += 1
            self.step_id = 0
            info = {"episode_id": f"e{self.episode}", "step_id": 0}
            return {"ok": True, "obs": OBS, "info": info, "env": ENV}
        if path == "/control/step":
            self.step_id += 1
            info["step_id"] = self.step_id
            return {
                "ok": True,
                "obs": OBS,
                "info": info,
                "reward": 0.5,
                "terminated": False,
                "truncated": False,
            }
        if path == "/control/observe":
            return {"ok": True, "obs": OBS, "info": info}
        if path == "/control/configure":
            return {"ok": True, "pending": False, "env": ENV}
        return {"ok": True}

    def paths(self):
        return [c[0] for c in self.calls]


def send_bytes(handler, data: bytes, headers: dict | None = None) -> None:
    """Answer a GET with ``data``, honouring ``Range: bytes=start-[end]``."""
    m = re.fullmatch(r"bytes=(\d+)-(\d*)", handler.headers.get("Range") or "")
    if m:
        start = int(m.group(1))
        end = min(int(m.group(2)) if m.group(2) else len(data) - 1, len(data) - 1)
        if start >= len(data):
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{len(data)}")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        handler.send_response(206)
        handler.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        data = data[start : end + 1]
    else:
        handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    for k, v in (headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(data)


def send_json(handler, status: int, reply: dict) -> None:
    data = json.dumps(reply).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    """Every test gets its own render cache and temporary directory: nothing
    reaches the user's cache directory or /tmp."""
    root = tmp_path / "user-cache"
    # Playwright finds its browsers under the user cache directory too: keep
    # them where they are.
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        if sys.platform.startswith("linux"):
            monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(Path(xdg) / "ms-playwright"))
    # So is this package's headless browser (render.browsers_dir): a test that
    # launches one uses the user's. None downloads one unless it says so.
    if not os.environ.get("DIGIARENA_BROWSERS_DIR"):
        if sys.platform == "darwin":
            user_cache = Path.home() / "Library" / "Caches"
        elif os.name == "nt":
            user_cache = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        else:
            user_cache = Path(xdg)
        monkeypatch.setenv("DIGIARENA_BROWSERS_DIR", str(user_cache / "digiarena" / "browsers"))
    if os.environ.get("WE_RENDER_AUTO_INSTALL") is None:
        monkeypatch.setenv("WE_RENDER_AUTO_INSTALL", "0")
    # The stand-ins for Chrome launch as the "chrome" channel: a browser
    # choice of the user's would turn them away (see test_render_chrome.py
    # for WE_TEST_BROWSER).
    monkeypatch.delenv("WE_RENDER_BROWSER", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    if sys.platform.startswith("linux"):
        monkeypatch.delenv("DIGIARENA_CACHE_DIR", raising=False)
    else:  # the user cache directory there does not follow XDG_CACHE_HOME
        monkeypatch.setenv("DIGIARENA_CACHE_DIR", str(root / "digiarena" / "render"))
    monkeypatch.delenv("DIGIARENA_CACHE_MAX_AGE_DAYS", raising=False)
    monkeypatch.delenv("DIGIARENA_CACHE_MAX_SIZE_GB", raising=False)
    systmp = tmp_path / "systmp"
    systmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(systmp))
    try:
        from world_engine_client import cache
    except ImportError:  # world-engine-client before 0.6 has no render cache
        return root
    monkeypatch.setattr(cache, "_warned", set())
    return root


def start_stub() -> ControlStub:
    """A ControlStub on a loopback port; ``state.close()`` stops it."""
    state = ControlStub()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append((self.path, body, dict(self.headers)))
            responder = state.responders.get(self.path)
            reply = responder(body) if responder else state.default(self.path, body)
            if reply is DROP:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            content_type = "application/json"
            if isinstance(reply, tuple):
                content_type, data = reply
            else:
                data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state.calls.append((self.path, None, dict(self.headers)))
            custom = state.gets.get(self.path)
            if custom is not None:
                custom(self)
                return
            data = state.files.get(self.path)
            if data is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            send_bytes(self, data)

        def log_message(self, *args):
            pass

    state.handler_class = Handler  # a test may set Handler.timeout: idle connections close
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"

    def close():
        server.shutdown()
        server.server_close()

    state.close = close
    return state


@pytest.fixture
def stub():
    state = start_stub()
    yield state
    state.close()
