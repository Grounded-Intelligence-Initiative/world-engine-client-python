"""Shared stubs: a programmable control server and a protocol-3 packet builder.

These tests exercise the client alone. The wire contract against a real engine
is covered in the server repository (tests/test_remote_*.py).
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    """Records every request; ``responders[path]`` overrides the default reply."""

    def __init__(self):
        self.calls = []
        self.responders = {}
        self.step_id = 0
        self.episode = 0
        self.files = {}  # GET path -> bytes (the /render/ prefix)

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


@pytest.fixture
def stub():
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
            data = state.files.get(self.path)
            if data is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()
