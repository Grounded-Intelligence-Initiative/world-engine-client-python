"""A stand-in for the hosting platform, or for a local ``world-engine serve``,
on top of the ControlStub (conftest.py).

Like the platform: the connect bundle says ``cache: "encrypted"``, an image
session of a client that does not list ``"dac1"`` in ``asset_encryption`` is
refused (``client_upgrade_required``, as a hosted engine does),
``/render/__key`` hands out the scene key, and ``/render/__asset/*`` answers
DAC1, with byte ranges on the ciphertext. Robot meshes (``/render/__robot/*``)
and the capture page stay plain. The ciphertext comes from the engine's
reference implementation (dac1_ref.py), not from the client's code.

``Platform(stub, encrypted=False)`` is a local engine: plain assets, ``cache:
"plain"``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import socket
import threading
import time

import dac1_ref as ref
from conftest import ENV, OBS, send_bytes, send_json

SECRET = "test-only-render-cache-secret-0000000000000000"
SCENE = "5f2b7d10-6c3e-4a8b-b1d9-0e4c7a2f6b31"
USER = "8d3c2f0e-1b7a-4c55-9e0d-2a6f4b1c9e77"
CAMERAS = {"front": {"width": 8, "height": 6}}


def body(magic: bytes, seed: str, size: int) -> bytes:
    return magic + ref.pattern(seed, size)


# What scene assets start with: a splat, a GLB, a WEM1 mesh, two textures.
ASSETS = {
    "/__asset/0.ply": body(b"ply\nformat binary_little_endian 1.0\n", "splat", 50_000),
    "/__asset/1.glb": body(b"glTF\x02\x00\x00\x00", "glb", 20_000),
    "/__asset/2.wem": body(b"WEM1", "wem", 9_000),
    "/__asset/3.png": body(b"\x89PNG\r\n\x1a\n", "png", 7_000),
    "/__asset/4.jpg": body(b"\xff\xd8\xff\xe0", "jpg", 6_000),
}
ROBOT = {"/__robot/0.bin": body(b"WEL1", "link", 5_000)}
MAGICS = (b"ply", b"glTF", b"WEM1", b"\x89PNG", b"\xff\xd8\xff")

INDEX = (
    b'<!doctype html><script type="module" crossorigin src="/assets/main-D-j9cs4x.js"></script>'
    b'<link rel="modulepreload" crossorigin href="/assets/sceneEnvironment-uFkXfr3L.js">'
)
PAGE = {
    "/render/index.html": INDEX,
    "/render/assets/main-D-j9cs4x.js": b"// main\n" * 100,
    "/render/assets/sceneEnvironment-uFkXfr3L.js": b"// environment\n" * 300,
}


def scene_key(epoch: str = "1.0", user: str = USER) -> bytes:
    return ref.scene_key(SECRET, user, SCENE, epoch)


class Platform:
    """Serves one scene through ``stub``. Counters and switches:

    * ``downloads[url]``: GETs of each asset (``ranges[url]`` the Range headers);
    * ``revoked``: the key route says ``asset_key_revoked``;
    * ``epoch``: change it to rotate the key;
    * ``delay``: seconds between 4 KiB pieces of an asset (a slow line);
    * ``cut[url] = n``: the next GET of ``url`` breaks off after ``n`` bytes.
    """

    def __init__(self, stub, *, encrypted: bool = True, assets: dict | None = None):
        self.stub = stub
        self.encrypted = encrypted
        self.assets = dict(ASSETS if assets is None else assets)
        self.assets.update(ROBOT)
        self.epoch = "1.0"
        self.revoked = False
        self.delay = 0.0
        self.cut: dict[str, int] = {}
        self.downloads: dict[str, int] = {}
        self.ranges: dict[str, list] = {}
        self.sent: dict[str, int] = {}  # bytes of each asset sent so far, all GETs
        self.key_requests = 0
        self.lock = threading.Lock()
        self._ciphers: dict[tuple[str, str], bytes] = {}
        stub.files.update(PAGE)
        stub.gets["/render/__key"] = self._key
        for url in self.assets:
            stub.gets["/render" + url] = self._asset(url)
        stub.responders["/control/connect"] = self._connect
        for path in ("/control/reset", "/control/observe"):
            stub.responders[path] = lambda body: {
                "ok": True,
                "obs": dict(OBS),
                "info": {"episode_id": "e1", "step_id": 0},
                "render_state": self.state(),
            }
        stub.responders["/control/step"] = lambda body: {
            "ok": True,
            "obs": dict(OBS),
            "info": {"episode_id": "e1", "step_id": 1},
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "render_state": self.state(),
        }

    # ----- what the engine says ---------------------------------------------- #
    def scene_init(self) -> dict:
        meshes = [
            {"id": f"m{i}", "url": url, "scale": [1, 1, 1]}
            for i, url in enumerate(sorted(self.assets))
            if url.startswith("/__asset/") and url != "/__asset/0.ply"
        ]
        init = {"type": "scene_init", "meshes": meshes, "cameras": [], "robot": {"links": []}}
        if "/__asset/0.ply" in self.assets:
            init["splat"] = {"url": "/__asset/0.ply"}
        for url in sorted(ROBOT):
            if url in self.assets:
                init["robot"]["links"].append({"name": "link0", "robot_id": "r", "url": url})
        return init

    def digests(self) -> dict:
        return {url: hashlib.sha256(data).hexdigest() for url, data in self.assets.items()}

    def bundle(self) -> dict:
        out = {
            "protocol": 1,
            "scene_init": self.scene_init(),
            "digests": self.digests(),
            "cache": "encrypted" if self.encrypted else "plain",
        }
        if self.encrypted:
            out["encryption"] = "dac1"
        return out

    def state(self) -> dict:
        return {
            "type": "state",
            "bodies": [],
            "robot": {},
            "cameras": [{"name": "front", "w": 8, "h": 6}],
        }

    def _connect(self, body: dict) -> dict:
        images = body.get("images") or body.get("cameras")
        formats = body.get("asset_encryption", [])
        if images and self.encrypted and (not isinstance(formats, list) or "dac1" not in formats):
            return {
                "ok": False,
                "code": "client_upgrade_required",
                "error": "this platform sends scene assets encrypted; this client cannot read them",
                "fixes": ["pip install -U 'world-engine-client[render]'"],
            }
        env = {**copy.deepcopy(ENV), "cameras": CAMERAS, "scene_id": "fixture"}
        reply = {
            "ok": True,
            "session_id": "sess",
            "proto": 3,
            "instance_id": "inst",
            "episode_id": "e0",
            "env": env,
        }
        if images:
            reply["render"] = self.bundle()
        return reply

    # ----- what the gateway sends -------------------------------------------- #
    def key(self) -> bytes:
        return scene_key(self.epoch)

    def cipher(self, url: str) -> bytes:
        """The DAC1 file of an asset under the current key."""
        k = (url, self.epoch)
        if k not in self._ciphers:
            data = self.assets[url]
            self._ciphers[k] = ref.encrypt(self.key(), hashlib.sha256(data).hexdigest(), data)
        return self._ciphers[k]

    def wire(self, url: str) -> bytes:
        if self.encrypted and url.startswith("/__asset/"):
            return self.cipher(url)
        return self.assets[url]

    def _key(self, handler) -> None:
        with self.lock:
            self.key_requests += 1
        if self.revoked:
            send_json(
                handler,
                403,
                {
                    "ok": False,
                    "code": "asset_key_revoked",
                    "error": "access to this scene's assets was revoked",
                    "fixes": [],
                },
            )
            return
        key = self.key()
        send_json(
            handler,
            200,
            {
                "v": 1,
                "alg": "A256GCM",
                "format": "dac1",
                "key_id": ref.key_id(key).hex(),
                "scope": SCENE,
                "key": base64.b64encode(key).decode(),
            },
        )

    def _asset(self, url: str):
        def answer(handler) -> None:
            with self.lock:
                self.downloads[url] = self.downloads.get(url, 0) + 1
                self.ranges.setdefault(url, []).append(handler.headers.get("Range"))
                cut = self.cut.pop(url, None)
            if self.revoked and self.encrypted and url.startswith("/__asset/"):
                send_json(
                    handler,
                    403,
                    {"ok": False, "code": "asset_key_revoked", "error": "revoked", "fixes": []},
                )
                return
            data = self.wire(url)
            digest = {"X-We-Digest": hashlib.sha256(self.assets[url]).hexdigest()}
            if cut is None and not self.delay:
                send_bytes(handler, data, digest)
                return

            def count(n):
                with self.lock:
                    self.sent[url] = self.sent.get(url, 0) + n

            stream(handler, data, digest, delay=self.delay, cut=cut, sent=count)

        return answer


def stream(handler, data: bytes, headers: dict, *, delay: float, cut: int | None, sent=None):
    """Send ``data`` in 4 KiB pieces, ``delay`` s apart; break the connection
    after ``cut`` bytes; ``sent(n)`` after each piece. Byte ranges as
    send_bytes."""
    import re

    sent_cb = sent
    m = re.fullmatch(r"bytes=(\d+)-(\d*)", handler.headers.get("Range") or "")
    start = int(m.group(1)) if m else 0
    end = (int(m.group(2)) if m and m.group(2) else len(data) - 1) if m else len(data) - 1
    handler.send_response(206 if m else 200)
    if m:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
    handler.send_header("Content-Length", str(end - start + 1))
    for k, v in headers.items():
        handler.send_header(k, v)
    handler.end_headers()
    sent, at = 0, start
    try:
        while at <= end:
            piece = data[at : min(at + 4096, end + 1)]
            if cut is not None and sent + len(piece) > cut:
                handler.wfile.write(piece[: cut - sent])
                handler.wfile.flush()
                handler.connection.shutdown(socket.SHUT_RDWR)
                handler.close_connection = True
                return
            handler.wfile.write(piece)
            handler.wfile.flush()
            if sent_cb is not None:
                sent_cb(len(piece))
            sent += len(piece)
            at += len(piece)
            if delay:
                time.sleep(delay)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True
