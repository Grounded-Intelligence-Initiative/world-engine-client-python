"""Render camera observations on this machine (``render="local"``).

The server only simulates physics. It sends the scene description once at
connect and the poses with every observation; a headless Chrome started here
runs the same capture page the server-side renderer runs, so the images match.
Headless Chrome is never a "background tab", so nothing throttles it.

The capture page, its scripts and the scene assets are fetched from the
server's ``/render/`` prefix with this session's credentials. Assets are cached
by content digest in a temporary directory that ``close()`` deletes, or in
``cache_dir`` to keep them across sessions.

Needs the ``render`` extra (Playwright, websockets, Pillow) and a Chrome or
Chromium: ``pip install 'world-engine-client[render]'`` then
``python -m playwright install chromium`` (or use an installed Chrome).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import mimetypes
import os
import shlex
import shutil
import sys
import tempfile
import threading
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .errors import WorldEngineClientError

# Hardware WebGL in headless Chrome, per platform. The Linux flags are the ones
# the server renderer is verified with (NVIDIA through Vulkan ANGLE); macOS uses
# Metal; Windows uses Chrome's default (D3D11). WE_RENDER_CHROME_ARGS overrides.
if sys.platform == "darwin":
    _CHROME_ARGS = ["--headless=new", "--use-angle=metal", "--ignore-gpu-blocklist"]
elif sys.platform.startswith("linux"):
    _CHROME_ARGS = [
        "--headless=new",
        "--ozone-platform=headless",
        "--use-angle=vulkan",
        "--enable-features=Vulkan",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
        "--no-sandbox",
    ]
else:
    _CHROME_ARGS = ["--headless=new", "--ignore-gpu-blocklist"]

_INSTALL_FIXES = [
    "pip install 'world-engine-client[render]' && python -m playwright install chromium",
    "or pass images=False for a state-only session",
]


class RendererError(WorldEngineClientError):
    """Local rendering is unavailable or failed. The physics step itself is
    unaffected: ``env.observe()`` renders the current state again."""


def available() -> bool:
    """True when the ``render`` extra is installed (Chrome is checked on start)."""
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("playwright", "websockets", "PIL"))


Fetch = Callable[[str], "io.BufferedIOBase"]


class LocalRenderer:
    """Headless Chrome running the capture page; ``render(state)`` -> images.

    ``fetch(path)`` opens ``<server>/render<path>`` with the session's
    credentials and returns a readable response."""

    def __init__(
        self,
        fetch: Fetch,
        scene_init: dict,
        digests: dict[str, str],
        *,
        cache_dir: str | os.PathLike | None = None,
        allow_software: bool = False,
        chrome_args: list[str] | None = None,
        timeout: float = 300.0,
    ):
        if not available():
            raise RendererError(
                "local rendering needs the render extra",
                code="renderer_unavailable",
                fixes=_INSTALL_FIXES,
            )
        self._fetch = fetch
        self._scene_init = scene_init
        self._digests = dict(digests)
        self._own_cache = cache_dir is None
        self._cache = Path(cache_dir or tempfile.mkdtemp(prefix="we_render_"))
        self._cache.mkdir(parents=True, exist_ok=True)
        self._allow_software = allow_software
        override = os.environ.get("WE_RENDER_CHROME_ARGS")
        self._args = chrome_args or (shlex.split(override) if override else _CHROME_ARGS)
        self._timeout = timeout
        self.gpu = "(unknown)"
        self.software = False
        self.lighting: str | None = None
        self._frame_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._ws = None
        self._ready = None
        self._page_error: dict | None = None
        self._console: list[str] = []
        self._http = None
        self._download_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        try:
            self._call(self._start(), timeout + 30)
        except BaseException:
            self.close()
            raise

    # ----- public ------------------------------------------------------------- #
    def render(self, state: dict) -> dict[str, np.ndarray]:
        """Pose the scene as ``state`` (a ``render_state`` from the server) and
        return ``{camera name: (H, W, 3) uint8}``."""
        if self._ws is None:
            raise RendererError("renderer is closed", code="renderer_unavailable")
        header, payload = self._call(self._render(dict(state)), self._timeout)
        from PIL import Image

        images = {}
        for spec in header.get("images", []):
            chunk = payload[spec["offset"] : spec["offset"] + spec["length"]]
            with Image.open(io.BytesIO(chunk)) as img:
                images[spec["name"]] = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        self.last_timings = header.get("timings", {})
        return images

    def close(self) -> None:
        if self._loop.is_running():
            try:
                self._call(self._stop(), 30)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
        if self._own_cache:
            shutil.rmtree(self._cache, ignore_errors=True)

    def __enter__(self) -> LocalRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----- loop thread ---------------------------------------------------------- #
    def _call(self, coro, timeout):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except TimeoutError as exc:
            future.cancel()
            raise RendererError(
                "local renderer timed out: " + " | ".join(self._console[-5:]),
                code="renderer_unavailable",
            ) from exc

    async def _start(self):
        import websockets
        from playwright.async_api import async_playwright

        self._ready = asyncio.Event()

        async def handler(ws):
            self._ws = ws
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    buf = bytes(raw)
                    size = int.from_bytes(buf[:4], "little")
                    header = json.loads(buf[4 : 4 + size].decode("utf-8"))
                    fut = self._pending.pop(header.get("frame_id"), None)
                    if fut is not None and not fut.done():
                        fut.set_result((header, buf[4 + size :]))
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "ready":
                    self.gpu = msg.get("gpu", self.gpu)
                    self.software = bool(msg.get("software"))
                    self.lighting = msg.get("lighting")
                    self._ready.set()
                elif msg.get("type") == "error":
                    self._page_error = msg
                    self.gpu = msg.get("gpu", self.gpu)
                    self._ready.set()

        self._server = await websockets.serve(handler, "127.0.0.1", 0, max_size=None)
        port = self._server.sockets[0].getsockname()[1]
        self._pw = await async_playwright().start()
        self._browser = await self._launch()
        self._page = await self._browser.new_page()
        self._page.on(
            "console",
            lambda m: m.type in ("warning", "error") and self._console.append(m.text[:300]),
        )
        self._page.on("pageerror", lambda e: self._console.append(f"pageerror: {e}"))
        # A real loopback HTTP server serves the page: Chrome's local network
        # access checks block a page that is not itself on loopback (including
        # one answered by request interception) from opening ws://127.0.0.1.
        self._http = _ProxyServer(self)
        origin = f"http://127.0.0.1:{self._http.port}"
        query = f"captureOnly=1&ws=ws://127.0.0.1:{port}/"
        if self._allow_software:
            query += "&allowSoftware=1"
        await self._page.goto(f"{origin}/index.html?{query}", wait_until="load")
        for _ in range(int(self._timeout * 10)):
            if self._ws is not None:
                break
            await asyncio.sleep(0.1)
        if self._ws is None:
            raise RendererError(
                "the capture page did not start: " + " | ".join(self._console[-5:]),
                code="renderer_unavailable",
            )
        await self._ws.send(json.dumps(self._scene_init))
        try:
            await asyncio.wait_for(self._ready.wait(), self._timeout)
        except TimeoutError as exc:
            raise RendererError(
                "scene loading timed out: " + " | ".join(self._console[-5:]),
                code="renderer_unavailable",
            ) from exc
        if self._page_error is not None:
            raise RendererError(
                f"Chrome has no hardware WebGL here ({self.gpu}); software rendering "
                "is very slow, so it is off by default",
                code="software_renderer",
                fixes=[
                    "check chrome://gpu, or set WE_RENDER_CHROME_ARGS for your GPU",
                    "pass render_options={'allow_software': True} to accept slow rendering",
                    "or pass images=False for a state-only session",
                ],
            )
        if self.software:
            warnings.warn(
                f"rendering camera images in software ({self.gpu}); expect a few frames per second",
                RuntimeWarning,
                stacklevel=2,
            )

    async def _launch(self):
        errors = []
        for channel in ("chrome", None):  # an installed Chrome, else Playwright's
            try:
                return await self._pw.chromium.launch(
                    channel=channel,
                    headless=True,
                    args=self._args,
                    ignore_default_args=["--disable-gpu", "--headless"],
                )
            except Exception as exc:  # noqa: BLE001 - try the next browser
                errors.append(str(exc).splitlines()[0])
        raise RendererError(
            "no Chrome or Chromium to render with: " + " | ".join(errors),
            code="renderer_unavailable",
            fixes=_INSTALL_FIXES,
        )

    async def _render(self, state):
        self._frame_id += 1
        state["frame_id"] = self._frame_id
        fut = self._loop.create_future()
        self._pending[self._frame_id] = fut
        await self._ws.send(json.dumps(state))
        return await fut

    def _provide(self, path: str) -> tuple[Path | bytes, str]:
        """The body for ``path`` (a cached file or bytes) and its content type."""
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".js"):
            ctype = "text/javascript"
        digest = self._digests.get(path)
        if digest is None:
            return self._read(path), ctype
        cached = self._cache / digest
        with self._download_lock:
            if not cached.is_file():
                self._download(path, digest, cached)
        return cached, ctype

    def _read(self, path: str) -> bytes:
        with self._fetch(path) as response:
            return response.read()

    def _download(self, path: str, digest: str, target: Path) -> None:
        partial = target.with_suffix(".part")
        h = hashlib.sha256()
        with self._fetch(path) as response, open(partial, "wb") as out:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != digest:
            partial.unlink(missing_ok=True)
            raise RendererError(f"asset {path} does not match its digest", code="asset_mismatch")
        partial.replace(target)

    async def _stop(self):
        for name in ("_browser", "_pw"):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    await (obj.close() if name == "_browser" else obj.stop())
                except Exception:
                    pass
        server = getattr(self, "_server", None)
        if server is not None:
            server.close()
            await server.wait_closed()
        if self._http is not None:
            self._http.stop()
            self._http = None
        self._ws = None


class _ProxyServer:
    """Loopback HTTP server for the capture page: each GET is answered from
    the digest cache or fetched from the server's /render/ prefix."""

    def __init__(self, renderer: LocalRenderer):
        import http.server

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                path = self.path.split("?", 1)[0]
                try:
                    body, ctype = renderer._provide(path)
                except FileNotFoundError:
                    self.send_error(404)
                    return
                except Exception as exc:  # noqa: BLE001 - reported in the console tail
                    renderer._console.append(f"fetch {path}: {exc}")
                    self.send_error(502)
                    return
                size = body.stat().st_size if isinstance(body, Path) else len(body)
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                if isinstance(body, Path):
                    with open(body, "rb") as f:
                        shutil.copyfileobj(f, self.wfile, 1 << 20)
                else:
                    self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
