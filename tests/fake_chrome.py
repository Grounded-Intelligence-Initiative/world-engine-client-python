"""Chrome and the capture page, played over real sockets.

``install()`` replaces Playwright with a stand-in. The renderer's own loopback
HTTP server and its real WebSocket server stay. The page does what the
engine's capture page does: it connects to the ``?ws=`` URL as given, and on
``scene_init`` it GETs every asset URL from the page's origin with the extra
headers its context was given (Playwright adds those to every request).
Each ``state`` is answered with one PNG per camera.

What the page received is in ``SIM.received`` (URL -> bytes), what failed in
``SIM.failed`` (URL -> HTTP status or error). Used in-process by the tests and
by ``render_worker.py`` in other processes.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from types import ModuleType, SimpleNamespace


def asset_urls(scene_init: dict) -> list[str]:
    """Every ``url`` in a scene_init, in order, without repeats."""
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "url" and isinstance(value, str) and value not in found:
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(scene_init)
    return found


class Sim:
    def __init__(self):
        self.reset()

    def reset(self):
        self.received: dict[str, bytes] = {}
        self.failed: dict[str, object] = {}
        self.headers: list[dict] = []  # extra_http_headers of each page
        self.ws_urls: list[str] = []
        self.goto_urls: list[str] = []
        self.goto_error: str | None = None  # goto raises this, {url} filled in
        self.gpu = "ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 4090/PCIe/SSE2, OpenGL ES 3.2)"
        self.color = (40, 120, 200)
        self.pages: list[FakePage] = []

    def playwright(self):
        async def start():
            return SimpleNamespace(chromium=SimpleNamespace(launch=self.launch), stop=_done)

        return SimpleNamespace(start=start)

    async def launch(self, channel=None, args=(), **_):
        return FakeBrowser(self, list(args))


async def _done():
    return None


SIM = Sim()


class FakeBrowser:
    def __init__(self, sim: Sim, args):
        self.sim, self.args, self.tasks = sim, args, []

    async def new_page(self, **options):
        page = FakePage(self, dict(options.get("extra_http_headers") or {}))
        self.sim.pages.append(page)
        return page

    async def close(self):
        for task in self.tasks:
            task.cancel()


class FakePage:
    def __init__(self, browser: FakeBrowser, headers: dict):
        self.browser, self.headers, self.handlers = browser, headers, {}
        browser.sim.headers.append(headers)

    def on(self, event, handler):
        self.handlers[event] = handler

    async def goto(self, url, **_):
        self.browser.sim.goto_urls.append(url)
        if self.browser.sim.goto_error is not None:
            raise RuntimeError(self.browser.sim.goto_error.format(url=url))
        parts = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parts.query)
        origin = f"{parts.scheme}://{parts.netloc}"
        sim = self.browser.sim
        # Load the page and the scripts it names, as Chrome does.
        for path in [parts.path]:
            html = await self._fetch(origin, path)
            if html is None:
                return
            for script in re.findall(rb'(?:src|href)="(/assets/[^"]+)"', html):
                await self._fetch(origin, script.decode())
        ws_url = query["ws"][0]
        sim.ws_urls.append(ws_url)
        task = asyncio.get_running_loop().create_task(self._run(origin, ws_url))
        self.browser.tasks.append(task)

    async def _fetch(self, origin: str, path: str) -> bytes | None:
        sim = self.browser.sim
        try:
            data = await asyncio.to_thread(self._get, origin, path)
        except urllib.error.HTTPError as exc:
            sim.failed[path] = exc.code
            return None
        except Exception as exc:  # noqa: BLE001 - recorded for the test
            sim.failed[path] = repr(exc)
            return None
        sim.received[path] = data
        return data

    def _get(self, origin: str, path: str) -> bytes:
        request = urllib.request.Request(origin + path, headers=self.headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    async def _run(self, origin: str, ws_url: str):
        import websockets

        sim = self.browser.sim
        async with websockets.connect(ws_url, max_size=None) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "scene_init":
                    for path in asset_urls(msg):
                        await self._fetch(origin, path)
                    await ws.send(json.dumps({"type": "ready", "gpu": sim.gpu, "software": False}))
                elif msg.get("type") == "state":
                    await ws.send(frame(msg, sim.color))


def frame(state: dict, color) -> bytes:
    """The capture page's reply: [u32 header length][header][images]."""
    from PIL import Image

    images, blobs, offset = [], [], 0
    for cam in state.get("cameras") or [{"name": "front", "w": 8, "h": 6}]:
        buf = io.BytesIO()
        Image.new("RGB", (cam["w"], cam["h"]), tuple(color)).save(buf, "PNG")
        data = buf.getvalue()
        images.append(
            {
                "name": cam["name"],
                "w": cam["w"],
                "h": cam["h"],
                "offset": offset,
                "length": len(data),
            }
        )
        blobs.append(data)
        offset += len(data)
    header = json.dumps({"frame_id": state["frame_id"], "images": images}).encode()
    return len(header).to_bytes(4, "little") + header + b"".join(blobs)


def install(monkeypatch=None) -> Sim:
    """Replace Playwright (only) with the stand-in; returns the shared Sim.
    With ``monkeypatch`` the change is undone after the test. render.available
    is replaced by a function of no arguments, as the engine's tests do: the
    client must keep calling it so."""
    from world_engine_client import render

    SIM.reset()
    playwright = ModuleType("playwright")
    async_api = SimpleNamespace(async_playwright=SIM.playwright)
    if monkeypatch is not None:
        monkeypatch.setattr(render, "available", lambda: True)
        monkeypatch.setitem(sys.modules, "playwright", playwright)
        monkeypatch.setitem(sys.modules, "playwright.async_api", async_api)
        monkeypatch.delenv("WE_RENDER_CHROME_ARGS", raising=False)
    else:
        render.available = lambda: True
        sys.modules["playwright"] = playwright
        sys.modules["playwright.async_api"] = async_api
    return SIM
