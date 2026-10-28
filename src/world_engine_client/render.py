"""Render camera observations on this machine (``render="local"``).

The server only simulates physics. It sends the scene description once at
connect and the poses with every observation; a headless Chrome started here
runs the same capture page the server-side renderer runs, so the images match.
Headless Chrome is never a "background tab", so nothing throttles it.

The capture page, its scripts and the scene assets are fetched from the
server's ``/render/`` prefix with this session's credentials, before Chrome
starts. Assets are kept in the render cache (:mod:`world_engine_client.cache`)
between sessions. The hosting platform sends them encrypted (DAC1); they stay
encrypted on disk, and the key for them is held in memory only.

Chrome reads the page and the assets from a loopback HTTP server here, which
decrypts in memory, and sends its images back over a loopback WebSocket. Other
processes on this machine can reach both ports, so both want this session's
random token: Playwright adds it to every request the page makes, and the
WebSocket URL the page gets ends in it (BUG-32).

Needs the ``render`` extra (Playwright, websockets, Pillow, cryptography) and a
Chrome or Chromium: ``pip install 'world-engine-client[render]'``. An installed Google
Chrome is used first; otherwise Playwright's Chromium, which is downloaded
automatically the first time (``WE_RENDER_AUTO_INSTALL=0`` turns that off;
``python -m playwright install chromium`` does it by hand).

Without a hardware GPU the images are rendered in software (SwiftShader), with
a RuntimeWarning; ``allow_software=False`` refuses that instead. A hardware
adapter that renders slowly gets one RuntimeWarning too (see ``_SLOW_STEPS``).
"""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import logging
import mimetypes
import os
import re
import secrets
import shlex
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable
from http import HTTPStatus

import numpy as np

from .cache import _HASHED_PAGE_FILE, SceneAssets
from .errors import RendererError

# Hardware WebGL in headless Chrome, per platform, tried in order until one
# loads the scene on a GPU and renders a frame. On Linux, ANGLE over native
# EGL/GLES works on NVIDIA RTX 4090 (driver 580), where ANGLE over Vulkan hangs
# on the first frame; the Vulkan set is the one the server renderer was
# verified with on an RTX 6000 Ada. macOS uses Metal; Windows Chrome's default
# (D3D11). WE_RENDER_CHROME_ARGS or chrome_args replaces the list.
if sys.platform == "darwin":
    _CANDIDATES = [["--headless=new", "--use-angle=metal", "--ignore-gpu-blocklist"]]
elif sys.platform.startswith("linux"):
    _CANDIDATES = [
        ["--headless=new", "--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist"],
        [
            "--headless=new",
            "--ozone-platform=headless",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--use-gl=angle",
            "--ignore-gpu-blocklist",
        ],
    ]
    _CANDIDATES = [c + ["--no-sandbox"] for c in _CANDIDATES]
else:
    _CANDIDATES = [["--headless=new", "--ignore-gpu-blocklist"]]
_SOFTWARE = ["--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
# A GPU path that hangs, hangs on the first frame. Only hardware rendering is
# held to this; software rendering is slow, not stuck, and gets the full timeout.
_PROBE_SECONDS = 15.0
_PAGE_ERROR_GRACE = 5.0  # seconds a page that threw before connecting gets to connect
# Launching Chrome and loading the scene, over all flag sets together, may take
# ``timeout`` plus this many seconds. Without a shared limit, a scene that never
# loads would cost a full timeout per flag set. A software first frame gets the
# whole ``timeout`` on top, and a Chromium download does not count.
_START_GRACE = 30.0
# Software adapters by their WebGL renderer name: SwiftShader, Mesa's llvmpipe
# and lavapipe, Windows' WARP ("Microsoft Basic Render Driver").
_SOFTWARE_GPU = re.compile(
    r"swiftshader|llvmpipe|lavapipe|softpipe|basic render driver|software", re.IGNORECASE
)

# Speed check of hardware rendering (software rendering has its own warning).
# This browser path is slow even on a fast GPU: an RTX 4090 renders three
# 256x256 cameras of 0.4M-1.6M splat scenes at 6.8-17.2 steps/s (render only,
# 2026-10-26). Below half of the slowest of those, rendering limits the session
# and one RuntimeWarning per process says so. Timed after the first frames,
# which carry one-time costs; the median keeps a single slow frame from counting.
_SLOW_STEPS = 3.0
_SPEED_SKIP = 2  # renders after connect that are not timed
_SPEED_SAMPLES = 10  # renders timed after those
_slow_warned = False

_INSTALL_FIXES = [
    "pip install 'world-engine-client[render]' (Chromium downloads on first use)",
    "or pass images=False for a state-only session",
]
_SYSTEM_DEPS_FIX = (
    "on Linux, install Chromium's system libraries: "
    "python -m playwright install --with-deps chromium (needs sudo)"
)
_install_lock = threading.Lock()
_installed = False

# Playwright sends the session token in this header with every request the
# page makes; the loopback HTTP server refuses requests without it.
_TOKEN_HEADER = "X-World-Engine-Token"
# What the loopback server fetches from the server for the page, besides the
# session's assets: the page and the files of its build.
_PAGE_FILE = re.compile(r"/index\.html|/assets/[A-Za-z0-9_-][A-Za-z0-9_.-]*")

# websockets logs every handshake, its path included, when its logger is set
# to DEBUG, and the path carries the session token. The WebSocket server logs
# here instead, and this logger stops at INFO.
_WS_LOG = logging.getLogger("world_engine_client.render.ws")
_WS_LOG.setLevel(logging.INFO)
log = logging.getLogger("world_engine_client.render")


def _same_secret(sent: str, expected: str) -> bool:
    """Compare a received token with the session's in constant time."""
    return hmac.compare_digest(sent.encode("utf-8", "replace"), expected.encode("utf-8"))


def _browser_missing(message: str) -> bool:
    return "Executable doesn't exist" in message or "playwright install" in message


def _system_deps_missing(message: str) -> bool:
    text = message.lower()
    return "missing dependencies" in text or "shared libraries" in text


def _auto_install_enabled(option: bool | None) -> bool:
    """Download Chromium when no browser is found, unless turned off by the
    option or WE_RENDER_AUTO_INSTALL=0."""
    if option is not None:
        return option
    return os.environ.get("WE_RENDER_AUTO_INSTALL", "1") != "0"


def install_chromium() -> None:
    """Download Playwright's Chromium, at most once per process.

    Used when neither an installed Chrome nor Playwright's Chromium is
    available, so ``pip install 'world-engine-client[render]'`` is the only
    step most users need. Progress goes to the terminal."""
    global _installed
    with _install_lock:
        if _installed:
            return
        print(
            "world-engine-client: downloading Chromium for camera rendering "
            "(a download of about 300 MB, only once) ...",
            file=sys.stderr,
            flush=True,
        )
        done = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
        if done.returncode != 0:
            raise RendererError(
                "downloading Chromium failed",
                code="renderer_unavailable",
                fixes=["python -m playwright install chromium", _SYSTEM_DEPS_FIX],
            )
        _installed = True


_RENDER_MODULES = ("playwright", "websockets", "PIL")


def _modules_found(modules: tuple[str, ...]) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in modules)


def available() -> bool:
    """True when the ``render`` extra is installed (Chrome is checked on start).
    The extra includes cryptography, which the hosting platform's encrypted
    scene assets need. A local ``world-engine serve`` sends its scene plain:
    for that the rest is enough (:func:`_available_plain`)."""
    return _modules_found(_RENDER_MODULES + ("cryptography",))


def _available_plain() -> bool:
    """True when a scene that arrives plain can be rendered here: the render
    extra apart from cryptography. Kept apart from :func:`available`, which
    tests elsewhere replace with a function of no arguments."""
    return _modules_found(_RENDER_MODULES)


def candidate_args(chrome_args=None, allow_software=True) -> list[list[str]]:
    """The Chrome flag sets to try, in order. An explicit set (argument or
    WE_RENDER_CHROME_ARGS) is the only one tried; software rendering comes last
    unless ``allow_software`` is False."""
    override = os.environ.get("WE_RENDER_CHROME_ARGS")
    if chrome_args or override:
        return [list(chrome_args or shlex.split(override))]
    candidates = [list(c) for c in _CANDIDATES]
    if allow_software:
        candidates.append(_SOFTWARE + ["--no-sandbox"])
    return candidates


Fetch = Callable[[str], "io.BufferedIOBase"]


class LocalRenderer:
    """Headless Chrome running the capture page; ``render(state)`` -> images.

    ``fetch(path)`` opens ``<server>/render<path>`` with the session's
    credentials and returns a readable response. ``bundle`` is the server's
    render bundle: its ``cache`` says whether assets arrive encrypted.
    ``platform`` is True for a session through the hosting platform (it never
    gets plain assets written to disk). Assets are downloaded, or found in the
    render cache (``cache_dir``, by default the user cache directory), and
    verified before Chrome starts. ``scene_name`` is the name the cache lists
    the scene under.

    Without hardware WebGL the page renders in software after a
    RuntimeWarning; ``allow_software=False`` makes that an error. ``timeout``
    bounds loading the page, loading the scene and each frame. The start as a
    whole, over all Chrome flag sets, gets ``timeout`` + 30 s, and ``timeout``
    more for a software first frame; downloading Chromium or the scene does
    not count."""

    def __init__(
        self,
        fetch: Fetch,
        scene_init: dict,
        digests: dict[str, str],
        *,
        bundle: dict | None = None,
        platform: bool = False,
        origin: str = "",
        scene_id: str | None = None,
        scene_name: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        allow_software: bool = True,
        chrome_args: list[str] | None = None,
        auto_install: bool | None = None,
        timeout: float = 300.0,
    ):
        mode = (bundle or {}).get("cache") or "plain"
        if mode not in ("encrypted", "plain"):
            raise RendererError(
                f"the server keeps scene assets in a way this client does not know ({mode!r})",
                code="client_upgrade_required",
                fixes=["pip install -U 'world-engine-client[render]'"],
            )
        if mode == "encrypted" and (bundle or {}).get("encryption", "dac1") != "dac1":
            raise RendererError(
                f"the server encrypts scene assets as {bundle.get('encryption')!r}; "
                "this client reads DAC1",
                code="client_upgrade_required",
                fixes=["pip install -U 'world-engine-client[render]'"],
            )
        if not available() and (mode == "encrypted" or not _available_plain()):
            needs = (
                "the server's encrypted scene assets need the render extra (cryptography)"
                if _available_plain()
                else "local rendering needs the render extra"
            )
            raise RendererError(needs, code="renderer_unavailable", fixes=_INSTALL_FIXES)
        self._fetch = fetch
        self._scene_init = scene_init
        self._digests = dict(digests)
        # Other processes here can reach the loopback ports: only the page
        # gets this token, so only the page is served (BUG-32).
        self._token = secrets.token_urlsafe(32)
        self.assets: SceneAssets | None = None
        self._allow_software = allow_software
        self._candidates = candidate_args(chrome_args, allow_software)
        self._auto_install = _auto_install_enabled(auto_install)
        self.chrome_args: list[str] | None = None  # the set that worked
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
        self._failures: list[str] = []  # "<flag set> -> <error>" for each set that failed
        self._http = None
        self._page_kept: dict[str, bytes] = {}  # page files served from memory
        self._render_seconds: list[float] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        try:
            self.assets = SceneAssets(
                fetch,
                self._digests,
                encrypted=mode == "encrypted",
                platform=platform,
                cache_dir=cache_dir,
                scene_name=scene_name,
                scene_id=scene_id,
                origin=origin,
            )
            # Downloads have their own limit (no new bytes for 60 s): a large
            # scene on a slow line does not count against the start. So do the
            # page's scripts: Playwright gives the page 30 s to load.
            try:
                self.assets.prepare()
                self._page_kept = self._prefetch_page()
            except (OSError, sqlite3.Error) as exc:  # a full disk, a broken index
                raise RendererError(
                    f"the render cache failed: {exc}",
                    code="renderer_unavailable",
                    fixes=[
                        "set DIGIARENA_CACHE_DIR to a writable directory on a local disk",
                        "or pass images=False for a state-only session",
                    ],
                ) from exc
            # The start may move this: see _attempt and _launch.
            self._deadline = time.monotonic() + timeout + _START_GRACE
            self._wait_for_start()
        except BaseException:
            self.close()
            raise

    # ----- public ------------------------------------------------------------- #
    def render(self, state: dict) -> dict[str, np.ndarray]:
        """Pose the scene as ``state`` (a ``render_state`` from the server) and
        return ``{camera name: (H, W, 3) uint8}``."""
        if self._ws is None:
            raise RendererError("renderer is closed", code="renderer_unavailable")
        started = time.perf_counter()
        header, payload = self._call(self._render(dict(state)), self._timeout)
        from PIL import Image

        images = {}
        for spec in header.get("images", []):
            chunk = payload[spec["offset"] : spec["offset"] + spec["length"]]
            with Image.open(io.BytesIO(chunk)) as img:
                images[spec["name"]] = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        self.last_timings = header.get("timings", {})
        self._check_speed(time.perf_counter() - started)
        return images

    def close(self) -> None:
        if self._loop.is_running():
            try:
                self._call(self._stop(), 30)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
        if not self._thread.is_alive() and not self._loop.is_closed():
            self._loop.close()
        if self.assets is not None:
            self.assets.close()  # the key and anything held in memory

    def __enter__(self) -> LocalRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----- loop thread ---------------------------------------------------------- #
    def _wait_for_start(self) -> None:
        """Run the start on the loop thread until it ends or the deadline
        passes. The start can move the deadline, so it is read every second."""
        began = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(self._start(), self._loop)
        while not future.done() and time.monotonic() < self._deadline:
            try:
                future.result(max(0.0, min(self._deadline - time.monotonic(), 1.0)))
            except TimeoutError:
                pass
        if future.done():
            future.result()
            return
        future.cancel()
        detail = " | ".join(self._failures or self._console[-5:]) or "no flag set finished"
        detail = self._redact(detail)
        raise RendererError(
            f"the local renderer did not start within {time.monotonic() - began:.0f} s: {detail}",
            code="renderer_unavailable",
        )

    def _call(self, coro, timeout):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except TimeoutError as exc:
            future.cancel()
            raise RendererError(
                "local renderer timed out: " + self._redact(" | ".join(self._console[-5:])),
                code="renderer_unavailable",
            ) from exc

    async def _start(self):
        import websockets
        from playwright.async_api import async_playwright

        self._ready = asyncio.Event()

        async def handler(ws):
            # Only a client that knows the token gets here: the page, also
            # when it reconnects.
            self._ws = ws
            async for raw in ws:
                self._receive(raw)

        self._server = await websockets.serve(
            handler,
            "127.0.0.1",
            0,
            max_size=None,
            process_request=self._check_ws_token,
            logger=_WS_LOG,
        )
        self._ws_port = self._server.sockets[0].getsockname()[1]
        self._pw = await async_playwright().start()
        # A real loopback HTTP server serves the page: Chrome's local network
        # access checks block a page that is not itself on loopback (including
        # one answered by request interception) from opening ws://127.0.0.1.
        self._http = _ProxyServer(self)
        await self._try_candidates()

    async def _check_ws_token(self, connection_or_path, request_or_headers):
        """websockets' process_request hook: a handshake whose path is not
        ``/<token>`` gets 403 and never reaches the handler. websockets 14 and
        later pass (connection, request), older releases (path, headers)."""
        legacy = isinstance(connection_or_path, str)
        path = connection_or_path if legacy else request_or_headers.path
        if _same_secret(path, "/" + self._token):
            return None
        if legacy:
            return HTTPStatus.FORBIDDEN, [], b"missing or wrong session token\n"
        return connection_or_path.respond(HTTPStatus.FORBIDDEN, "missing or wrong session token\n")

    def _redact(self, text: str) -> str:
        # The page URL carries the token, and console lines and start errors
        # end up in error messages and logs.
        return text.replace(self._token, "<token>") if getattr(self, "_token", "") else text

    def _receive(self, raw) -> None:
        """One message from the capture page: a frame, or ready/error."""
        if isinstance(raw, (bytes, bytearray)):
            buf = bytes(raw)
            size = int.from_bytes(buf[:4], "little")
            header = json.loads(buf[4 : 4 + size].decode("utf-8"))
            fut = self._pending.pop(header.get("frame_id"), None)
            if fut is not None and not fut.done():
                fut.set_result((header, buf[4 + size :]))
            return
        msg = json.loads(raw)
        if msg.get("type") == "ready":
            self.gpu = msg.get("gpu", self.gpu)
            # The page knows SwiftShader and llvmpipe; Windows' WARP is software too.
            found = _SOFTWARE_GPU.search(str(self.gpu))
            self.software = bool(msg.get("software")) or found is not None
            self.lighting = msg.get("lighting")
            self._ready.set()
        elif msg.get("type") == "error":
            self._page_error = msg
            self.gpu = msg.get("gpu", self.gpu)
            self._ready.set()

    async def _try_candidates(self):
        """Try the Chrome flag sets in order until one renders a frame."""
        failures = []
        for index, args in enumerate(self._candidates):
            last = index == len(self._candidates) - 1
            try:
                await self._attempt(args, accept_software=self._allow_software and last)
                self.chrome_args = args
                return
            except RendererError as exc:
                failures.append((exc, args))
                self._failures.append(self._redact(f"{' '.join(args)} -> {exc}"))
                await self._close_browser()
        no_hardware = any(e.code == "software_renderer" for e, _ in failures)
        if no_hardware and not self._allow_software:
            raise RendererError(
                f"Chrome has no hardware WebGL here ({self.gpu}), and allow_software=False "
                "turns software rendering off",
                code="software_renderer",
                fixes=[
                    "check chrome://gpu, or set WE_RENDER_CHROME_ARGS for your GPU",
                    "leave allow_software at its default (True) to accept slow software rendering",
                    "or pass images=False for a state-only session",
                ],
            )
        raise RendererError(
            "no Chrome configuration rendered a frame: " + " | ".join(self._failures),
            code="renderer_unavailable",
        )

    async def _attempt(self, args: list[str], *, accept_software: bool):
        """Launch Chrome with ``args``, load the scene and render one frame."""
        self._ws, self._page_error, self.software = None, None, False
        self._ready = asyncio.Event()
        self._browser = await self._launch(args)
        # Playwright adds the header to every request the page makes: the
        # page, its scripts and the scene assets. A fresh context each time:
        # nothing of the page is kept on disk.
        self._page = await self._browser.new_page(extra_http_headers={_TOKEN_HEADER: self._token})
        self._page.on(
            "console",
            lambda m: (
                m.type in ("warning", "error") and self._console.append(self._redact(m.text[:300]))
            ),
        )
        failed_at = []  # when the page first threw an uncaught error

        def page_error(error):
            self._console.append(self._redact(f"pageerror: {error}"))
            failed_at.append(time.monotonic())

        self._page.on("pageerror", page_error)
        origin = f"http://127.0.0.1:{self._http.port}"
        # The page opens the ?ws= URL as given, so the token rides in its path.
        query = f"captureOnly=1&ws=ws://127.0.0.1:{self._ws_port}/{self._token}"
        if accept_software:
            query += "&allowSoftware=1"
        try:
            await self._page.goto(f"{origin}/index.html?{query}", wait_until="load")
        except Exception as exc:  # noqa: BLE001 - Playwright's errors name the URL, token included
            raise RendererError(
                "the capture page did not load: " + self._redact(str(exc))[:1000],
                code="renderer_unavailable",
            ) from None
        deadline = time.monotonic() + self._timeout
        while self._ws is None and time.monotonic() < deadline:
            # The page opens its socket at the end of its start-up script. An
            # uncaught error with no socket soon after (no WebGL with these
            # flags, for one) means it never will.
            if failed_at and time.monotonic() - failed_at[0] > _PAGE_ERROR_GRACE:
                break
            await asyncio.sleep(0.1)
        if self._ws is None:
            raise RendererError(
                "the capture page did not start: " + self._redact(" | ".join(self._console[-5:])),
                code="renderer_unavailable",
            )
        await self._ws.send(json.dumps(self._scene_init))
        try:
            await asyncio.wait_for(self._ready.wait(), self._timeout)
        except TimeoutError as exc:
            raise RendererError(
                "scene loading timed out: " + self._redact(" | ".join(self._console[-5:])),
                code="renderer_unavailable",
            ) from exc
        if self._page_error is not None or (self.software and not self._allow_software):
            raise RendererError(f"software renderer ({self.gpu})", code="software_renderer")
        limit = _PROBE_SECONDS
        if self.software:
            # Software rendering is slow, not stuck: warn now, since the first
            # frame of a real scene can take minutes, and give it the timeout.
            warnings.warn(
                f"no hardware GPU for camera images, rendering in software ({self.gpu}). "
                "Real scenes take seconds per step; pass images=False for a state-only "
                "session, or check chrome://gpu if this machine has a GPU",
                RuntimeWarning,
                stacklevel=1,
            )
            limit = self._timeout
            # However long loading took, this frame gets the whole timeout.
            self._deadline = max(self._deadline, time.monotonic() + limit + _START_GRACE)
        probe = {
            "type": "state",
            "bodies": [],
            "robot": {"ee_pos": [0, 0, 0], "ee_quat": [0, 0, 0, 1], "gripper": 0.0},
            "cameras": self._scene_init.get("cameras", []),
        }
        try:
            await asyncio.wait_for(self._render(probe), limit)
        except TimeoutError as exc:
            raise RendererError(
                f"the first frame did not render within {limit:.0f} s ({self.gpu})",
                code="renderer_unavailable",
            ) from exc

    def _check_speed(self, seconds: float) -> None:
        """Time the first renders; warn once per process when a hardware adapter
        renders below _SLOW_STEPS. Software rendering has its own warning."""
        global _slow_warned
        needed = _SPEED_SKIP + _SPEED_SAMPLES
        if self.software or _slow_warned or len(self._render_seconds) >= needed:
            return
        self._render_seconds.append(seconds)
        if len(self._render_seconds) < needed:
            return
        rate = 1.0 / statistics.median(self._render_seconds[_SPEED_SKIP:])
        if rate < _SLOW_STEPS:
            _slow_warned = True
            warnings.warn(
                f"camera rendering is slow here: {rate:.1f} steps/s over {_SPEED_SAMPLES} steps "
                f"on {self.gpu} (a hardware adapter, not software), below {_SLOW_STEPS:g} "
                "steps/s. Pass images=False for a state-only session, or cameras=[...] to "
                "render fewer cameras",
                RuntimeWarning,
                stacklevel=1,
            )

    async def _close_browser(self):
        browser, self._browser = getattr(self, "_browser", None), None
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass
        self._ws = None
        self._pending.clear()

    async def _launch(self, args: list[str]):
        errors = []
        for channel in ("chrome", None):  # an installed Chrome, else Playwright's
            try:
                return await self._open(channel, args)
            except Exception as exc:  # noqa: BLE001 - try the next browser
                errors.append(str(exc))
        if self._auto_install and _browser_missing(errors[-1]):
            # A download is slow, not stuck: it does not count against the start.
            deadline, self._deadline = self._deadline, float("inf")
            downloading = time.monotonic()
            try:
                await asyncio.to_thread(install_chromium)
            finally:
                self._deadline = deadline + time.monotonic() - downloading
            try:
                return await self._open(None, args)
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(str(exc))
        if _system_deps_missing(errors[-1]):
            fixes = [_SYSTEM_DEPS_FIX]
        elif _browser_missing(errors[-1]):
            # Only reached with the automatic download turned off.
            fixes = [
                "python -m playwright install chromium (automatic download is off)",
                "or pass images=False for a state-only session",
            ]
        else:
            fixes = list(_INSTALL_FIXES)
        raise RendererError(
            "no Chrome or Chromium to render with: "
            + " | ".join(e.splitlines()[0] for e in errors),
            code="renderer_unavailable",
            fixes=fixes,
        )

    async def _open(self, channel: str | None, args: list[str]):
        return await self._pw.chromium.launch(
            channel=channel,
            headless=True,
            args=args,
            ignore_default_args=["--disable-gpu", "--headless"],
        )

    async def _render(self, state):
        self._frame_id += 1
        state["frame_id"] = self._frame_id
        fut = self._loop.create_future()
        self._pending[self._frame_id] = fut
        await self._ws.send(json.dumps(state))
        return await fut

    def _prefetch_page(self) -> dict[str, bytes]:
        """index.html and the scripts it names, before Chrome starts. Returns
        what the page gets from memory this session: index.html, and the
        scripts as well when there is no cache to keep them in. A failure is
        left to the page load, which fetches the file again and reports it."""
        kept: dict[str, bytes] = {}
        try:
            html = kept["/index.html"] = self.assets.page_file("/index.html")
            for ref in sorted(set(re.findall(rb'(?:src|href)="(/assets/[^"]+)"', html))):
                path = ref.decode("ascii", "replace")
                if _HASHED_PAGE_FILE.fullmatch(path):
                    data = self.assets.page_file(path)
                    if self.assets.cache is None:
                        kept[path] = data
        except Exception as exc:  # noqa: BLE001 - see above
            log.info("fetching the capture page before Chrome starts failed: %s", exc)
        return kept

    def _provide(self, path: str) -> tuple[int, object, str]:
        """``(length, body, content type)`` for a page request: ``body`` is
        bytes, or an iterator of verified plaintext pieces of a scene asset.
        Raises FileNotFoundError for what the page may not have."""
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".js"):
            ctype = "text/javascript"
        if path in self._digests:
            return self.assets.length(path), self.assets.open(path), ctype
        if not _PAGE_FILE.fullmatch(path):
            # Fetched with this session's credentials: only the page's own
            # files, never another asset, /__key or a path in another
            # spelling (the server may decode %5F%5Fkey to __key).
            raise FileNotFoundError(path)
        data = self._page_kept.get(path)
        if data is None:
            data = self.assets.page_file(path)
        return len(data), data, ctype

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
    """Loopback HTTP server for the capture page, for requests that carry the
    session token: scene assets come decrypted from the render cache (in
    memory; nothing plain is written), the page's files from the cache or the
    server's /render/ prefix. Nothing it sends may be cached."""

    def __init__(self, renderer: LocalRenderer):
        import http.server

        token = renderer._token

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self, with_body: bool) -> None:
                # Other local processes can reach this port: without the
                # session token they get nothing, and nothing is fetched for
                # them with the session's credentials (BUG-32).
                if not _same_secret(self.headers.get(_TOKEN_HEADER, ""), token):
                    self.send_error(HTTPStatus.FORBIDDEN, "missing or wrong session token")
                    return
                path = self.path.split("?", 1)[0]
                try:
                    length, body, ctype = renderer._provide(path)
                except FileNotFoundError:
                    self.send_error(404)
                    return
                except Exception as exc:  # noqa: BLE001 - reported in the console tail
                    renderer._console.append(f"fetch {path}: {exc}")
                    self.send_error(502)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if not with_body:
                    return
                if isinstance(body, bytes):
                    self.wfile.write(body)
                    return
                try:
                    for piece in body:
                        self.wfile.write(piece)
                except (ConnectionError, OSError):
                    return  # the page went away
                except Exception as exc:  # noqa: BLE001 - a cache file changed under us
                    renderer._console.append(f"serve {path}: {exc}")
                    self.close_connection = True

            def do_GET(self):  # noqa: N802 - http.server API
                self._answer(True)

            def do_HEAD(self):  # noqa: N802 - http.server API
                self._answer(False)

            def log_message(self, *args):
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
