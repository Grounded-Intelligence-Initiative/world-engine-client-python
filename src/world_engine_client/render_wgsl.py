"""Render camera observations on this machine with renderer D (wgpu-py and WGSL).

The server only simulates physics. At connect it sends the scene description
(``scene_init``) and a digest per asset, and with every observation the poses
(``render_state``). :class:`WGSLRenderer` downloads the assets into the render
cache (:mod:`world_engine_client.cache`; the hosting platform's arrive
encrypted and stay encrypted on disk), decrypts them into memory
(:mod:`world_engine_client.assets`), uploads them to the GPU and draws every
camera with the WGSL pipeline this package ships (``world_engine_client.wgsl``,
a copy of the engine's; a server never sends shaders). ``render(state)``
returns ``{camera name: (H, W, 3) uint8}``, read straight from the GPU into
numpy, with no browser and no image encoding. Each camera has its own size.

The adapter: a hardware GPU first, then a software one (Mesa's lavapipe, the
bundled SwiftShader of world-engine-swiftshader, Windows' WARP), each probed
in a subprocess; see :mod:`world_engine_client.wgsl.adapters`. A software
adapter renders with one RuntimeWarning; rendering below real time (1/0.034 s,
about 29.4 steps/s, timed after a warm-up) gets one RuntimeWarning too. When
nothing renders, :class:`NoRenderAdapter` says how to fix that, and
:func:`world_engine_client.connect` falls back to the headless Chrome path
(``render.LocalRenderer``) where it is installed; so it does for a scene D
cannot draw (:class:`SceneNotDrawable`), and on a machine whose GPU Vulkan
does not reach but Chrome reaches through OpenGL (see :func:`chrome_may_be_faster`).

Needs the ``render`` extra (wgpu, cryptography for the platform's encrypted
assets, Pillow for mesh textures): ``pip install 'world-engine-client[render]'``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from . import assets as asset_io
from .cache import SceneAssets, directory
from .errors import RendererError

WGSL_DIR = Path(__file__).resolve().parent / "wgsl"
PIPELINE = json.loads((WGSL_DIR / "pipeline.json").read_text())["pipeline"]
_lock = threading.Lock()


class NoRenderAdapter(RendererError):
    """No GPU and no software adapter renders here (``code`` renderer_unavailable,
    or software_renderer when ``allow_software=False`` left out a software one
    that might have)."""


class SceneNotDrawable(RendererError):
    """Renderer D cannot draw this scene: an asset it does not read (``code``
    asset_mismatch: a PLY or GLB it does not take), or more than the device's
    memory holds (renderer_unavailable). The browser path may still draw it."""


def available() -> bool:
    """True when this machine has what renderer D needs: wgpu and numpy (the
    ``render`` extra); the platform's encrypted assets also need cryptography."""
    from .render import _modules_found

    return _modules_found(("wgpu", "numpy"))


def _mode_of(bundle: dict | None) -> str:
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
    from .render import crypto_available

    if mode == "encrypted" and not crypto_available():
        raise RendererError(
            "the server's encrypted scene assets need the render extra (cryptography)",
            code="renderer_unavailable",
            fixes=[
                "pip install 'world-engine-client[render]'",
                "or pass images=False for a state-only session",
            ],
        )
    return mode


def _cache_dir(cache_dir) -> Path | None:
    """The render cache's directory, made as the render cache makes it (this
    user only); None when there is no usable one (then nothing is kept)."""
    from .cache import _private_dir

    try:
        root = directory(cache_dir)
        _private_dir(root)
        return root
    except Exception:  # noqa: BLE001 - no usable cache directory: probe every time
        return None


def _no_adapter(exc, allow_software: bool) -> NoRenderAdapter:
    code = "renderer_unavailable"
    fixes = list(exc.fixes)
    if not allow_software:
        code = "software_renderer"
        fixes.insert(0, "leave allow_software at its default (True) to render in software")
    return NoRenderAdapter(str(exc), code=code, fixes=fixes)


def find_adapter(
    *, adapter: str = "auto", allow_software: bool = True, cache_dir=None, timeout: float = 120.0
):
    """The adapter D would render with here, without opening it in this
    process: the search's ``AdapterChoice`` (each step probed in a subprocess,
    the result kept next to the render cache for a day). Raises NoRenderAdapter."""
    from .wgsl import adapters

    root = _cache_dir(cache_dir)
    with _lock:
        try:
            return adapters.search(
                prefer=adapter,
                allow_software=allow_software,
                cache_file=root / "adapters.json" if root else None,
                timeout=timeout,
            )
        except adapters.NoAdapterError as exc:
            raise _no_adapter(exc, allow_software) from None


def open_adapter(
    *,
    adapter: str = "auto",
    allow_software: bool = True,
    cache_dir=None,
    timeout: float = 120.0,
    choice=None,
):
    """The wgpu adapter D renders with and how it was found:
    ``(adapter, AdapterChoice)``. Probes once per environment (the result is
    kept next to the render cache for a day), or opens ``choice`` (from
    :func:`find_adapter`). Emits the software warning. Raises NoRenderAdapter."""
    from .wgsl import adapters

    root = _cache_dir(cache_dir)
    with _lock:  # one search at a time in this process (the instance is shared)
        try:
            found, choice = adapters.open_adapter(
                choice=choice,
                prefer=adapter,
                allow_software=allow_software,
                cache_file=root / "adapters.json" if root else None,
                timeout=timeout,
            )
        except adapters.NoAdapterError as exc:
            raise _no_adapter(exc, allow_software) from None
    if choice.software:
        adapters.warn_once("software", adapters.software_message(choice), stacklevel=4)
    return found, choice


# ------------------------------------------- D on the CPU, Chrome on the GPU
_BROWSER_FILE = "browser.json"


def _browser_key(adapter: str, allow_software: bool, chrome_args) -> str:
    import hashlib
    import importlib.metadata as md

    from .wgsl import adapters

    try:
        playwright = md.version("playwright")
    except Exception:  # noqa: BLE001
        playwright = None
    parts = [
        adapters.environment_key(adapter, allow_software),
        chrome_args,
        os.environ.get("WE_RENDER_CHROME_ARGS"),
        playwright,
    ]
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()


def chrome_may_be_faster(
    choice, *, adapter: str = "auto", allow_software: bool = True, cache_dir=None, chrome_args=None
) -> bool:
    """True when D would draw on the CPU (``choice`` is a software adapter)
    on a machine that shows a GPU device Vulkan does not reach (no Vulkan
    driver for the GPU, a virtual GPU, an old one): headless Chrome may reach
    that GPU through OpenGL, and is then much faster than D on the CPU. Not
    when the caller asked for a software adapter or one by name, nor when an
    earlier session here found Chrome on software too (kept a day)."""
    from .wgsl import adapters

    if not choice.software or adapter not in ("auto", "low-power"):
        return False
    if not adapters.gpu_devices():
        return False
    root = _cache_dir(cache_dir)
    if root is None:
        return True
    try:
        rec = json.loads((root / _BROWSER_FILE).read_text())
        fresh = time.time() - float(rec.get("time", 0)) <= adapters.CACHE_SECONDS
        known = rec.get("key") == _browser_key(adapter, allow_software, chrome_args) and fresh
        return not (known and not rec.get("hardware"))
    except (OSError, ValueError, TypeError, AttributeError):
        return True


def remember_browser(
    *,
    hardware: bool,
    gpu: str | None,
    error: str | None = None,
    adapter: str = "auto",
    allow_software: bool = True,
    cache_dir=None,
    chrome_args=None,
) -> None:
    """Keep, for this environment, whether headless Chrome reached the GPU
    (its WebGL renderer ``gpu``), or why it did not (``error``)."""
    root = _cache_dir(cache_dir)
    if root is None:
        return
    rec = {
        "key": _browser_key(adapter, allow_software, chrome_args),
        "time": time.time(),
        "hardware": bool(hardware),
        "gpu": gpu,
        "error": error,
    }
    try:
        tmp = root / f"{_BROWSER_FILE}.{os.getpid()}.part"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, root / _BROWSER_FILE)
    except OSError:
        pass


class WGSLRenderer:
    """Renderer D for one session; ``render(state)`` -> images.

    ``fetch(path)`` opens ``<server>/render<path>`` with the session's
    credentials. ``bundle`` is the server's render bundle (its ``cache`` says
    whether assets arrive encrypted), ``platform`` True for a session through
    the hosting platform. Assets are downloaded, or found in the render cache
    (``cache_dir``), and verified before anything is drawn; ``scene_name`` is the
    name the cache lists the scene under.

    ``adapter``: ``auto`` (a discrete GPU, an integrated one, then software),
    ``low-power`` (integrated first), ``software``, or ``name:SUBSTR`` (only an
    adapter whose name contains SUBSTR, e.g. ``name:Intel``).
    ``allow_software`` False refuses software adapters. ``color_format``
    overrides the colour target (see wgsl.render_wgsl.COLOR_FORMATS).
    ``timeout`` bounds each adapter probe. ``opened``: an ``(adapter,
    AdapterChoice)`` from :func:`open_adapter`, to skip the search."""

    def __init__(
        self,
        fetch,
        scene_init: dict,
        digests: dict[str, str],
        *,
        bundle: dict | None = None,
        platform: bool = False,
        origin: str = "",
        scene_id: str | None = None,
        scene_name: str | None = None,
        cache_dir=None,
        allow_software: bool = True,
        timeout: float = 300.0,
        adapter: str = "auto",
        color_format: str | None = None,
        opened=None,
    ):
        from .wgsl import adapters
        from .wgsl.budget import SplatMemoryError
        from .wgsl.render_wgsl import create_device
        from .wgsl.scene import SceneRenderer

        mode = _mode_of(bundle)
        self.assets: SceneAssets | None = None
        self._scene = None
        self.device = None
        if opened is None:
            opened = open_adapter(
                adapter=adapter,
                allow_software=allow_software,
                cache_dir=cache_dir,
                timeout=min(float(timeout), 120.0),
            )
        self._adapter, self.choice = opened
        self.software = bool(self.choice.software)
        self.gpu = self.choice.label
        try:
            self.device = create_device(self._adapter)
            self.assets = SceneAssets(
                fetch,
                dict(digests),
                encrypted=mode == "encrypted",
                platform=platform,
                cache_dir=cache_dir,
                scene_name=scene_name,
                scene_id=scene_id,
                origin=origin,
            )
            try:
                self.assets.prepare()
            except (OSError, sqlite3.Error) as exc:  # a full disk, a broken index
                raise RendererError(
                    f"the render cache failed: {exc}",
                    code="renderer_unavailable",
                    fixes=[
                        "set DIGIARENA_CACHE_DIR to a writable directory on a local disk",
                        "or pass images=False for a state-only session",
                    ],
                ) from exc
            started = time.perf_counter()
            try:
                self._scene = SceneRenderer(
                    scene_init,
                    self._read,
                    device=self.device,
                    color_format=color_format,
                    wgsl_dir=WGSL_DIR,
                )
            except SplatMemoryError as exc:
                raise SceneNotDrawable(str(exc), code="renderer_unavailable") from None
            except ValueError as exc:  # a PLY or GLB this renderer cannot read
                raise SceneNotDrawable(
                    f"the scene's assets cannot be drawn: {exc}", code="asset_mismatch"
                ) from None
            self.load_seconds = time.perf_counter() - started
        except BaseException:
            self.close()
            raise
        self._speed = adapters.SpeedCheck(self.choice)
        self.last_timings: dict = {}

    def _read(self, url: str):
        return asset_io.read(self.assets, url)

    # ----- public ------------------------------------------------------------- #
    @property
    def info(self) -> dict:
        """What renders this session's images, for ``info["render"]``."""
        c = self.choice
        out = {
            "renderer": "wgsl",
            "pipeline": PIPELINE,
            "adapter": c.device,
            "adapter_type": c.adapter_type,
            "backend": c.backend,
            "software": c.software,
            "step": c.step,
            "color_format": self._scene.color_format if self._scene else None,
        }
        if c.software:
            failed = [f"{t['step']}: {t.get('error')}" for t in c.trail if not t.get("ok")]
            out["reason"] = "; ".join(failed) or "no hardware adapter"
        if self._speed.rate is not None:
            out["steps_per_s"] = round(self._speed.rate, 2)
        return out

    def render(self, state: dict) -> dict[str, np.ndarray]:
        """Pose the scene as ``state`` (a ``render_state`` from the server) and
        return ``{camera name: (H, W, 3) uint8}``."""
        if self._scene is None:
            raise RendererError("renderer is closed", code="renderer_unavailable")
        started = time.perf_counter()
        try:
            images = self._scene.render(state)
        except Exception as exc:
            if type(exc).__module__.split(".")[0] == "wgpu":  # a lost device, a GPU error
                raise RendererError(
                    f"rendering failed on {self.gpu}: {exc}", code="renderer_unavailable"
                ) from exc
            raise
        seconds = time.perf_counter() - started
        self.last_timings = {"render_ms": seconds * 1e3, **self._scene.last_timing}
        self._speed.add(seconds)
        return images

    def close(self) -> None:
        if self._scene is not None:
            self._scene.close()
            self._scene = None
        if self.assets is not None:
            self.assets.close()  # the key and anything held in memory
        if self.device is not None:
            try:
                self.device.destroy()
            except Exception:  # noqa: BLE001 - already lost
                pass
            self.device = None

    def __enter__(self) -> WGSLRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def env_options(options: dict) -> tuple[dict, dict, str]:
    """Split ``connect(render_options=...)`` into (D's options, the browser
    path's options, the renderer asked for: auto, wgsl or chrome). D's own
    options (adapter, color_format) do not go to the browser path, nor its
    (chrome_args, auto_install) to D; any other name goes to both, and the
    renderer that starts refuses one it does not know."""
    opts = dict(options or {})
    renderer = opts.pop("renderer", None) or os.environ.get("WE_RENDERER") or "auto"
    if renderer not in ("auto", "wgsl", "chrome"):
        raise ValueError("render_options['renderer'] must be 'auto', 'wgsl' or 'chrome'")
    d_only, chrome_only = ("adapter", "color_format"), ("chrome_args", "auto_install")
    d = {k: v for k, v in opts.items() if k not in chrome_only}
    if "adapter" not in d and os.environ.get("WE_RENDER_ADAPTER"):
        d["adapter"] = os.environ["WE_RENDER_ADAPTER"]
    chrome = {k: v for k, v in opts.items() if k not in d_only}
    return d, chrome, renderer
