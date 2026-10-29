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

Pipeline wgsl/1 draws the splats as Spark does and the meshes as the engine's
capture page does: three.js's MeshStandardMaterial from the objects' WEM1 (the
engine compiles each GLB) and the robot links, lit by the scene's own lighting
baked at its light probe (or its HDRI, or three's room environment), plus the
key light's shadow, blended above 8 bits and rounded to 8 bits once.

The adapter: a hardware GPU first, then a software one (Mesa's lavapipe, the
bundled SwiftShader of world-engine-swiftshader, Windows' WARP), each probed
in a subprocess; see :mod:`world_engine_client.wgsl.adapters`. A software
adapter renders with one RuntimeWarning; rendering below real time (1/0.034 s,
about 29.4 steps/s, timed after a warm-up) gets one RuntimeWarning too. When
D fails on the adapter it opened (:class:`PipelineError`: a pipeline the
backend's compiler rejects, a validation error), :func:`start_renderer` takes
the next adapter of the search, with one RuntimeWarning. When nothing
renders, :class:`NoRenderAdapter` says how to fix that, and
:func:`world_engine_client.connect` falls back to the headless Chrome path
(``render.LocalRenderer``) where it is installed; so it does for a scene D
cannot draw (:class:`SceneNotDrawable`), and on a machine whose GPU Vulkan
does not reach but Chrome reaches through OpenGL (see :func:`chrome_may_be_faster`).

Needs the ``render`` extra (wgpu, cryptography for the platform's encrypted
assets, Pillow for mesh textures): ``pip install 'world-engine-client[render]'``.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import sqlite3
import struct
import sys
import threading
import time
import traceback
import warnings
import weakref
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


class PipelineError(NoRenderAdapter):
    """Renderer D fails on the adapter it opened (``choice``): a pipeline that
    does not build there (a shader the backend's compiler rejects, FXC on
    D3D12 say), a validation error. ``error`` says what wgpu raised.
    :func:`start_renderer` then takes the next adapter of the search."""

    def __init__(self, message: str, *, choice, error: str, **kw):
        super().__init__(message, code="renderer_unavailable", **kw)
        self.choice = choice
        self.error = error


class SceneNotDrawable(RendererError):
    """Renderer D cannot draw this scene as the browser path would: an asset it
    does not read (``code`` asset_mismatch: a splat PLY, a WEM1 mesh or an HDRI
    it does not take, a mesh the server sent without its WEM1, a WEM1 whose GLB
    has content three draws and D does not, such as GPU instances or lines),
    or more than the device's memory holds (renderer_unavailable). The browser
    path may still draw it."""


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
    warn: bool = True,
):
    """The wgpu adapter D renders with and how it was found:
    ``(adapter, AdapterChoice)``. Probes once per environment (the result is
    kept next to the render cache for a day), or opens ``choice`` (from
    :func:`find_adapter`). Emits the software warning unless ``warn`` is
    False. Raises NoRenderAdapter."""
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
    if choice.software and warn:
        adapters.warn_once("software", _software_message(choice, adapter), stacklevel=4)
    return found, choice


def _asked_for(adapter: str) -> bool:
    """True when ``adapter`` asks for a software adapter, or names one."""
    return isinstance(adapter, str) and (adapter == "software" or adapter.startswith("name:"))


def _software_message(choice, adapter: str) -> str:
    """The warning for a software adapter. When the caller asked for one, it
    says that, not that Vulkan reaches no GPU (the search did not look for one)."""
    from .wgsl import adapters

    if not _asked_for(adapter):
        return adapters.software_message(choice)
    msg = (
        f"rendering camera images on the CPU with {choice.label}, as adapter={adapter!r} "
        "asks. Expect a step or two per second on real scenes; pass images=False for a "
        "state-only session"
    )
    if choice.step == "swiftshader":
        msg += ", or apt install mesa-vulkan-drivers for a faster CPU renderer (lavapipe)"
    return msg


def _speed_check(choice, adapter: str):
    """The vendored SpeedCheck for a session on ``choice``. For a software
    adapter the caller asked for, its warning leaves out the hint that Vulkan
    does not reach this machine's NVIDIA GPU, as _software_message does: the
    vendored slow_message reads ``choice.software`` only to add that hint."""
    from .wgsl import adapters

    if choice.software and _asked_for(adapter):
        choice = dataclasses.replace(choice, software=False)
    return adapters.SpeedCheck(choice)


def start_renderer(
    fetch,
    scene_init: dict,
    digests: dict[str, str],
    *,
    choice,
    adapter: str = "auto",
    allow_software: bool = True,
    cache_dir=None,
    timeout: float = 300.0,
    **options,
):
    """:class:`WGSLRenderer` on ``choice`` (from :func:`find_adapter`); when D
    fails on that adapter (:class:`PipelineError`: a pipeline its compiler
    rejects, a validation error), on the next one of the search, as often as
    one fails, with one RuntimeWarning that names each adapter and its error.
    The search passes over those adapters from then on in this process
    (``adapters.mark_failed``) and keeps the one it takes for the next session.
    When the process's wgpu instance does not have that one (another Vulkan
    loader's), its best adapter left is tried. Raises NoRenderAdapter, naming
    the failures, when no adapter is left: connect() then draws with the
    browser path. ``options`` go to WGSLRenderer."""
    from .wgsl import adapters

    search = {"adapter": adapter, "allow_software": allow_software, "cache_dir": cache_dir}
    probe_timeout = min(float(timeout), 120.0)
    # The reasons, not the exceptions: kept here, their tracebacks and this
    # frame would hold each other, and the device D failed on, until a
    # garbage collection.
    failures: list[str] = []

    def failed_on() -> str:
        return "; ".join(failures)

    while True:
        try:
            opened = open_adapter(choice=choice, timeout=probe_timeout, warn=not failures, **search)
            renderer = WGSLRenderer(
                fetch, scene_init, digests, timeout=timeout, opened=opened, **search, **options
            )
        except PipelineError as exc:
            adapters.mark_failed(exc.choice, exc.error)
            failures.append(exc.reason)
            try:
                choice = find_adapter(timeout=probe_timeout, **search)
            except NoRenderAdapter as none:
                raise NoRenderAdapter(
                    f"{failed_on()}; and {none.reason}",
                    code=none.code,
                    fixes=[_UPGRADE, *none.fixes],
                ) from None
            if adapters.failed(choice):  # a search that does not pass it over
                raise NoRenderAdapter(
                    f"{failed_on()}; and the search found no other adapter",
                    code="renderer_unavailable",
                    fixes=[_UPGRADE, "or pass images=False for a state-only session"],
                ) from None
            continue
        except NoRenderAdapter as exc:
            if not failures:
                raise
            raise NoRenderAdapter(
                f"{failed_on()}; and {exc.reason}", code=exc.code, fixes=[_UPGRADE, *exc.fixes]
            ) from None
        except SceneNotDrawable as exc:
            if not failures:
                raise
            raise SceneNotDrawable(f"{failed_on()}; and {exc.reason}", code=exc.code) from None
        if failures:
            c = renderer.choice
            renderer.passed_over = failed_on()
            msg = f"{failed_on()}. Renderer D draws camera images with {c.label} instead"
            if c.software:
                adapters.mark_warned("software")  # this warning says it
                msg += (
                    ", on the CPU: expect a step or two per second on real scenes, or pass "
                    "images=False for a state-only session"
                )
            warnings.warn(msg, RuntimeWarning, stacklevel=5)
        return renderer


_UPGRADE = (
    "pip install -U 'world-engine-client[render]' (a newer release may draw with this adapter)"
)


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
        from .wgsl.budget import SplatMemoryError
        from .wgsl.render_wgsl import WgslRenderer, create_device

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
        self._prefer = adapter  # info["render"]["reason"] says when it asks for software
        self.software = bool(self.choice.software)
        self.gpu = self.choice.label
        self.passed_over = ""  # the adapters D failed on before this one (start_renderer)
        outer = sys.exception()  # the caller's, when it calls from an except block
        try:
            self.device = create_device(self._adapter)
            self.assets = SceneAssets(
                fetch,
                drawn_assets(scene_init, digests),
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
                # strict: what D would leave out or light otherwise than the
                # capture page raises, so the browser path draws the scene
                self._scene = WgslRenderer(
                    scene_init,
                    self._read,
                    device=self.device,
                    color_format=color_format,
                    wgsl_dir=WGSL_DIR,
                    strict=True,
                )
            except SplatMemoryError as exc:
                raise SceneNotDrawable(str(exc), code="renderer_unavailable") from None
            except (ValueError, KeyError, IndexError, struct.error) as exc:  # unreadable asset
                raise SceneNotDrawable(
                    f"the scene's assets cannot be drawn: {exc}", code="asset_mismatch"
                ) from None
            self.load_seconds = time.perf_counter() - started
        except BaseException as exc:
            # The frames exc passed through hold the device; without their
            # locals it goes in close(), and not at a garbage collection.
            _clear_frames(exc, outer)
            self.close()
            if type(exc).__module__.split(".")[0] == "wgpu":  # a pipeline, a validation error
                error = f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:600]
                raise PipelineError(
                    f"renderer D cannot draw on {self.gpu}: {error}",
                    choice=self.choice,
                    error=error,
                ) from exc
            raise
        self._speed = _speed_check(self.choice, adapter)
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
        if self._scene is not None:
            out["lighting"] = self._scene.info.get("lighting")
        reasons = [self.passed_over] if self.passed_over else []
        if c.software and _asked_for(self._prefer):
            reasons.append(f"adapter={self._prefer!r} asks for a software adapter")
        elif c.software:
            failed = [f"{t['step']}: {t.get('error')}" for t in c.trail if not t.get("ok")]
            reasons.append("; ".join(failed) or "no hardware adapter")
        if reasons:
            out["reason"] = "; ".join(reasons)
        if self._speed.rate is not None:
            out["steps_per_s"] = round(self._speed.rate, 2)
        return out

    def render(self, state: dict) -> dict[str, np.ndarray]:
        """Pose the scene as ``state`` (a ``render_state`` from the server) and
        return ``{camera name: (H, W, 3) uint8}``."""
        if self._scene is None:
            raise RendererError("renderer is closed", code="renderer_unavailable")
        outer = sys.exception()
        started = time.perf_counter()
        try:
            images = self._scene.render(state)
        except Exception as exc:
            if type(exc).__module__.split(".")[0] == "wgpu":  # a lost device, a GPU error
                # so that close(), even while the caller handles this error
                # (a with block's exit), lets the device go
                _clear_frames(exc, outer)
                raise RendererError(
                    f"rendering failed on {self.gpu}: {exc}", code="renderer_unavailable"
                ) from exc
            raise
        seconds = time.perf_counter() - started
        self.last_timings = {"render_ms": seconds * 1e3, **self._scene.last_timing}
        self._speed.add(seconds)
        return images

    def close(self) -> None:
        """Free the scene's GPU memory, the key and the device, at once.
        Calling it again does nothing."""
        scene, self._scene = self._scene, None
        device, self.device = self.device, None
        self._adapter = None
        try:
            if scene is not None:
                scene.close()
            if self.assets is not None:
                self.assets.close()  # the key and anything held in memory
        finally:
            scene = None  # its buffers and textures go here
            if device is not None:
                try:
                    device.destroy()
                except Exception:  # noqa: BLE001 - already lost
                    pass
                left = _unlink(device)
                device = None
                if left is not None and left() is not None and not sys.is_finalizing():
                    gc.collect()  # a cycle _unlink does not know (another wgpu-py)

    def __enter__(self) -> WGSLRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _clear_frames(exc: BaseException, outer: BaseException | None) -> None:
    """Clear the locals of the finished frames that ``exc`` passed through,
    and of those the exceptions it chains passed through, back to ``outer``
    (the exception the caller was handling, if any, which is left as it is).

    wgpu-py raises its errors with the error in a local of the frame that
    raises it. That reference cycle keeps every frame of the error's
    traceback, and the device in their locals, until Python's cyclic garbage
    collector runs."""
    seen: set[int] = set()
    todo: list[BaseException | None] = [exc]
    while todo:
        e = todo.pop()
        if e is None or e is outer or id(e) in seen:
            continue
        seen.add(id(e))
        traceback.clear_frames(e.__traceback__)  # passes over the frames still running
        todo += (e.__cause__, e.__context__)


def _unlink(device):
    """Break the reference cycles that keep a destroyed wgpu-py 0.32 device
    alive, and return a weak reference to it (None if it takes none).

    The device's ``_device`` is the device itself, and its queue's
    ``_device`` is the device too, so dropping the last reference to a device
    frees neither. The native device, with its memory, threads and file
    descriptors (tens of MiB a session on a GPU), then stays until Python's
    cyclic garbage collector runs. Without those two references the device,
    its queue and its adapter go with the caller's last reference."""
    queue = getattr(device, "_queue", None)
    for obj in (queue, device):
        try:
            if getattr(obj, "_device", None) is device:
                obj._device = None
        except Exception:  # noqa: BLE001 - another wgpu-py: the caller collects
            pass
    try:
        return weakref.ref(device)
    except TypeError:
        return None


def drawn_assets(scene_init: dict, digests: dict[str, str]) -> dict[str, str]:
    """The digests of the assets renderer D reads: the splat, the objects' WEM1,
    the robot links, an HDRI or lighting asset. Not the objects' GLBs (the
    browser path's), so a session of D does not download them; when D cannot
    draw the scene, the browser path downloads the rest."""
    urls = {(scene_init.get(key) or {}).get("url") for key in ("splat", "hdri", "lighting")}
    for mesh in scene_init.get("meshes") or []:
        urls.add(mesh.get("wem"))
    for link in (scene_init.get("robot") or {}).get("links") or []:
        urls.add(link.get("url"))
    return {u: d for u, d in digests.items() if u in urls}


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
