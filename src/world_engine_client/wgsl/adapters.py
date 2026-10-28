"""Which adapter renderer D draws with: the search order, the warnings, the error.

Design doc 16 §9.9 (runtime) and task book §2.6 item 10, §8.1 item 3: a machine
without a GPU renders in software without being asked, with a warning; only
when nothing renders is there an error, and it says how to fix that. Linux, the
first that renders wins:

  1. ``system``: the system Vulkan loader and a hardware adapter
  2. ``bundled-loader``: the loader of world-engine-swiftshader with the system's
     drivers (ICDs), a hardware adapter. Covers a container that has the NVIDIA
     driver but not libvulkan1.
  3. ``lavapipe``: the system loader and a software adapter (Mesa's lavapipe)
  4. ``swiftshader``: the bundled loader and the bundled SwiftShader driver

Windows: a hardware adapter (D3D12 or Vulkan), then WARP (``warp``, the
"Microsoft Basic Render Driver" that ``force_fallback_adapter`` gives). macOS:
Metal. A browser path, where one is kept, comes after these; the caller decides.

wgpu-py makes one wgpu instance per process at the first enumeration, and that
instance keeps the Vulkan loader it loaded. So each step is probed in a
subprocess (``_probe.py``), which renders a little the way the renderer does;
the result can be cached (``cache_file``) against a fingerprint of the
environment. Then this process is set up for the step that rendered (the
bundled loader preloaded, the bundled ICD named) before its own first
enumeration, and opens that adapter.

Everything here uses only the standard library until :func:`open_adapter`
imports wgpu. It imports nothing from the engine or from world-engine-client
(world-engine-client ships a copy of this directory).
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROBE = Path(__file__).resolve().with_name("_probe.py")
NATIVE = ("Vulkan", "Metal", "D3D12")
HARDWARE = ("DiscreteGPU", "IntegratedGPU")
# One control step is 0.034 s (task book §2.1 item 3): below this many steps a
# second, rendering cannot keep up with real time.
REAL_TIME_STEPS = 1.0 / 0.034
PROBE_TIMEOUT = 120.0
CACHE_SECONDS = 86400.0
_ENV_KEYS = (
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
    "VK_ADD_DRIVER_FILES",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
    "VK_LOADER_LAYERS_ENABLE",
    "VK_INSTANCE_LAYERS",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "CUDA_VISIBLE_DEVICES",
    "LP_NUM_THREADS",
    "LD_LIBRARY_PATH",
    "XDG_DATA_DIRS",
    "XDG_DATA_HOME",
)
_ICD_DIRS = (
    "/usr/share/vulkan/icd.d",
    "/usr/local/share/vulkan/icd.d",
    "/etc/vulkan/icd.d",
    "/usr/share/vulkan/implicit_layer.d",
    "/etc/vulkan/implicit_layer.d",
)

_warned: set[str] = set()
# The choice this module opened the process's wgpu instance with (see open_adapter).
_process_choice: AdapterChoice | None = None


class NoAdapterError(RuntimeError):
    """No adapter renders here. ``fixes`` lists what to do, ``trail`` what was tried."""

    def __init__(self, message: str, fixes: list[str], trail: list[dict]):
        super().__init__(message)
        self.fixes = fixes
        self.trail = trail


@dataclass
class AdapterChoice:
    """The step of the search that rendered, and the adapter it found."""

    step: str  # system | bundled-loader | lavapipe | swiftshader | warp | current
    device: str
    adapter_type: str  # DiscreteGPU, IntegratedGPU, CPU, ...
    backend: str  # Vulkan, Metal, D3D12
    vendor: str = ""
    description: str = ""
    software: bool = False
    loader: str | None = None  # a loader to preload (the bundled one)
    icd: str | None = None  # an ICD manifest to name (the bundled SwiftShader)
    pick: str = "hardware"  # how _probe.py found it
    trail: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    cached: bool = False

    @property
    def label(self) -> str:
        return f"{self.device} ({self.adapter_type}, {self.backend})"

    def as_dict(self) -> dict:
        return asdict(self)


def bundled_swiftshader() -> tuple[str, str] | None:
    """(loader, ICD manifest) of the installed world-engine-swiftshader, or None."""
    try:
        mod = importlib.import_module("world_engine_swiftshader")
        loader, icd = str(mod.loader_path()), str(mod.icd_path())
    except Exception:  # noqa: BLE001 - not installed, or not for this machine
        return None
    if Path(loader).is_file() and Path(icd).is_file():
        return loader, icd
    return None


def nvidia_present() -> bool:
    return os.path.exists("/dev/nvidiactl")


def instance_created() -> bool:
    """True when this process already has its wgpu instance (and its loader)."""
    if "wgpu" not in sys.modules:
        return False
    try:
        from wgpu.backends.wgpu_native import _helpers
    except Exception:  # noqa: BLE001 - another wgpu backend, or none loaded
        return False
    return getattr(_helpers, "_the_instance", None) is not None


def loaded_vulkan() -> str | None:
    """The libvulkan.so.1 mapped into this process, if any (Linux)."""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libvulkan.so" in line:
                    return line.split()[-1]
    except OSError:
        pass
    return None


# --------------------------------------------------------------------- probing
def _steps(prefer: str, allow_software: bool, bundle) -> list[tuple[str, dict]]:
    """(step name, _probe.py options) in search order."""
    name = prefer[5:] if prefer.startswith("name:") else None
    hw = f"name:{name}" if name else "hardware"
    order = "integrated" if prefer in ("low-power", "integrated") else "discrete"
    want_hw = prefer != "software"
    steps: list[tuple[str, dict]] = []
    if sys.platform.startswith("linux"):
        if want_hw:
            steps.append(("system", {"pick": hw, "prefer": order}))
            if bundle:
                steps.append(
                    ("bundled-loader", {"pick": hw, "prefer": order, "preload": bundle[0]})
                )
        if allow_software and not name:
            steps.append(("lavapipe", {"pick": "software"}))
            if bundle:
                steps.append(
                    ("swiftshader", {"pick": "software", "preload": bundle[0], "icd": bundle[1]})
                )
    elif sys.platform == "win32":
        if want_hw:
            steps.append(("system", {"pick": hw, "prefer": order}))
        if allow_software and not name:
            steps.append(("warp", {"pick": "fallback"}))
    else:
        if want_hw:
            steps.append(("system", {"pick": hw, "prefer": order}))
    return steps


def run_probe(options: dict, timeout: float = PROBE_TIMEOUT) -> dict:
    """Run _probe.py once in a fresh interpreter; its JSON result, or an error."""
    fd, out = tempfile.mkstemp(prefix="we_probe_", suffix=".json")
    os.close(fd)
    cmd = [sys.executable, "-I", "-B", str(PROBE), "--out", out]
    for k in ("pick", "prefer", "preload", "icd"):
        if options.get(k):
            cmd += [f"--{k}", str(options[k])]
    env = dict(os.environ)
    if options.get("icd"):
        env.pop("VK_ADD_DRIVER_FILES", None)
    t = time.monotonic()
    try:
        done = subprocess.run(
            cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout
        )
        try:
            with open(out) as f:
                res = json.load(f)
        except (OSError, ValueError):
            tail = (done.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
            res = {
                "ok": False,
                "error": f"the probe exited with {done.returncode}"
                + (f": {' | '.join(tail)}" if tail else ""),
            }
    except subprocess.TimeoutExpired:
        res = {"ok": False, "error": f"the probe did not finish within {timeout:.0f} s"}
    except OSError as exc:
        res = {"ok": False, "error": f"the probe did not start: {exc}"}
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
    res["subprocess_s"] = round(time.monotonic() - t, 3)
    return res


def _fingerprint(prefer: str, allow_software: bool, bundle) -> str:
    """What the search result depends on: the interpreter and wgpu, the
    drivers and loader on disk, the environment variables that steer them."""
    parts: list = [
        sys.executable,
        sys.version,
        sys.platform,
        prefer,
        allow_software,
        bundle,
        PROBE.stat().st_mtime_ns if PROBE.exists() else 0,
        {k: os.environ.get(k) for k in _ENV_KEYS},
    ]
    try:
        import importlib.metadata as md

        parts.append(md.version("wgpu"))
    except Exception:  # noqa: BLE001
        parts.append(None)
    for d in _ICD_DIRS:
        try:
            parts.append(sorted((p.name, p.stat().st_mtime_ns) for p in Path(d).iterdir()))
        except OSError:
            parts.append(None)
    for p in (
        "/dev/nvidiactl",
        "/proc/driver/nvidia/version",
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
    ):
        try:
            st = os.stat(p)
            parts.append((p, st.st_mtime_ns, st.st_size))
        except OSError:
            parts.append((p, None))
    try:
        parts.append(sorted(os.listdir("/dev/dri")))
    except OSError:
        parts.append(None)
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()


def _read_cache(cache_file, key: str) -> AdapterChoice | None:
    if not cache_file:
        return None
    try:
        with open(cache_file) as f:
            rec = json.load(f)
        if rec.get("key") != key or time.time() - float(rec.get("time", 0)) > CACHE_SECONDS:
            return None
        choice = AdapterChoice(**rec["choice"])
    except (OSError, ValueError, TypeError, KeyError):
        return None
    choice.cached = True
    return choice


def _write_cache(cache_file, key: str, choice: AdapterChoice) -> None:
    if not cache_file:
        return
    try:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{cache_file}.{os.getpid()}.part"
        with open(tmp, "w") as f:
            json.dump({"key": key, "time": time.time(), "choice": choice.as_dict()}, f)
        os.replace(tmp, cache_file)
    except OSError:
        pass


def search(
    *,
    prefer: str = "auto",
    allow_software: bool = True,
    cache_file: str | os.PathLike | None = None,
    timeout: float = PROBE_TIMEOUT,
) -> AdapterChoice:
    """Probe the steps in order; the first that renders. Raises NoAdapterError.

    prefer: ``auto`` (discrete GPU, then integrated, then software),
    ``low-power`` (integrated first), ``software`` (skip the GPU steps) or
    ``name:SUBSTR`` (only an adapter whose name contains SUBSTR, no software
    fallback). allow_software False stops after the GPU steps."""
    bundle = bundled_swiftshader()
    key = _fingerprint(prefer, allow_software, bundle)
    hit = _read_cache(cache_file, key)
    if hit is not None:
        return hit
    trail: list[dict] = []
    t0 = time.monotonic()
    for step, opts in _steps(prefer, allow_software, bundle):
        res = run_probe(opts, timeout)
        picked = (res.get("picked") or {}).get("adapter") or {}
        trail.append(
            {
                "step": step,
                "ok": bool(res.get("ok")),
                "adapters": [a.get("device") for a in res.get("adapters", [])],
                "picked": picked.get("device"),
                "error": res.get("error")
                or "; ".join(
                    f"{(t.get('adapter') or {}).get('device', t.get('kind'))}: {t['error']}"
                    for t in res.get("tried", [])
                    if t.get("error")
                )
                or None,
                "seconds": res.get("subprocess_s"),
            }
        )
        if res.get("ok"):
            choice = AdapterChoice(
                step=step,
                device=picked.get("device") or "?",
                adapter_type=picked.get("adapter_type") or "?",
                backend=picked.get("backend_type") or "?",
                vendor=picked.get("vendor") or "",
                description=picked.get("description") or "",
                software=picked.get("adapter_type") == "CPU" or step == "warp",
                loader=opts.get("preload"),
                icd=opts.get("icd"),
                pick=opts["pick"],
                trail=trail,
                seconds=round(time.monotonic() - t0, 3),
            )
            _write_cache(cache_file, key, choice)
            return choice
    raise NoAdapterError(no_adapter_message(trail), no_adapter_fixes(bundle), trail)


# ------------------------------------------------------------- this process
def open_adapter(
    choice: AdapterChoice | None = None,
    *,
    prefer: str = "auto",
    allow_software: bool = True,
    cache_file: str | os.PathLike | None = None,
    timeout: float = PROBE_TIMEOUT,
):
    """Set up this process for ``choice`` (search() when None) and return
    ``(adapter, choice)``. When this process already has its wgpu instance,
    nothing can be set up any more: the best adapter it can see is used, and
    choice.step is "current". A cached choice that no longer opens is searched
    for again. Raises NoAdapterError."""
    if instance_created():
        return _reuse(prefer, allow_software)
    kw = {"prefer": prefer, "allow_software": allow_software, "timeout": timeout}
    if choice is None:
        choice = search(cache_file=cache_file, **kw)
    try:
        return _open(choice), choice
    except NoAdapterError:
        if not choice.cached:
            raise
    # The cached result no longer holds (a driver changed under the same
    # fingerprint): search again, unless the instance exists by now.
    if instance_created():
        return _reuse(prefer, allow_software)
    fresh = search(cache_file=None, **kw)
    _write_cache(cache_file, _fingerprint(prefer, allow_software, bundled_swiftshader()), fresh)
    return _open(fresh), fresh


def _reuse(prefer: str, allow_software: bool):
    """This process has its wgpu instance already. When this module made it and
    the step it chose fits the request, that adapter again (with that step).
    Else the best adapter the instance sees, step "current"."""
    prev = _process_choice
    fits = prev is not None and (allow_software or not prev.software)
    if fits:
        if prefer.startswith("name:"):
            fits = prefer[5:].lower() in prev.device.lower()
        elif prefer == "software":
            fits = prev.software
        elif prefer in ("low-power", "integrated"):
            fits = prev.adapter_type == "IntegratedGPU"
    if fits:
        import wgpu

        pool = (
            [wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)]
            if prev.pick == "fallback"
            else wgpu.gpu.enumerate_adapters_sync()
        )
        for ad in pool:
            if ad is not None and _matches(ad, prev):
                return ad, AdapterChoice(**{**prev.as_dict(), "cached": True})
    return _from_current(prefer, allow_software)


def _matches(ad, choice: AdapterChoice) -> bool:
    i = ad.info
    return (
        i.get("device") == choice.device
        and i.get("backend_type") == choice.backend
        and i.get("adapter_type") == choice.adapter_type
    )


def _open(choice: AdapterChoice):
    global _process_choice
    import wgpu
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    if choice.loader:
        loaded = loaded_vulkan()
        if loaded and Path(loaded).resolve() != Path(choice.loader).resolve():
            raise NoAdapterError(
                f"this process already loaded {loaded}, so it cannot use the bundled Vulkan "
                f"loader the search chose ({choice.step})",
                ["start the renderer before anything else in the process uses Vulkan"],
                choice.trail,
            )
        try:
            ctypes.CDLL(choice.loader, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise NoAdapterError(
                f"the bundled Vulkan loader does not load: {exc}",
                no_adapter_fixes(None),
                choice.trail,
            ) from None
    saved = {
        k: os.environ.get(k) for k in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_ADD_DRIVER_FILES")
    }
    try:
        if choice.icd:
            os.environ["VK_ICD_FILENAMES"] = os.environ["VK_DRIVER_FILES"] = choice.icd
            os.environ.pop("VK_ADD_DRIVER_FILES", None)
        set_instance_extras(backends=["Primary"])
        if choice.pick == "fallback":
            ads = [wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)]
        else:
            ads = list(wgpu.gpu.enumerate_adapters_sync())
    finally:
        # The loader read these when the instance was created; the rest of the
        # process, and its children, get the environment they had.
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for ad in ads:
        if ad is not None and _matches(ad, choice):
            _process_choice = choice
            return ad
    seen = [f"{a.info.get('device')} ({a.info.get('adapter_type')})" for a in ads if a is not None]
    raise NoAdapterError(
        f"the adapter the search chose ({choice.label}, step {choice.step}) is not there "
        f"in this process; it sees {seen or 'none'}",
        no_adapter_fixes(bundled_swiftshader()),
        choice.trail,
    )


def _from_current(prefer: str, allow_software: bool):
    """The best adapter of an instance this process created earlier."""
    import wgpu

    ads = [a for a in wgpu.gpu.enumerate_adapters_sync() if a.info.get("backend_type") in NATIVE]
    name = prefer[5:].lower() if prefer.startswith("name:") else None
    if name:
        ads = [a for a in ads if name in (a.info.get("device") or "").lower()]
    first = "IntegratedGPU" if prefer in ("low-power", "integrated") else "DiscreteGPU"
    hw = sorted(
        (a for a in ads if a.info.get("adapter_type") in HARDWARE),
        key=lambda a: a.info.get("adapter_type") != first,
    )
    sw = [a for a in ads if a.info.get("adapter_type") == "CPU"]
    pool = ([] if prefer == "software" else hw) + (sw if allow_software else [])
    if not pool:
        raise NoAdapterError(
            "this process created its wgpu instance before the renderer started, and that "
            "instance has no adapter the renderer can use",
            no_adapter_fixes(bundled_swiftshader()),
            [],
        )
    ad = pool[0]
    i = ad.info
    choice = AdapterChoice(
        step="current",
        device=i.get("device") or "?",
        adapter_type=i.get("adapter_type") or "?",
        backend=i.get("backend_type") or "?",
        vendor=i.get("vendor") or "",
        description=i.get("description") or "",
        software=i.get("adapter_type") == "CPU",
    )
    return ad, choice


# ---------------------------------------------------------- what users read
def no_adapter_message(trail: list[dict]) -> str:
    tried = (
        "; ".join(f"{t['step']}: {t.get('error') or 'nothing rendered'}" for t in trail)
        or "no step applies to this platform"
    )
    return f"no GPU or software adapter renders camera images here ({tried})"


def no_adapter_fixes(bundle) -> list[str]:
    if sys.platform == "win32":
        return [
            "update the graphics driver (WARP, Windows' software renderer, comes with "
            "Windows 10 and 11)",
            "or pass images=False for a state-only session",
        ]
    if sys.platform == "darwin":
        return [
            "D renders with Metal: macOS 11 or later",
            "or pass images=False for a state-only session",
        ]
    fixes = [
        "apt install libvulkan1 mesa-vulkan-drivers (the Vulkan loader and Mesa's CPU renderer)"
    ]
    if bundle is None:
        fixes.append(
            "pip install world-engine-swiftshader (a software Vulkan for Linux x86_64 and "
            "aarch64; world-engine-client[render] includes it there)"
        )
    if nvidia_present():
        fixes.append(nvidia_hint())
    else:
        fixes.append(
            "in a container with a GPU: give it the GPU and its graphics libraries "
            "(docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ...)"
        )
    fixes.append("or pass images=False for a state-only session")
    return fixes


def nvidia_hint() -> str:
    return (
        "this machine has an NVIDIA GPU that Vulkan does not reach: install libx11-6 "
        "libxext6 libegl1 (the NVIDIA Vulkan driver loads them), and in a container add "
        "graphics to NVIDIA_DRIVER_CAPABILITIES (for example NVIDIA_DRIVER_CAPABILITIES=all)"
    )


def software_message(choice: AdapterChoice) -> str:
    msg = (
        f"no hardware GPU for camera images: rendering on the CPU with {choice.label}. "
        "Expect a step or two per second on real scenes; pass images=False for a "
        "state-only session"
    )
    if choice.step == "swiftshader":
        msg += ", or apt install mesa-vulkan-drivers for a faster CPU renderer (lavapipe)"
    if nvidia_present():
        msg += ". " + nvidia_hint()[0].upper() + nvidia_hint()[1:]
    return msg


def slow_message(choice: AdapterChoice, rate: float, steps: int) -> str:
    msg = (
        f"camera rendering is slower than real time here: {rate:.1f} steps/s over {steps} "
        f"steps on {choice.label}, below {REAL_TIME_STEPS:.1f} steps/s (one step is 0.034 s). "
        "Pass images=False for a state-only session, or render fewer or smaller cameras"
    )
    if choice.software and nvidia_present():
        msg += ". " + nvidia_hint()[0].upper() + nvidia_hint()[1:]
    return msg


def warn_once(kind: str, message: str, stacklevel: int = 3) -> bool:
    """One RuntimeWarning of each kind per process. True when it was emitted."""
    if kind in _warned:
        return False
    _warned.add(kind)
    warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)
    return True


class SpeedCheck:
    """Times renders after a warm-up; one warning per process when the median
    rate is below real time."""

    def __init__(self, choice: AdapterChoice, skip: int = 3, samples: int = 10):
        self.choice, self.skip, self.samples = choice, skip, samples
        self.seconds: list[float] = []
        self.rate: float | None = None

    def add(self, seconds: float) -> None:
        if self.rate is not None:
            return
        self.seconds.append(seconds)
        if len(self.seconds) < self.skip + self.samples:
            return
        timed = sorted(self.seconds[self.skip :])
        mid = len(timed) // 2
        median = timed[mid] if len(timed) % 2 else 0.5 * (timed[mid - 1] + timed[mid])
        self.rate = 1.0 / median if median > 0 else float("inf")
        if self.rate < REAL_TIME_STEPS:
            warn_once("slow", slow_message(self.choice, self.rate, self.samples), stacklevel=4)
