"""What the bench ran on: the system, the CPU, every GPU with its driver, the
Python packages, the power source, and the environment variables that steer
adapter choice. Each probe is a short command or file read; a probe that is
missing or fails is recorded as such, never fatal."""

from __future__ import annotations

import ctypes
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Environment variables that decide which GPU a renderer gets.
STEERING = (
    "WE_RENDER_ADAPTER",
    "WE_RENDER_BROWSER",
    "WE_RENDER_CHROME_ARGS",
    "WE_RENDERER",
    "DIGIARENA_BROWSERS_DIR",
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
    "VK_LOADER_DRIVERS_SELECT",
    "MESA_VK_DEVICE_SELECT",
    "DRI_PRIME",
    "__NV_PRIME_RENDER_OFFLOAD",
    "__GLX_VENDOR_LIBRARY_NAME",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "__VK_LAYER_NV_optimus",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
)
PACKAGES = (
    "world-engine-client",
    "world-engine-swiftshader",
    "wgpu",
    "numpy",
    "Pillow",
    "playwright",
    "websockets",
    "cryptography",
)

# Lists every adapter wgpu sees, in its own process so the bench's processes
# make their own wgpu instance.
_ADAPTERS = """
import json, wgpu
out = []
for a in wgpu.gpu.enumerate_adapters_sync():
    info = dict(a.info)
    info["limits"] = {k: a.limits[k] for k in ("max-storage-buffer-binding-size",
        "max-compute-workgroup-storage-size", "max-texture-dimension-2d") if k in a.limits}
    info["features"] = sorted(str(f) for f in a.features)
    out.append(info)
print(json.dumps(out))
"""


def run(cmd: list[str], timeout: float = 20.0) -> dict:
    """A command's output, or why there is none."""
    if not shutil.which(cmd[0]):
        return {"cmd": " ".join(cmd), "missing": True}
    try:
        done = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"cmd": " ".join(cmd), "error": f"{type(exc).__name__}: {exc}"}
    out = {"cmd": " ".join(cmd), "returncode": done.returncode, "stdout": done.stdout[-20000:]}
    if done.stderr.strip():
        out["stderr"] = done.stderr[-4000:]
    return out


def read(path: str, limit: int = 20000) -> str | None:
    try:
        return Path(path).read_text(errors="replace")[:limit]
    except OSError:
        return None


def os_info() -> dict:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "node": platform.node(),
    }
    if sys.platform == "darwin":
        info["mac_ver"] = platform.mac_ver()[0]
        info["sw_vers"] = run(["sw_vers"]).get("stdout")
    elif os.name == "nt":
        info["win32_ver"] = list(platform.win32_ver())
        info["win32_edition"] = platform.win32_edition()
    else:
        text = read("/etc/os-release") or ""
        for line in text.splitlines():
            if line.startswith("PRETTY_NAME="):
                info["distribution"] = line.split("=", 1)[1].strip().strip('"')
    return info


def cpu_info() -> dict:
    info = {"logical_cores": os.cpu_count(), "processor": platform.processor()}
    if sys.platform == "darwin":
        info["model"] = (
            run(["sysctl", "-n", "machdep.cpu.brand_string"]).get("stdout") or ""
        ).strip()
        info["physical_cores"] = (
            run(["sysctl", "-n", "hw.physicalcpu"]).get("stdout") or ""
        ).strip()
        mem = (run(["sysctl", "-n", "hw.memsize"]).get("stdout") or "").strip()
        info["memory_bytes"] = int(mem) if mem.isdigit() else None
    elif os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            info["model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError as exc:
            info["model_error"] = str(exc)
        info["memory_bytes"] = _windows_memory()
    else:
        text = read("/proc/cpuinfo") or ""
        for line in text.splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
        mem = read("/proc/meminfo") or ""
        for line in mem.splitlines():
            if line.startswith("MemTotal:"):
                info["memory_bytes"] = int(line.split()[1]) * 1024
    if hasattr(os, "getloadavg"):
        info["load_average"] = [round(x, 2) for x in os.getloadavg()]
    return info


def _windows_memory() -> int | None:
    class Status(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except (AttributeError, OSError):
        pass
    return None


def gpus() -> dict:
    """Every GPU the system lists, with its driver, and every adapter wgpu sees."""
    info: dict = {"wgpu_adapters": wgpu_adapters()}
    if sys.platform == "darwin":
        sp = run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=60)
        try:
            info["system_profiler"] = json.loads(sp.get("stdout") or "")
        except ValueError:
            info["system_profiler"] = sp
    elif os.name == "nt":
        query = (
            "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, "
            "DriverDate, AdapterCompatibility, VideoProcessor, AdapterRAM, PNPDeviceID, "
            "Status | ConvertTo-Json"
        )
        ps = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", query], timeout=60)
        try:
            info["video_controllers"] = json.loads(ps.get("stdout") or "")
        except ValueError:
            info["video_controllers"] = ps
        info["dxdiag_hint"] = "dxdiag /t <file> lists every display device in detail"
    else:
        info["lspci"] = run(["lspci", "-nnk"])
        if "stdout" in info["lspci"]:
            blocks, keep = info["lspci"]["stdout"].split("\n"), []
            for i, line in enumerate(blocks):
                if any(
                    k in line for k in ("VGA compatible", "3D controller", "Display controller")
                ):
                    keep.append(line)
                    for nxt in blocks[i + 1 : i + 4]:
                        if nxt.startswith(("\t", " ")):
                            keep.append(nxt)
            info["lspci"]["stdout"] = "\n".join(keep)
        drm = []
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            if "-" in card.name:
                continue
            dev = card / "device"
            driver = dev / "driver"
            drm.append(
                {
                    "card": card.name,
                    "vendor": read(str(dev / "vendor"), 32),
                    "device": read(str(dev / "device"), 32),
                    "driver": driver.resolve().name if driver.exists() else None,
                }
            )
        info["drm_cards"] = drm
        info["render_nodes"] = sorted(p.name for p in Path("/dev/dri").glob("renderD*"))
        info["nvidia_smi"] = run(
            ["nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id", "--format=csv,noheader"]
        )
        info["vulkaninfo"] = run(["vulkaninfo", "--summary"], timeout=60)
        info["kernel"] = read("/proc/version", 400)
    return info


def wgpu_adapters() -> list | dict:
    try:
        done = subprocess.run(
            [sys.executable, "-c", _ADAPTERS],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if done.returncode != 0:
        return {"error": done.stderr[-4000:]}
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": done.stdout[-2000:] + done.stderr[-2000:]}


def power() -> dict:
    if sys.platform == "darwin":
        return {"pmset": (run(["pmset", "-g", "batt"]).get("stdout") or "").strip()}
    if os.name == "nt":

        class Power(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("SystemStatusFlag", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = Power()
        try:
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                return {
                    "ac_line": {0: "offline", 1: "online"}.get(status.ACLineStatus, "unknown"),
                    "battery_percent": status.BatteryLifePercent,
                }
        except (AttributeError, OSError):
            pass
        return {}
    out = {}
    for supply in sorted(Path("/sys/class/power_supply").glob("*")):
        online, status = read(str(supply / "online"), 8), read(str(supply / "status"), 32)
        out[supply.name] = {"online": (online or "").strip(), "status": (status or "").strip()}
    return out


def packages() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    try:
        import world_engine_client

        out["world_engine_client_file"] = world_engine_client.__file__
    except ImportError as exc:
        out["world_engine_client_file"] = f"not importable: {exc}"
    return out


def browsers() -> dict:
    """What the browser path would launch here."""
    try:
        from world_engine_client import render

        out = {"dir": str(render.browsers_dir()), "chrome_installed": render.chrome_installed()}
        shell = render.shell_executable()
    except Exception as exc:  # noqa: BLE001 - an older client, or none
        return {"error": f"{type(exc).__name__}: {exc}"}
    out["shell"] = str(shell) if shell else None
    return out


def collect() -> dict:
    return {
        "os": os_info(),
        "cpu": cpu_info(),
        "gpus": gpus(),
        "power": power(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "packages": packages(),
        "browsers": browsers(),
        "environment": {k: os.environ[k] for k in STEERING if k in os.environ},
    }
