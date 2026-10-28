#!/usr/bin/env python
"""Download size of the render extra on Linux, from PyPI's metadata.

Design doc 16 §9.6 (M-D3): what renderer D adds to an install that already has
numpy and Pillow (wgpu and what it needs, cryptography, world-engine-swiftshader)
must stay within 15 MB on x86_64 and 17 MB on aarch64. The browser path
(Playwright, websockets), still in the extra as the fallback, is listed apart.

    python scripts/render_size.py --wheel-dir DIR [--python 3.12] [--check]

The versions are the ones uv resolves for each platform today (``uv pip
compile``); the sizes are those of the wheels pip would download. The
world-engine-swiftshader wheels are taken from --wheel-dir (a build), or from
PyPI once published. --check exits 1 when a limit is exceeded.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMITS_MB = {"x86_64": 15.0, "aarch64": 17.0}
BASE = {"numpy", "pillow"}  # already installed without the render extra
BROWSER = {"playwright", "websockets", "greenlet", "pyee", "typing-extensions"}
SWS = "world-engine-swiftshader"


def resolve(arch: str, python: str) -> dict[str, str]:
    """name -> version of world-engine-client[render] for manylinux_2_28 on arch."""
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "req.in"
        lines = []
        for spec in render_requirements():
            name = re.split(r"[ ;<>=!~\[]", spec, maxsplit=1)[0].lower()
            if name != SWS:
                lines.append(spec)
        req.write_text("\n".join(lines) + "\nnumpy\n")
        done = subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                str(req),
                "--quiet",
                "--no-header",
                "--no-annotate",
                "--python-version",
                python,
                "--python-platform",
                f"{arch}-manylinux_2_28",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    out = {}
    for line in done.stdout.splitlines():
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+).*", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def render_requirements() -> list[str]:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return project["optional-dependencies"]["render"]


def wheel_size(name: str, version: str, arch: str, python: str) -> tuple[str, int]:
    """(file name, bytes) of the wheel pip picks for CPython ``python`` on
    manylinux_2_28 ``arch``."""
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as r:
        files = json.load(r)["urls"]
    cp = "cp" + python.replace(".", "")
    minor = int(python.split(".")[1])

    def rank(fn: str):
        if not fn.endswith(".whl"):
            return None
        _, _, py, abi, plat = fn[:-4].rsplit("-", 4)
        plats = plat.split(".")
        if not any(p == "any" or (p.endswith(arch) and _glibc_ok(p)) for p in plats):
            return None
        if py == cp and abi in (cp, "none"):
            score = 0
        elif abi == "abi3" and py.startswith("cp3") and int(py[3:] or 0) <= minor:
            score = 1
        elif py in ("py3", "py2.py3") and abi == "none":
            score = 2
        else:
            return None
        return (score, 0 if "any" not in plats else 1)

    best = None
    for f in files:
        k = rank(f["filename"])
        if k is not None and (best is None or k < best[0]):
            best = (k, f["filename"], f["size"])
    if best is None:
        raise SystemExit(f"no {name} {version} wheel for {cp} manylinux_2_28_{arch}")
    return best[1], best[2]


def _glibc_ok(plat: str) -> bool:
    m = re.match(r"manylinux_(\d+)_(\d+)_", plat)
    if m:
        return (int(m.group(1)), int(m.group(2))) <= (2, 28)
    return plat.startswith(("manylinux2014_", "manylinux1_", "manylinux2010_"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wheel-dir", type=Path, help="world-engine-swiftshader wheels to measure")
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    a = ap.parse_args(argv)
    report, failed = {}, False
    for arch in ("x86_64", "aarch64"):
        versions = resolve(arch, a.python)
        rows = []
        for name, version in sorted(versions.items()):
            if name in BASE:
                continue
            fn, size = wheel_size(name, version, arch, a.python)
            rows.append(
                {
                    "name": name,
                    "version": version,
                    "file": fn,
                    "bytes": size,
                    "part": "browser" if name in BROWSER else "d",
                }
            )
        sws = sorted(
            (a.wheel_dir or Path("/nonexistent")).glob(f"world_engine_swiftshader-*{arch}.whl")
        )
        if sws:
            rows.append(
                {"name": SWS, "file": sws[-1].name, "bytes": sws[-1].stat().st_size, "part": "d"}
            )
        else:
            rows.append({"name": SWS, "file": None, "bytes": None, "part": "d"})
        d_bytes = sum(r["bytes"] or 0 for r in rows if r["part"] == "d")
        browser = sum(r["bytes"] for r in rows if r["part"] == "browser")
        ok = sws != [] and d_bytes / 1e6 <= LIMITS_MB[arch]
        failed |= not ok
        report[arch] = {
            "rows": rows,
            "renderer_d_mb": round(d_bytes / 1e6, 2),
            "browser_path_mb": round(browser / 1e6, 2),
            "limit_mb": LIMITS_MB[arch],
            "ok": ok,
        }
        print(
            f"{arch} (CPython {a.python}): renderer D adds {d_bytes / 1e6:.2f} MB "
            f"(limit {LIMITS_MB[arch]:g}); the browser path {browser / 1e6:.2f} MB more"
        )
        for r in rows:
            size = f"{r['bytes'] / 1e6:6.2f} MB" if r["bytes"] else "   (no wheel)"
            print(f"  {r['part']:7s} {size}  {r['file'] or r['name']}")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1))
    return 1 if a.check and failed else 0


if __name__ == "__main__":
    sys.exit(main())
