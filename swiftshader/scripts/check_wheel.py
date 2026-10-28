#!/usr/bin/env python
"""Check a world-engine-swiftshader wheel before it is published.

* the tag is py3-none-manylinux_2_28_<arch>, and the libraries are ELF files of
  that architecture;
* the files are there, match the SHA-256 in build_info.json, and need no glibc
  newer than 2.28; the ICD manifest names the driver by a relative path;
* every licence text NOTICE.md names is in the wheel;
* the wheel is at most --max-mb;
* with --render: renders with it. The wheel is unpacked into a temporary
  directory and world-engine-client's adapter probe (its own check: a compute
  dispatch, a blended draw into the renderer's colour target, a readback) runs
  in a fresh process with this loader and this driver only. Needs wgpu and numpy.

    python swiftshader/scripts/check_wheel.py WHEEL [--max-mb 9] [--render] [--out RESULT.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "src" / "world_engine_client" / "wgsl" / "_probe.py"
MACHINES = {62: "x86_64", 183: "aarch64"}
PKG = "world_engine_swiftshader"


def check(wheel: Path, max_mb: float) -> tuple[dict, list[str]]:
    problems: list[str] = []
    m = re.fullmatch(
        r"world_engine_swiftshader-([^-]+)-py3-none-manylinux_2_28_(x86_64|aarch64)\.whl",
        wheel.name,
    )
    if not m:
        return {}, [f"unexpected wheel name {wheel.name}"]
    version, arch = m.groups()
    z = zipfile.ZipFile(wheel)
    names = set(z.namelist())
    report = {"wheel": wheel.name, "version": version, "arch": arch, "bytes": wheel.stat().st_size}
    if wheel.stat().st_size > max_mb * 1e6:
        problems.append(f"{wheel.stat().st_size / 1e6:.2f} MB, above {max_mb:g} MB")
    lib = f"{PKG}/lib/"
    for n in (
        "libvulkan.so.1",
        "libvk_swiftshader.so",
        "vk_swiftshader_icd.json",
        "build_info.json",
    ):
        if lib + n not in names:
            problems.append(f"missing {lib + n}")
    if problems:
        return report, problems
    info = json.loads(z.read(lib + "build_info.json"))
    report["backend"] = info.get("backend")
    report["files"] = {}
    for n, entry in info.get("files", {}).items():
        data = z.read(lib + n)
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            problems.append(f"{n} does not match build_info.json")
        glibc = entry.get("max_glibc")
        if glibc and tuple(map(int, glibc.split("."))) > (2, 28):
            problems.append(f"{n} needs GLIBC_{glibc}")
        if n.endswith((".so", ".so.1")):
            if (
                data[:4] != b"\x7fELF"
                or MACHINES.get(int.from_bytes(data[18:20], "little")) != arch
            ):
                problems.append(f"{n} is not an ELF file for {arch}")
        report["files"][n] = {"bytes": len(data), "max_glibc": glibc}
    want = {"x86_64": "Subzero", "aarch64": "LLVM"}[arch]
    if info.get("backend") != want:
        problems.append(f"backend {info.get('backend')!r}, expected {want} on {arch}")
    icd = json.loads(z.read(lib + "vk_swiftshader_icd.json"))
    if icd.get("ICD", {}).get("library_path") != "./libvk_swiftshader.so":
        problems.append(f"the ICD manifest names {icd.get('ICD', {}).get('library_path')!r}")
    notice = (
        z.read(f"{PKG}/licenses/NOTICE.md").decode() if f"{PKG}/licenses/NOTICE.md" in names else ""
    )
    if not notice:
        problems.append("no licenses/NOTICE.md")
    for text in sorted(set(re.findall(r"[A-Za-z0-9.-]+-LICENSE\.(?:txt|TXT|md)", notice))):
        if f"{PKG}/licenses/{text}" not in names:
            problems.append(f"NOTICE.md names {text}, which is not in the wheel")
    return report, problems


def render(wheel: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(wheel).extractall(tmp)
        lib = Path(tmp) / PKG / "lib"
        out = Path(tmp) / "probe.json"
        env = {k: v for k, v in os.environ.items() if not k.startswith("VK_")}
        done = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(PROBE),
                "--out",
                str(out),
                "--pick",
                "software",
                "--preload",
                str(lib / "libvulkan.so.1"),
                "--icd",
                str(lib / "vk_swiftshader_icd.json"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        try:
            res = json.loads(out.read_text())
        except (OSError, ValueError):
            res = {
                "ok": False,
                "error": f"probe exited with {done.returncode}: {done.stderr[-800:]}",
            }
    picked = res.get("picked") or {}
    return {
        "ok": bool(res.get("ok")),
        "adapters": [a.get("device") for a in res.get("adapters", [])],
        "device": (picked.get("adapter") or {}).get("device"),
        "color_format": picked.get("color_format"),
        "seconds": res.get("seconds"),
        "error": res.get("error")
        or [t.get("error") for t in res.get("tried", []) if t.get("error")],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--max-mb", type=float, default=9.0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    report, problems = check(a.wheel, a.max_mb)
    if a.render and not problems:
        report["render"] = render(a.wheel)
        if not report["render"]["ok"]:
            problems.append(f"rendering with the wheel failed: {report['render']['error']}")
        elif "SwiftShader" not in (report["render"]["device"] or ""):
            problems.append(f"the probe rendered on {report['render']['device']}, not SwiftShader")
    report["problems"] = problems
    print(json.dumps(report, indent=1))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
