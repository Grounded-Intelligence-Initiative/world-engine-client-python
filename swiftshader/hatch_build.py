"""Wheel tag of world-engine-swiftshader: py3-none-manylinux_2_28_<arch>.

The package holds native libraries but no Python extension, so the tag names
any Python 3 and the platform the libraries were built for: the architecture
is read from the ELF header of libvk_swiftshader.so (WE_SWIFTSHADER_PLATFORM
overrides). A wheel is refused when scripts/build.sh has not put the
libraries in place, or when they need a glibc newer than manylinux_2_28.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

NEEDED = ("libvulkan.so.1", "libvk_swiftshader.so", "vk_swiftshader_icd.json", "build_info.json")
MACHINES = {62: "x86_64", 183: "aarch64"}


def _machine(path: Path) -> str:
    head = path.read_bytes()[:20]
    if head[:4] != b"\x7fELF":
        raise RuntimeError(f"{path} is not an ELF file")
    e_machine = int.from_bytes(head[18:20], "little")
    if e_machine not in MACHINES:
        raise RuntimeError(f"{path} is built for ELF machine {e_machine}, not x86_64 or aarch64")
    return MACHINES[e_machine]


class PlatformTag(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        lib = Path(self.root) / "src" / "world_engine_swiftshader" / "lib"
        missing = [n for n in NEEDED if not (lib / n).is_file()]
        if missing:
            raise RuntimeError(f"missing {', '.join(missing)} in {lib}: run scripts/build.sh first")
        info = json.loads((lib / "build_info.json").read_text())
        arch = _machine(lib / "libvk_swiftshader.so")
        if _machine(lib / "libvulkan.so.1") != arch:
            raise RuntimeError("the loader and the driver are built for different machines")
        # A release is manylinux_2_28; a development build elsewhere (a cross build
        # against a newer glibc) names its own tag and is held to it.
        platform = os.environ.get("WE_SWIFTSHADER_PLATFORM") or f"manylinux_2_28_{arch}"
        m = re.fullmatch(rf"manylinux_(\d+)_(\d+)_{arch}", platform)
        if not m:
            raise RuntimeError(f"WE_SWIFTSHADER_PLATFORM={platform} for {arch} libraries")
        allowed = (int(m.group(1)), int(m.group(2)))
        for name, entry in info.get("files", {}).items():
            glibc = entry.get("max_glibc")
            if glibc and tuple(map(int, glibc.split("."))) > allowed:
                raise RuntimeError(f"{name} needs GLIBC_{glibc}, newer than {platform}")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform}"
