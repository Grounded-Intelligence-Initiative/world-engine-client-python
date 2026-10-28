"""SwiftShader (a Vulkan implementation that runs on the CPU) and the Khronos
Vulkan loader, for Linux machines where world-engine-client finds no GPU.

world-engine-client's renderer draws camera images with wgpu, which needs a
Vulkan loader on Linux. Minimal systems and many containers have none, and
machines without a GPU have no Vulkan driver either. world-engine-client's
``render`` extra installs this package on Linux x86_64 and aarch64; the
renderer uses it only when neither a GPU nor the system's own software Vulkan
(Mesa's lavapipe) renders: it preloads :func:`loader_path` and points the
loader at :func:`icd_path`, in a fresh process. Nothing happens at import.

The files are built from pinned sources (``SWIFTSHADER_COMMIT``,
``VULKAN_SDK``) in the manylinux_2_28 image, with libstdc++ and libgcc linked
statically. Their licences are in :func:`licenses_dir` (see NOTICE.md there).
"""

from __future__ import annotations

import json
from pathlib import Path

__version__ = "0.1.0"
SWIFTSHADER_COMMIT = "cddb5c91b23ff70a1f607ea31610aa0d7f4b5ad5"
VULKAN_SDK = "vulkan-sdk-1.4.363.0"

_HERE = Path(__file__).resolve().parent
_LIB = _HERE / "lib"


def loader_path() -> Path:
    """The Khronos Vulkan loader (libvulkan.so.1) of this package."""
    return _LIB / "libvulkan.so.1"


def icd_path() -> Path:
    """The ICD manifest of SwiftShader's Vulkan driver (its library path is
    relative to the manifest)."""
    return _LIB / "vk_swiftshader_icd.json"


def library_path() -> Path:
    """SwiftShader's Vulkan driver (libvk_swiftshader.so)."""
    return _LIB / "libvk_swiftshader.so"


def licenses_dir() -> Path:
    return _HERE / "licenses"


def build_info() -> dict:
    """How the files were built: sources, Reactor backend (Subzero on x86_64,
    LLVM 10 on aarch64), compiler, and the size and SHA-256 of each file."""
    try:
        return json.loads((_LIB / "build_info.json").read_text())
    except (OSError, ValueError):
        return {}
