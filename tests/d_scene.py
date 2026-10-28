"""A small real scene for renderer D (render protocol 2), served by the
platform stand-in (platform_stub.py): the golden splat PLY, a red box off the
origin (its GLB, which the browser path draws, and the engine's WEM1 of it,
which renderer D draws), one robot link (WEL1), two cameras of different sizes.

``DPlatform(stub, encrypted=..., hdri=...)`` answers protocol 2 to a client that
offers it with pipeline wgsl/1, protocol 1 otherwise (as the engine does), and
sends a render_state with every observation; ``hdri`` (the bytes of an HDRI,
e.g. ``hdr()``) lights the scene with it. Setting ``box`` moves the box."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from platform_stub import Platform

GOLDEN = Path(__file__).resolve().parent / "data" / "wgsl_golden"
CAMS = json.loads((GOLDEN / "cameras.json").read_text())
# the golden cameras look at the origin from about 1.9 m; the box sits in view
BOX_AT = [0.25, -0.1, 0.05]


def _box(size: float) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned cube: 24 vertices (float32), triangles counter-clockwise
    from outside (uint32, flat)."""
    h = size / 2
    v, f = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[axis] = sign
            u = np.zeros(3)
            u[(axis + 1) % 3] = 1.0
            w = np.cross(n, u)
            base = len(v)
            v += [(n + a * u + b * w) * h for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            f += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.asarray(v, np.float32), np.asarray(f, np.uint32)


def box_glb(size=0.2, rgba=(1.0, 0.0, 0.0, 1.0)) -> bytes:
    """A GLB of an axis-aligned cube, one material (linear base colour)."""
    pos, idx = _box(size)
    bin_ = pos.tobytes() + idx.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": list(rgba)}}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos.nbytes},
            {"buffer": 0, "byteOffset": pos.nbytes, "byteLength": idx.nbytes},
        ],
        "buffers": [{"byteLength": len(bin_)}],
    }
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(bin_), b"BIN\x00")
    body += bin_
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def box_wem1(size=0.2, rgba=(1.0, 0.0, 0.0, 1.0)) -> bytes:
    """What the engine compiles box_glb(size, rgba) into (world_engine/assets/
    wem1.py, glb_to_wem1): WEM1, version 1. The GLB has no normals, so renderer
    D shades it flat; glTF's defaults make it metallic with roughness 1.
    tests/test_wgsl_vendored.py checks this against the engine's converter when
    it has an engine checkout."""
    pos, idx = _box(size)
    body, parts = 0, []

    def add(raw: bytes) -> int:
        nonlocal body
        off = body
        raw += b"\0" * (-len(raw) % 4)
        parts.append(raw)
        body += len(raw)
        return off

    p64 = pos.astype(np.float64)
    lo, hi = p64.min(0), p64.max(0)
    prim = {
        "vertices": len(pos),
        "triangles": len(idx) // 3,
        "center": [float(v) for v in (lo + hi) / 2],
        "position": [add(pos.astype("<f4").tobytes()), int(pos.size)],
        "index": [add(idx.astype("<u4").tobytes()), int(idx.size)],
        "material": 0,
    }
    material = {
        "name": "",
        "type": "standard",
        "color": [float(v) for v in rgba[:3]],
        "opacity": float(rgba[3]),
        "metalness": 1.0,
        "roughness": 1.0,
        "side": "front",
        "alpha_mode": "OPAQUE",
        "alpha_test": 0.0,
        "normal_scale": [1.0, 1.0],
        "ao_intensity": 1.0,
        "emissive": [0.0, 0.0, 0.0],
        "maps": {},
    }
    doc = {
        "version": 1,
        "generator": "world_engine.assets.wem1",
        "primitives": [prim],
        "materials": [material],
        "images": [],
        "bounds": [lo.tolist(), hi.tolist()],
        "unsupported": [],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    data = b"".join(parts)
    return b"WEM1" + struct.pack("<III", 1, len(js), len(data)) + js + data


def link_wel1() -> bytes:
    """A small grey tetrahedron as a preset robot link (the engine's WEL1)."""
    v = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], "<f4")
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], "<u4")
    return b"WEL1" + struct.pack("<IIBBBx", 4, 4, 0, 0, 0) + v.tobytes() + f.tobytes()


def hdr() -> bytes:
    """A 128 x 64 Radiance .hdr of whole RGBE bytes: a sky gradient and a sun
    (the engine's tests/wgsl_pin.py has the same)."""
    h, w = 64, 128
    y, x = np.mgrid[0:h, 0:w]
    px = np.zeros((h, w, 4), np.uint8)
    px[..., 0] = 60 + (186 * x) // (w - 1)
    px[..., 1] = 90 + (120 * (h - 1 - y)) // (h - 1)
    px[..., 2] = 200
    px[..., 3] = 128
    px[12:22, 80:94] = (240, 200, 160, 132)
    return f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode() + px.tobytes()


ASSETS = {
    "/__asset/0.ply": (GOLDEN / "scene.ply").read_bytes(),
    "/__asset/1.glb": box_glb(),
    "/__asset/1.wem1": box_wem1(),
}
ROBOT = {"/__robot/0.bin": link_wel1()}
HDRI = "/__asset/hdri.hdr"


class DPlatform(Platform):
    def __init__(self, stub, *, encrypted: bool = True, hdri: bytes | None = None):
        assets = dict(ASSETS)
        if hdri is not None:
            assets[HDRI] = hdri
        super().__init__(stub, encrypted=encrypted, assets=assets)
        self.assets.update(ROBOT)
        self.hdri = hdri is not None
        self.box = list(BOX_AT)
        self.offers: list[dict] = []

    def scene_init(self) -> dict:
        return {
            "type": "scene_init",
            "scene_id": "fixture",
            "splat": {
                "url": "/__asset/0.ply",
                "pos": [0, 0, 0],
                "quat": [0, 0, 0, 1],
                "scale": 1.0,
            },
            "meshes": [
                {
                    "id": "box",
                    "url": "/__asset/1.glb",
                    "wem": "/__asset/1.wem1",
                    "scale": [1, 1, 1],
                    "pos": BOX_AT,
                    "quat": [0, 0, 0, 1],
                }
            ],
            "hdri": {"url": HDRI, "strength": 1.3} if self.hdri else None,
            "probe_center": None,
            "robot": {
                "ee_proxy_size": [0.04, 0.08, 0.04],
                "links": [{"name": "link0", "robot_id": "r", "url": "/__robot/0.bin"}],
            },
            "cameras": CAMS,
        }

    def state(self) -> dict:
        return {
            "type": "state",
            "bodies": [{"id": "box", "pos": list(self.box), "quat": [0, 0, 0, 1]}],
            "robot": {"links": {"link0": {"pos": [-0.3, 0.3, 0.0], "quat": [0, 0, 0, 1]}}},
            "cameras": CAMS,
        }

    def _connect(self, body: dict) -> dict:
        self.offers.append(dict(body))
        reply = super()._connect(body)
        bundle = reply.get("render")
        if bundle is not None and 2 in (body.get("render_protocols") or []):
            if "wgsl/1" in (body.get("pipelines") or []):
                bundle.update(protocol=2, pipeline={"version": "wgsl/1"})
        return reply
