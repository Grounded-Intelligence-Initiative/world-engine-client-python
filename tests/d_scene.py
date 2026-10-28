"""A small real scene for renderer D (render protocol 2), served by the
platform stand-in (platform_stub.py): the golden splat PLY, a red GLB box off
the origin, one robot link (WEL1), two cameras of different sizes.

``DPlatform(stub, encrypted=...)`` answers protocol 2 to a client that offers
it with pipeline wgsl/1, protocol 1 otherwise (as the engine does), and sends a
render_state with every observation. ``set_box(pos)`` moves the box."""

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


def box_glb(size=0.2, rgba=(1.0, 0.0, 0.0, 1.0)) -> bytes:
    """A GLB of an axis-aligned cube, one material (linear base colour)."""
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
    pos = np.asarray(v, np.float32)
    idx = np.asarray(f, np.uint32)
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


def link_wel1() -> bytes:
    """A small grey tetrahedron as a preset robot link (the engine's WEL1)."""
    v = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], "<f4")
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], "<u4")
    return b"WEL1" + struct.pack("<IIBBBx", 4, 4, 0, 0, 0) + v.tobytes() + f.tobytes()


ASSETS = {
    "/__asset/0.ply": (GOLDEN / "scene.ply").read_bytes(),
    "/__asset/1.glb": box_glb(),
}
ROBOT = {"/__robot/0.bin": link_wel1()}


class DPlatform(Platform):
    def __init__(self, stub, *, encrypted: bool = True):
        super().__init__(stub, encrypted=encrypted, assets=dict(ASSETS))
        self.assets.update(ROBOT)
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
                    "scale": [1, 1, 1],
                    "pos": BOX_AT,
                    "quat": [0, 0, 0, 1],
                }
            ],
            "hdri": None,
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
