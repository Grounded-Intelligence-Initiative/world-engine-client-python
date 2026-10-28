"""The bench scene: r2_400k with only its splat, the trace that poses its
cameras, and the local stand-in of a server's /render/ route that both paths
read from.

The splat is made here, byte for byte. Design doc 16's round-2 scene r2_400k
(DW/bench/round2/scenes_gen/make_scenes.py) copies its splat from the first
round's fixture400k, which is the output of make_ply() in the DigiArena
website's scripts/e2e_incremental_save.py with n=400000: 400k Gaussians
uniformly in [-1, 1]^3, all grey (f_dc 0.5), opacity logit 2, log scale -5
(about 6.7 mm), no rotation. That generator is copied below. numpy's
RandomState stream is frozen across numpy versions, so every machine makes
the same 22,400,362 bytes, checked against SPLAT_SHA256.

The trace is r2_400k's (DW/bench/round2/traces/r2_400k, recorded 2026-10-26):
21 warm-up states and 200 timed ones, three 256x256 cameras, of which the
wrist camera moves with the arm. Splat only: no meshes, no robot links, and
the end-effector box has size 0 and sits 170 m away, so every pixel of every
image is splat region (design doc 16 §9.10 item 5).

r2_400k is a uniform grey fog: seen from outside, or from the wrist camera
inside it, every pixel saturates to the same grey (163, standard deviation
under 0.5 over an image). Its images show whether a renderer draws at all,
not how exactly. The quality numbers that the thresholds judge come from a
second scene, ``room200k``, made here as well: a box room (a checkered floor,
four banded walls, two boxes, floaters) of 200k flat splats, opaque and
translucent, with SH degree 3. It is drawn with the same cameras and states.
Its generator uses only RandomState.uniform and exact arithmetic (+, -, *, /,
sqrt, floor; no sin or exp, whose last bits differ between platforms), so it
is byte for byte the same everywhere too (ROOM_SHA256).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRACE = HERE / "trace"
CAPTURE_PAGE = HERE / "capture_page"
SPLAT_URL = "/__asset/0.ply"
SPLAT_COUNT = 400_000
SPLAT_SHA256 = "c1ab11fcbbf2adb0eed554809cdcae1ada3e80dee657db32a723fb0319839d80"
PROPS = [
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]


def make_splat(n: int = SPLAT_COUNT) -> bytes:
    """The PLY of r2_400k's splat (see the module docstring)."""
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {p}\n" for p in PROPS) + "end_header\n"
    rows = np.zeros((n, len(PROPS)), dtype="<f4")
    rows[:, :3] = np.random.RandomState(0).uniform(-1, 1, (n, 3))
    rows[:, 3:6] = 0.5
    rows[:, 6] = 2.0
    rows[:, 7:10] = -5.0
    rows[:, 10] = 1.0
    return header.encode() + rows.tobytes()


C0 = 0.28209479177387814  # the SH degree-0 basis constant
ROOM_COUNT = 200_000
ROOM_SHA256 = "c3c111af1fbbab9ae3733d6e1b18cc5ad9c248347e2f4f9d349b5fa3daabbcb0"
ROOM_PROPS = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)
_S = np.sqrt(0.5)
# Quaternions (w, x, y, z) that turn a splat's local z axis to a face normal.
_TO_NORMAL = {
    "+z": (1.0, 0.0, 0.0, 0.0),
    "-z": (0.0, 1.0, 0.0, 0.0),
    "+x": (_S, 0.0, _S, 0.0),
    "-x": (_S, 0.0, -_S, 0.0),
    "+y": (_S, -_S, 0.0, 0.0),
    "-y": (_S, _S, 0.0, 0.0),
}


def make_room(n: int = ROOM_COUNT) -> bytes:
    """The PLY of room200k (see the module docstring): exact arithmetic only."""
    rs = np.random.RandomState(7)

    def u(*shape):
        return rs.uniform(0.0, 1.0, shape)

    parts = []  # (pos, quat wxyz, log scale, rgb, opacity logit)

    def face(count, origin, e1, e2, normal, colour, scale=(-5.8, -4.6)):
        st = u(count, 2)
        o, a, b = (np.asarray(v, float) for v in (origin, e1, e2))
        nrm = np.zeros(3)
        nrm["xyz".index(normal[1])] = 1.0 if normal[0] == "+" else -1.0
        pos = o + st[:, :1] * a + st[:, 1:] * b + (u(count, 1) - 0.5) * 0.002 * nrm
        # a random turn about the normal: (c, 0, 0, s) normalised, then to the face
        cs = u(count, 2) * 2.0 - 1.0
        norm = np.sqrt(cs[:, 0] * cs[:, 0] + cs[:, 1] * cs[:, 1])
        norm = np.where(norm < 1e-6, 1.0, norm)
        c, s = cs[:, 0] / norm, cs[:, 1] / norm
        w1, x1, y1, z1 = _TO_NORMAL[normal]
        quat = np.stack([w1 * c - z1 * s, x1 * c + y1 * s, y1 * c - x1 * s, w1 * s + z1 * c], 1)
        lo, hi = scale
        logs = np.concatenate([lo + (hi - lo) * u(count, 2), -8.0 + 1.0 * u(count, 1)], 1)
        rgb = colour(pos) * (0.94 + 0.12 * u(count, 1))
        opaque = u(count) < 0.78
        logit = np.where(opaque, 2.0 + 3.0 * u(count), -3.0 + 3.0 * u(count))
        parts.append((pos, quat, logs, rgb, logit))

    def flat(c):
        return lambda p: np.broadcast_to(np.asarray(c, float), (len(p), 3))

    def floor_colour(p):
        cell = (np.floor(p[:, 0] * 4.0) + np.floor(p[:, 1] * 4.0)) % 2.0
        dark = np.asarray([0.62, 0.45, 0.30]) * (0.85 + 0.1 * (p[:, :1] + 1.5) / 3.0)
        light = np.asarray([0.80, 0.70, 0.55]) * (0.85 + 0.1 * (p[:, 1:2] + 1.5) / 3.0)
        return np.where(cell[:, None] > 0.5, dark, light)

    def wall_colour(base):
        def colour(p):
            band = np.floor(p[:, 2] * 5.0) % 2.0
            shade = (1.0 - 0.2 * p[:, 2:3] / 1.6) * np.where(band[:, None] > 0.5, 0.85, 1.0)
            return np.asarray(base) * shade

        return colour

    def stripes(base):
        def colour(p):
            stripe = np.floor((p[:, 0] + p[:, 1] + p[:, 2]) * 12.0) % 2.0
            return np.asarray(base) * np.where(stripe[:, None] > 0.5, 0.7, 1.0)

        return colour

    def box(count, lo, hi, colour):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        d = hi - lo
        faces = [  # origin, e1, e2, normal (no bottom face: it stands on the floor)
            (lo + [0, 0, d[2]], [d[0], 0, 0], [0, d[1], 0], "+z"),
            (lo, [d[0], 0, 0], [0, 0, d[2]], "-y"),
            (lo + [0, d[1], 0], [d[0], 0, 0], [0, 0, d[2]], "+y"),
            (lo, [0, d[1], 0], [0, 0, d[2]], "-x"),
            (lo + [d[0], 0, 0], [0, d[1], 0], [0, 0, d[2]], "+x"),
        ]
        each = count // len(faces)
        for origin, e1, e2, normal in faces:
            face(each, origin, e1, e2, normal, colour, scale=(-6.2, -5.2))

    share = {"floor": 0.40, "walls": 0.32, "box_a": 0.10, "box_b": 0.10}
    k = {name: int(n * v) for name, v in share.items()}
    face(k["floor"], [-1.5, -1.5, 0.0], [3.0, 0, 0], [0, 3.0, 0], "+z", floor_colour)
    walls = [  # origin, e1, e2, normal into the room, colour
        ([-1.5, -1.5, 0.0], [0, 3.0, 0], [0, 0, 1.6], "+x", [0.55, 0.62, 0.75]),
        ([1.5, -1.5, 0.0], [0, 3.0, 0], [0, 0, 1.6], "-x", [0.75, 0.60, 0.55]),
        ([-1.5, -1.5, 0.0], [3.0, 0, 0], [0, 0, 1.6], "+y", [0.60, 0.72, 0.58]),
        ([-1.5, 1.5, 0.0], [3.0, 0, 0], [0, 0, 1.6], "-y", [0.78, 0.76, 0.62]),
    ]
    for origin, e1, e2, normal, base in walls:
        face(k["walls"] // 4, origin, e1, e2, normal, wall_colour(base), scale=(-5.4, -4.3))
    box(k["box_a"], [0.35, -0.15, 0.0], [0.65, 0.15, 0.3], stripes([0.85, 0.22, 0.15]))
    box(k["box_b"], [-0.55, 0.35, 0.0], [-0.3, 0.75, 0.45], stripes([0.20, 0.35, 0.85]))
    # floaters: translucent, small, anywhere in the room, any orientation
    rest = n - sum(len(part[0]) for part in parts)
    pos = np.stack([-1.4 + 2.8 * u(rest), -1.4 + 2.8 * u(rest), 0.05 + 1.4 * u(rest)], 1)
    q = u(rest, 4) * 2.0 - 1.0
    qn = np.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / np.where(qn < 1e-6, 1.0, qn)[:, None]
    grey = 0.5 + 0.2 * u(rest, 1)
    rgb = np.concatenate([grey, grey * (0.95 + 0.1 * u(rest, 1)), grey], 1)
    parts.append((pos, q, -5.0 + 1.5 * u(rest, 3), rgb, -4.0 + 3.0 * u(rest)))

    pos = np.concatenate([p[0] for p in parts])
    quat = np.concatenate([p[1] for p in parts])
    logs = np.concatenate([p[2] for p in parts])
    rgb = np.clip(np.concatenate([p[3] for p in parts]), 0.02, 0.98)
    logit = np.concatenate([p[4] for p in parts])
    order = rs.permutation(n)  # captures are not stored surface by surface
    cols = [
        pos[order],
        np.zeros((n, 3)),
        ((rgb - 0.5) / C0)[order],
        0.06 * (u(n, 45) - 0.5),  # SH degree 3: a little view dependence
        logit[order][:, None],
        logs[order],
        quat[order],
    ]
    data = np.concatenate(cols, 1).astype("<f4")
    assert data.shape == (n, len(ROOM_PROPS)), data.shape
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {p}\n" for p in ROOM_PROPS) + "end_header\n"
    return header.encode() + data.tobytes()


SCENES = {"r2_400k": (make_splat, SPLAT_SHA256), "room200k": (make_room, ROOM_SHA256)}


def splat(name: str = "r2_400k") -> bytes:
    """The PLY of a bench scene, checked against its SHA-256."""
    make, want = SCENES[name]
    data = make()
    got = hashlib.sha256(data).hexdigest()
    if want is not None and got != want:
        raise SystemExit(f"the generated splat of {name} has sha256 {got}, not {want}")
    return data


def trace(name: str = "r2_400k") -> tuple[dict, list[dict], int]:
    """(scene_init of scene ``name``, states, number of warm-up states). Both
    scenes have the one trace: the same cameras in the same states."""
    scene_init = json.loads((TRACE / "scene_init.json").read_text())
    scene_init["scene_id"] = f"{name}_splat"
    states = [json.loads(line) for line in (TRACE / "states.jsonl").read_text().splitlines()]
    meta = json.loads((TRACE / "meta.json").read_text())
    return scene_init, states, int(meta["warmup_states"])


class Response:
    """What a server's /render/ route answers, from memory: read(), the
    headers SceneAssets reads, a Range start."""

    def __init__(self, data: bytes, start: int = 0):
        self._body = io.BytesIO(data[start:])
        self.status = 206 if start else 200
        self._headers = {"Content-Length": str(len(data) - start)}
        if start:
            self._headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"

    def read(self, *args):
        return self._body.read(*args)

    def getheader(self, name, default=None):
        return self._headers.get(name, default)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


PNG_ENCODE = (b'convertToBlob({type:"image/jpeg",quality:s})', b'convertToBlob({type:"image/png"})')


class Server:
    """``fetch(path, headers)`` for both renderers: the splat, and the capture
    page (``/index.html``, ``/assets/...``) from ``page_dir``. ``lossless``
    serves the page with PNG in place of its JPEG encoding (same pixels as a
    readback), for the Spark reference."""

    def __init__(
        self, name: str = "r2_400k", page_dir: Path = CAPTURE_PAGE, lossless: bool = False
    ):
        self.assets = {SPLAT_URL: splat(name)}
        self.digests = {SPLAT_URL: SCENES[name][1]}
        self.page_dir = Path(page_dir)
        self.lossless = lossless
        self.requests: list[str] = []

    def __call__(self, path: str, headers: dict | None = None):
        self.requests.append(path)
        start = 0
        if headers and headers.get("Range", "").startswith("bytes="):
            start = int(headers["Range"][6:].split("-")[0])
        if path in self.assets:
            return Response(self.assets[path], start)
        if path == "/index.html" or path.startswith("/assets/"):
            file = (self.page_dir / path.lstrip("/")).resolve()
            if self.page_dir.resolve() not in file.parents or not file.is_file():
                raise FileNotFoundError(path)
            data = file.read_bytes()
            if self.lossless and file.name.startswith("main-") and file.suffix == ".js":
                if data.count(PNG_ENCODE[0]) != 1:
                    raise SystemExit(f"{file}: the JPEG encoding to replace is not there once")
                data = data.replace(*PNG_ENCODE)
            return Response(data, start)
        raise FileNotFoundError(path)
