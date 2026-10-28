"""Lighting of D's mesh pass: what the capture page lights its meshes with.

Standalone (numpy only; nothing from world_engine), like ``splat_data``.
``frontend/src/sceneRenderer.ts`` picks one of three set-ups, and so does
``WgslRenderer``:

* ``scene``: the environment baked from the splats at the light probe
  (``sceneEnvironment.ts``): a 128^2 cube map of the splats (8-bit, linear,
  mipmapped), its light probe (``lighting.ts`` dominantLightDirection: mean
  luminance and the luminance^4-weighted direction above the horizon), PMREM
  prefiltered; environment intensity clamp(0.08 / mean, 1, 2.5), no rotation,
  the key light from the probe direction at 0.3. A probe that is black, or
  darker than a mean of 0.03, keeps the studio set-up.
* ``studio``: three's RoomEnvironment prefiltered with sigma 0.04, intensity
  0.7, rotated so the room's +y is the world's +z, key light from (2, -2, 4)
  at 0.9. Also when there is no splat.
* ``hdri``: an equirectangular .hdr/.exr (only when the engine sends one, which
  WE_FEATURE_HDRI gates), prefiltered, intensity = its strength, the studio
  rotation, no key light.

A prefiltered environment can also come as a lighting asset
(``scene_init.lighting``, design doc 16 §9.2 step 8): the PMREM atlas plus the
intensities and the key light, in the WELI1 format below; the host then skips
the bake. Nothing produces it in the product yet (the editor will, when a scene
is saved); the bench writes it from the capture page's own bake.

WELI1 (little-endian): ``"WELI"`` | u32 version (1) | u32 json bytes | u32 data
bytes | JSON (UTF-8, space-padded to 4) | the atlas, RGBA float16, rows in
three's order (row 0 is v = 0). JSON: width, height, max_mip, intensity,
rotation (three's Euler, radians), key (dir, intensity, colour) or null, mode,
probe.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

STUDIO_ENV_INTENSITY = 0.7
STUDIO_KEY_INTENSITY = 0.9
STUDIO_KEY_DIR = (2.0, -2.0, 4.0)
SCENE_ENV_INTENSITY = 1.0
SCENE_ENV_MIN_MEAN = 0.03
SCENE_ENV_TARGET_MEAN = 0.08
SCENE_ENV_MAX_BOOST = 2.5
SCENE_KEY_INTENSITY = 0.3
PROBE_HEIGHT = 0.6
ENV_ROTATION_ZUP = (math.pi / 2, 0.0, 0.0)
BAKE_SIZE = 128  # sceneEnvironment.ts bake(size)
BAKE_NEAR = 0.05
BAKE_FAR = 200.0
STUDIO_SIZE = 256  # PMREMGenerator.fromScene default size
STUDIO_SIGMA = 0.04
# The splat sizes of the capture page's bake (sceneEnvironment.ts). The page asks
# SceneEnvironmentBaker to reuse the scene's SparkRenderer, but the baker only
# reuses one whose enableLod equals the splat mesh's LoD mode: Spark 2.1.0
# defaults enableLod to true and the page's PLY is not a LoD mesh, so the baker
# makes its own SparkRenderer, with maxPixelRadius 256 (and encodeLinear). Spark
# sizes splats (the 0.3 px^2 blur, its alpha compensation, the cap) in the pixels
# of its renderSize, the canvas's drawing buffer, not the 128^2 cube target; the
# canvas is the headless window then, 1280 x 720 (Playwright's default viewport)
# at pixel ratio 1. Against the page's read-back faces (DW/bench/m-d2) these
# constants give 1.619 / 255 per face on r2_room; the scene renderer's 512 px cap
# at the same size gives 1.862, and 5 to 13 on the golden scene. Should the
# page's reuse ever take effect (a LoD splat mesh, or a change to that check),
# its bake becomes 512 px at 1280 x 720.
CAPTURE_PAGE_CANVAS = (1280, 720)
BAKE_RENDER_SIZE = CAPTURE_PAGE_CANVAS
BAKE_MAX_PIXEL_RADIUS = 256.0


def euler_xyz(e) -> np.ndarray:
    """three's Matrix4.makeRotationFromEuler for order XYZ (3x3)."""
    a, b, c = (float(v) for v in e)
    ca, sa, cb, sb, cc, sc = (
        math.cos(a),
        math.sin(a),
        math.cos(b),
        math.sin(b),
        math.cos(c),
        math.sin(c),
    )
    return np.array(
        [
            [cb * cc, -cb * sc, sb],
            [ca * sc + sa * sb * cc, ca * cc - sa * sb * sc, -sa * cb],
            [sa * sc - ca * sb * cc, sa * cc + ca * sb * sc, ca * cb],
        ]
    )


def env_map_rotation(e) -> np.ndarray:
    """three's envMapRotation uniform for a PMREM environment (WebGLMaterials:
    the scene's Euler negated, "accommodate left-handed frame")."""
    return euler_xyz([-float(v) for v in e])


def cube_face_direction(face: int, s, t) -> np.ndarray:
    """lighting.ts cubeFaceDirection: world direction of texel (s, t) in [-1, 1]^2
    of a read-back cube face (WebGL cube convention, t grows with the row)."""
    s, t = np.broadcast_arrays(np.asarray(s, np.float64), np.asarray(t, np.float64))
    one = np.ones_like(s)
    table = {
        0: (one, -t, -s),
        1: (-one, -t, s),
        2: (s, one, t),
        3: (s, -one, -t),
        4: (s, -t, one),
        5: (-s, -t, -one),
    }
    return np.stack(table[face], -1)


@dataclass
class Probe:
    dir: np.ndarray
    mean: float
    peak: float


def dominant_light_direction(faces: np.ndarray, above_horizon: bool = True) -> Probe | None:
    """lighting.ts dominantLightDirection over six (size, size, 4) uint8 faces in
    read-back order (row 0 is the GL bottom row)."""
    faces = np.asarray(faces)
    size = faces.shape[1]
    c = (2 * (np.arange(size) + 0.5)) / size - 1
    s, t = np.meshgrid(c, c)  # s along columns, t along rows
    solid = np.power(1 + s * s + t * t, -1.5)
    acc = np.zeros(3)
    wsum = lsum = asum = 0.0
    peak = 0.0
    for f in range(6):
        px = faces[f].astype(np.float64)
        lum = (0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]) / 255
        d = cube_face_direction(f, s, t)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        lsum += float((lum * solid).sum())
        asum += float(solid.sum())
        peak = max(peak, float(lum.max()))
        keep = np.ones_like(lum, bool) if not above_horizon else d[..., 2] > 0.05
        w = lum**4 * solid * keep
        acc += (d * w[..., None]).sum((0, 1))
        wsum += float(w.sum())
    if wsum <= 1e-6 or float(acc @ acc) < 1e-8:
        return None
    return Probe(acc / np.linalg.norm(acc), lsum / asum, peak)


# The bake is used when its probe's mean luminance reaches SCENE_ENV_MIN_MEAN;
# the two set-ups look very different. The host blends its bake above 8 bits and
# rounds once: the RTX 4090, lavapipe and SwiftShader bake the indoor probe
# within 0.003 % of each other. Blended in an 8-bit target, as the capture page
# blends, the RTX 4090's indoor probe is 0.8 % brighter and lavapipe's 1.2 %
# brighter than that. Within this fraction of the threshold the host warns.
PROBE_MARGIN_WARN = 0.05


def probe_report(p: Probe) -> dict:
    """The probe for info["render"]: direction, mean and peak luminance, and the
    mean's relative margin over SCENE_ENV_MIN_MEAN (negative: below, studio)."""
    return {
        "dir": [float(v) for v in p.dir],
        "mean": float(p.mean),
        "peak": float(p.peak),
        "threshold": SCENE_ENV_MIN_MEAN,
        "margin": float(p.mean / SCENE_ENV_MIN_MEAN - 1.0),
    }


def scene_intensity(mean: float) -> float:
    return min(SCENE_ENV_MAX_BOOST, max(SCENE_ENV_INTENSITY, SCENE_ENV_TARGET_MEAN / mean))


def _look_quat_xyzw(forward, up) -> list[float]:
    """xyzw quaternion of a GL camera (looks down -z, y up) facing ``forward``
    with ``up`` (three's Matrix4.lookAt for a camera)."""
    f = np.asarray(forward, np.float64)
    z = -f / np.linalg.norm(f)
    x = np.cross(np.asarray(up, np.float64), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.stack([x, y, z], 1)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        k = 2.0 * math.sqrt(1.0 + tr)
        q = [(m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k, k / 4]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k_ = (i + 1) % 3, (i + 2) % 3
        r = 2.0 * math.sqrt(1.0 + m[i, i] - m[j, j] - m[k_, k_])
        q = [0.0, 0.0, 0.0, (m[k_, j] - m[j, k_]) / r]
        q[i] = r / 4
        q[j] = (m[j, i] + m[i, j]) / r
        q[k_] = (m[k_, i] + m[i, k_]) / r
    return [float(v) for v in q]


# Bake cameras: one per GL cube face (+x, -x, +y, -y, +z, -z), facing the face
# axis, with the up vector that makes D's top-down image the face image in
# read-back order once its rows are reversed (see bake_cameras).
_FACE_AXES = [
    ((1, 0, 0), (0, -1, 0)),
    ((-1, 0, 0), (0, -1, 0)),
    ((0, 1, 0), (0, 0, 1)),
    ((0, -1, 0), (0, 0, -1)),
    ((0, 0, 1), (0, -1, 0)),
    ((0, 0, -1), (0, -1, 0)),
]


def bake_cameras(center, size: int = BAKE_SIZE) -> list[dict]:
    """Six render_state cameras (90 degrees, size^2) for the cube faces of the
    bake at ``center``. Each face image comes back top-down; reversed
    top-to-bottom it is the face as three reads it back (row 0 = t of -1):
    the GL cube convention puts world -y (for the side faces) at t = +1, so
    the camera's up is world -y there, and D's top row (its up) lands at
    t = -1 after the reversal. The tests check every face against
    cube_face_direction."""
    out = []
    for i, (fwd, up) in enumerate(_FACE_AXES):
        out.append(
            {
                "name": f"bake{i}",
                "pos": [float(v) for v in center],
                "quat": _look_quat_xyzw(fwd, up),
                "fov_y_deg": 90.0,
                "w": int(size),
                "h": int(size),
                "near": BAKE_NEAR,
                "far": BAKE_FAR,
            }
        )
    return out


@dataclass
class Lighting:
    """What lights the meshes: a CubeUV environment (atlas) with an intensity
    and rotation, and an optional directional key light."""

    mode: str  # scene | studio | hdri | asset | uniform | none
    atlas: np.ndarray | None = None  # (H, W, 4) float32, three's row order
    max_mip: int = 0
    env_intensity: float = 1.0
    env_rotation: tuple = (0.0, 0.0, 0.0)  # three's scene.environmentRotation (Euler XYZ)
    key_dir: tuple | None = None  # world direction towards the light
    key_intensity: float = 0.0
    key_color: tuple = (1.0, 1.0, 1.0)
    info: dict = field(default_factory=dict)
    source: object = None  # GPU-side: what to prefilter (set by the host)

    @staticmethod
    def uniform(radiance: float = 1.0) -> Lighting:
        """A constant environment (every direction ``radiance``), no key light:
        the default for hand-built scenes and tests."""
        return Lighting("uniform", atlas=None, env_intensity=float(radiance))

    def rotation_matrix(self) -> np.ndarray:
        return env_map_rotation(self.env_rotation)


def atlas_size(cube_size: int) -> tuple[int, int, int]:
    """(width, height, max_mip) of the PMREM atlas for a cube of cube_size
    (PMREMGenerator._setSize, _allocateTargets)."""
    lod_max = int(math.floor(math.log2(cube_size)))
    cs = 2**lod_max
    return 3 * max(cs, 16 * 7), 4 * cs, lod_max


# ---------------------------------------------------------------- WELI1 assets
WELI_MAGIC = b"WELI"


def write_weli1(path, lighting: Lighting) -> None:
    """Store a prefiltered environment and its light set-up (bench and tests)."""
    a = np.ascontiguousarray(lighting.atlas, np.float32)
    h, w = a.shape[:2]
    doc = {
        "width": w,
        "height": h,
        "max_mip": int(lighting.max_mip),
        "intensity": float(lighting.env_intensity),
        "rotation": [float(v) for v in lighting.env_rotation],
        "key": (
            None
            if lighting.key_dir is None
            else {
                "dir": [float(v) for v in lighting.key_dir],
                "intensity": float(lighting.key_intensity),
                "color": [float(v) for v in lighting.key_color],
            }
        ),
        "mode": lighting.info.get("source_mode", lighting.mode),
        "probe": lighting.info.get("probe"),
    }
    jb = json.dumps(doc).encode("utf-8")
    jb += b" " * ((4 - len(jb) % 4) % 4)
    data = a.astype(np.float16).tobytes()
    with open(path, "wb") as f:
        f.write(WELI_MAGIC + struct.pack("<III", 1, len(jb), len(data)) + jb + data)


def read_weli1(data: bytes) -> Lighting:
    if data[:4] != WELI_MAGIC:
        raise ValueError("not a WELI1 lighting asset")
    version, jn, dn = struct.unpack("<III", data[4:16])
    if version != 1:
        raise ValueError(f"WELI version {version}; this host reads version 1")
    doc = json.loads(data[16 : 16 + jn].decode("utf-8"))
    w, h = int(doc["width"]), int(doc["height"])
    atlas = np.frombuffer(data, np.float16, w * h * 4, 16 + jn).reshape(h, w, 4).astype(np.float32)
    key = doc.get("key")
    return Lighting(
        "asset",
        atlas=atlas,
        max_mip=int(doc["max_mip"]),
        env_intensity=float(doc["intensity"]),
        env_rotation=tuple(doc.get("rotation") or (0.0, 0.0, 0.0)),
        key_dir=None if key is None else tuple(key["dir"]),
        key_intensity=0.0 if key is None else float(key["intensity"]),
        key_color=(1.0, 1.0, 1.0) if key is None else tuple(key.get("color", (1, 1, 1))),
        info={"asset_mode": doc.get("mode"), "probe": doc.get("probe")},
    )


# ------------------------------------------------------------------ HDR images
def read_hdr(data: bytes) -> np.ndarray:
    """Radiance RGBE (.hdr) -> (H, W, 3) float32, rows top to bottom, decoded as
    three's HDRLoader does."""
    if not (data.startswith(b"#?RADIANCE") or data.startswith(b"#?RGBE")):
        raise ValueError("not a Radiance .hdr file")
    pos = 0
    while True:
        end = data.index(b"\n", pos)
        line = data[pos:end].strip()
        pos = end + 1
        if line.startswith(b"-Y") or line.startswith(b"+Y"):
            parts = line.split()
            h, w = int(parts[1]), int(parts[3])
            flip = parts[0] == b"+Y"
            break
    out = np.zeros((h, w, 4), np.uint8)
    buf = memoryview(data)
    for y in range(h):
        if (
            w >= 8
            and w < 32768
            and buf[pos] == 2
            and buf[pos + 1] == 2
            and (buf[pos + 2] & 0x80) == 0
        ):
            pos += 4
            for ch in range(4):
                x = 0
                while x < w:
                    n = buf[pos]
                    pos += 1
                    if n > 128:
                        n -= 128
                        out[y, x : x + n, ch] = buf[pos]
                        pos += 1
                    else:
                        out[y, x : x + n, ch] = np.frombuffer(buf[pos : pos + n], np.uint8)
                        pos += n
                    x += n
        else:
            out[y] = np.frombuffer(buf[pos : pos + 4 * w], np.uint8).reshape(w, 4)
            pos += 4 * w
    # three's HDRLoader: byte * 2^(e - 128) / 255 (not the format's / 256)
    e = out[..., 3].astype(np.float64)
    f = np.power(2.0, e - 128.0) / 255.0
    rgb = out[..., :3].astype(np.float64) * f[..., None]
    if flip:
        rgb = rgb[::-1]
    return rgb.astype(np.float32)


def read_exr(data: bytes) -> np.ndarray:
    """OpenEXR scanline images with no, RLE, ZIPS or ZIP compression -> (H, W, 3)
    float32. PIZ, PXR24, B44 and DWA raise (the host then keeps the scene bake)."""
    if data[:4] != b"\x76\x2f\x31\x01":
        raise ValueError("not an OpenEXR file")
    pos = 8
    header: dict = {}
    while data[pos] != 0:
        end = data.index(b"\0", pos)
        name = data[pos:end].decode()
        pos = end + 1
        end = data.index(b"\0", pos)
        typ = data[pos:end].decode()
        pos = end + 1
        size = struct.unpack("<i", data[pos : pos + 4])[0]
        pos += 4
        header[name] = (typ, data[pos : pos + size])
        pos += size
    pos += 1
    comp = header["compression"][1][0]
    if comp not in (0, 1, 2, 3):
        raise ValueError(f"EXR compression {comp} is not supported (only none, RLE, ZIPS, ZIP)")
    raw = header["channels"][1]
    chans = []
    q = 0
    while raw[q] != 0:
        end = raw.index(b"\0", q)
        nm = raw[q:end].decode()
        q = end + 1
        pt = struct.unpack("<i", raw[q : q + 4])[0]
        q += 16
        chans.append((nm, pt))
    xmin, ymin, xmax, ymax = struct.unpack("<iiii", header["dataWindow"][1])
    w, h = xmax - xmin + 1, ymax - ymin + 1
    lines = {0: 1, 1: 1, 2: 1, 3: 16}[comp]
    nblocks = (h + lines - 1) // lines
    offsets = struct.unpack(f"<{nblocks}Q", data[pos : pos + 8 * nblocks])
    sizes = {0: 4, 1: 2, 2: 4}
    img = {nm: np.zeros((h, w), np.float32) for nm, _ in chans}
    for off in offsets:
        y0, n = struct.unpack("<ii", data[off : off + 8])
        blk = data[off + 8 : off + 8 + n]
        ny = min(lines, ymax + 1 - y0)
        expect = sum(sizes[pt] * w for _, pt in chans) * ny
        if comp in (2, 3):
            blk = zlib.decompress(blk)
        elif comp == 1:
            out = bytearray()
            i = 0
            while i < len(blk):
                c = struct.unpack("b", blk[i : i + 1])[0]
                i += 1
                if c < 0:
                    out += blk[i : i - c]
                    i -= c
                else:
                    out += blk[i : i + 1] * (c + 1)
                    i += 1
            blk = bytes(out)
        if comp in (1, 2, 3) and len(blk) == expect:
            t = np.frombuffer(blk, np.uint8).astype(np.int32)
            t = np.cumsum(np.concatenate([t[:1], t[1:] - 128])) & 255
            t = t.astype(np.uint8)
            half = (len(t) + 1) // 2
            inter = np.empty(len(t), np.uint8)
            inter[0::2] = t[:half]
            inter[1::2] = t[half:]
            blk = inter.tobytes()
        p = 0
        for yy in range(ny):
            for nm, pt in sorted(chans):
                nb = sizes[pt] * w
                row = blk[p : p + nb]
                p += nb
                dt = {0: "<u4", 1: "<f2", 2: "<f4"}[pt]
                img[nm][y0 - ymin + yy] = np.frombuffer(row, dt).astype(np.float32)
    get = lambda k: img.get(k, np.zeros((h, w), np.float32))  # noqa: E731
    return np.stack([get("R"), get("G"), get("B")], -1)


def read_hdri(data: bytes) -> np.ndarray:
    """An .hdr or .exr by its magic bytes (the compiler stores either as lighting.exr)."""
    if data[:4] == b"\x76\x2f\x31\x01":
        return read_exr(data)
    return read_hdr(data)


def half_float_like_three(x) -> np.ndarray:
    """three's DataUtils.toHalfFloat on an array, as HDRLoader and EXRLoader fill
    their HalfFloatType data: clamp to +-65504, round to float32, then drop the
    mantissa bits a half cannot hold (round towards zero; numpy's astype rounds
    to nearest, which is up to one half-float step brighter)."""
    x = np.clip(np.asarray(x, np.float32), -65504.0, 65504.0)
    h = x.astype(np.float16)
    away = np.abs(h.astype(np.float32)) > np.abs(x)
    h.view(np.uint16)[away] -= 1
    return h
