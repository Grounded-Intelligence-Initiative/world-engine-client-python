"""3DGS splat data for D's WGSL renderer: PLY reading and the GPU layouts.

Standalone (numpy only, nothing from world_engine) so M-D3 can move it into
world-engine-client next to render_wgsl.py (design doc 16 §9.3 calls that
module ``assets.py``).

GPU layouts (see frontend/src/render/wgsl/README.md):

* geometry, 3 x vec4f per splat: (mean.xyz, opacity), (cov xx, xy, xz, yy),
  (cov yz, zz, 0, 0); world-space 3D covariance R S S^T R^T;
* SH, f16 pairs in u32 words: coefficient-major (k, channel), coefficient 0 is
  the DC term, ``budget.sh_words(degree)`` words per splat.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "<u2",
    "uint16": "<u2",
    "short": "<i2",
    "int16": "<i2",
    "uint": "<u4",
    "uint32": "<u4",
    "int": "<i4",
    "int32": "<i4",
}


_PLY_HEADER_MAX = 1 << 16
_END_HEADER = re.compile(rb"end_header\r?\n")


def _read_ply_vertices(src) -> np.ndarray:
    """The vertex element of a binary little-endian PLY as a structured array.

    ``src`` is a path, or the file's bytes (bytes, bytearray, memoryview or a
    uint8 array): a client that decrypts a scene asset in memory reads it from
    there, without writing it to disk. The array then views those bytes. The
    vertex element must come first (3DGS PLYs have only that one)."""
    if isinstance(src, (bytes, bytearray, memoryview, np.ndarray)):
        buf = np.frombuffer(src, np.uint8) if not isinstance(src, np.ndarray) else src.reshape(-1)
        if buf.dtype != np.uint8:
            raise TypeError("PLY bytes must be uint8")
        name = "the PLY"
        head = buf[:_PLY_HEADER_MAX].tobytes()
    else:
        buf, name = None, str(src)
        with open(src, "rb") as f:
            head = f.read(_PLY_HEADER_MAX)
    m = _END_HEADER.search(head)
    if m is None:
        raise ValueError(f"{name}: PLY header has no end_header")
    offset = m.end()
    lines = head[: m.start()].split(b"\n")
    if lines[0].strip() != b"ply":
        raise ValueError(f"{name}: not a PLY file")
    fmt = None
    n = None
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in lines[1:]:
        tok = line.decode("ascii", "replace").split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            if n is None and tok[1] != "vertex":
                raise ValueError(f"{name}: element {tok[1]!r} before the vertex element")
            in_vertex = tok[1] == "vertex" and n is None
            if in_vertex:
                n = int(tok[2])
        elif tok[0] == "property" and in_vertex:
            if tok[1] == "list":
                raise ValueError(f"{name}: list properties in the vertex element")
            if tok[1] not in _PLY_TYPES:
                raise ValueError(f"{name}: PLY type {tok[1]!r}")
            props.append((tok[2], _PLY_TYPES[tok[1]]))
    if fmt != "binary_little_endian":
        raise ValueError(f"{name}: PLY format {fmt!r}, expected binary_little_endian")
    if not n or not props:
        raise ValueError(f"{name}: no vertex element")
    dtype = np.dtype(props)
    if buf is None:
        with open(src, "rb") as f:
            f.seek(offset)
            data = np.fromfile(f, dtype=dtype, count=n)
    else:
        have = (len(buf) - offset) // dtype.itemsize
        if have < n:
            raise ValueError(f"{name}: {have} of {n} vertices")
        data = np.frombuffer(buf, dtype=dtype, count=n, offset=offset)
    if len(data) != n:
        raise ValueError(f"{name}: {len(data)} of {n} vertices")
    return data


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1),
        ],
        1,
    )


def _spark_quat_roundtrip(q: np.ndarray) -> np.ndarray:
    """wxyz quaternions through Spark's encodeQuatOctXy88R8 / decodeQuatOctXy88R8."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(q[:, :1] < 0, -q, q)
    theta = 2.0 * np.arccos(np.clip(q[:, 0], -1.0, 1.0))
    xyz = q[:, 1:]
    norm = np.linalg.norm(xyz, axis=1)
    axis = np.where(
        norm[:, None] < 1e-6, np.array([[1.0, 0.0, 0.0]]), xyz / np.maximum(norm, 1e-30)[:, None]
    )
    s = np.abs(axis).sum(1)
    px, py = axis[:, 0] / s, axis[:, 1] / s
    neg = axis[:, 2] < 0
    px2 = np.where(neg, (1 - np.abs(py)) * np.where(px >= 0, 1.0, -1.0), px)
    py2 = np.where(neg, (1 - np.abs(px)) * np.where(py >= 0, 1.0, -1.0), py)
    qu = np.round((px2 * 0.5 + 0.5) * 255)
    qv = np.round((py2 * 0.5 + 0.5) * 255)
    qa = np.round(theta * 255 / math.pi)
    fx, fy = (qu / 255 - 0.5) * 2, (qv / 255 - 0.5) * 2
    fz = 1 - (np.abs(fx) + np.abs(fy))
    t = np.maximum(-fz, 0)
    fx = fx + np.where(fx >= 0, -t, t)
    fy = fy + np.where(fy >= 0, -t, t)
    ax = np.stack([fx, fy, fz], 1)
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    half = qa / 255 * math.pi / 2
    return np.concatenate([np.cos(half)[:, None], ax * np.sin(half)[:, None]], 1)


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return _quat_wxyz_to_matrix(np.array([[w / n, x / n, y / n, z / n]]))[0]


@dataclass
class Splats:
    """Activated 3DGS parameters (the PLY's logs and logits already applied)."""

    means: np.ndarray  # (N, 3)
    scales: np.ndarray  # (N, 3)
    quats: np.ndarray  # (N, 4) w, x, y, z, unit length
    opacities: np.ndarray  # (N,)
    sh: np.ndarray  # (N, K, 3), K = (degree + 1) ** 2; sh[:, 0] is the DC term
    # world -> splat-object rotation: SH coefficients live in the PLY's frame
    sh_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    world_scale: float = 1.0

    @property
    def n(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        return int(round(math.sqrt(self.sh.shape[1]))) - 1

    def with_sh_degree(self, degree: int) -> Splats:
        k = (degree + 1) ** 2
        if k > self.sh.shape[1]:
            raise ValueError(f"scene has SH degree {self.sh_degree}, asked for {degree}")
        return Splats(
            self.means,
            self.scales,
            self.quats,
            self.opacities,
            self.sh[:, :k],
            self.sh_rotation,
            self.world_scale,
        )

    def subset(self, index) -> Splats:
        return Splats(
            self.means[index],
            self.scales[index],
            self.quats[index],
            self.opacities[index],
            self.sh[index],
            self.sh_rotation,
            self.world_scale,
        )

    def finite_mask(self, geometry: np.ndarray | None = None) -> np.ndarray:
        """True for the splats the renderer can draw: the float32 GPU geometry
        (position, opacity, world covariance) and the SH coefficients are all
        finite. A non-finite scale or rotation, a zero rotation quaternion and a
        covariance that overflows float32 all show up in the covariance.
        ``geometry``: this scene's gpu_geometry(), if already computed.

        The shaders must not see NaN or Inf: WGSL lets an implementation assume
        neither occurs, so what such a splat does differs between adapters."""
        if geometry is None:
            with np.errstate(all="ignore"):
                geometry = self.gpu_geometry()
        return np.isfinite(geometry).all(1) & np.isfinite(self.sh.reshape(self.n, -1)).all(1)

    def placed(self, pos=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), scale=1.0) -> Splats:
        """The splats under scene_init.splat's placement (the editor's splat_align)."""
        r = quat_xyzw_to_matrix(quat_xyzw)
        if np.allclose(r, np.eye(3)) and float(scale) == 1.0 and not np.any(pos):
            return self
        means = (float(scale) * self.means.astype(np.float64)) @ r.T + np.asarray(pos, float)
        qr = np.array(quat_xyzw, float)
        qw = np.array([qr[3], qr[0], qr[1], qr[2]]) / np.linalg.norm(qr)
        q = self.quats.astype(np.float64)
        w1, x1, y1, z1 = qw
        w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        quats = np.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            1,
        )
        return Splats(
            means.astype(np.float32),
            (self.scales * float(scale)).astype(np.float32),
            quats.astype(np.float32),
            self.opacities,
            self.sh,
            self.sh_rotation @ r.T,
            self.world_scale * float(scale),
        )

    def spark_packed(self) -> Splats:
        """The splats as Spark 2.1.0 stores them (opt-in profile, not the default).

        Spark's SplatMesh keeps a PLY in PackedSplats with DEFAULT_SPLAT_ENCODING:
        centre as f16, each scale as an 8-bit log over [-12, 9], the rotation as an
        8+8-bit octahedral axis and an 8-bit angle, opacity and base colour in 8
        bits, SH degree 1 / 2 / 3 as 7 / 8 / 6-bit signed values over [-1, 1]; its
        accumulator then keeps half the opacity in 8 bits. Rendering this copy
        brings D closer to the editor's Spark image (r2_room: 0.95 instead of 1.57
        per 255, splat region, lossless reference), at the cost of the PLY's
        precision. Spark's per-frame re-quantization of camera-relative centres is
        not reproduced.

        The opacity is capped at 1. Spark decodes an opacity byte of 255 as
        2 * 128 / 255 = 1.0039 and then takes its extended-opacity branch for
        alpha > 1 (a wider quad and another falloff, splatVertex/splatFragment);
        D has no such branch, and an alpha above 1 would make the blend factor
        one-minus-src-alpha negative. The difference is below 1/255."""
        ln_min, ln_max = -12.0, 9.0
        means = self.means.astype(np.float16).astype(np.float32)
        step = (ln_max - ln_min) / 254.0
        u = np.clip(np.round((np.log(self.scales.astype(np.float64)) - ln_min) / step) + 1, 1, 255)
        scales = np.exp(ln_min + (u - 1) * step).astype(np.float32)
        quats = _spark_quat_roundtrip(self.quats.astype(np.float64)).astype(np.float32)
        ua = np.clip(np.round(self.opacities.astype(np.float64) * 255), 0, 255)
        opac = np.minimum(2.0 * np.round(0.5 * ua) / 255.0, 1.0).astype(np.float32)
        sh = self.sh.astype(np.float64).copy()
        c0 = 0.28209479177387814
        base = np.clip(np.round((0.5 + c0 * sh[:, 0]) * 255), 0, 255)
        sh[:, 0] = (base / 255.0 - 0.5) / c0
        for lo, hi, levels in ((1, 4, 63), (4, 9, 127), (9, 16, 31)):
            if sh.shape[1] > lo:
                sh[:, lo:hi] = np.round(np.clip(sh[:, lo:hi] * levels, -levels, levels)) / levels
        return Splats(
            means, scales, quats, opac, sh.astype(np.float32), self.sh_rotation, self.world_scale
        )

    def covariances(self) -> np.ndarray:
        """(N, 6) world covariance xx, xy, xz, yy, yz, zz (float64 math)."""
        q = self.quats.astype(np.float64)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        m = _quat_wxyz_to_matrix(q) * self.scales.astype(np.float64)[:, None, :]
        c = m @ m.transpose(0, 2, 1)
        return np.stack([c[:, 0, 0], c[:, 0, 1], c[:, 0, 2], c[:, 1, 1], c[:, 1, 2], c[:, 2, 2]], 1)

    def gpu_geometry(self) -> np.ndarray:
        """(N, 12) float32, the geometry buffer layout."""
        out = np.zeros((self.n, 12), np.float32)
        cov = self.covariances()
        out[:, 0:3] = self.means
        out[:, 3] = self.opacities
        out[:, 4:8] = cov[:, 0:4]
        out[:, 8:10] = cov[:, 4:6]
        return out

    def gpu_sh(self, step: int = 1 << 18) -> np.ndarray:
        """(N, sh_words) uint32: f16 pairs, coefficient-major, low half first.
        Converted ``step`` splats at a time, so the temporaries stay small (a
        1.5M-splat SH3 scene would need 0.6 GB of float64 at once)."""
        flat = self.sh.reshape(self.n, -1)
        per = flat.shape[1]
        out = np.empty((self.n, (per + 1) // 2), np.uint32)
        for s in range(0, self.n, step):
            # float32 to float16 rounds as float64 to float16 does (the float64
            # holds the float32 exactly), and +-65504 are float32 values
            halves = np.clip(flat[s : s + step], -65504.0, 65504.0).astype(np.float16)
            if per % 2:
                halves = np.concatenate([halves, np.zeros((len(halves), 1), np.float16)], 1)
            u16 = np.ascontiguousarray(halves).view(np.uint16).astype(np.uint32)
            out[s : s + step] = u16[:, 0::2] | (u16[:, 1::2] << 16)
        return out


def read_splat_ply(path, max_sh_degree: int = 3) -> Splats:
    """Read a 3DGS PLY (x y z, f_dc_*, f_rest_*, opacity, scale_*, rot_*) from a
    path or from the file's bytes (see _read_ply_vertices)."""
    v = _read_ply_vertices(path)
    if not isinstance(path, (str, Path)):
        path = "the PLY"
    names = set(v.dtype.names or ())
    need = {
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
    }
    missing = sorted(need - names)
    if missing:
        raise ValueError(f"{path}: not a 3DGS PLY, missing {', '.join(missing)}")

    def col(*cols):
        return np.stack([v[c].astype(np.float64) for c in cols], 1)

    n_rest = len([c for c in names if c.startswith("f_rest_")])
    degree = int(round(math.sqrt(n_rest / 3 + 1))) - 1 if n_rest else 0
    if 3 * ((degree + 1) ** 2 - 1) != n_rest:
        raise ValueError(f"{path}: {n_rest} f_rest_* properties is no SH degree")
    degree = min(degree, max_sh_degree)
    k = (degree + 1) ** 2
    # Filled column by column in float32: the values are those of a float64
    # detour (a float or double field rounds to float32 once either way), without
    # its 1 GB of temporaries on a 1.5M-splat SH3 scene. f_rest_{c * m + j} is
    # coefficient j + 1 of channel c, m = n_rest / 3.
    sh = np.empty((len(v), k, 3), np.float32)
    for c in range(3):
        sh[:, 0, c] = v[f"f_dc_{c}"]
        for j in range(k - 1):
            sh[:, 1 + j, c] = v[f"f_rest_{c * (n_rest // 3) + j}"]
    q = col("rot_0", "rot_1", "rot_2", "rot_3")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    return Splats(
        means=col("x", "y", "z").astype(np.float32),
        scales=np.exp(col("scale_0", "scale_1", "scale_2")).astype(np.float32),
        quats=q.astype(np.float32),
        opacities=(1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))).astype(np.float32),
        sh=sh,
    )


def write_splat_ply(path: str | Path, s: Splats) -> None:
    """Write a standard 3DGS PLY (float32, the inverse of read_splat_ply)."""
    n_rest = 3 * (s.sh.shape[1] - 1)
    names = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(n_rest)]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    cols = [s.means.astype(np.float64), np.zeros((s.n, 3)), s.sh[:, 0].astype(np.float64)]
    if n_rest:
        cols.append(s.sh[:, 1:].astype(np.float64).transpose(0, 2, 1).reshape(s.n, n_rest))
    op = np.clip(s.opacities.astype(np.float64), 1e-7, 1 - 1e-7)
    cols += [
        np.log(op / (1 - op))[:, None],
        np.log(s.scales.astype(np.float64)),
        s.quats.astype(np.float64),
    ]
    data = np.concatenate(cols, 1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {s.n}\n"
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())
