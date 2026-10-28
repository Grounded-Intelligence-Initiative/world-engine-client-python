"""Meshes for D's mesh pass: what the capture page draws, as GPU-ready arrays.

Standalone like ``splat_data`` (numpy, and Pillow for textures; nothing from
world_engine), so M-D3 can move it into world-engine-client. It reads the
scene_init a render session sends (design doc 16 §8.3, doc 14 §7.1):

* ``meshes[]``: an object with ``wem`` (the engine's WEM1 of its GLB,
  ``world_engine/assets/wem1.py``), ``scale`` (per axis, on top of the file),
  ``pos``/``quat`` (the authored pose; moving bodies then follow
  ``render_state.bodies``), ``local`` (``{pos, quat}``, the display mesh's offset
  from its body: world = body pose x local), ``box`` (``[hx, hy, hz]`` half
  sizes: a grey proxy box when there is no file); an entry with no file and no
  box is the capture page's 6 cm grey proxy box;
* ``robot.links[]``: WEL1 link meshes posed by ``render_state.robot.links``; no
  links means the blue end-effector proxy box at ``robot.ee_pos``;
* ``surfaces[]``: ``{id, mesh, uv_per_m}``, a belt whose maps scroll with
  ``render_state.surfaces[].travel`` (doc 14 §7.2: offset = offset0 - L g travel,
  each component mod 1, repeat wrapping on the scrolled maps).

Materials follow three 0.180 as the capture page sets them up: GLTFLoader's
MeshStandardMaterial for objects (made flat-shaded without normals, vertex
coloured with COLOR_0, normalScale.y negated with derivative tangents),
``robotMaterialFor(rgba, "painted")`` of ``frontend/src/robotMesh.ts`` for
robot links, ``MeshStandardMaterial({color, metalness: 0, roughness: 0.85})``
for proxy boxes. The capture page forces every GLB material opaque; so does
this (a BLEND material keeps GLTFLoader's depthWrite false), and the mesh pass
draws in three's opaque order (MeshScene.camera_order).
"""

from __future__ import annotations

import io
import json
import math
import struct
import warnings
from dataclasses import dataclass, field

import numpy as np

VERTEX_FLOATS = 20  # pos 3, normal 3, uv0 2, uv1 2, colour 4, tangent 4, owner (u32), pad
VERTEX_BYTES = 4 * VERTEX_FLOATS
OWNER_BYTES = 128  # model + normal matrix (mat4x4f each)
MATERIAL_SLOT = 512  # one material's uniform (272 B), at a 256-aligned offset
MAP_SLOTS = ("map", "normalMap", "roughnessMap", "metalnessMap", "aoMap", "emissiveMap")
SRGB_SLOTS = ("map", "emissiveMap")

# Material.flags.x bits (mesh.wgsl MF_*)
MF_MAP = 1
MF_NORMAL_MAP = 2
MF_ROUGHNESS_MAP = 4
MF_METALNESS_MAP = 8
MF_AO_MAP = 16
MF_EMISSIVE_MAP = 32
MF_VERTEX_RGB = 64
MF_VERTEX_RGBA = 128
MF_FLAT = 256
MF_DOUBLE_SIDED = 512
MF_TANGENT = 1024
MF_UNLIT = 2048
MF_RECEIVE_SHADOW = 4096
_SLOT_FLAG = {
    "map": MF_MAP,
    "normalMap": MF_NORMAL_MAP,
    "roughnessMap": MF_ROUGHNESS_MAP,
    "metalnessMap": MF_METALNESS_MAP,
    "aoMap": MF_AO_MAP,
    "emissiveMap": MF_EMISSIVE_MAP,
}


def srgb_to_linear(c):
    """three's SRGBToLinear (Color.setRGB(..., SRGBColorSpace), Color.setHex)."""
    c = np.asarray(c, np.float64)
    return np.where(c < 0.04045, c * 0.0773993808, np.power(c * 0.9478672986 + 0.0521327014, 2.4))


def hex_to_linear(h: int) -> list[float]:
    return [
        float(v)
        for v in srgb_to_linear([(h >> 16 & 255) / 255, (h >> 8 & 255) / 255, (h & 255) / 255])
    ]


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_matrix(pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_xyzw_to_matrix(quat)
    m[:3, 3] = [float(v) for v in pos]
    return m


@dataclass
class TextureRef:
    """One map of a material: an image, three's sampler state and uv transform."""

    image: int  # index into MeshScene.images
    channel: int = 0  # 0: uv (TEXCOORD_0), 1: uv1
    mag: str = "linear"
    min: str = "linear_mipmap_linear"
    wrap_s: str = "repeat"
    wrap_t: str = "repeat"
    offset: tuple[float, float] = (0.0, 0.0)
    repeat: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)

    def matrix(self, offset=None) -> np.ndarray:
        """three's Texture.matrix (Matrix3.setUvTransform), 3x3 row-major."""
        tx, ty = self.offset if offset is None else offset
        sx, sy = self.repeat
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        cx, cy = self.center
        return np.array(
            [
                [sx * c, sx * s, -sx * (c * cx + s * cy) + cx + tx],
                [-sy * s, sy * c, -sy * (-s * cx + c * cy) + cy + ty],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def mipmaps(self) -> bool:
        return self.min not in ("linear", "nearest")


@dataclass
class Material:
    """MeshStandardMaterial (or MeshBasicMaterial for ``unlit``) parameters."""

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)  # linear
    opacity: float = 1.0
    metalness: float = 0.0
    roughness: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)  # linear, times intensity
    side: str = "front"  # front | double
    alpha_test: float = 0.0
    normal_scale: tuple[float, float] = (1.0, 1.0)
    ao_intensity: float = 1.0
    maps: dict[str, TextureRef] = field(default_factory=dict)
    vertex_colors: int = 0  # 0, 3 (rgb) or 4 (rgba, USE_COLOR_ALPHA)
    flat: bool = False
    tangents: bool = False
    unlit: bool = False
    depth_write: bool = True
    cast_shadow: bool = True
    receive_shadow: bool = True
    name: str = ""

    def masked_shadow(self) -> bool:
        """three gives the shadow pass's depth material the caster's map and
        alphaTest (WebGLShadowMap.getDepthMaterial): a cast shadow with holes."""
        return self.cast_shadow and self.alpha_test > 0 and self.maps.get("map") is not None

    def flags(self) -> int:
        f = 0
        for slot, ref in self.maps.items():
            if ref is not None:
                f |= _SLOT_FLAG[slot]
        if self.vertex_colors == 4:
            f |= MF_VERTEX_RGBA
        elif self.vertex_colors == 3:
            f |= MF_VERTEX_RGB
        if self.flat:
            f |= MF_FLAT
        if self.side == "double":
            f |= MF_DOUBLE_SIDED
        if self.tangents:
            f |= MF_TANGENT
        if self.unlit:
            f |= MF_UNLIT
        if self.receive_shadow:
            f |= MF_RECEIVE_SHADOW
        return f

    def uniform(self, offsets: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
        """The Material struct of mesh.wgsl (272 bytes) as uint32 words."""
        u = np.zeros(MATERIAL_SLOT // 4, np.float32)
        u[0:3] = self.color
        u[3] = self.opacity
        u[4:7] = self.emissive
        u[8:12] = (self.metalness, self.roughness, self.ao_intensity, self.alpha_test)
        u[12:14] = self.normal_scale
        w = u.view(np.uint32)
        w[16] = self.flags()
        ch = 0
        for k, slot in enumerate(MAP_SLOTS):
            ref = self.maps.get(slot)
            if ref is not None and ref.channel == 1:
                ch |= 1 << k
        w[17] = ch
        for k, slot in enumerate(MAP_SLOTS):
            ref = self.maps.get(slot)
            m = np.eye(3) if ref is None else ref.matrix((offsets or {}).get(slot))
            u[20 + 8 * k : 23 + 8 * k] = m[0]
            u[24 + 8 * k : 27 + 8 * k] = m[1]
        return w


@dataclass
class Primitive:
    """Triangles of one owner with one material; arrays in the owner's frame."""

    positions: np.ndarray  # (n, 3) float32
    indices: np.ndarray  # (m, 3) uint32
    material: int
    normals: np.ndarray | None = None
    uv0: np.ndarray | None = None
    uv1: np.ndarray | None = None
    colors: np.ndarray | None = None  # (n, 3 or 4) float32, linear
    tangents: np.ndarray | None = None  # (n, 4)
    # where three measures the draw's depth for its opaque sort (the geometry's
    # bounding-sphere centre through its node, in the owner's frame); None: the
    # centre of the vertices' box
    center: np.ndarray | None = None
    # primitives with the same block share one copy of their vertices (the
    # material groups of a robot link: one geometry, its index ranges split);
    # they hold the same arrays
    block: int | None = None

    def sort_point(self) -> np.ndarray:
        if self.center is not None:
            return np.asarray(self.center, np.float64)
        if not len(self.positions):
            return np.zeros(3)
        p = self.positions.astype(np.float64)
        return (p.min(0) + p.max(0)) / 2


@dataclass
class Owner:
    """Something with a pose: a body, a robot link, the end-effector proxy, or
    a static mesh. Its world matrix is pose x local x diag(scale)."""

    kind: str  # body | link | ee
    key: str
    pos: tuple = (0.0, 0.0, 0.0)  # pose until the first state that names it
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    local: np.ndarray = field(default_factory=lambda: np.eye(4))
    scale: tuple = (1.0, 1.0, 1.0)


@dataclass
class Surface:
    """A scrolling belt: the maps of these materials move with ``travel``."""

    id: str
    materials: list[int]
    uv_per_m: tuple[float, float]


def compute_vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """three's BufferGeometry.computeVertexNormals for an indexed geometry:
    each face's cross((c - b), (a - b)) added to its three vertices, normalised."""
    v = np.asarray(v, np.float64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(c - b, a - b)
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.where(ln > 0, ln, 1.0)).astype(np.float32)


def box_primitive(w: float, h: float, d: float, material: int) -> Primitive:
    """THREE.BoxGeometry(w, h, d): 24 vertices, outward normals, uvs."""
    pos, nrm, uv, idx = [], [], [], []
    # buildPlane(u, v, w, udir, vdir, width, height, depth) as BoxGeometry calls it
    planes = [
        (2, 1, 0, -1, -1, d, h, w),  # px
        (2, 1, 0, 1, -1, d, h, -w),  # nx
        (0, 2, 1, 1, 1, w, d, h),  # py
        (0, 2, 1, 1, -1, w, d, -h),  # ny
        (0, 1, 2, 1, -1, w, h, d),  # pz
        (0, 1, 2, -1, -1, w, h, -d),  # nz
    ]
    for u_ax, v_ax, w_ax, udir, vdir, width, height, depth in planes:
        base = len(pos)
        for iy in range(2):
            for ix in range(2):
                p = [0.0, 0.0, 0.0]
                p[u_ax] = (ix * width - width / 2) * udir
                p[v_ax] = (iy * height - height / 2) * vdir
                p[w_ax] = depth / 2
                n = [0.0, 0.0, 0.0]
                n[w_ax] = 1.0 if depth > 0 else -1.0
                pos.append(p)
                nrm.append(n)
                uv.append([ix, 1 - iy])
        a, b, c, e = base, base + 2, base + 3, base + 1
        idx += [[a, b, e], [b, c, e]]
    return Primitive(
        np.asarray(pos, np.float32),
        np.asarray(idx, np.uint32),
        material,
        normals=np.asarray(nrm, np.float32),
        uv0=np.asarray(uv, np.float32),
    )


def decode_wel1(data: bytes) -> dict:
    """A robot link's WEL1 buffer (frontend/src/robotMesh.ts parseRobotLink)."""
    if data[:4] != b"WEL1":
        raise ValueError("robot link: bad magic")
    nv, nf, hc, hn, hg = struct.unpack("<IIBBBx", data[4:16])
    off = 16
    verts = np.frombuffer(data, "<f4", nv * 3, off).reshape(nv, 3).copy()
    off += nv * 12
    faces = np.frombuffer(data, "<u4", nf * 3, off).reshape(nf, 3).copy()
    off += nf * 12
    out = {"vertices": verts, "faces": faces, "colors": None, "normals": None, "groups": None}
    if hc == 1:
        out["colors"] = np.frombuffer(data, "u1", nv * 3, off).reshape(nv, 3).copy()
        off += nv * 3
    if hn == 1 and off + nv * 3 <= len(data):
        n8 = np.frombuffer(data, "i1", nv * 3, off).reshape(nv, 3).astype(np.float32)
        out["normals"] = np.maximum(n8 / 127.0, -1.0).astype(np.float32)
        off += nv * 3
    if hg == 1 and off + 4 <= len(data):
        n = struct.unpack("<I", data[off : off + 4])[0]
        off += 4
        rows = []
        for _ in range(n):
            if off + 12 > len(data):
                break
            rows.append(struct.unpack("<IIBBBB", data[off : off + 12]))
            off += 12
        out["groups"] = rows
    return out


def robot_material(rgba, finish: str = "painted") -> Material:
    """robotMaterialFor (frontend/src/robotMesh.ts) for the "painted" finish:
    dielectric, roughness 0.4 for saturated accents, 0.45 for dark parts, else
    0.55. The capture page passes no finish, so every robot is "painted"; the
    editor's "glossy-plastic" needs MeshPhysicalMaterial's clearcoat, which D does
    not draw (it falls back to "painted" with a warning)."""
    r, g, b, a = (int(x) for x in rgba)
    if finish != "painted":
        warnings.warn(
            f"robot finish {finish!r}: D draws robots with the 'painted' finish", RuntimeWarning, 2
        )
    sat = (max(r, g, b) - min(r, g, b)) / 255
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    roughness = 0.55
    if sat > 0.3:
        roughness = 0.4
    elif luma < 0.2:
        roughness = 0.45
    # transparent links (a < 255) are drawn opaque (the capture page sorts them
    # with three's transparent pass, which D does not have)
    return Material(
        color=tuple(srgb_to_linear([r / 255, g / 255, b / 255]).tolist()),
        metalness=0.0,
        roughness=roughness,
        name=f"robot {r},{g},{b},{a}",
    )


_SRGB_TABLE = srgb_to_linear(np.arange(256) / 255.0).astype(np.float32)


def decode_wem1(data: bytes) -> tuple[dict, bytes]:
    if data[:4] != b"WEM1":
        raise ValueError("not a WEM1 buffer")
    version, jn, bn = struct.unpack("<III", data[4:16])
    if version != 1:
        raise ValueError(f"WEM1 version {version}; this host reads version 1")
    doc = json.loads(bytes(data[16 : 16 + jn]).decode("utf-8"))
    return doc, data[16 + jn : 16 + jn + bn]


def _arr(body: bytes, ref, dtype, cols) -> np.ndarray:
    off, count = ref
    return np.frombuffer(body, dtype, count, off).reshape(-1, cols).copy()


def decode_image(raw: bytes) -> np.ndarray:
    """PNG / JPEG / WebP bytes -> (H, W, 4) uint8, EXIF orientation applied
    (Chrome's createImageBitmap default)."""
    from PIL import Image, ImageOps

    im = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im)
    return np.asarray(im.convert("RGBA"), np.uint8).copy()


def mip_chain(rgba: np.ndarray, srgb: bool) -> list[np.ndarray]:
    """Mip levels like WebGL's generateMipmap: halve (floor) until 1x1, 2x2
    averages (in linear light for sRGB textures, as GL filters sRGB formats)."""
    levels = [rgba]
    cur = rgba.astype(np.float64) / 255.0
    if srgb:
        cur[..., :3] = srgb_to_linear(cur[..., :3])
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        h, w = cur.shape[:2]
        nh, nw = max(1, h // 2), max(1, w // 2)
        ys = np.minimum(np.arange(nh) * 2, h - 1)
        xs = np.minimum(np.arange(nw) * 2, w - 1)
        ys1 = np.minimum(ys + 1, h - 1)
        xs1 = np.minimum(xs + 1, w - 1)
        cur = 0.25 * (cur[ys][:, xs] + cur[ys1][:, xs] + cur[ys][:, xs1] + cur[ys1][:, xs1])
        out = cur.copy()
        if srgb:
            c = out[..., :3]
            out[..., :3] = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
        levels.append(np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8))
    return levels


@dataclass
class TextureImage:
    rgba: np.ndarray  # (H, W, 4) uint8, level 0
    srgb: bool
    mipmaps: bool

    def levels(self) -> list[np.ndarray]:
        return mip_chain(self.rgba, self.srgb) if self.mipmaps else [self.rgba]

    def gpu_bytes(self) -> int:
        return sum(lv.nbytes for lv in self.levels()) if self.mipmaps else self.rgba.nbytes


class MeshScene:
    """Everything the mesh pass draws, from a scene_init and a byte fetcher."""

    def __init__(self):
        self.owners: list[Owner] = []
        self.owner_index: dict[tuple[str, str], int] = {}
        self.primitives: list[tuple[int, Primitive]] = []  # (owner, primitive)
        self.materials: list[Material] = []
        self.images: list[TextureImage] = []
        self.surfaces: list[Surface] = []
        self.ee_proxy: int | None = None
        self.warnings: list[str] = []
        self._robot_materials: dict[tuple, int] = {}
        self._blocks = 0
        self._vb = None

    # ------------------------------------------------------------ building
    def owner(self, kind: str, key: str, **kw) -> int:
        k = (kind, key)
        if k not in self.owner_index:
            self.owner_index[k] = len(self.owners)
            self.owners.append(Owner(kind, key, **kw))
        return self.owner_index[k]

    def add_material(self, m: Material) -> int:
        self.materials.append(m)
        return len(self.materials) - 1

    def add_primitive(self, owner: int, p: Primitive) -> None:
        self.primitives.append((owner, p))
        self._vb = None
        self.__dict__.pop("_owner_corners", None)

    def _texture(
        self, images: list, src_index: int, slot: str, ref: dict, image_of: dict
    ) -> TextureRef:
        """A map of one WEM1 file. image_of belongs to that file's add_wem1 call:
        (image, sRGB, mipmaps) -> index into self.images, so the materials of
        one file share an upload and no file ever sees another file's images."""
        srgb = slot in SRGB_SLOTS
        tr = TextureRef(
            image=-1,
            channel=1 if ref.get("channel", 0) >= 1 else 0,
            mag=ref.get("mag", "linear"),
            min=ref.get("min", "linear_mipmap_linear"),
            wrap_s=ref.get("wrap_s", "repeat"),
            wrap_t=ref.get("wrap_t", "repeat"),
            offset=tuple(ref.get("offset", (0.0, 0.0))),
            repeat=tuple(ref.get("repeat", (1.0, 1.0))),
            rotation=float(ref.get("rotation", 0.0)),
        )
        key = (src_index, srgb, tr.mipmaps)
        if key not in image_of:
            image_of[key] = len(self.images)
            self.images.append(TextureImage(images[src_index], srgb, tr.mipmaps))
        tr.image = image_of[key]
        return tr

    def add_wem1(self, owner: int, data: bytes) -> list[int]:
        """The primitives of a WEM1 object; returns the materials it made."""
        doc, body = decode_wem1(data)
        for u in doc.get("unsupported") or []:
            self.warnings.append(f"WEM1: {u} not drawn as three draws it")
        decoded = []
        for im in doc.get("images", []):
            raw = body[im["data"] : im["data"] + im["bytes"]]
            try:
                decoded.append(decode_image(raw) if raw else None)
            except Exception as e:  # pragma: no cover - a broken image is drawn white
                self.warnings.append(f"WEM1 image not decoded ({e})")
                decoded.append(None)
        image_of: dict[tuple[int, bool, bool], int] = {}
        made: dict[tuple, int] = {}
        out = []
        for p in doc["primitives"]:
            md = doc["materials"][p["material"]]
            has_t = "tangent" in p
            vc = int(p.get("color_components", 0)) if "color" in p else 0
            flat = "normal" not in p
            key = (p["material"], has_t, vc, flat)
            if key not in made:
                maps = {}
                for slot, ref in (md.get("maps") or {}).items():
                    src = ref["image"]
                    if src < len(decoded) and decoded[src] is not None:
                        maps[slot] = self._texture(decoded, src, slot, ref, image_of)
                ns = list(md.get("normal_scale", [1.0, 1.0]))
                if not has_t:
                    ns[1] = -ns[1]  # GLTFLoader assignFinalMaterial, derivative tangents
                mode = md.get("alpha_mode", "OPAQUE")
                mat = Material(
                    color=tuple(md["color"]),
                    opacity=float(md.get("opacity", 1.0)),
                    metalness=float(md.get("metalness", 1.0)),
                    roughness=float(md.get("roughness", 1.0)),
                    emissive=tuple(md.get("emissive", (0.0, 0.0, 0.0))),
                    side=md.get("side", "front"),
                    alpha_test=float(md.get("alpha_test", 0.0)),
                    normal_scale=(ns[0], ns[1]),
                    ao_intensity=float(md.get("ao_intensity", 1.0)),
                    maps=maps,
                    vertex_colors=vc,
                    flat=flat,
                    tangents=has_t,
                    unlit=md.get("type") == "basic",
                    depth_write=mode != "BLEND",
                    name=md.get("name", ""),
                )
                made[key] = self.add_material(mat)
            mi = made[key]
            out.append(mi)
            n = int(p["vertices"])
            prim = Primitive(
                _arr(body, p["position"], "<f4", 3),
                _arr(body, p["index"], "<u4", 3),
                mi,
                normals=_arr(body, p["normal"], "<f4", 3) if "normal" in p else None,
                uv0=_arr(body, p["uv0"], "<f4", 2) if "uv0" in p else None,
                uv1=_arr(body, p["uv1"], "<f4", 2) if "uv1" in p else None,
                colors=_arr(body, p["color"], "<f4", vc) if vc else None,
                tangents=_arr(body, p["tangent"], "<f4", 4) if has_t else None,
                center=np.asarray(p["center"], np.float64) if "center" in p else None,
            )
            if len(prim.positions) != n:
                raise ValueError("WEM1 primitive: vertex count disagrees with its arrays")
            self.add_primitive(owner, prim)
        return sorted(set(out))

    def add_wel1(self, owner: int, data: bytes, finish: str = "painted") -> None:
        g = decode_wel1(data)
        v, f = g["vertices"], g["faces"]
        normals = g["normals"] if g["normals"] is not None else compute_vertex_normals(v, f)
        groups = g["groups"]
        if groups:
            cache = self._robot_materials  # one material per colour, as many links share them
            self._blocks += 1
            for start, count, r, gg, b, a in groups:
                key = (r, gg, b, a, finish)
                if key not in cache:
                    cache[key] = self.add_material(robot_material(key[:4], finish))
                sub = f[start : start + count]
                self.add_primitive(
                    owner,
                    Primitive(v, sub.copy(), cache[key], normals=normals, block=self._blocks),
                )
        else:
            colors = None
            if g["colors"] is not None:
                colors = _SRGB_TABLE[g["colors"]]
                mat = Material(color=(1.0, 1.0, 1.0), metalness=0.0, roughness=0.5, vertex_colors=3)
            else:
                mat = Material(color=tuple(hex_to_linear(0xBFC4CC)), metalness=0.0, roughness=0.5)
            mi = self.add_material(mat)
            self.add_primitive(owner, Primitive(v, f, mi, normals=normals, colors=colors))

    def add_box(self, owner: int, size, color_hex: int = 0x9AA0A6) -> int:
        mi = self.add_material(
            Material(color=tuple(hex_to_linear(color_hex)), metalness=0.0, roughness=0.85)
        )
        self.add_primitive(owner, box_primitive(*[float(x) for x in size], mi))
        return mi

    @classmethod
    def from_scene_init(
        cls, scene_init: dict, fetch, *, glb_to_wem1=None, strict: bool = False
    ) -> MeshScene:
        """fetch(url) -> bytes. glb_to_wem1: converts a GLB the engine did not
        compile (an older server); without it such a mesh is skipped with a
        warning, and so is a robot link that cannot be fetched (the capture
        page skips it too). strict: raise ValueError for those instead (a
        client that can let the browser path draw the scene)."""
        s = cls()
        mesh_mats: dict[str, list[int]] = {}
        for m in scene_init.get("meshes") or []:
            mid = str(m["id"])
            local = None
            if m.get("local"):
                local = pose_matrix(
                    m["local"].get("pos", (0, 0, 0)), m["local"].get("quat", (0, 0, 0, 1))
                )
            kw = {
                "pos": tuple(m.get("pos") or (0.0, 0.0, 0.0)),
                "quat": tuple(m.get("quat") or (0.0, 0.0, 0.0, 1.0)),
                "scale": tuple(m.get("scale") or (1.0, 1.0, 1.0)),
            }
            if local is not None:
                kw["local"] = local
            oi = s.owner("body", mid, **kw)
            url = m.get("url") or ""
            if m.get("wem"):
                mesh_mats[mid] = s.add_wem1(oi, fetch(m["wem"]))
            elif url and url.lower().endswith(".glb") and glb_to_wem1 is not None:
                mesh_mats[mid] = s.add_wem1(oi, glb_to_wem1(fetch(url)))
            elif url:
                if strict:
                    raise ValueError(f"mesh {mid}: no WEM1 for {url}")
                s.warnings.append(f"mesh {mid}: no WEM1 for {url}; not drawn")
            elif m.get("box"):
                hx, hy, hz = (float(x) for x in m["box"])
                mesh_mats[mid] = [s.add_box(oi, (2 * hx, 2 * hy, 2 * hz))]
            else:
                mesh_mats[mid] = [s.add_box(oi, (0.06, 0.06, 0.06))]
        robot = scene_init.get("robot") or {}
        finish = robot.get("finish") or "painted"
        n_links = 0
        for ln in robot.get("links") or []:
            try:
                data = fetch(ln["url"])
            except Exception as e:  # the capture page logs and skips a link it cannot load
                if strict:
                    raise
                s.warnings.append(f"robot link {ln.get('name')}: {e}")
                continue
            s.add_wel1(s.owner("link", str(ln["name"])), data, finish)
            n_links += 1
        if n_links == 0:
            sz = robot.get("ee_proxy_size") or [0.04, 0.08, 0.04]
            s.ee_proxy = s.owner("ee", "ee")
            s.add_box(s.ee_proxy, sz, 0x3DA5FF)
        for sf in scene_init.get("surfaces") or []:
            mats = mesh_mats.get(str(sf.get("mesh") or sf.get("id")), [])
            g = sf.get("uv_per_m") or [0.0, 0.0]
            s.surfaces.append(Surface(str(sf["id"]), list(mats), (float(g[0]), float(g[1]))))
            for mi in mats:  # attachSurface: RepeatWrapping on the scrolled maps
                for ref in s.materials[mi].maps.values():
                    ref.wrap_s = ref.wrap_t = "repeat"
        return s

    # ------------------------------------------------------------- arrays
    def draw_key(self, material: int) -> tuple:
        """Layout of the index buffer: casters first, alpha-tested casters after
        the others, single-sided before double-sided, then material. The
        triangles of one material are one range (its primitives in order), and
        the shadow pass draws the plain casters as one range per culling mode,
        the alpha-tested ones material by material. The camera pass does not
        draw in this order (see camera_order)."""
        m = self.materials[material]
        return (not m.cast_shadow, m.masked_shadow(), m.side == "double", material)

    def camera_order(self) -> list[int]:
        """Materials in the order the capture page draws them: three sorts its
        opaque list by material.id, the order the materials were made, and the
        page makes them object by object in scene_init's order (meshes and
        proxy boxes, then the robot links, then the end-effector box), as here.
        Within a GLB, GLTFLoader's order (base materials when their textures
        arrive, clones for derivative tangents, vertex colours or flat shading
        when a mesh is assembled) is taken as the order of first use. The order
        decides what shows where a material that writes no depth (glTF BLEND,
        drawn opaque by the page) overlaps another, and which coplanar surface
        wins; for the draws of one such material three's next key, the depth
        of each draw's centre, is applied per camera by the mesh pass."""
        return list(range(len(self.materials)))

    def primitive_ranges(self, material: int) -> list[tuple[int, int, int]]:
        """(first index, index count, primitive) of each primitive of a material,
        inside the material's range of the index buffer."""
        self.vertex_arrays()
        return self._ranges.get(material, [])

    def vertex_arrays(self) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
        """(vertices (V, 20) float32 words, indices (I,) uint32, draws) where a
        draw is (first index, index count, material), one per material, in the
        index buffer's order (draw_key). The owner of a vertex is its word 18
        (uint32), with the owner's scale baked into position, normal and tangent
        as three's matrixWorld applies it."""
        if self._vb is not None:
            return self._vb
        vs, tris = [], []
        base = 0
        block_base: dict[int, int] = {}
        for oi, p in self.primitives:
            n = len(p.positions)
            sc = np.asarray(self.owners[oi].scale, np.float64)
            if p.block is not None and p.block in block_base:  # vertices already in the buffer
                tri = p.indices.astype(np.uint32)
                if np.prod(np.sign(sc)) < 0:
                    tri = tri[:, [0, 2, 1]]
                tris.append(
                    (self.draw_key(p.material), len(tris), (tri + block_base[p.block]).reshape(-1))
                )
                continue
            if p.block is not None:
                block_base[p.block] = base
            rec = np.zeros((n, VERTEX_FLOATS), np.float32)
            rec[:, 0:3] = p.positions * sc
            if p.normals is not None:
                rec[:, 3:6] = p.normals / np.where(sc == 0, 1.0, sc)
            if p.uv0 is not None:
                rec[:, 6:8] = p.uv0
            if p.uv1 is not None:
                rec[:, 8:10] = p.uv1
            rec[:, 10:14] = 1.0
            if p.colors is not None:
                rec[:, 10 : 10 + p.colors.shape[1]] = p.colors
            if p.tangents is not None:
                rec[:, 14:17] = p.tangents[:, :3] * sc
                rec[:, 17] = p.tangents[:, 3]
            rec.view(np.uint32)[:, 18] = oi
            tri = p.indices.astype(np.uint32)
            if np.prod(np.sign(sc)) < 0:  # mirrored: three flips the front face
                tri = tri[:, [0, 2, 1]]
            vs.append(rec)
            tris.append((self.draw_key(p.material), len(tris), (tri + base).reshape(-1)))
            base += n
        if not vs:
            vs = [np.zeros((1, VERTEX_FLOATS), np.float32)]
        ids, draws, first = [], [], 0
        self._ranges = {}
        for key in sorted({k for k, _, _ in tris}):
            mi = key[-1]
            start = first
            for _, pi, t in (x for x in tris if x[0] == key):
                ids.append(t)
                self._ranges.setdefault(mi, []).append((first, int(t.size), pi))
                first += t.size
            draws.append((start, int(first - start), mi))
        idx = np.concatenate(ids).astype(np.uint32) if ids else np.zeros(0, np.uint32)
        self._vb = (np.concatenate(vs), idx, draws)
        return self._vb

    def bounds(self, matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """World AABB of every owner's geometry under (n_owners, 4, 4) matrices
        (Box3.expandByObject: each geometry's box, transformed)."""
        if not hasattr(self, "_owner_corners"):
            boxes: dict[int, list] = {}
            for oi, p in self.primitives:
                if len(p.positions):
                    sc = np.asarray(self.owners[oi].scale, np.float64)
                    q = p.positions.astype(np.float64) * sc
                    pl, ph = q.min(0), q.max(0)
                    b = boxes.setdefault(oi, [pl, ph])
                    b[0] = np.minimum(b[0], pl)
                    b[1] = np.maximum(b[1], ph)
            ids = sorted(boxes)
            corners = np.zeros((len(ids), 8, 3))
            for k, oi in enumerate(ids):
                a, b = boxes[oi]
                corners[k] = [
                    [x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])
                ]
            self._owner_corners = (np.asarray(ids, np.int64), corners)
        ids, corners = self._owner_corners
        if not len(ids):
            return None
        m = np.asarray(matrices)[ids]
        w = np.einsum("nij,nkj->nki", m[:, :3, :3], corners) + m[:, None, :3, 3]
        return w.reshape(-1, 3).min(0), w.reshape(-1, 3).max(0)


def owner_poses(scene: MeshScene, state: dict, poses: dict[int, np.ndarray]) -> None:
    """Update the persistent owner poses from a render_state, as the capture
    page does: a body or link the state names takes its pose, one it does not
    name keeps the last; the end-effector proxy follows robot.ee_pos/ee_quat."""
    idx = scene.owner_index
    for b in state.get("bodies") or []:
        oi = idx.get(("body", str(b["id"])))
        if oi is not None:
            poses[oi] = pose_matrix(b["pos"], b["quat"])
    robot = state.get("robot") or {}
    for name, p in (robot.get("links") or {}).items():
        oi = idx.get(("link", str(name)))
        if oi is not None:
            poses[oi] = pose_matrix(p["pos"], p["quat"])
    if scene.ee_proxy is not None and "ee_pos" in robot:
        poses[scene.ee_proxy] = pose_matrix(robot["ee_pos"], robot.get("ee_quat", (0, 0, 0, 1)))


def owner_matrices(scene: MeshScene, poses: dict[int, np.ndarray]) -> np.ndarray:
    """(n_owners, 4, 4) pose x local (scale is in the vertices)."""
    out = np.tile(np.eye(4), (max(1, len(scene.owners)), 1, 1))
    for oi, o in enumerate(scene.owners):
        p = poses.get(oi)
        if p is None:
            p = pose_matrix(o.pos, o.quat)
        out[oi] = p @ o.local
    return out


def owner_uniforms(matrices: np.ndarray) -> np.ndarray:
    """The Owner array of mesh.wgsl: model, then the inverse transpose of its
    3x3 (the matrices are rigid here, scale lives in the vertices, but a
    local offset may scale), both column-major."""
    m = np.asarray(matrices, np.float64)
    n = len(m)
    out = np.zeros((n, 2, 4, 4), np.float32)
    out[:, 0] = m.transpose(0, 2, 1)
    lin = m[:, :3, :3]
    ok = np.abs(np.linalg.det(lin)) > 1e-30
    inv = lin.copy()
    if ok.any():
        inv[ok] = np.linalg.inv(lin[ok])
    nm = np.tile(np.eye(4), (n, 1, 1))
    nm[:, :3, :3] = np.where(ok[:, None, None], inv.transpose(0, 2, 1), lin)
    out[:, 1] = nm.transpose(0, 2, 1)
    return out


def surface_offsets(scene: MeshScene, state: dict) -> dict[int, dict[str, tuple[float, float]]]:
    """Per scrolled material, per map: the texture offset of doc 14 §7.2
    (offset0 - L g travel, each component mod 1)."""
    travel = {str(s["id"]): float(s.get("travel", 0.0)) for s in state.get("surfaces") or []}
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for sf in scene.surfaces:
        if sf.id not in travel:
            continue
        t = travel[sf.id]
        g = np.array(sf.uv_per_m, np.float64) * t
        for mi in sf.materials:
            offs = out.setdefault(mi, {})
            for slot, ref in scene.materials[mi].maps.items():
                m = ref.matrix()
                off = np.array(ref.offset, np.float64) - m[:2, :2] @ g
                offs[slot] = (
                    float(off[0] - math.floor(off[0])),
                    float(off[1] - math.floor(off[1])),
                )
    return out
