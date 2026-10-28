"""A scene as the client renderer gets it, drawn with D's WGSL renderer.

``SceneRenderer(scene_init, read_asset, device=...)`` builds the splats and the
meshes of a ``scene_init`` (the message every page that draws a scene gets,
``world_engine/serving/messages.py``); ``render(state)`` poses them as a
``render_state`` says and returns ``{camera name: (H, W, 3) uint8}``, the
interface of world-engine-client's LocalRenderer. ``read_asset(url)`` returns an
asset's bytes: a client reads them decrypted from its render cache, in memory,
and nothing here writes them anywhere.

What it draws (pipeline wgsl/1):

* the background splat (``scene_init.splat``) under its placement;
* the object meshes (``scene_init.meshes``): each GLB at its authored pose
  (``pos``/``quat``, BUG-18), scaled by ``scale``, moved by the state's
  ``bodies``; an optional ``local`` offset ({pos, quat}) of the drawn mesh in its
  body; a grey proxy box for an entry without ``url`` (``box`` half sizes, else
  6 cm);
* the preset robot's links (``robot.links``, WEL1 buffers), posed by the state's
  ``robot.links``; without links, the end-effector proxy box at ``ee_pos``/``ee_quat``.

Meshes go through the placeholder flat pass (``mesh_flat.wgsl``): one colour per
material, the glTF base colour (times the mean of its texture, when Pillow can
decode it) or the link's material colour, lit by one fixed light. M-D2's PBR
pass replaces it. Not drawn by wgsl/1: the scene lighting (``hdri``,
``probe_center``), textures as such, surface scrolling (``surfaces``).

Standalone like the rest of this directory (numpy and wgpu; Pillow optional).
"""

from __future__ import annotations

import base64
import io
import json
import struct
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .render_wgsl import MeshSet, SplatRenderer
from .splat_data import Splats, quat_xyzw_to_matrix, read_splat_ply

PROXY_RGBA = (0x9A / 255, 0xA0 / 255, 0xA6 / 255, 1.0)  # capture page: proxyBox(0x9aa0a6)
EE_PROXY_RGBA = (0x3D / 255, 0xA5 / 255, 0xFF / 255, 1.0)  # capture page: proxyBox(0x3da5ff)
LINK_RGBA = (0.75, 0.77, 0.8, 1.0)
GLB_DEFAULT_RGBA = (1.0, 1.0, 1.0, 1.0)  # glTF: baseColorFactor defaults to white
PROXY_SIZE = (0.06, 0.06, 0.06)
EE_PROXY_SIZE = (0.04, 0.08, 0.04)

_COMPONENTS = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass
class Part:
    """One piece of mesh in its owner's frame: vertices (n, 3), faces (m, 3),
    vertex normals (n, 3), display colour rgba 0-1 ((4,) or one per vertex)."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    rgba: np.ndarray


# ------------------------------------------------------------------- colours
def srgb_to_linear(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


# --------------------------------------------------------------- geometry
def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v, dtype=np.float64)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def box_part(size, rgba) -> Part:
    """An axis-aligned box of full size ``size`` centred on the origin, flat
    normals (24 vertices), faces counter-clockwise from outside."""
    hx, hy, hz = (0.5 * float(s) for s in size)
    v, f, n = [], [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            u = np.zeros(3)
            u[(axis + 1) % 3] = 1.0
            w = np.cross(normal, u)
            corners = [normal + a * u + b * w for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            base = len(v)
            v += [c * (hx, hy, hz) for c in corners]
            n += [normal] * 4
            f += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return Part(
        np.asarray(v, np.float32),
        np.asarray(f, np.uint32),
        np.asarray(n, np.float32),
        np.asarray(rgba),
    )


def pose_matrix(pos=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), scale=None) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    if scale is not None:
        m[:3, :3] = m[:3, :3] @ np.diag(np.broadcast_to(np.asarray(scale, float), (3,)))
    m[:3, 3] = np.asarray(pos, float)
    return m


def transform_part(p: Part, m: np.ndarray) -> Part:
    """``p`` with ``m`` (4x4, may scale) applied to its vertices and normals."""
    v = p.vertices.astype(np.float64) @ m[:3, :3].T + m[:3, 3]
    nm = np.linalg.inv(m[:3, :3]).T
    n = p.normals.astype(np.float64) @ nm.T
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    faces = p.faces
    if np.linalg.det(m[:3, :3]) < 0:  # a mirror flips the winding
        faces = faces[:, ::-1].copy()
    return Part(v.astype(np.float32), faces, n.astype(np.float32), p.rgba)


# ------------------------------------------------------------------- WEL1
def read_wel1(buf) -> list[Part]:
    """A preset robot link (messages.encode_robot_link): one Part per material
    group, else one with the vertex colours (or the default link colour)."""
    b = bytes(buf[:16])
    if b[:4] != b"WEL1":
        raise ValueError("not a WEL1 robot link buffer")
    nv, nf, has_c, has_n, has_g = struct.unpack_from("<IIBBBx", b, 4)
    mv = memoryview(buf)
    off = 16
    v = np.frombuffer(mv, "<f4", nv * 3, off).reshape(nv, 3).astype(np.float32)
    off += nv * 12
    f = np.frombuffer(mv, "<u4", nf * 3, off).reshape(nf, 3).astype(np.uint32)
    off += nf * 12
    col = None
    if has_c:
        col = np.frombuffer(mv, "u1", nv * 3, off).reshape(nv, 3)
        off += nv * 3
    if has_n:
        nrm = np.frombuffer(mv, "i1", nv * 3, off).reshape(nv, 3).astype(np.float32) / 127.0
        off += nv * 3
    else:
        nrm = vertex_normals(v, f)
    groups = []
    if has_g:
        (ng,) = struct.unpack_from("<I", mv, off)
        off += 4
        for _ in range(ng):
            s, c, r, g, bl, a = struct.unpack_from("<IIBBBB", mv, off)
            off += 12
            groups.append((s, c, (r / 255, g / 255, bl / 255, a / 255)))
    if groups:
        return [Part(v, f[s : s + c], nrm, np.asarray(rgba)) for s, c, rgba in groups if c]
    if col is not None:
        rgba = np.concatenate([col / 255.0, np.ones((nv, 1))], 1)
        return [Part(v, f, nrm, rgba)]
    return [Part(v, f, nrm, np.asarray(LINK_RGBA))]


# -------------------------------------------------------------------- GLB
class _Gltf:
    def __init__(self, data, image_mean: Callable[[bytes], np.ndarray | None] | None):
        b = memoryview(data)
        magic, version, length = struct.unpack_from("<4sII", b, 0)
        if magic != b"glTF":
            raise ValueError("not a binary glTF (GLB)")
        if version != 2:
            raise ValueError(f"glTF version {version}; glTF 2.0 is read")
        off = 12
        self.json, self.bin = None, None
        while off + 8 <= min(length, len(b)):
            clen, ctype = struct.unpack_from("<I4s", b, off)
            chunk = b[off + 8 : off + 8 + clen]
            if ctype == b"JSON":
                self.json = json.loads(bytes(chunk).decode("utf-8"))
            elif ctype == b"BIN\x00" and self.bin is None:
                self.bin = chunk
            off += 8 + clen + (-clen % 4)
        if self.json is None:
            raise ValueError("GLB without a JSON chunk")
        self._image_mean = image_mean
        self._image_cache: dict[int, np.ndarray | None] = {}

    def view_bytes(self, index: int) -> memoryview:
        bv = self.json["bufferViews"][index]
        if bv.get("buffer", 0) != 0 or self.bin is None:
            raise ValueError("only the GLB's own binary chunk is read")
        start = bv.get("byteOffset", 0)
        return self.bin[start : start + bv["byteLength"]]

    def accessor(self, index: int) -> np.ndarray:
        acc = self.json["accessors"][index]
        if "sparse" in acc:
            raise ValueError("sparse accessors are not read")
        dt = np.dtype(_COMPONENTS[acc["componentType"]]).newbyteorder("<")
        width = _WIDTH[acc["type"]]
        count = acc["count"]
        if "bufferView" not in acc:
            return np.zeros((count, width), dt)
        bv = self.json["bufferViews"][acc["bufferView"]]
        raw = self.view_bytes(acc["bufferView"])
        offset = acc.get("byteOffset", 0)
        item = dt.itemsize * width
        stride = bv.get("byteStride") or item
        need = offset + stride * (count - 1) + item if count else 0
        if need > len(raw):
            raise ValueError("an accessor runs past its buffer view")
        flat = np.frombuffer(raw, np.uint8)
        rows = np.lib.stride_tricks.as_strided(
            flat[offset:], shape=(count, item), strides=(stride, 1), writeable=False
        )
        out = np.ascontiguousarray(rows).view(dt).reshape(count, width)
        if acc.get("normalized") and dt.kind in "iu":
            out = np.maximum(out / float(np.iinfo(dt).max), -1.0)
        return out

    def material_rgba(self, index) -> np.ndarray:
        """Display colour of a material: base colour (linear) times the mean of
        its base colour texture, to sRGB."""
        if index is None:
            return np.asarray(GLB_DEFAULT_RGBA)
        m = self.json.get("materials", [])[index]
        pbr = m.get("pbrMetallicRoughness", {})
        lin = np.asarray(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), np.float64)
        tex = pbr.get("baseColorTexture")
        if tex is not None:
            mean = self.texture_mean(tex.get("index"))
            if mean is not None:
                lin = lin * mean
        return np.concatenate([linear_to_srgb(lin[:3]), [float(np.clip(lin[3], 0, 1))]])

    def texture_mean(self, tex_index) -> np.ndarray | None:
        """Mean linear rgba of a texture's image, or None (not decodable here)."""
        if tex_index is None or self._image_mean is None:
            return None
        try:
            src = self.json["textures"][tex_index].get("source")
        except (KeyError, IndexError):
            return None
        if src is None:
            return None
        if src not in self._image_cache:
            mean = None
            try:
                img = self.json["images"][src]
                if "bufferView" in img:
                    data = bytes(self.view_bytes(img["bufferView"]))
                elif str(img.get("uri", "")).startswith("data:"):
                    data = base64.b64decode(img["uri"].split(",", 1)[1])
                else:
                    data = None
                if data is not None:
                    mean = self._image_mean(data)
            except Exception:  # noqa: BLE001 - an image we cannot decode keeps the factor
                mean = None
            self._image_cache[src] = mean
        return self._image_cache[src]

    def parts(self) -> list[Part]:
        g = self.json
        nodes = g.get("nodes", [])
        if "scenes" in g and g["scenes"]:
            roots = g["scenes"][g.get("scene", 0)].get("nodes", [])
        else:
            children = {c for n in nodes for c in n.get("children", [])}
            roots = [i for i in range(len(nodes)) if i not in children]
        out: list[Part] = []
        stack = [(i, np.eye(4)) for i in roots]
        seen = 0
        while stack:
            i, parent = stack.pop()
            seen += 1
            if seen > 100_000:
                raise ValueError("the GLB's node graph does not end")
            n = nodes[i]
            if "matrix" in n:
                local = np.asarray(n["matrix"], np.float64).reshape(4, 4).T
            else:
                local = pose_matrix(
                    n.get("translation", (0, 0, 0)),
                    n.get("rotation", (0, 0, 0, 1)),
                    n.get("scale", (1, 1, 1)),
                )
            world = parent @ local
            if "mesh" in n:
                for p in self.mesh_parts(n["mesh"]):
                    out.append(transform_part(p, world))
            stack += [(c, world) for c in n.get("children", [])]
        return out

    def mesh_parts(self, index: int) -> list[Part]:
        out = []
        for prim in self.json["meshes"][index].get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue  # points and lines are not drawn
            if "KHR_draco_mesh_compression" in prim.get("extensions", {}):
                raise ValueError("Draco-compressed meshes are not read")
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            v = self.accessor(attrs["POSITION"]).astype(np.float32)
            if "indices" in prim:
                f = self.accessor(prim["indices"]).reshape(-1).astype(np.uint32)
            else:
                f = np.arange(len(v), dtype=np.uint32)
            f = f[: len(f) // 3 * 3].reshape(-1, 3)
            if len(f) == 0:
                continue
            if f.max() >= len(v):
                raise ValueError("an index points past the vertices")
            nrm = (
                self.accessor(attrs["NORMAL"]).astype(np.float32)
                if "NORMAL" in attrs
                else vertex_normals(v, f)
            )
            rgba = self.material_rgba(prim.get("material"))
            if "COLOR_0" in attrs:
                col = self.accessor(attrs["COLOR_0"]).astype(np.float64)
                if col.shape[1] == 3:
                    col = np.concatenate([col, np.ones((len(col), 1))], 1)
                lin = srgb_to_linear(rgba[:3]) * col[:, :3]  # COLOR_0 is linear
                rgba = np.concatenate([linear_to_srgb(lin), rgba[3] * col[:, 3:4]], 1)
            out.append(Part(v, f, nrm, rgba))
        return out


def _pillow_mean(data: bytes) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(io.BytesIO(data)) as im:
        im.draft("RGB", (64, 64))
        rgba = np.asarray(im.convert("RGBA").resize((64, 64)), np.float64) / 255.0
    lin = srgb_to_linear(rgba[..., :3]).reshape(-1, 3)
    return np.concatenate([lin.mean(0), [rgba[..., 3].mean()]])


def read_glb(data, *, decode_images: bool = True) -> list[Part]:
    """The triangles of a GLB in its scene frame, one Part per primitive."""
    return _Gltf(data, _pillow_mean if decode_images else None).parts()


# ----------------------------------------------------------- the scene
def _empty_splats() -> Splats:
    """One invisible splat: the renderer needs at least one."""
    return Splats(
        np.zeros((1, 3), np.float32),
        np.full((1, 3), 1e-3, np.float32),
        np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        np.zeros(1, np.float32),
        np.zeros((1, 1, 3), np.float32),
    )


def scene_splats(scene_init: dict, read_asset: Callable[[str], object]) -> Splats:
    sp = scene_init.get("splat") or {}
    if not sp.get("url"):
        return _empty_splats()
    data = read_asset(sp["url"])
    splats = read_splat_ply(data)
    del data
    return splats.placed(
        sp.get("pos", (0, 0, 0)), sp.get("quat", (0, 0, 0, 1)), sp.get("scale", 1.0)
    )


@dataclass
class SceneMeshes:
    parts: list[tuple[Part, int]]  # (part, owner index)
    owners: list[tuple[str, str]]  # ("body", id) | ("link", name) | ("ee", "proxy")
    rest: np.ndarray  # (n_owners, 4, 4): the pose of each owner before any state
    skipped: list[str]  # what could not be read, and why


def scene_meshes(
    scene_init: dict, read_asset: Callable[[str], object], *, decode_images: bool = True
) -> SceneMeshes:
    parts: list[tuple[Part, int]] = []
    owners: list[tuple[str, str]] = []
    rest: list[np.ndarray] = []
    skipped: list[str] = []

    def owner(key, pose) -> int:
        owners.append(key)
        rest.append(pose)
        return len(owners) - 1

    for m in scene_init.get("meshes") or []:
        mid = str(m.get("id"))
        scale = m.get("scale") or (1.0, 1.0, 1.0)
        local = m.get("local") or {}
        bake = pose_matrix(local.get("pos", (0, 0, 0)), local.get("quat", (0, 0, 0, 1)))
        if m.get("url"):
            try:
                got = read_glb(read_asset(m["url"]), decode_images=decode_images)
            except (ValueError, KeyError, IndexError, TypeError, struct.error) as exc:
                skipped.append(f"{mid}: {exc}")
                continue
            got = [transform_part(p, bake @ pose_matrix(scale=scale)) for p in got]
        else:
            size = [2.0 * float(x) for x in m["box"]] if m.get("box") else PROXY_SIZE
            got = [transform_part(box_part(size, PROXY_RGBA), bake)]
        i = owner(("body", mid), pose_matrix(m.get("pos", (0, 0, 0)), m.get("quat", (0, 0, 0, 1))))
        parts += [(p, i) for p in got]
    robot = scene_init.get("robot") or {}
    links = robot.get("links") or []
    for ln in links:
        name = str(ln.get("name"))
        try:
            got = read_wel1(read_asset(ln["url"]))
        except (ValueError, KeyError, struct.error) as exc:
            skipped.append(f"link {name}: {exc}")
            continue
        i = owner(("link", name), np.eye(4))
        parts += [(p, i) for p in got]
    if not links:
        size = robot.get("ee_proxy_size") or EE_PROXY_SIZE
        i = owner(("ee", "proxy"), np.eye(4))
        parts.append((box_part(size, EE_PROXY_RGBA), i))
    return SceneMeshes(parts, owners, np.asarray(rest).reshape(-1, 4, 4), skipped)


class OwnerPoses:
    """The pose of every mesh owner, kept across states: a state's ``bodies``
    move the objects (static ones keep their authored pose, which no state
    names), ``robot.links`` the links, ``robot.ee_pos``/``ee_quat`` the
    end-effector proxy. An owner a state does not mention keeps its last pose."""

    def __init__(self, owners: list[tuple[str, str]], rest: np.ndarray):
        self.index = {key: i for i, key in enumerate(owners)}
        self.poses = np.asarray(rest, np.float64).reshape(-1, 4, 4).copy()

    def update(self, state: dict) -> np.ndarray:
        idx = self.index
        for b in state.get("bodies") or []:
            i = idx.get(("body", str(b.get("id"))))
            if i is not None:
                self.poses[i] = pose_matrix(b["pos"], b["quat"])
        robot = state.get("robot") or {}
        for name, p in (robot.get("links") or {}).items():
            i = idx.get(("link", str(name)))
            if i is not None:
                self.poses[i] = pose_matrix(p["pos"], p["quat"])
        i = idx.get(("ee", "proxy"))
        if i is not None and robot.get("ee_pos") is not None:
            self.poses[i] = pose_matrix(robot["ee_pos"], robot.get("ee_quat", (0, 0, 0, 1)))
        return self.poses


class SceneRenderer:
    """A scene_init's splats and meshes on one device; ``render(state)`` gives
    ``{camera name: (H, W, 3) uint8}`` for the state's cameras.

    read_asset(url): the asset's bytes. device: a wgpu device (render_wgsl.
    create_device). The other options go to render_wgsl.SplatRenderer
    (color_format, memory_limit, batch, params, wgsl_dir, upload_chunk)."""

    def __init__(self, scene_init: dict, read_asset: Callable[[str], object], *, device, **options):
        decode_images = options.pop("decode_images", True)
        self.scene_init = scene_init
        cameras = list(scene_init.get("cameras") or [])
        if not cameras:
            raise ValueError("the scene has no cameras")
        meshes = scene_meshes(scene_init, read_asset, decode_images=decode_images)
        self.skipped = meshes.skipped
        if meshes.skipped:
            warnings.warn(
                "some meshes are not drawn: " + "; ".join(meshes.skipped[:5]),
                RuntimeWarning,
                stacklevel=2,
            )
        splats = scene_splats(scene_init, read_asset)
        mesh_set = None
        if meshes.parts:
            mesh_set = MeshSet.build(
                [(p.vertices, p.faces, p.normals, p.rgba, i) for p, i in meshes.parts],
                meshes.owners,
            )
        self.poses = OwnerPoses(meshes.owners, meshes.rest)
        self.n_mesh_parts = len(meshes.parts)
        self.renderer = SplatRenderer(splats, cameras, meshes=mesh_set, device=device, **options)
        del splats
        self.color_format = self.renderer.color_format

    def render(self, state: dict) -> dict[str, np.ndarray]:
        cams = list(state.get("cameras") or self.scene_init["cameras"])
        transforms = self.poses.update(state) if self.renderer.meshes is not None else None
        images = self.renderer.render_cameras(cams, transforms)
        return {c["name"]: im for c, im in zip(cams, images, strict=True)}

    @property
    def last_timing(self) -> dict:
        return self.renderer.last_timing

    def close(self) -> None:
        self.renderer.close()
