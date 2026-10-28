"""GPU side of D's mesh pass, shadow map and environment prefilter (wgpu-py).

Used by ``render_wgsl.SplatRenderer``; standalone like the rest of this package
(numpy, wgpu, and the sibling modules). The shaders are ``mesh.wgsl``,
``shadow.wgsl`` and ``pmrem.wgsl`` of the pipeline directory; the host contract
is in its README.

* ``Prefilter`` is three's PMREMGenerator on the GPU: a source (the bake's cube
  map, an equirectangular HDRI, or three's RoomEnvironment) into the CubeUV
  atlas the mesh pass samples.
* ``MeshPass`` holds the meshes' buffers, materials, textures and the shadow
  map; per frame it writes the owners' matrices, the frame constants and the
  scrolled materials, encodes the shadow pass, and draws into each camera's
  render pass before the splats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lighting import Lighting, atlas_size
from .mesh_data import (
    MAP_SLOTS,
    MATERIAL_SLOT,
    OWNER_BYTES,
    VERTEX_BYTES,
    MeshScene,
    owner_matrices,
    owner_poses,
    owner_uniforms,
    quat_xyzw_to_matrix,
    surface_offsets,
)

LOD_MIN = 4
EXTRA_LOD_SIGMA = (0.125, 0.215, 0.35, 0.446, 0.526, 0.582)
MAX_SAMPLES = 20
PHI = (1 + math.sqrt(5)) / 2
INV_PHI = 1 / PHI
AXIS_DIRECTIONS = [
    (-PHI, INV_PHI, 0),
    (PHI, INV_PHI, 0),
    (-INV_PHI, 0, PHI),
    (INV_PHI, 0, PHI),
    (0, PHI, -INV_PHI),
    (0, PHI, INV_PHI),
    (-1, 1, -1),
    (1, 1, -1),
    (-1, 1, 1),
    (1, 1, 1),
]
SLOT = 256
FRAME_BYTES = 256
ATLAS_FORMAT = "rgba16float"


def three_look_at(eye, target, up) -> np.ndarray:
    """Matrix4.lookAt (with three's nudge when up and the view are parallel):
    the 3x3 of the camera's world rotation (columns x, y, z)."""
    z = np.asarray(eye, np.float64) - np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    if float(z @ z) == 0.0:
        z = np.array([0.0, 0.0, 1.0])
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if float(x @ x) == 0.0:
        if abs(up[2]) == 1:
            z[0] += 0.0001
        else:
            z[2] += 0.0001
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


@dataclass(frozen=True)
class ShadowParams:
    """The editor's quality-tier key-light shadow (editorScene.ts): PCF over a
    1024^2 map, radius 6 texels, bias -0.0005, normalBias 0.02, the orthographic
    camera fitted to the meshes' bounds each frame."""

    size: int = 1024
    radius: float = 6.0
    bias: float = -0.0005
    normal_bias: float = 0.02


def shadow_matrices(lo, hi, light_dir) -> tuple[np.ndarray, np.ndarray]:
    """(view_proj, shadow matrix) of the key light's shadow camera, fitted as
    editorScene._fitShadowCamera: centre of the bounds, radius = max(half the
    diagonal, 0.5), the light at centre + dir (2 radius + 1), ortho half size
    1.15 radius, near 0.05, far 4 radius + 2; up +y (three's default)."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    center = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) * 0.5, 0.5)
    d = np.asarray(light_dir, np.float64)
    d = d / np.linalg.norm(d)
    eye = center + d * (radius * 2 + 1)
    rot = three_look_at(eye, center, (0.0, 1.0, 0.0))
    view = np.eye(4)
    view[:3, :3] = rot.T
    view[:3, 3] = -rot.T @ eye
    r = radius * 1.15
    near, far = 0.05, radius * 4 + 2
    left, right, top, bottom = -r, r, r, -r
    w, h, p = 1.0 / (right - left), 1.0 / (top - bottom), 1.0 / (far - near)
    proj = np.array(
        [
            [2 * w, 0, 0, -(right + left) * w],
            [0, 2 * h, 0, -(top + bottom) * h],
            [0, 0, -2 * p, -(far + near) * p],
            [0, 0, 0, 1],
        ]
    )
    vp = proj @ view
    bias = np.array([[0.5, 0, 0, 0.5], [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5], [0, 0, 0, 1.0]])
    return vp, bias @ vp


def _blur_constants(sigma: float, size_in: int) -> tuple[float, int, list[float]]:
    """_halfBlur: (dTheta, samples, 20 normalised weights)."""
    pixels = size_in - 1
    finite = math.isfinite(sigma)
    rpp = math.pi / (2 * pixels) if finite else 2 * math.pi / (2 * MAX_SAMPLES - 1)
    sigma_px = sigma / rpp
    samples = 1 + math.floor(3 * sigma_px) if finite else MAX_SAMPLES
    weights = []
    total = 0.0
    for i in range(MAX_SAMPLES):
        x = i / sigma_px
        wgt = math.exp(-x * x / 2)
        weights.append(wgt)
        if i == 0:
            total += wgt
        elif i < samples:
            total += 2 * wgt
    return rpp, min(samples, MAX_SAMPLES), [w / total for w in weights]


def _round(n: int) -> int:
    return max(16, (int(n) + 3) // 4 * 4)


def env_texture_bytes(lighting: Lighting | None) -> int:
    """Bytes of the environment atlas MeshPass installs for this lighting."""
    if lighting is None:
        return 8
    if isinstance(lighting.source, tuple):
        _, w, h, _ = lighting.source
        return 8 * w * h
    if lighting.atlas is None:
        w, h, _ = atlas_size(16)
        return 8 * w * h
    h, w = lighting.atlas.shape[:2]
    return 8 * w * h


def mesh_budget(
    scene: MeshScene, lighting: Lighting | None, shadows
) -> tuple[list[int], dict[str, int]]:
    """What MeshPass allocates, in its creation order (buffers, then textures),
    and the bytes by kind for budget.gpu_budget."""
    verts, idx, _ = scene.vertex_arrays()
    n_own = max(1, len(scene.owners))
    n_mat = max(1, len(scene.materials))
    bufs = [
        _round(verts.nbytes),
        _round(idx.nbytes if idx.size else 16),
        _round(OWNER_BYTES * n_own),
        _round(FRAME_BYTES),
        _round(MATERIAL_SLOT * n_mat),
        _round(64),
    ]
    maps = [4] + [img.gpu_bytes() for img in scene.images]
    env = env_texture_bytes(lighting)
    size = shadows.size if shadows else 1
    textures = maps + [8, 24, env, 4 * size * size]
    parts = {
        "mesh vertices and indices": bufs[0] + bufs[1],
        "mesh owners, materials, frame": sum(bufs[2:]),
        "mesh textures": sum(maps),
        "environment (CubeUV)": env + 32,
        "shadow map": 4 * size * size,
    }
    return bufs + textures, parts


class Prefilter:
    """three's PMREMGenerator on the GPU (pmrem.wgsl)."""

    def __init__(self, device, module):
        import wgpu

        self.wgpu = wgpu
        self.device = device
        S = wgpu.ShaderStage
        self.bgl = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": S.VERTEX | S.FRAGMENT,
                    "buffer": {"type": "uniform", "has_dynamic_offset": True},
                },
                {
                    "binding": 1,
                    "visibility": S.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 2,
                    "visibility": S.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "cube"},
                },
                {
                    "binding": 3,
                    "visibility": S.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {"binding": 4, "visibility": S.FRAGMENT, "sampler": {"type": "filtering"}},
            ]
        )
        pl = device.create_pipeline_layout(bind_group_layouts=[self.bgl])
        self.pipes = {
            name: device.create_render_pipeline(
                layout=pl,
                vertex={"module": module, "entry_point": "vs_plane"},
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                fragment={
                    "module": module,
                    "entry_point": name,
                    "targets": [{"format": ATLAS_FORMAT}],
                },
            )
            for name in ("fs_cube", "fs_equirect", "fs_blur", "fs_room")
        }
        self.sampler = device.create_sampler(
            mag_filter="linear",
            min_filter="linear",
            mipmap_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge",
        )
        T = wgpu.TextureUsage
        self.dummy2d = device.create_texture(
            size=(1, 1, 1), format=ATLAS_FORMAT, usage=T.TEXTURE_BINDING
        )
        self.dummycube = device.create_texture(
            size=(1, 1, 6), format="rgba8unorm", usage=T.TEXTURE_BINDING
        )
        self.dummycube_view = self.dummycube.create_view(dimension="cube")
        self._resources = [self.dummy2d, self.dummycube]

    def _targets(self, cube_size: int):
        w, h, lod_max = atlas_size(cube_size)
        T = self.wgpu.TextureUsage
        use = T.RENDER_ATTACHMENT | T.TEXTURE_BINDING | T.COPY_SRC
        atlas = self.device.create_texture(size=(w, h, 1), format=ATLAS_FORMAT, usage=use)
        ping = self.device.create_texture(size=(w, h, 1), format=ATLAS_FORMAT, usage=use)
        return atlas, ping, w, h, lod_max

    def _plan(self, lod_max: int):
        """_createPlanes: (size, sigma) per LOD."""
        lods = []
        lod = lod_max
        total = lod_max - LOD_MIN + 1 + len(EXTRA_LOD_SIGMA)
        for i in range(total):
            size = 2**lod
            sigma = 1.0 / size
            if i > lod_max - LOD_MIN:
                sigma = EXTRA_LOD_SIGMA[i - lod_max + LOD_MIN - 1]
            elif i == 0:
                sigma = 0.0
            lods.append((size, sigma))
            if lod > LOD_MIN:
                lod -= 1
        return lods

    def run(self, kind: str, source=None, cube_size: int | None = None, pre_blur: float = 0.0):
        """Prefilter; returns (atlas texture, width, height, max_mip). kind: cube
        (source: a cube texture view, cube_size its face size), equirect (source:
        a 2D texture view of width 4 x cube size), room (three's RoomEnvironment,
        256^2, pre_blur 0.04 as fromScene)."""
        wgpu = self.wgpu
        dev = self.device
        atlas, ping, w, h, lod_max = self._targets(cube_size)
        lods = self._plan(lod_max)
        cs = 2**lod_max
        geo = (1.0 / w, 1.0 / h, float(lod_max), 0.0)
        passes = []  # (pipeline, target, src_atlas, src_cube, src_eq, viewport, constants)

        def consts(
            plane_size,
            blur=(0.0, 0.0, 0.0, 0.0),
            pole=(0.0, 1.0, 0.0),
            weights=None,
            room=(0.0, 0.0),
        ):
            u = np.zeros(SLOT // 4, np.float32)
            u[0:4] = geo
            u[4] = 1.0 / (plane_size - 2)
            u[5] = plane_size
            u[8:12] = blur
            u[12:15] = pole
            if weights is not None:
                u[16:36] = weights
            u[36:38] = room
            return u

        top = (0, 0, 3 * cs, 2 * cs)
        if kind == "cube":
            passes.append(("fs_cube", atlas, None, source, None, top, consts(lods[0][0])))
        elif kind == "equirect":
            passes.append(("fs_equirect", atlas, None, None, source, top, consts(lods[0][0])))
        elif kind == "room":
            passes.append(
                ("fs_room", atlas, None, None, None, top, consts(lods[0][0], room=(cs, 0.1)))
            )
        else:
            raise ValueError(kind)
        pole_default = (0.0, 1.0, 0.0)

        def half_blur(t_in, t_out, lod_in, lod_out, sigma, latitudinal, pole):
            rpp, samples, wts = _blur_constants(sigma, lods[lod_in][0])
            out_size = lods[lod_out][0]
            x = 3 * out_size * (lod_out - lod_max + LOD_MIN if lod_out > lod_max - LOD_MIN else 0)
            y = 4 * (cs - out_size)
            passes.append(
                (
                    "fs_blur",
                    t_out,
                    t_in,
                    None,
                    None,
                    (x, y, 3 * out_size, 2 * out_size),
                    consts(
                        out_size,
                        blur=(
                            rpp,
                            float(samples),
                            1.0 if latitudinal else 0.0,
                            float(lod_max - lod_in),
                        ),
                        pole=pole,
                        weights=wts,
                    ),
                )
            )

        def blur(lod_in, lod_out, sigma, pole):
            half_blur(atlas, ping, lod_in, lod_out, sigma, True, pole)
            half_blur(ping, atlas, lod_out, lod_out, sigma, False, pole)

        if pre_blur > 0:
            blur(0, 0, pre_blur, pole_default)
        n = len(lods)
        for i in range(1, n):
            sigma = math.sqrt(lods[i][1] ** 2 - lods[i - 1][1] ** 2)
            pole = AXIS_DIRECTIONS[(n - i - 1) % len(AXIS_DIRECTIONS)]
            blur(i - 1, i, sigma, pole)

        ubuf = dev.create_buffer(
            size=SLOT * len(passes), usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        dev.queue.write_buffer(ubuf, 0, np.concatenate([p[6] for p in passes]))
        views = {}

        def view(t):
            if id(t) not in views:
                views[id(t)] = t.create_view()
            return views[id(t)]

        dummy2d = self.dummy2d.create_view()
        enc = dev.create_command_encoder()
        for k, (pipe, target, s_atlas, s_cube, s_eq, vp, _) in enumerate(passes):
            bg = dev.create_bind_group(
                layout=self.bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": SLOT}},
                    {"binding": 1, "resource": view(s_atlas) if s_atlas is not None else dummy2d},
                    {
                        "binding": 2,
                        "resource": s_cube if s_cube is not None else self.dummycube_view,
                    },
                    {"binding": 3, "resource": s_eq if s_eq is not None else dummy2d},
                    {"binding": 4, "resource": self.sampler},
                ],
            )
            rp = enc.begin_render_pass(
                color_attachments=[{"view": view(target), "load_op": "load", "store_op": "store"}]
            )
            rp.set_pipeline(self.pipes[pipe])
            rp.set_bind_group(0, bg, [k * SLOT])
            rp.set_viewport(vp[0], vp[1], vp[2], vp[3], 0.0, 1.0)
            rp.set_scissor_rect(vp[0], vp[1], vp[2], vp[3])
            rp.draw(36)
            rp.end()
        dev.queue.submit([enc.finish()])
        ping.destroy()
        ubuf.destroy()
        return atlas, w, h, lod_max

    def read(self, atlas, w: int, h: int) -> np.ndarray:
        """(h, w, 4) float32 copy of an atlas (tests and the bench)."""
        wgpu = self.wgpu
        bpr = (w * 8 + 255) // 256 * 256
        buf = self.device.create_buffer(
            size=bpr * h, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        enc = self.device.create_command_encoder()
        enc.copy_texture_to_buffer(
            {"texture": atlas},
            {"buffer": buf, "bytes_per_row": bpr, "rows_per_image": h},
            (w, h, 1),
        )
        self.device.queue.submit([enc.finish()])
        buf.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(buf.read_mapped(copy=True), np.uint8).reshape(h, bpr)[:, : w * 8]
        buf.unmap()
        buf.destroy()
        return raw.copy().view(np.float16).reshape(h, w, 4).astype(np.float32)

    def destroy(self) -> None:
        for r in self._resources:
            r.destroy()
        self._resources = []


def cube_mips(faces: np.ndarray) -> list[np.ndarray]:
    """The bake cube's mip chain (generateMipmap on an 8-bit linear cube):
    2x2 averages, rounded, down to 1x1. faces: (6, s, s, 4) uint8."""
    levels = [faces]
    cur = faces.astype(np.float64)
    while cur.shape[1] > 1:
        cur = 0.25 * (
            cur[:, 0::2, 0::2] + cur[:, 1::2, 0::2] + cur[:, 0::2, 1::2] + cur[:, 1::2, 1::2]
        )
        levels.append(np.clip(np.round(cur), 0, 255).astype(np.uint8))
    return levels


def upload_texture(device, levels: list[np.ndarray], fmt: str, layers: int = 1, cube: bool = False):
    """A texture (2D or a 6-layer cube) with these mip levels, rows as given."""
    import wgpu

    T = wgpu.TextureUsage
    first = levels[0]
    if layers > 1:
        h, w = first.shape[1:3]
    else:
        h, w = first.shape[:2]
    bpp = first.dtype.itemsize * first.shape[-1]
    tex = device.create_texture(
        size=(w, h, layers),
        format=fmt,
        usage=T.TEXTURE_BINDING | T.COPY_DST,
        mip_level_count=len(levels),
    )
    for lvl, data in enumerate(levels):
        lw = max(1, w >> lvl)
        lh = max(1, h >> lvl)
        arr = np.ascontiguousarray(data)
        device.queue.write_texture(
            {"texture": tex, "mip_level": lvl, "origin": (0, 0, 0)},
            arr,
            {"offset": 0, "bytes_per_row": lw * bpp, "rows_per_image": lh},
            (lw, lh, layers),
        )
    view = tex.create_view(dimension="cube") if cube else tex.create_view()
    return tex, view


_WRAP = {"repeat": "repeat", "clamp": "clamp-to-edge", "mirror": "mirror-repeat"}
_MIN = {
    "nearest": ("nearest", "nearest"),
    "linear": ("linear", "nearest"),
    "nearest_mipmap_nearest": ("nearest", "nearest"),
    "linear_mipmap_nearest": ("linear", "nearest"),
    "nearest_mipmap_linear": ("nearest", "linear"),
    "linear_mipmap_linear": ("linear", "linear"),
}


class MeshPass:
    """The meshes of one renderer: buffers, materials, textures, shadow map."""

    def __init__(
        self,
        device,
        sources: dict[str, str],
        scene: MeshScene,
        lighting: Lighting | None,
        shadows: ShadowParams | None,
        *,
        cams_buf,
        make_buffer,
    ):
        """make_buffer(size, usage, data=None): the renderer's buffer factory
        (it keeps the buffer, uploads large data through its staging buffer)."""
        import wgpu

        self.wgpu = wgpu
        self.device = device
        self.scene = scene
        self.shadows = shadows
        self.lighting = lighting
        self._make_buffer = make_buffer
        self._textures: list = []
        self.cams_buf = cams_buf
        U = wgpu.BufferUsage
        S = wgpu.ShaderStage
        T = wgpu.TextureUsage
        dev = device
        self.mesh_module = dev.create_shader_module(code=sources["mesh"], label="mesh")
        self.shadow_module = dev.create_shader_module(code=sources["shadow"], label="shadow")
        self.pmrem_module = dev.create_shader_module(code=sources["pmrem"], label="pmrem")

        verts, idx, draws = scene.vertex_arrays()
        self.draws = draws
        # the camera pass: materials in three's opaque order; a material that
        # writes no depth is drawn primitive by primitive, nearest centre first
        # (three's next sort key), per camera
        span = {mi: (first, count) for first, count, mi in draws}
        self.camera_draws: list[tuple[int, int, int, list | None]] = []
        self._split_points: dict[int, tuple[int, np.ndarray]] = {}
        for mi in scene.camera_order():
            if mi not in span or not span[mi][1]:
                continue
            split = None
            if not scene.materials[mi].depth_write:
                split = scene.primitive_ranges(mi)
                for _, _, pi in split:
                    oi, prim = scene.primitives[pi]
                    sc = np.asarray(scene.owners[oi].scale, np.float64)
                    self._split_points[pi] = (oi, prim.sort_point() * sc)
            self.camera_draws.append((mi, span[mi][0], span[mi][1], split))
        # shadow casters: consecutive ranges (the index buffer puts them first) by
        # culling mode; alpha-tested casters one material at a time (their map)
        self.shadow_draws: list[list] = []
        self.masked_shadow_draws: list[tuple[str, int, int, int]] = []
        for first, count, mi in draws:
            m = scene.materials[mi]
            if not m.cast_shadow or not count:
                continue
            cull = "none" if m.side == "double" else "front"
            if m.masked_shadow():
                self.masked_shadow_draws.append((cull, first, count, mi))
                continue
            last = self.shadow_draws[-1] if self.shadow_draws else None
            if last and last[0] == cull and last[1] + last[2] == first:
                last[2] += count
            else:
                self.shadow_draws.append([cull, first, count])
        self.n_owners = max(1, len(scene.owners))
        self.vb = self._buf(verts, U.VERTEX)
        self.ib = self._buf(idx if idx.size else np.zeros(4, np.uint32), U.INDEX)
        self.owners_buf = self._buf(None, U.STORAGE | U.COPY_DST, size=OWNER_BYTES * self.n_owners)
        self.frame_buf = self._buf(None, U.UNIFORM | U.COPY_DST, size=FRAME_BYTES)
        n_mat = max(1, len(scene.materials))
        self.mat_buf = self._buf(None, U.UNIFORM | U.COPY_DST, size=MATERIAL_SLOT * n_mat)
        if scene.materials:
            dev.queue.write_buffer(
                self.mat_buf, 0, np.concatenate([m.uniform() for m in scene.materials])
            )
        self.shadow_frame = self._buf(None, U.UNIFORM | U.COPY_DST, size=64)

        # textures (level 0 rows top-down: uv (0, 0) is the image's first row, flipY false)
        self.white = self._texture([np.full((1, 1, 4), 255, np.uint8)], "rgba8unorm")
        self.tex_views = []
        for img in scene.images:
            fmt = "rgba8unorm-srgb" if img.srgb else "rgba8unorm"
            self.tex_views.append(self._texture(img.levels(), fmt))
        self.samplers: dict = {}

        # environment
        self.prefilter = Prefilter(dev, self.pmrem_module)
        self.env_view = None
        self.env_geo = (1.0, 1.0, 0.0)
        self.set_lighting(lighting)

        # shadow map
        size = shadows.size if shadows else 1
        self.shadow_tex = dev.create_texture(
            size=(size, size, 1),
            format="depth32float",
            usage=T.RENDER_ATTACHMENT | T.TEXTURE_BINDING,
        )
        self._textures.append(self.shadow_tex)
        self.shadow_view = self.shadow_tex.create_view()

        VF = S.VERTEX | S.FRAGMENT
        self.bgl0 = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
                {
                    "binding": 1,
                    "visibility": VF,
                    "buffer": {"type": "uniform", "has_dynamic_offset": True},
                },
                {"binding": 2, "visibility": S.VERTEX, "buffer": {"type": "read-only-storage"}},
                {"binding": 3, "visibility": VF, "buffer": {"type": "uniform"}},
                {"binding": 4, "visibility": S.FRAGMENT, "texture": {"sample_type": "float"}},
                {"binding": 5, "visibility": S.FRAGMENT, "sampler": {"type": "filtering"}},
                {"binding": 6, "visibility": S.FRAGMENT, "texture": {"sample_type": "depth"}},
            ]
        )
        mat_entries = [{"binding": 0, "visibility": S.FRAGMENT, "buffer": {"type": "uniform"}}]
        for k in range(6):
            mat_entries.append(
                {"binding": 1 + k, "visibility": S.FRAGMENT, "texture": {"sample_type": "float"}}
            )
        for k in range(6):
            mat_entries.append(
                {"binding": 7 + k, "visibility": S.FRAGMENT, "sampler": {"type": "filtering"}}
            )
        self.bgl1 = dev.create_bind_group_layout(entries=mat_entries)
        self.layout = dev.create_pipeline_layout(bind_group_layouts=[self.bgl0, self.bgl1])
        self.env_sampler = dev.create_sampler(
            mag_filter="linear",
            min_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
        )
        self.mat_groups = [self._material_group(i) for i in range(len(scene.materials))]
        self._pipes: dict = {}
        self.sbgl = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": S.VERTEX, "buffer": {"type": "uniform"}},
                {"binding": 1, "visibility": S.VERTEX, "buffer": {"type": "read-only-storage"}},
            ]
        )
        self.sbg = dev.create_bind_group(
            layout=self.sbgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self.shadow_frame, "offset": 0, "size": 64}},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.owners_buf,
                        "offset": 0,
                        "size": OWNER_BYTES * self.n_owners,
                    },
                },
            ],
        )
        self.spl = dev.create_pipeline_layout(bind_group_layouts=[self.sbgl])
        self.spl_masked = dev.create_pipeline_layout(bind_group_layouts=[self.sbgl, self.bgl1])
        self.poses: dict[int, np.ndarray] = {}
        self.group0 = None
        self.last_frame: dict = {}
        self._owner_mats = owner_matrices(scene, self.poses)

    # ------------------------------------------------------------- set-up
    def _buf(self, data, usage, size=None):
        if data is not None:
            return self._make_buffer(0, usage, data)
        return self._make_buffer(int(size), usage)

    def _texture(self, levels, fmt):
        tex, view = upload_texture(self.device, levels, fmt)
        self._textures.append(tex)
        return view

    def _sampler(self, ref):
        key = (ref.mag, ref.min, ref.wrap_s, ref.wrap_t)
        if key not in self.samplers:
            mn, mip = _MIN.get(ref.min, ("linear", "linear"))
            self.samplers[key] = self.device.create_sampler(
                mag_filter="nearest" if ref.mag == "nearest" else "linear",
                min_filter=mn,
                mipmap_filter=mip,
                address_mode_u=_WRAP.get(ref.wrap_s, "repeat"),
                address_mode_v=_WRAP.get(ref.wrap_t, "repeat"),
                lod_max_clamp=32.0 if ref.mipmaps else 0.0,
            )
        return self.samplers[key]

    def _material_group(self, i: int):
        m = self.scene.materials[i]
        entries = [
            {
                "binding": 0,
                "resource": {"buffer": self.mat_buf, "offset": i * MATERIAL_SLOT, "size": 272},
            }
        ]
        default_sampler = self._default_sampler()
        for k, slot in enumerate(MAP_SLOTS):
            ref = m.maps.get(slot)
            view = self.tex_views[ref.image] if ref is not None else self.white
            entries.append({"binding": 1 + k, "resource": view})
        for k, slot in enumerate(MAP_SLOTS):
            ref = m.maps.get(slot)
            entries.append(
                {
                    "binding": 7 + k,
                    "resource": self._sampler(ref) if ref is not None else default_sampler,
                }
            )
        return self.device.create_bind_group(layout=self.bgl1, entries=entries)

    def _default_sampler(self):
        if not hasattr(self, "_dsamp"):
            self._dsamp = self.device.create_sampler(mag_filter="linear", min_filter="linear")
        return self._dsamp

    def set_lighting(self, lighting: Lighting | None) -> None:
        """Install the environment: an atlas array (three's layout), or a GPU
        atlas the renderer prefiltered (lighting.source = (texture, w, h, max_mip)),
        or a constant environment."""
        self.lighting = lighting
        old = getattr(self, "_env_tex", None)
        n_before = len(self._textures)
        if lighting is None:
            self.env_view = self._texture([np.zeros((1, 1, 4), np.float16)], ATLAS_FORMAT)
            self.env_geo = (1.0, 1.0, 0.0)
        elif isinstance(lighting.source, tuple):
            tex, w, h, mm = lighting.source
            self._textures.append(tex)
            self.env_view = tex.create_view()
            self.env_geo = (1.0 / w, 1.0 / h, float(mm))
        else:
            self._install_atlas(lighting)
        self._env_tex = self._textures[n_before]
        if old is not None and old is not self._env_tex:
            self._textures.remove(old)
            old.destroy()

    def _install_atlas(self, lighting: Lighting) -> None:
        if lighting.atlas is None:  # constant environment
            w, h, mm = atlas_size(16)
            atlas = np.ones((h, w, 4), np.float32)
        else:
            atlas = lighting.atlas
            h, w = atlas.shape[:2]
            mm = lighting.max_mip
        self.env_view = self._texture(
            [np.ascontiguousarray(atlas.astype(np.float16))], ATLAS_FORMAT
        )
        self.env_geo = (1.0 / w, 1.0 / h, float(mm))

    def pipeline(self, cull: str, depth_write: bool, color_format: str):
        key = ("mesh", cull, depth_write, color_format)
        if key not in self._pipes:
            self._pipes[key] = self.device.create_render_pipeline(
                layout=self.layout,
                vertex={
                    "module": self.mesh_module,
                    "entry_point": "vs_mesh",
                    "buffers": [_VERTEX_LAYOUT],
                },
                primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": cull},
                depth_stencil={
                    "format": "depth32float",
                    "depth_write_enabled": depth_write,
                    "depth_compare": "less-equal",
                },
                fragment={
                    "module": self.mesh_module,
                    "entry_point": "fs_mesh",
                    "targets": [{"format": color_format}],
                },
            )
        return self._pipes[key]

    def shadow_pipeline(self, cull: str, masked: bool = False):
        """Depth only; masked: alpha-tested casters (fs_shadow_masked discards
        below the material's alphaTest, the material's bind group in group 1)."""
        key = ("shadow", cull, masked)
        if key not in self._pipes:
            kw = {}
            if masked:
                kw["fragment"] = {
                    "module": self.shadow_module,
                    "entry_point": "fs_shadow_masked",
                    "targets": [],
                }
            self._pipes[key] = self.device.create_render_pipeline(
                layout=self.spl_masked if masked else self.spl,
                vertex={
                    "module": self.shadow_module,
                    "entry_point": "vs_shadow_masked" if masked else "vs_shadow",
                    "buffers": [_SHADOW_MASKED_LAYOUT if masked else _SHADOW_LAYOUT],
                },
                # y is flipped (rows in GL order), so a counter-clockwise front face
                # is clockwise here: cull "front" removes the casters' front faces
                primitive={"topology": "triangle-list", "front_face": "cw", "cull_mode": cull},
                depth_stencil={
                    "format": "depth32float",
                    "depth_write_enabled": True,
                    "depth_compare": "less-equal",
                },
                **kw,
            )
        return self._pipes[key]

    def bind_frame(self, disp_buf, slot_size: int) -> None:
        """Group 0 for the current camera set (the dispatch table is per set)."""
        self.group0 = self.device.create_bind_group(
            layout=self.bgl0,
            entries=[
                {"binding": 0, "resource": {"buffer": self.cams_buf, "offset": 0, "size": 96 * 16}},
                {"binding": 1, "resource": {"buffer": disp_buf, "offset": 0, "size": slot_size}},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.owners_buf,
                        "offset": 0,
                        "size": OWNER_BYTES * self.n_owners,
                    },
                },
                {
                    "binding": 3,
                    "resource": {"buffer": self.frame_buf, "offset": 0, "size": FRAME_BYTES},
                },
                {"binding": 4, "resource": self.env_view},
                {"binding": 5, "resource": self.env_sampler},
                {"binding": 6, "resource": self.shadow_view},
            ],
        )

    # -------------------------------------------------------------- frame
    def prepare(self, enc, state: dict, timer=None) -> None:
        """Owners, frame constants and scrolled materials for this state; the
        shadow pass (timed as stage "shadow" when a StageTimer is given)."""
        dev = self.device
        sc = self.scene
        owner_poses(sc, state, self.poses)
        mats = owner_matrices(sc, self.poses)
        self._owner_mats = mats
        dev.queue.write_buffer(self.owners_buf, 0, owner_uniforms(mats))
        for mi, offs in surface_offsets(sc, state).items():
            dev.queue.write_buffer(self.mat_buf, mi * MATERIAL_SLOT, sc.materials[mi].uniform(offs))
        L = self.lighting
        f = np.zeros(FRAME_BYTES // 4, np.float32)
        fw = f.view(np.uint32)
        light_on = L is not None and L.key_dir is not None and L.key_intensity > 0
        if light_on:
            d = np.asarray(L.key_dir, np.float64)
            f[0:3] = d / np.linalg.norm(d)
            f[3] = 1.0
            f[4:7] = np.asarray(L.key_color, np.float64) * L.key_intensity
        if L is not None:
            f[8] = L.env_intensity
            f[9] = 1.0
            rot = L.rotation_matrix()
            f[12:15], f[16:19], f[20:23] = rot[0], rot[1], rot[2]
        f[24:27] = self.env_geo
        shadow_on = bool(self.shadows) and light_on
        self.last_frame = {"shadow": False}
        if shadow_on:
            b = sc.bounds(mats)
            if b is None:
                shadow_on = False
            else:
                vp, sm = shadow_matrices(b[0], b[1], L.key_dir)
                f[28:44] = sm.T.reshape(-1)
                f[44:48] = (
                    self.shadows.bias,
                    self.shadows.normal_bias,
                    self.shadows.radius,
                    float(self.shadows.size),
                )
                dev.queue.write_buffer(self.shadow_frame, 0, vp.T.astype(np.float32).reshape(-1))
                self.last_frame = {"shadow": True, "bounds": [b[0].tolist(), b[1].tolist()]}
        fw[48] = 1 if shadow_on else 0
        fw[49] = 1  # Neutral tone mapping
        fw[50] = 1  # sRGB output
        f[52] = 1.0  # toneMappingExposure
        dev.queue.write_buffer(self.frame_buf, 0, f)
        if shadow_on:
            rp = enc.begin_render_pass(
                color_attachments=[],
                depth_stencil_attachment={
                    "view": self.shadow_view,
                    "depth_load_op": "clear",
                    "depth_store_op": "store",
                    "depth_clear_value": 1.0,
                },
                timestamp_writes=timer.writes("shadow") if timer else None,
            )
            rp.set_bind_group(0, self.sbg)
            rp.set_vertex_buffer(0, self.vb)
            rp.set_index_buffer(self.ib, "uint32")
            for cull, first, count in self.shadow_draws:
                rp.set_pipeline(self.shadow_pipeline(cull))
                rp.draw_indexed(count, 1, first, 0, 0)
            for cull, first, count, mi in self.masked_shadow_draws:
                rp.set_pipeline(self.shadow_pipeline(cull, masked=True))
                rp.set_bind_group(1, self.mat_groups[mi])
                rp.draw_indexed(count, 1, first, 0, 0)
            rp.end()

    def split_order(self, split: list, camera: dict | None) -> list:
        """The primitive ranges of a material that writes no depth, as three
        draws them for this camera: by the depth of each draw's centre along
        the view direction (three's z, front to back), then in creation order."""
        if camera is None or len(split) < 2:
            return split
        fwd = quat_xyzw_to_matrix(camera["quat"]) @ np.array([0.0, 0.0, -1.0])
        eye = np.asarray(camera["pos"], np.float64)
        keyed = []
        for k, (first, count, pi) in enumerate(split):
            oi, c = self._split_points[pi]
            m = self._owner_mats[oi]
            keyed.append((float(fwd @ (m[:3, :3] @ c + m[:3, 3] - eye)), k, (first, count, pi)))
        return [x[2] for x in sorted(keyed)]

    def draw(
        self, rp, disp_offset: int, color_format: str = "rgba8unorm", camera: dict | None = None
    ) -> None:
        """The meshes into a camera's render pass, in three's opaque order
        (camera_draws; camera: the render_state camera, for the per-draw depth
        order of materials without depth writes)."""
        if not self.draws:
            return
        rp.set_vertex_buffer(0, self.vb)
        rp.set_index_buffer(self.ib, "uint32")
        rp.set_bind_group(0, self.group0, [disp_offset])
        sc = self.scene
        current = None
        for mi, first, count, split in self.camera_draws:
            m = sc.materials[mi]
            pipe = self.pipeline(
                "none" if m.side == "double" else "back", m.depth_write, color_format
            )
            if pipe is not current:
                rp.set_pipeline(pipe)
                current = pipe
            rp.set_bind_group(1, self.mat_groups[mi])
            if split is None:
                rp.draw_indexed(count, 1, first, 0, 0)
            else:
                for f, c, _ in self.split_order(split, camera):
                    rp.draw_indexed(c, 1, f, 0, 0)

    def gpu_bytes(self) -> int:
        tot = 0
        for t in self._textures + self.prefilter._resources:
            s = t.size
            fmt = str(t.format)
            bpp = 8 if "16float" in fmt else 4
            lv = t.mip_level_count
            tot += sum(max(1, s[0] >> i) * max(1, s[1] >> i) * s[2] * bpp for i in range(lv))
        return tot

    def destroy(self) -> None:
        for t in self._textures:
            t.destroy()
        self._textures = []
        self.prefilter.destroy()


_VERTEX_LAYOUT = {
    "array_stride": VERTEX_BYTES,
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "float32x3", "offset": 12, "shader_location": 1},
        {"format": "float32x2", "offset": 24, "shader_location": 2},
        {"format": "float32x2", "offset": 32, "shader_location": 3},
        {"format": "float32x4", "offset": 40, "shader_location": 4},
        {"format": "float32x4", "offset": 56, "shader_location": 5},
        {"format": "uint32", "offset": 72, "shader_location": 6},
    ],
}
_SHADOW_LAYOUT = {
    "array_stride": VERTEX_BYTES,
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "uint32", "offset": 72, "shader_location": 6},
    ],
}
_SHADOW_MASKED_LAYOUT = {
    "array_stride": VERTEX_BYTES,
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "float32x2", "offset": 24, "shader_location": 2},
        {"format": "float32x2", "offset": 32, "shader_location": 3},
        {"format": "uint32", "offset": 72, "shader_location": 6},
    ],
}
