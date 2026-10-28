"""D's WGSL renderer: the Python host (wgpu-py) of pipeline wgsl/1 (M-D1 to M-D3).

Design doc 16 §9. The WGSL comes from one place, ``frontend/src/render/wgsl/``
(``pipeline.json`` lists the modules); this host never carries shader text.
It imports nothing from world_engine, only numpy, wgpu, Pillow and its sibling
modules: world-engine-client ships a copy of this directory and of the WGSL
(its ``scripts/sync_wgsl.py``), and ``default_wgsl_dir`` finds the shaders next
to this file there.

Per frame, one command buffer. Cameras go in batches (as many as one binding of
sort keys holds, at most 16): per batch one compute pass culls, compacts and
sorts every camera of the batch at once (workgroup_id.z = camera); before the
cameras, the key light's shadow map (when there are meshes and a key light);
then per camera a compute pass projects the sorted splats into records, a
render pass draws the meshes (three's MeshStandardMaterial, ``mesh.wgsl``) and
then the splats into a colour target of more than 8 bits (rgba16float on a GPU,
rgba32float on a CPU adapter: ``default_color_format``), a resolve pass rounds
it to rgba8unorm once (``resolve.wgsl``), and a copy puts that into one
readback buffer. One map at the end gives every camera as (H, W, 3) uint8.

``SplatRenderer`` takes the parts (splats, a ``MeshScene``, a ``Lighting``);
``WgslRenderer`` builds them from a session's scene_init and asset bytes the
way the capture page does (meshes from WEM1, robot links from WEL1, lighting
with the bake at the light probe and its fallbacks) and renders render_state
dicts: world-engine-client's renderer D and the engine's ``wgsl`` backend both
draw through it. The adapter search is ``adapters.py``; which scenes wgsl/1
draws, the engine's ``serving/render_protocol.py``. Not here: the browser host
(M-D4).
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .budget import (
    MAX_CAMS,
    ST_WORDS,
    UNIFORM_SLOT,
    WEBGPU_DEFAULT_LIMITS,
    WG,
    Budget,
    Layout,
    SplatMemoryError,
    batches,
    check_budget,
    gpu_budget,
    memory_error_message,
    readback_row_bytes,
    sh_coeffs,
    sh_words,
)
from .lighting import (
    BAKE_MAX_PIXEL_RADIUS,
    BAKE_RENDER_SIZE,
    BAKE_SIZE,
    ENV_ROTATION_ZUP,
    PROBE_HEIGHT,
    PROBE_MARGIN_WARN,
    SCENE_ENV_MIN_MEAN,
    SCENE_KEY_INTENSITY,
    STUDIO_ENV_INTENSITY,
    STUDIO_KEY_DIR,
    STUDIO_KEY_INTENSITY,
    STUDIO_SIGMA,
    STUDIO_SIZE,
    Lighting,
    bake_cameras,
    dominant_light_direction,
    half_float_like_three,
    probe_report,
    read_hdri,
    read_weli1,
    scene_intensity,
)
from .mesh_data import Material, MeshScene, Primitive, srgb_to_linear
from .mesh_pass import (
    MeshPass,
    ShadowParams,
    cube_mips,
    mesh_budget,
    upload_texture,
)
from .splat_data import Splats, quat_xyzw_to_matrix

PIPELINE = "wgsl/1"
NATIVE_BACKENDS = ("Vulkan", "Metal", "D3D12")
# state segment layout (u32 words, common.wgsl ST_*)
ST_COUNT = 0
ST_PROJECT_ARGS = 8
ST_DRAW_ARGS = 72
# Uploads go through one staging buffer of this size, a copy at a time.
# create_buffer_with_data and queue.write_buffer keep a staging copy of the
# whole array until the next submit completes (the 1.5M-splat room's 240 MiB
# of splat data and quad indices: 979 MiB per process right after construction
# instead of 787 MiB, RTX 4090).
UPLOAD_CHUNK = 16 << 20
SMALL_UPLOAD = 1 << 20  # arrays up to this (and up to upload_chunk) use queue.write_buffer

# Colour targets the mesh and splat passes can draw into, and their bytes per
# pixel. Anything but rgba8unorm is resolved to rgba8unorm before the readback
# (resolve.wgsl), so the image is rounded to 8 bits once. An 8-bit target
# rounds every blended layer to 8 bits, and adapters round differently: on the
# 1.5M-splat room the RTX 4090's 8-bit result is 1.07/255 darker than exact
# blending, SwiftShader's is not, and the two differ by 1.77/255 in the splat
# region. With rgba16float on the GPU and rgba32float on the CPU adapters they
# differ by 0.10 (2026-10-28, design doc 16 §9.4).
COLOR_FORMATS = {"rgba8unorm": 4, "rgb10a2unorm": 4, "rgba16float": 8, "rgba32float": 16}


def default_color_format(adapter_type: str | None, features) -> str:
    """The colour target for an adapter: rgba16float on a GPU (on an RTX 4090
    its draw pass costs 1.17 times rgba8unorm's, and a whole frame of the room
    scene, three 256x256 cameras, 1.04 to 1.07 times: 1.02 to 1.10 over rounds
    measured in one process and in separate ones, 2026-10-28; rgb10a2unorm's
    draw pass costs 1.45 and rgba32float's 6.4); rgba32float on a CPU adapter
    whose device has float32-blendable (as fast as 8 bits there), else
    rgb10a2unorm (lavapipe blends rgba16float with a bias of -1.1/255)."""
    if adapter_type == "CPU":
        return "rgba32float" if "float32-blendable" in set(features or ()) else "rgb10a2unorm"
    return "rgba16float"


def default_wgsl_dir() -> Path:
    """WE_WGSL_DIR; else this directory when the shaders sit next to this module
    (world-engine-client's copy); else the engine checkout's
    frontend/src/render/wgsl."""
    env = os.environ.get("WE_WGSL_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for d in (here, here.parents[2] / "frontend" / "src" / "render" / "wgsl"):
        if (d / "pipeline.json").is_file():
            return d
    return here


def load_pipeline(wgsl_dir: str | Path | None = None) -> tuple[dict, dict[str, str]]:
    """(manifest, {module name: WGSL source}) from a pipeline directory."""
    d = Path(wgsl_dir) if wgsl_dir else default_wgsl_dir()
    manifest = json.loads((d / "pipeline.json").read_text())
    if manifest.get("pipeline") != PIPELINE:
        raise RuntimeError(
            f"{d}: pipeline {manifest.get('pipeline')!r}, this host speaks {PIPELINE}"
        )
    modules = {
        name: "\n".join((d / f).read_text() for f in files)
        for name, files in manifest["modules"].items()
    }
    return manifest, modules


def adapter_summary(adapter) -> dict:
    i = adapter.info
    return {
        k: i.get(k) for k in ("vendor", "device", "adapter_type", "backend_type", "description")
    }


def find_adapter(kind: str = "auto"):
    """Development adapter choice (M-D3 does the real probe order).

    auto: a hardware adapter on Vulkan/Metal/D3D12, else a software one;
    hardware / software: only that kind; name:SUBSTR: by device name. Never an
    OpenGL adapter: NVIDIA's GL adapter has no compute (doc 16 §6.1)."""
    import wgpu

    ads = [
        a
        for a in wgpu.gpu.enumerate_adapters_sync()
        if a.info.get("backend_type") in NATIVE_BACKENDS
    ]
    hw = [a for a in ads if a.info.get("adapter_type") in ("DiscreteGPU", "IntegratedGPU")]
    hw.sort(key=lambda a: a.info.get("adapter_type") != "DiscreteGPU")
    sw = [a for a in ads if a.info.get("adapter_type") == "CPU"]
    if kind.startswith("name:"):
        s = kind[5:].lower()
        hit = [a for a in ads if s in (a.info.get("device") or "").lower()]
        if hit:
            return hit[0]
    elif kind == "hardware" and hw:
        return hw[0]
    elif kind == "software" and sw:
        return sw[0]
    elif kind == "auto" and (hw or sw):
        return (hw or sw)[0]
    names = [
        f"{a.info.get('device')} ({a.info.get('adapter_type')}, {a.info.get('backend_type')})"
        for a in ads
    ]
    raise RuntimeError(f"no {kind!r} WebGPU adapter; found: {names or 'none'}")


def device_features(adapter, extra=()) -> list[str]:
    """The optional features a device for this adapter asks for: float32-blendable
    on a CPU adapter that has it (its colour target is then rgba32float, see
    default_color_format), plus ``extra`` where the adapter has them."""
    have = set(adapter.features)
    want = list(extra)
    if adapter.info.get("adapter_type") == "CPU":
        want.append("float32-blendable")
    return [f for f in dict.fromkeys(want) if f in have]


def create_device(adapter, limits: dict | None = None, features: list | None = None):
    """A device with exactly the WebGPU default limits (doc 16 §9.2): the same
    chunking on every adapter, and nothing that only some adapters allow.
    ``features`` None asks for device_features(adapter): float32-blendable on a
    CPU adapter, for its colour target, and nothing else; a list asks for
    exactly those (the stage timer adds timestamps)."""
    req = dict(WEBGPU_DEFAULT_LIMITS if limits is None else limits)
    for k, v in req.items():
        have = adapter.limits.get(k)
        if have is not None and have < v:
            raise RuntimeError(
                f"{adapter.info.get('device')}: limit {k} is {have}, below the WebGPU default {v}"
            )
    feats = device_features(adapter) if features is None else list(features)
    return adapter.request_device_sync(required_limits=req, required_features=feats)


@dataclass(frozen=True)
class SplatParams:
    """Spark 2.1.0 SparkRenderer defaults (the capture page sets none of them)."""

    max_std_dev: float = math.sqrt(8.0)
    max_pixel_radius: float = 512.0
    min_pixel_radius: float = 0.0
    min_alpha: float = 0.5 / 255.0
    blur: float = 0.3
    pre_blur: float = 0.0
    clip_xy: float = 1.4
    clear_rgb: tuple[int, int, int] = (0x0B, 0x0D, 0x10)  # the capture page's clear colour


class MeshSet:
    """M-D1's placeholder mesh input, kept for the tests and benches that build
    meshes by hand: ``build(parts, owners)`` with parts (v (n, 3), f (m, 3),
    normals (n, 3), rgba 0-1 (display-referred), owner index) and owners
    (kind, key) gives a MeshScene of plain MeshStandardMaterials (colour
    decoded from sRGB, metalness 0, roughness 0.5)."""

    @staticmethod
    def build(parts, owners) -> MeshScene:
        sc = MeshScene()
        for kind, key in owners:
            sc.owner(str(kind), str(key))
        for v, f, nrm, rgba, oi in parts:
            col = srgb_to_linear(np.asarray(rgba, float)[:3])
            mi = sc.add_material(Material(color=tuple(col.tolist()), metalness=0.0, roughness=0.5))
            sc.add_primitive(
                int(oi),
                Primitive(
                    np.asarray(v, np.float32),
                    np.asarray(f, np.uint32).reshape(-1, 3),
                    mi,
                    normals=np.asarray(nrm, np.float32),
                ),
            )
        return sc


def camera_uniform(c: dict) -> np.ndarray:
    """A render_state camera (GL convention: looks down -z, y up; quat xyzw) as
    the Camera struct of common.wgsl (x right, y down, z forward)."""
    w, h = int(c["w"]), int(c["h"])
    r = quat_xyzw_to_matrix(c["quat"]) @ np.diag([1.0, -1.0, -1.0])
    f = 0.5 * h / math.tan(math.radians(float(c["fov_y_deg"])) / 2.0)
    u = np.zeros(24, np.float64)
    u[0:3], u[4:7], u[8:11] = r[:, 0], r[:, 1], r[:, 2]
    rs = c.get("spark_render_size")  # Spark sizes splats in these pixels (the bake)
    if rs and (int(rs[0]), int(rs[1])) != (w, h):
        u[3], u[7] = float(rs[0]) / w, float(rs[1]) / h
    u[12:15] = c["pos"]
    u[16:20] = (f, f, w / 2.0, h / 2.0)
    u[20:24] = (w, h, float(c["near"]), float(c["far"]))
    return u.astype(np.float32)


def _quat_to_matrix4(p: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_xyzw_to_matrix(p["quat"])
    m[:3, 3] = p["pos"]
    return m


def _ub(b, dyn=False, vis=None):
    import wgpu

    vis = wgpu.ShaderStage.COMPUTE if vis is None else vis
    return {
        "binding": b,
        "visibility": vis,
        "buffer": {"type": "uniform", "has_dynamic_offset": dyn},
    }


def _sb(b, rw, vis=None):
    import wgpu

    vis = wgpu.ShaderStage.COMPUTE if vis is None else vis
    return {
        "binding": b,
        "visibility": vis,
        "buffer": {"type": "storage" if rw else "read-only-storage"},
    }


def _res(b, offset=0, size=None):
    r = {"buffer": b, "offset": offset}
    if size is not None:
        r["size"] = size
    return r


# compute kernels: module, group-1 bindings (number, writable)
KERNELS = {
    "cull": ("preprocess", [(0, False), (2, True), (3, True)]),
    "scan_groups": ("preprocess", [(3, True), (6, True), (10, True)]),
    "compact": ("preprocess", [(0, False), (2, True), (3, True), (4, True), (5, True)]),
    "project_sorted": ("preprocess", [(0, False), (1, False), (7, False), (8, False), (9, True)]),
    "sort_histogram": ("sort", [(0, False), (4, True), (6, False)]),
    "sort_scan": ("sort", [(4, True), (5, True), (6, False)]),
    "sort_scatter": (
        "sort",
        [(0, False), (1, False), (2, True), (3, True), (4, True), (5, True), (6, False)],
    ),
}


class StageTimer:
    """GPU time per stage from timestamp queries (needs the 'timestamp-query'
    feature; development profiling only). Each pass gets a begin/end pair;
    ``read`` sums the pairs per stage name, in ms. wgpu-native reports raw
    ticks, taken here as nanoseconds (1 tick = 1 ns on NVIDIA and lavapipe)."""

    def __init__(self, device, max_passes: int = 64):
        import wgpu

        self.count = 2 * max_passes
        self.qs = device.create_query_set(type="timestamp", count=self.count)
        U = wgpu.BufferUsage
        self.b_resolve = device.create_buffer(
            size=8 * self.count, usage=U.QUERY_RESOLVE | U.COPY_SRC
        )
        self.b_read = device.create_buffer(size=8 * self.count, usage=U.MAP_READ | U.COPY_DST)
        self.names: list[str] = []

    def writes(self, name: str) -> dict:
        i = 2 * len(self.names)
        if i + 2 > self.count:
            raise RuntimeError("StageTimer: too many passes")
        self.names.append(name)
        return {
            "query_set": self.qs,
            "beginning_of_pass_write_index": i,
            "end_of_pass_write_index": i + 1,
        }

    def resolve(self, enc) -> None:
        n = 2 * len(self.names)
        enc.resolve_query_set(self.qs, 0, n, self.b_resolve, 0)
        enc.copy_buffer_to_buffer(self.b_resolve, 0, self.b_read, 0, 8 * n)

    def read(self) -> dict:
        import wgpu

        n = len(self.names)
        self.b_read.map_sync(wgpu.MapMode.READ)
        t = (
            np.frombuffer(self.b_read.read_mapped(copy=True), np.uint64)[: 2 * n]
            .astype(np.float64)
            .reshape(n, 2)
        )
        self.b_read.unmap()
        out: dict[str, float] = {}
        for name, (a, b) in zip(self.names, t, strict=True):
            out[name] = out.get(name, 0.0) + (b - a) / 1e6
        out = {k: round(v, 4) for k, v in out.items()}
        out["span"] = round(float((t[:, 1].max() - t[:, 0].min()) / 1e6), 4) if n else 0.0
        self.names = []
        return out


class ComputeKernels:
    """The compute pipelines of the pipeline directory, with explicit layouts:
    group 0 = Params, the camera array, Dispatch (dynamic offset); group 1 = the
    kernel's storage buffers (KERNELS)."""

    def __init__(self, device, sources: dict[str, str]):
        self.device = device
        self.modules = {k: device.create_shader_module(code=v, label=k) for k, v in sources.items()}
        self.bgl0 = device.create_bind_group_layout(entries=[_ub(0), _ub(1), _ub(2, True)])
        self.pipe, self.bgl1 = {}, {}
        for name, (mod, entries) in KERNELS.items():
            bgl = device.create_bind_group_layout(entries=[_sb(b, rw) for b, rw in entries])
            pl = device.create_pipeline_layout(bind_group_layouts=[self.bgl0, bgl])
            self.pipe[name] = device.create_compute_pipeline(
                layout=pl, compute={"module": self.modules[mod], "entry_point": name}
            )
            self.bgl1[name] = bgl

    def group0(self, params_buf, cams_buf, disp_buf):
        return self.device.create_bind_group(
            layout=self.bgl0,
            entries=[
                {"binding": 0, "resource": _res(params_buf, 0, UNIFORM_SLOT)},
                {"binding": 1, "resource": _res(cams_buf, 0, 96 * MAX_CAMS)},
                {"binding": 2, "resource": _res(disp_buf, 0, UNIFORM_SLOT)},
            ],
        )

    def group1(self, kernel: str, mapping: dict):
        return self.device.create_bind_group(
            layout=self.bgl1[kernel],
            entries=[{"binding": b, "resource": _res(buf)} for b, buf in sorted(mapping.items())],
        )

    def sort_groups(self, sb: dict) -> dict:
        """Bind groups of the four radix passes over a batch's buffers (a -> b -> a -> b -> a)."""
        ab = [
            (sb["keys_a"], sb["vals_a"], sb["keys_b"], sb["vals_b"]),
            (sb["keys_b"], sb["vals_b"], sb["keys_a"], sb["vals_a"]),
        ]
        return {
            "hist": [
                self.group1("sort_histogram", {0: ki, 4: sb["hist"], 6: sb["state"]})
                for ki, _, _, _ in ab
            ],
            "scan": self.group1("sort_scan", {4: sb["hist"], 5: sb["totals"], 6: sb["state"]}),
            "scatter": [
                self.group1(
                    "sort_scatter",
                    {0: ki, 1: vi, 2: ko, 3: vo, 4: sb["hist"], 5: sb["totals"], 6: sb["state"]},
                )
                for ki, vi, ko, vo in ab
            ],
        }

    def encode_sort(
        self, cp, bg0, groups: dict, args_buf, args_offset: int, n_seg: int, pass_disp0: int
    ) -> None:
        """Four stable radix passes over the n_seg segments of a batch; the result
        is back in keys_a / vals_a. args_buf at args_offset holds the dispatch size
        (tiles of the largest segment, 1, n_seg)."""
        for pas in range(4):
            cp.set_bind_group(0, bg0, [(pass_disp0 + pas) * UNIFORM_SLOT])
            cp.set_pipeline(self.pipe["sort_histogram"])
            cp.set_bind_group(1, groups["hist"][pas % 2])
            cp.dispatch_workgroups_indirect(args_buf, args_offset)
            cp.set_pipeline(self.pipe["sort_scan"])
            cp.set_bind_group(1, groups["scan"])
            cp.dispatch_workgroups(256, 1, n_seg)
            cp.set_pipeline(self.pipe["sort_scatter"])
            cp.set_bind_group(1, groups["scatter"][pas % 2])
            cp.dispatch_workgroups_indirect(args_buf, args_offset)


def sort_segments(
    device, segments: list[tuple[np.ndarray, np.ndarray]], *, wgsl_dir=None, stale_fill: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sort (keys, values) uint32 segments with the GPU radix sort, all segments
    in one batch as the renderer sorts its cameras. Every segment gets room for
    the longest one; the room past a segment's end is filled with ``stale_fill``,
    so a leak across a segment boundary shows. Returns the sorted pairs."""
    import wgpu

    U = wgpu.BufferUsage
    _, sources = load_pipeline(wgsl_dir)
    k = ComputeKernels(device, {m: sources[m] for m in ("preprocess", "sort")})
    nseg = len(segments)
    cap = max(1, max(len(kv[0]) for kv in segments))
    tiles_cap = (cap + 1023) // 1024
    prm = np.zeros(UNIFORM_SLOT // 4, np.uint32)
    prm[0] = cap  # n_splats: the segment stride
    prm[13] = tiles_cap
    b_params = device.create_buffer_with_data(data=prm, usage=U.UNIFORM)
    b_cams = device.create_buffer(size=96 * MAX_CAMS, usage=U.UNIFORM)
    disp = np.zeros((4, UNIFORM_SLOT // 4), np.uint32)
    disp[:, 4] = [0, 8, 16, 24]
    b_disp = device.create_buffer_with_data(data=disp, usage=U.UNIFORM)
    bg0 = k.group0(b_params, b_cams, b_disp)
    st = U.STORAGE | U.COPY_SRC | U.COPY_DST
    ka = np.full(nseg * cap, stale_fill, np.uint32)
    va = np.full(nseg * cap, stale_fill, np.uint32)
    state = np.zeros(nseg * ST_WORDS, np.uint32)
    for s, (keys, vals) in enumerate(segments):
        ka[s * cap : s * cap + len(keys)] = keys
        va[s * cap : s * cap + len(keys)] = vals
        state[s * ST_WORDS + ST_COUNT] = len(keys)
    args = np.array([max((len(kv[0]) + 1023) // 1024 for kv in segments), 1, nseg, 0], np.uint32)
    sb = {
        "keys_a": device.create_buffer_with_data(data=ka, usage=st),
        "vals_a": device.create_buffer_with_data(data=va, usage=st),
        "keys_b": device.create_buffer_with_data(
            data=np.full(nseg * cap, stale_fill, np.uint32), usage=st
        ),
        "vals_b": device.create_buffer_with_data(
            data=np.full(nseg * cap, stale_fill, np.uint32), usage=st
        ),
        "hist": device.create_buffer(size=4 * nseg * 256 * tiles_cap, usage=st),
        "totals": device.create_buffer(size=4 * nseg * 256, usage=st),
        "state": device.create_buffer_with_data(data=state, usage=st),
    }
    b_args = device.create_buffer_with_data(data=args, usage=st | U.INDIRECT)
    groups = k.sort_groups(sb)
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    k.encode_sort(cp, bg0, groups, b_args, 0, nseg, 0)
    cp.end()
    device.queue.submit([enc.finish()])
    ko = np.frombuffer(device.queue.read_buffer(sb["keys_a"]), np.uint32)
    vo = np.frombuffer(device.queue.read_buffer(sb["vals_a"]), np.uint32)
    out = [
        (ko[s * cap : s * cap + len(kv[0])].copy(), vo[s * cap : s * cap + len(kv[0])].copy())
        for s, kv in enumerate(segments)
    ]
    for b in list(sb.values()) + [b_args, b_params, b_cams, b_disp]:
        b.destroy()
    return out


class SplatRenderer:
    """Render a splat scene, and meshes when given, from up to 16 cameras, each
    with its own size. ``render(state)`` takes a render_state-shaped dict and
    returns ``{camera name: (H, W, 3) uint8}``.

    meshes: a MeshScene (mesh_data). lighting: what lights them (lighting.Lighting;
    None: a constant environment of radiance 1 and no key light). shadows: the
    key light's shadow map (True or a ShadowParams: the editor's quality tier;
    None: none).

    batch: cameras sorted together (None: as many as one binding of keys holds).
    binding_limit: lower the storage binding limit below the device's (tests use
    it to force many chunks). memory_limit: refuse scenes whose estimated device
    memory (``Budget.device_estimate``, buffers plus the allocator's and the
    driver's share) exceeds it, before allocating; checked again when the camera
    sizes change. upload_chunk: bytes per staging copy (tests make it small).
    color_format: the colour target the passes draw into (COLOR_FORMATS; None:
    default_color_format for the adapter and the device's features); anything
    but rgba8unorm is resolved to 8 bits per camera before the readback.
    Splats with non-finite values are dropped, with one RuntimeWarning.
    """

    def __init__(
        self,
        splats: Splats,
        cameras: list[dict],
        *,
        meshes: MeshScene | None = None,
        lighting: Lighting | None = None,
        shadows: ShadowParams | bool | None = None,
        adapter="auto",
        device=None,
        batch: int | None = None,
        params: SplatParams | None = None,
        binding_limit: int | None = None,
        memory_limit: int | None = None,
        wgsl_dir: str | Path | None = None,
        depth_readback: bool = False,
        profile: bool = False,
        upload_chunk: int = UPLOAD_CHUNK,
        color_format: str | None = None,
    ):
        import wgpu

        self._wgpu = wgpu
        if device is None:
            self.adapter = find_adapter(adapter) if isinstance(adapter, str) else adapter
            extra = ["timestamp-query"] if profile else []
            self.device = create_device(self.adapter, features=device_features(self.adapter, extra))
        else:
            self.device = device
            self.adapter = device.adapter
        self.adapter_info = adapter_summary(self.adapter)
        if color_format is None:
            color_format = default_color_format(
                self.adapter_info.get("adapter_type"), self.device.features
            )
        if color_format not in COLOR_FORMATS:
            raise ValueError(
                f"color_format must be one of {sorted(COLOR_FORMATS)}, not {color_format!r}"
            )
        if color_format == "rgba32float" and "float32-blendable" not in self.device.features:
            raise ValueError("an rgba32float colour target needs a device with float32-blendable")
        self.color_format = color_format
        self.resolves = color_format != "rgba8unorm"
        self.params = params or SplatParams()
        lim = self.device.limits
        bind = min(lim["max-storage-buffer-binding-size"], lim["max-buffer-size"])
        if binding_limit is not None:
            bind = min(bind, int(binding_limit))
        self.binding_limit = bind
        self.meshes = meshes
        if meshes is not None and lighting is None:
            lighting = Lighting.uniform(1.0)
        self.lighting = lighting
        self.shadow_params = ShadowParams() if shadows is True else (shadows or None)
        self.mesh = None
        self.depth_readback = depth_readback
        self.memory_limit = memory_limit
        self.upload_chunk = max(256, int(upload_chunk) // 4 * 4)
        with np.errstate(all="ignore"):
            geom = splats.gpu_geometry()
            keep = splats.finite_mask(geom)
        self.dropped_splats = int((~keep).sum())
        if self.dropped_splats:
            warnings.warn(
                f"{self.dropped_splats:,} of {splats.n:,} splats have a non-finite position, "
                "scale, rotation, opacity or SH coefficient (or a zero rotation); they are "
                "not drawn",
                RuntimeWarning,
                stacklevel=2,
            )
            splats = splats.subset(keep)
            geom = geom[keep]
        self.n_splats = splats.n
        self.sh_degree = splats.sh_degree
        sizes = [(int(c["w"]), int(c["h"])) for c in cameras]
        self._mesh_kw = {}
        if meshes is not None:
            bufs, parts = mesh_budget(meshes, lighting, self.shadow_params)
            self._mesh_kw = {"mesh_buffers": bufs, "mesh_parts": parts}
        self._batch_arg = batch
        self.budget: Budget = self._plan(sizes)
        self.layout: Layout = self.budget.layout
        self.batch = self.budget.batch
        self._name = self.adapter_info.get("device") or "the adapter"
        check_budget(self.budget, memory_limit, self._name)
        self.manifest, sources = load_pipeline(wgsl_dir)
        self._buffers: list = []  # the scene's buffers, kept for the renderer's life
        self._cam_buffers: list = []  # the camera set's buffers and targets, see _set_cameras
        self._cam_textures: list = []
        self._stage = None
        self.sizes: list[tuple[int, int]] = []
        try:
            self._build(geom, splats.gpu_sh(), splats.sh_rotation, sources)
            del geom
            self._set_cameras(sizes)
        except SplatMemoryError:
            self.close()
            raise
        except (wgpu.GPUOutOfMemoryError, MemoryError) as e:
            self.close()
            raise SplatMemoryError(
                memory_error_message(self.budget, "out of memory", self._name)
            ) from e
        finally:
            self._drop_stage()
        self.last_timing: dict = {}
        self._timer = None
        if profile and "timestamp-query" in self.device.features:
            self._timer = StageTimer(self.device)

    def _plan(self, sizes: list[tuple[int, int]], sort_batch: int | None = None) -> Budget:
        """The budget for these camera sizes; sort_batch: the sort buffers already
        allocated hold this many cameras (a new camera set on a built renderer)."""
        return gpu_budget(
            self.n_splats,
            self.sh_degree,
            sizes,
            batch=self._batch_arg,
            sort_batch=sort_batch,
            depth_readback=self.depth_readback,
            binding_limit=self.binding_limit,
            buffer_limit=self.device.limits["max-buffer-size"],
            color_bytes=COLOR_FORMATS[self.color_format],
            resolve=self.resolves,
            **self._mesh_kw,
        )

    # ------------------------------------------------------------------ setup
    def _buffer(self, size: int, usage, data: np.ndarray | None = None, owner: list | None = None):
        """A buffer of at least ``size`` bytes (rounded up to 4, never past the
        binding limit), filled with ``data`` through the staging buffer."""
        U = self._wgpu.BufferUsage
        size = max(16, (int(size) + 3) // 4 * 4)
        if data is None:
            b = self.device.create_buffer(size=size, usage=usage)
        else:
            raw = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
            if raw.nbytes % 4:
                raw = np.concatenate([raw, np.zeros(4 - raw.nbytes % 4, np.uint8)])
            size = max(size, raw.nbytes)
            b = self.device.create_buffer(size=size, usage=usage | U.COPY_DST)
            if raw.nbytes <= min(SMALL_UPLOAD, self.upload_chunk):
                self.device.queue.write_buffer(b, 0, raw)
            else:
                self._upload(b, raw)
        (self._buffers if owner is None else owner).append(b)
        return b

    def _drop_stage(self) -> None:
        if self._stage is not None:
            self._stage.destroy()
            self._stage = None

    def _upload(self, buf, raw: np.ndarray) -> None:
        """Copy raw (bytes) to the start of buf, upload_chunk bytes at a time
        through one MAP_WRITE staging buffer. Mapping the staging buffer waits
        for the previous copy, so no more than one chunk is ever in flight."""
        wgpu = self._wgpu
        if self._stage is None:
            self._stage = self.device.create_buffer(
                size=self.upload_chunk, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
            )
        for off in range(0, raw.nbytes, self.upload_chunk):
            n = min(self.upload_chunk, raw.nbytes - off)
            self._stage.map_sync(wgpu.MapMode.WRITE, 0, n)
            self._stage.write_mapped(raw[off : off + n], 0)
            self._stage.unmap()
            enc = self.device.create_command_encoder()
            enc.copy_buffer_to_buffer(self._stage, 0, buf, off, n)
            self.device.queue.submit([enc.finish()])

    def _build(
        self, geom: np.ndarray, shw: np.ndarray, sh_rotation, sources: dict[str, str]
    ) -> None:
        wgpu = self._wgpu
        dev = self.device
        U = wgpu.BufferUsage
        S = wgpu.ShaderStage
        L = self.layout
        p = self.params
        B = self.batch

        prm = np.zeros(28, np.uint32)
        prm[0:4] = (L.n_splats, sh_coeffs(L.sh_degree), sh_words(L.sh_degree), L.n_groups)
        prm[4:11] = np.array(
            [
                p.max_std_dev,
                p.max_pixel_radius,
                p.min_pixel_radius,
                p.min_alpha,
                p.blur,
                p.pre_blur,
                p.clip_xy,
            ],
            np.float32,
        ).view(np.uint32)
        prm[11:14] = (L.rec_chunk, L.n_rec_chunks, L.tiles_cap)
        rot = np.asarray(sh_rotation, np.float64)
        rot_rows = np.zeros(12, np.float32)
        rot_rows[0:3], rot_rows[4:7], rot_rows[8:11] = rot[0], rot[1], rot[2]
        prm[16:28] = rot_rows.view(np.uint32)
        self.b_params = self._buffer(UNIFORM_SLOT, U.UNIFORM, prm)
        self.b_cams = self._buffer(96 * MAX_CAMS, U.UNIFORM | U.COPY_DST)
        self.src = L.source_ranges()
        self.recs_r = L.rec_ranges()

        # splat data, one geometry + one SH buffer per source chunk
        self.b_geom, self.b_sh = [], []
        for start, ln in self.src:
            self.b_geom.append(
                self._buffer(geom[start : start + ln].nbytes, U.STORAGE, geom[start : start + ln])
            )
            self.b_sh.append(
                self._buffer(shw[start : start + ln].nbytes, U.STORAGE, shw[start : start + ln])
            )

        # one batch of cameras: segment `seg` of each array belongs to camera seg of the batch
        n = L.capacity
        st = U.STORAGE | U.COPY_SRC | U.COPY_DST  # copies: diagnostics and tests
        self.fb = {
            "flags": self._buffer(4 * B * L.n_splats, st),
            "groups": self._buffer(4 * B * L.n_groups, st),
            "keys_a": self._buffer(4 * B * n, st),
            "vals_a": self._buffer(4 * B * n, st),
            "keys_b": self._buffer(4 * B * n, st),
            "vals_b": self._buffer(4 * B * n, st),
            "hist": self._buffer(4 * B * 256 * L.tiles_cap, st),
            "totals": self._buffer(4 * B * 256, st),
            "state": self._buffer(4 * B * ST_WORDS, st | U.INDIRECT),
        }
        self.b_recs = [self._buffer(32 * ln, st) for _, ln in self.recs_r]
        quads = np.arange(L.rec_chunk, dtype=np.uint32)[:, None] * 4
        self.b_quad_index = self._buffer(
            0, U.INDEX, (quads + np.array([0, 1, 2, 2, 1, 3], np.uint32)).ravel()
        )
        if self.meshes is not None:
            self.mesh = MeshPass(
                dev,
                sources,
                self.meshes,
                self.lighting,
                self.shadow_params,
                cams_buf=self.b_cams,
                make_buffer=self._buffer,
            )

        # pipelines and bind groups
        self.kernels = ComputeKernels(dev, {m: sources[m] for m in ("preprocess", "sort", "splat")})
        mods = self.kernels.modules
        self.pipe = self.kernels.pipe
        g1 = self.kernels.group1
        fb = self.fb
        self.bg_cull = [g1("cull", {0: gb, 2: fb["flags"], 3: fb["groups"]}) for gb in self.b_geom]
        self.bg_compact = [
            g1(
                "compact",
                {0: gb, 2: fb["flags"], 3: fb["groups"], 4: fb["keys_a"], 5: fb["vals_a"]},
            )
            for gb in self.b_geom
        ]
        self.bg_sort = self.kernels.sort_groups(fb)
        self.bg_project = [
            [
                g1(
                    "project_sorted",
                    {0: self.b_geom[k], 1: self.b_sh[k], 7: fb["vals_a"], 8: fb["state"], 9: rb},
                )
                for rb in self.b_recs
            ]
            for k in range(len(self.src))
        ]

        VF = S.VERTEX | S.FRAGMENT
        self.bgl_draw = dev.create_bind_group_layout(
            entries=[
                _ub(0, vis=VF),
                _ub(1, vis=VF),
                _sb(2, False, vis=S.VERTEX),
                _ub(3, True, vis=VF),
            ]
        )
        dpl = dev.create_pipeline_layout(bind_group_layouts=[self.bgl_draw])
        blend = {
            "color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
            "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
        }
        fmt = self.color_format
        self.p_splat = dev.create_render_pipeline(
            layout=dpl,
            vertex={"module": mods["splat"], "entry_point": "vs_splat"},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": "depth32float",
                "depth_write_enabled": False,
                "depth_compare": "less-equal",
            },
            fragment={
                "module": mods["splat"],
                "entry_point": "fs_splat",
                "targets": [{"format": fmt, "blend": blend}],
            },
        )
        if self.resolves:
            # the colour target, read texel by texel into rgba8unorm (resolve.wgsl)
            self.bgl_resolve = dev.create_bind_group_layout(
                entries=[
                    {
                        "binding": 0,
                        "visibility": S.FRAGMENT,
                        "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"},
                    }
                ]
            )
            m_resolve = dev.create_shader_module(code=sources["resolve"], label="resolve")
            self.p_resolve = dev.create_render_pipeline(
                layout=dev.create_pipeline_layout(bind_group_layouts=[self.bgl_resolve]),
                vertex={"module": m_resolve, "entry_point": "vs_resolve"},
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                fragment={
                    "module": m_resolve,
                    "entry_point": "fs_resolve",
                    "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
                },
            )

    def _draw_group(self, storage_buf):
        return self.device.create_bind_group(
            layout=self.bgl_draw,
            entries=[
                {"binding": 0, "resource": _res(self.b_params, 0, UNIFORM_SLOT)},
                {"binding": 1, "resource": _res(self.b_cams, 0, 96 * MAX_CAMS)},
                {"binding": 2, "resource": _res(storage_buf)},
                {"binding": 3, "resource": _res(self.b_disp, 0, UNIFORM_SLOT)},
            ],
        )

    def _set_cameras(self, sizes: list[tuple[int, int]]) -> None:
        """Targets, readback layout and the Dispatch table for this camera set.
        A new set of sizes first gets its budget checked against memory_limit;
        then the previous set's buffers and textures are destroyed."""
        wgpu = self._wgpu
        if len(sizes) > MAX_CAMS:
            raise ValueError(f"{len(sizes)} cameras; a renderer draws at most {MAX_CAMS}")
        if not sizes:
            raise ValueError("no cameras")
        if self.sizes:  # a new camera set on a built renderer
            budget = self._plan(sizes, sort_batch=self.batch)
            check_budget(budget, self.memory_limit, self._name)
            self._release_cameras()
            self.budget = budget
        try:
            self._make_cameras(sizes)
        except (wgpu.GPUOutOfMemoryError, MemoryError) as e:
            self._release_cameras()
            raise SplatMemoryError(
                memory_error_message(self.budget, "out of memory", self._name)
            ) from e
        finally:
            self._drop_stage()

    def _release_cameras(self) -> None:
        for r in self._cam_buffers + self._cam_textures:
            r.destroy()
        self._cam_buffers, self._cam_textures = [], []
        self.sizes = []

    def _texture(self, w: int, h: int, fmt, sampled: bool = False):
        T = self._wgpu.TextureUsage
        usage = T.RENDER_ATTACHMENT | T.COPY_SRC | (T.TEXTURE_BINDING if sampled else 0)
        t = self.device.create_texture(size=(w, h, 1), format=fmt, usage=usage)
        self._cam_textures.append(t)
        return t

    def _make_cameras(self, sizes: list[tuple[int, int]]) -> None:
        wgpu = self._wgpu
        U = wgpu.BufferUsage
        own = self._cam_buffers
        self.color = [self._texture(w, h, self.color_format, self.resolves) for w, h in sizes]
        self.depth = [self._texture(w, h, wgpu.TextureFormat.depth32float) for w, h in sizes]
        self.color_views = [t.create_view() for t in self.color]
        self.depth_views = [t.create_view() for t in self.depth]
        # what is copied to the readback buffer: the colour target itself, or its
        # 8-bit resolve
        self.readout = self.color
        if self.resolves:
            self.readout = [self._texture(w, h, wgpu.TextureFormat.rgba8unorm) for w, h in sizes]
            self.readout_views = [t.create_view() for t in self.readout]
            self.bg_resolve = [
                self.device.create_bind_group(
                    layout=self.bgl_resolve, entries=[{"binding": 0, "resource": v}]
                )
                for v in self.color_views
            ]
        self.bpr = [readback_row_bytes(w) for w, _ in sizes]
        self.rb_off = [0]
        for (_, h), b in zip(sizes, self.bpr, strict=True):
            self.rb_off.append(self.rb_off[-1] + b * h)
        self.b_read = self._buffer(max(256, self.rb_off[-1]), U.MAP_READ | U.COPY_DST, owner=own)
        self.b_read_depth = None
        if self.depth_readback:
            self.b_read_depth = self._buffer(
                max(256, self.rb_off[-1]), U.MAP_READ | U.COPY_DST, owner=own
            )
        # Dispatch table: per (batch, source chunk); 4 sort passes; per (camera, source chunk, record chunk)
        self.batches = batches(len(sizes), self.batch)
        n_k, n_r = len(self.src), len(self.recs_r)
        rows = []
        self.d_batch = {}
        for b, (c0, _) in enumerate(self.batches):
            for k, (start, ln) in enumerate(self.src):
                self.d_batch[b, k] = len(rows)
                rows.append(dict(start=start, len=ln, group0=start // WG, cam0=c0, args=4 * b))
        self.d_sort = len(rows)
        for pas in range(4):
            rows.append(dict(shift=8 * pas))
        self.d_cam = {}
        for c0, nb in self.batches:
            for s in range(nb):
                for k in range(n_k):
                    for r in range(n_r):
                        self.d_cam[c0 + s, k, r] = len(rows)
                        start, ln = self.src[k]
                        rows.append(
                            dict(
                                start=start,
                                len=ln,
                                group0=start // WG,
                                rec_start=self.recs_r[r][0],
                                cam=c0 + s,
                                seg=s,
                            )
                        )
        order = ["start", "len", "group0", "rec_start", "shift", "cam0", "cam", "seg", "args"]
        disp = np.zeros((len(rows), UNIFORM_SLOT // 4), np.uint32)
        for i, row in enumerate(rows):
            for j, key in enumerate(order):
                disp[i, j] = row.get(key, 0)
        self.b_disp = self._buffer(disp.nbytes, U.UNIFORM, disp, owner=own)
        self.b_sort_args = self._buffer(
            16 * len(self.batches), U.STORAGE | U.INDIRECT | U.COPY_DST, owner=own
        )
        self.sort_args_init = np.zeros((len(self.batches), 4), np.uint32)
        for b, (_, nb) in enumerate(self.batches):
            self.sort_args_init[b] = (0, 1, nb, 0)
        self.bg0 = self.kernels.group0(self.b_params, self.b_cams, self.b_disp)
        self.bg_scan_groups = self.kernels.group1(
            "scan_groups", {3: self.fb["groups"], 6: self.fb["state"], 10: self.b_sort_args}
        )
        self.bg_draw = [self._draw_group(rb) for rb in self.b_recs]
        if self.mesh is not None:
            self.mesh.bind_frame(self.b_disp, UNIFORM_SLOT)
        self.sizes = list(sizes)

    # ------------------------------------------------------------------ frame
    def render(self, state: dict) -> dict[str, np.ndarray]:
        """render_state -> {camera name: (H, W, 3) uint8}."""
        cams = list(state["cameras"])
        images = self.render_cameras(cams, state)
        return {c["name"]: im for c, im in zip(cams, images, strict=True)}

    def render_cameras(
        self,
        cams: list[dict],
        state: dict | None = None,
        *,
        draw_meshes: bool = True,
        clear_rgb: tuple[int, int, int] | None = None,
    ) -> list[np.ndarray]:
        """Render these cameras; the meshes take their poses from ``state``
        (bodies, robot links, surfaces; the poses persist across calls, as the
        capture page keeps an object's last pose). draw_meshes False and
        clear_rgb are for the environment bake."""
        wgpu = self._wgpu
        dev = self.device
        sizes = [(int(c["w"]), int(c["h"])) for c in cams]
        if sizes != self.sizes:
            self._set_cameras(sizes)
        t0 = time.perf_counter()
        cu = np.zeros((MAX_CAMS, 24), np.float32)
        for i, c in enumerate(cams):
            cu[i] = camera_uniform(c)
        dev.queue.write_buffer(self.b_cams, 0, cu)
        dev.queue.write_buffer(self.b_sort_args, 0, self.sort_args_init)
        timer = self._timer
        enc = dev.create_command_encoder()
        meshes = self.mesh is not None and draw_meshes
        if meshes:
            self.mesh.prepare(enc, state or {}, timer)
        n_r = len(self.recs_r)
        clear_rgb = self.params.clear_rgb if clear_rgb is None else clear_rgb
        clear = tuple(x / 255.0 for x in clear_rgb) + (1.0,)
        slot = UNIFORM_SLOT
        for b, (c0, nb) in enumerate(self.batches):
            # cull, compact and sort every camera of the batch (workgroup_id.z = camera)
            cp = enc.begin_compute_pass(timestamp_writes=timer.writes("compact") if timer else None)
            cp.set_pipeline(self.pipe["cull"])
            for k, (_, ln) in enumerate(self.src):
                cp.set_bind_group(0, self.bg0, [self.d_batch[b, k] * slot])
                cp.set_bind_group(1, self.bg_cull[k])
                cp.dispatch_workgroups((ln + WG - 1) // WG, 1, nb)
            cp.set_pipeline(self.pipe["scan_groups"])
            cp.set_bind_group(0, self.bg0, [self.d_batch[b, 0] * slot])
            cp.set_bind_group(1, self.bg_scan_groups)
            cp.dispatch_workgroups(1, 1, nb)
            cp.set_pipeline(self.pipe["compact"])
            for k, (_, ln) in enumerate(self.src):
                cp.set_bind_group(0, self.bg0, [self.d_batch[b, k] * slot])
                cp.set_bind_group(1, self.bg_compact[k])
                cp.dispatch_workgroups((ln + WG - 1) // WG, 1, nb)
            if timer:
                cp.end()
                cp = enc.begin_compute_pass(timestamp_writes=timer.writes("sort"))
            self.kernels.encode_sort(
                cp, self.bg0, self.bg_sort, self.b_sort_args, 16 * b, nb, self.d_sort
            )
            cp.end()
            self._after_sort(enc, b, nb)
            for s in range(nb):
                ci = c0 + s
                w, h = sizes[ci]
                st_off = 4 * s * ST_WORDS
                cp = enc.begin_compute_pass(
                    timestamp_writes=timer.writes("project") if timer else None
                )
                cp.set_pipeline(self.pipe["project_sorted"])
                for k in range(len(self.src)):
                    for r in range(n_r):
                        cp.set_bind_group(0, self.bg0, [self.d_cam[ci, k, r] * slot])
                        cp.set_bind_group(1, self.bg_project[k][r])
                        cp.dispatch_workgroups_indirect(
                            self.fb["state"], st_off + 4 * (ST_PROJECT_ARGS + 4 * r)
                        )
                cp.end()
                rp = enc.begin_render_pass(
                    color_attachments=[
                        {
                            "view": self.color_views[ci],
                            "load_op": "clear",
                            "store_op": "store",
                            "clear_value": clear,
                        }
                    ],
                    depth_stencil_attachment={
                        "view": self.depth_views[ci],
                        "depth_load_op": "clear",
                        "depth_store_op": "store",
                        "depth_clear_value": 1.0,
                    },
                    timestamp_writes=timer.writes("draw") if timer else None,
                )
                if meshes:
                    self.mesh.draw(
                        rp, self.d_cam[ci, 0, 0] * slot, self.color_format, camera=cams[ci]
                    )
                self._draw_splats(rp, ci, s)
                rp.end()
                if self.resolves:
                    rp = enc.begin_render_pass(
                        color_attachments=[
                            {
                                "view": self.readout_views[ci],
                                "load_op": "clear",
                                "store_op": "store",
                                "clear_value": (0.0, 0.0, 0.0, 1.0),
                            }
                        ],
                        timestamp_writes=timer.writes("resolve") if timer else None,
                    )
                    rp.set_pipeline(self.p_resolve)
                    rp.set_bind_group(0, self.bg_resolve[ci])
                    rp.draw(3)
                    rp.end()
                enc.copy_texture_to_buffer(
                    {"texture": self.readout[ci]},
                    {
                        "buffer": self.b_read,
                        "offset": self.rb_off[ci],
                        "bytes_per_row": self.bpr[ci],
                        "rows_per_image": h,
                    },
                    (w, h, 1),
                )
                if self.b_read_depth is not None:
                    enc.copy_texture_to_buffer(
                        {"texture": self.depth[ci], "aspect": "depth-only"},
                        {
                            "buffer": self.b_read_depth,
                            "offset": self.rb_off[ci],
                            "bytes_per_row": self.bpr[ci],
                            "rows_per_image": h,
                        },
                        (w, h, 1),
                    )
        if timer:
            timer.resolve(enc)
        dev.queue.submit([enc.finish()])
        t1 = time.perf_counter()
        self.b_read.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(self.b_read.read_mapped(copy=True), np.uint8)
        self.b_read.unmap()
        out = []
        for ci, (w, h) in enumerate(sizes):
            px = raw[self.rb_off[ci] : self.rb_off[ci] + self.bpr[ci] * h].reshape(h, self.bpr[ci])
            out.append(np.ascontiguousarray(px[:, : 4 * w].reshape(h, w, 4)[:, :, :3]))
        t2 = time.perf_counter()
        self.last_timing = {"encode_submit_ms": (t1 - t0) * 1e3, "wait_read_ms": (t2 - t1) * 1e3}
        if timer:
            self.last_timing["gpu_ms"] = timer.read()
        return out

    def _after_sort(self, enc, batch: int, n_cams: int) -> None:
        """Hook after a batch's cull, compact and sort pass (the draw A/B bench,
        scripts/wgsl/draw_ab.py, computes its draw arguments here)."""

    def _draw_splats(self, rp, ci: int, seg: int) -> None:
        """The splats of camera ci (segment seg of its batch): one non-instanced
        indexed draw per record chunk, 4 vertices and 6 indices per quad, with the
        arguments scan_groups wrote (design doc 16 §9.10 item 2)."""
        st_off = 4 * seg * ST_WORDS
        rp.set_pipeline(self.p_splat)
        rp.set_index_buffer(self.b_quad_index, "uint32")
        for r in range(len(self.recs_r)):
            rp.set_bind_group(0, self.bg_draw[r], [self.d_cam[ci, 0, r] * UNIFORM_SLOT])
            rp.draw_indexed_indirect(self.fb["state"], st_off + 4 * (ST_DRAW_ARGS + 8 * r))

    def read_depth(self) -> list[np.ndarray]:
        """Depth of the last frame per camera, (H, W) float32 (needs depth_readback=True).
        Splats never write depth, so depth < 1 marks pixels whose nearest surface is a mesh."""
        if self.b_read_depth is None:
            raise RuntimeError("construct with depth_readback=True")
        self.b_read_depth.map_sync(self._wgpu.MapMode.READ)
        raw = np.frombuffer(self.b_read_depth.read_mapped(copy=True), np.float32)
        self.b_read_depth.unmap()
        out = []
        for ci, (w, h) in enumerate(self.sizes):
            o = self.rb_off[ci] // 4
            out.append(
                raw[o : o + self.bpr[ci] // 4 * h].reshape(h, self.bpr[ci] // 4)[:, :w].copy()
            )
        return out

    def visible_counts(self) -> list[int]:
        """Visible splats per camera of the last batch rendered (diagnostic)."""
        v = np.frombuffer(self.device.queue.read_buffer(self.fb["state"]), np.uint32).reshape(
            -1, ST_WORDS
        )
        nb = self.batches[-1][1]
        return [int(x) for x in v[:nb, ST_COUNT]]

    def gpu_bytes(self) -> int:
        """Bytes of the buffers and textures this renderer holds now. Not the
        process's device memory: see budget.Budget.device_estimate."""
        tot = sum(b.size for b in self._buffers + self._cam_buffers)
        per_px = COLOR_FORMATS[self.color_format] + 4 + (4 if self.resolves else 0)
        tot += sum(per_px * w * h for w, h in self.sizes)  # colour, depth32float, 8-bit resolve
        if self.mesh is not None:
            tot += self.mesh.gpu_bytes()  # textures: maps, environment, shadow map
        return tot

    def live_resources(self) -> int:
        """Buffers and textures held (diagnostic: constant across camera resizes)."""
        return len(self._buffers) + len(self._cam_buffers) + len(self._cam_textures)

    # ------------------------------------------------------- environment
    def bake_faces(
        self, center, *, size: int = BAKE_SIZE, render_size=None, max_pixel_radius=None
    ) -> np.ndarray:
        """The capture page's environment bake (sceneEnvironment.ts): the splats
        seen from ``center`` on the six 90-degree faces of a size^2 cube, colours
        decoded to linear (Spark's encodeLinear) and blended in 8 bits over the
        page's clear colour as a linear render target stores it, no meshes.
        render_size: Spark's renderSize for the splat sizes (None: the face);
        max_pixel_radius: the bake renderer's cap (None: this renderer's).
        Returns (6, size, size, 4) uint8 faces in three's read-back order (GL
        cube faces +x -x +y -y +z -z, row 0 at t = -1). The renderer's next
        render() goes back to its own cameras and cap."""
        cams = bake_cameras(center, size)
        if render_size is not None:
            for c in cams:
                c["spark_render_size"] = [int(render_size[0]), int(render_size[1])]
        self._set_encode_linear(1)
        if max_pixel_radius is not None:
            self._set_max_pixel_radius(float(max_pixel_radius))
        try:
            imgs = self.render_cameras(cams, None, draw_meshes=False, clear_rgb=(1, 1, 1))
        finally:
            self._set_encode_linear(0)
            if max_pixel_radius is not None:
                self._set_max_pixel_radius(self.params.max_pixel_radius)
        faces = np.stack([im[::-1] for im in imgs])
        alpha = np.full(faces.shape[:3] + (1,), 255, np.uint8)
        return np.concatenate([faces, alpha], -1)

    def _set_encode_linear(self, on: int) -> None:
        self.device.queue.write_buffer(self.b_params, 14 * 4, np.array([on], np.uint32))

    def _set_max_pixel_radius(self, px: float) -> None:
        self.device.queue.write_buffer(self.b_params, 5 * 4, np.array([px], np.float32))

    def prefilter_cube(self, faces: np.ndarray):
        """PMREM of a cube given as (6, s, s, 4) uint8 faces in read-back order,
        mipmapped as generateMipmap does: (atlas texture, width, height, max_mip)."""
        tex, view = upload_texture(self.device, cube_mips(faces), "rgba8unorm", layers=6, cube=True)
        try:
            return self.mesh.prefilter.run("cube", view, cube_size=faces.shape[1])
        finally:
            tex.destroy()

    def prefilter_equirect(self, rgb: np.ndarray):
        """PMREM of an equirectangular HDR image ((H, W, 3) float32, rows top to
        bottom, as RGBELoader / EXRLoader decode it): uploaded as half floats the
        way those loaders convert them (HalfFloatType), with three's flipY (row 0
        at v = 1)."""
        h, w = rgb.shape[:2]
        rgba = np.concatenate([np.asarray(rgb, np.float32), np.ones((h, w, 1), np.float32)], -1)
        tex, view = upload_texture(
            self.device, [np.ascontiguousarray(half_float_like_three(rgba[::-1]))], "rgba16float"
        )
        try:
            return self.mesh.prefilter.run("equirect", view, cube_size=w // 4)
        finally:
            tex.destroy()

    def prefilter_studio(self):
        """three's RoomEnvironment through PMREMGenerator.fromScene(room, 0.04)."""
        return self.mesh.prefilter.run("room", None, cube_size=STUDIO_SIZE, pre_blur=STUDIO_SIGMA)

    def set_lighting(self, lighting: Lighting) -> None:
        """Light the meshes with this (the environment texture is replaced)."""
        self.lighting = lighting
        if self.mesh is None:
            return
        self.mesh.set_lighting(lighting)
        self.mesh.bind_frame(self.b_disp, UNIFORM_SLOT)
        # the budget follows the new environment, and so do the camera sets planned later
        bufs, parts = mesh_budget(self.meshes, lighting, self.shadow_params)
        self._mesh_kw = {"mesh_buffers": bufs, "mesh_parts": parts}
        self.budget = self._plan(self.sizes, sort_batch=self.batch)

    def close(self) -> None:
        self._release_cameras()
        if self.mesh is not None:
            self.mesh.destroy()
            self.mesh = None
        for b in self._buffers:
            b.destroy()
        self._buffers = []


def _no_splats() -> Splats:
    """One fully transparent splat: the splat pass of a scene without a 3DGS
    background draws nothing (min_alpha culls it)."""
    return Splats(
        np.zeros((1, 3), np.float32),
        np.full((1, 3), 1e-3, np.float32),
        np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        np.zeros(1, np.float32),
        np.zeros((1, 1, 3), np.float32),
    )


class WgslRenderer:
    """D's rendering of a client-render session, the way the capture page does
    it (frontend/src/sceneRenderer.ts): from the session's scene_init and its
    asset bytes to ``render(state) -> {camera: (H, W, 3) uint8}``.

    fetch(url) returns the bytes (or a file path) of a scene_init URL: the
    splat PLY, the objects' WEM1 (``meshes[].wem``), the robot links' WEL1, the
    lighting asset or HDRI. Lighting, in the capture page's order: a lighting
    asset (``scene_init.lighting``) when there is one; else an HDRI
    (``scene_init.hdri``); else the bake at ``scene_init.probe_center`` (at
    PROBE_HEIGHT above the origin when absent), falling back to the studio set-up
    when the probe is black or dark, or when there is no splat. ``info`` says
    which (for ``info["render"]``).

    shadows: the key light's shadow map (the editor's quality tier; the capture
    page has none). bake_render_size, bake_max_pixel_radius: Spark's renderSize
    and cap in the page's bake (lighting.BAKE_RENDER_SIZE explains them).
    glb_to_wem1: converts a mesh URL that has no WEM1 (benches with recorded
    traces; the engine compiles WEM1 itself). strict: a part of the scene this
    host would leave out or light otherwise than the capture page (a GLB
    without its WEM1, a robot link it cannot fetch, an HDRI or lighting asset
    it cannot read) raises ValueError instead of a RuntimeWarning
    (world-engine-client: the browser path then draws the scene). The other
    options go to SplatRenderer (color_format, wgsl_dir, memory_limit, ...).

    fetch may return bytes, a bytearray, a memoryview (world-engine-client's
    assets decrypted in memory; the splat PLY is read from it without a copy)
    or a path."""

    def __init__(
        self,
        scene_init: dict,
        fetch,
        *,
        adapter="auto",
        device=None,
        shadows: ShadowParams | bool | None = True,
        lighting: Lighting | None = None,
        bake_render_size=BAKE_RENDER_SIZE,
        bake_max_pixel_radius=BAKE_MAX_PIXEL_RADIUS,
        glb_to_wem1=None,
        max_sh_degree: int = 3,
        strict: bool = False,
        **renderer_kw,
    ):
        def get(url):
            data = fetch(url)
            if isinstance(data, (str, Path)):
                return Path(data).read_bytes()
            if isinstance(data, memoryview):  # the readers below slice and decode bytes
                return data.tobytes()
            return data

        self.scene_init = scene_init
        self.strict = strict
        self.info: dict = {"pipeline": PIPELINE}
        self.baked_faces: np.ndarray | None = None
        sp = scene_init.get("splat") or {}
        self.has_splat = bool(sp.get("url"))
        if self.has_splat:
            data = fetch(sp["url"])
            splats = read_splat_ply_any(data, max_sh_degree)
            del data  # the plaintext: the splats are parsed out of it
            splats = splats.placed(
                sp.get("pos", (0, 0, 0)), sp.get("quat", (0, 0, 0, 1)), sp.get("scale", 1.0)
            )
        else:
            splats = _no_splats()
        self.meshes = MeshScene.from_scene_init(
            scene_init, get, glb_to_wem1=glb_to_wem1, strict=strict
        )
        for w in self.meshes.warnings:
            warnings.warn(w, RuntimeWarning, stacklevel=2)
        self.renderer = SplatRenderer(
            splats,
            scene_init["cameras"],
            meshes=self.meshes,
            lighting=Lighting.uniform(1.0),
            shadows=shadows,
            adapter=adapter,
            device=device,
            **renderer_kw,
        )
        t0 = time.perf_counter()
        self._bake_args = {
            "render_size": bake_render_size,
            "max_pixel_radius": bake_max_pixel_radius,
        }
        del splats
        try:
            light = lighting or self._lighting(scene_init, get)
            self.renderer.set_lighting(light)
        except BaseException:
            self.renderer.close()
            raise
        self.lighting = light
        self.info["lighting"] = light.mode
        self.info["lighting_detail"] = dict(light.info)
        self.info["lighting_s"] = round(time.perf_counter() - t0, 3)

    def _lighting(self, si: dict, get) -> Lighting:
        r = self.renderer
        if si.get("lighting") and si["lighting"].get("url"):
            try:
                return read_weli1(get(si["lighting"]["url"]))
            except Exception as e:  # pragma: no cover - a broken asset falls back
                if self.strict:
                    raise ValueError(f"the scene's lighting asset is not read: {e}") from e
                warnings.warn(f"lighting asset not used ({e}); baking instead", RuntimeWarning, 3)
        hdri = si.get("hdri")
        if hdri and hdri.get("url"):
            try:
                atlas = r.prefilter_equirect(read_hdri(get(hdri["url"])))
                return Lighting(
                    "hdri",
                    env_intensity=float(hdri.get("strength", 1.0)),
                    env_rotation=ENV_ROTATION_ZUP,
                    source=atlas,
                    info={"hdri": hdri["url"]},
                )
            except Exception as e:
                if self.strict:
                    raise ValueError(f"the scene's HDRI is not read: {e}") from e
                warnings.warn(f"HDRI not used ({e}); keeping the default lights", RuntimeWarning, 3)
                self.info["hdri_error"] = str(e)
        if self.has_splat:
            pc = si.get("probe_center")
            center = (
                [float(v) for v in pc]
                if pc is not None and len(pc) >= 3
                else [0.0, 0.0, PROBE_HEIGHT]
            )
            t0 = time.perf_counter()
            faces = r.bake_faces(center, **self._bake_args)
            self.baked_faces = faces  # the read-back faces (tests and the bench)
            probe = dominant_light_direction(faces)
            info = {"center": center, "bake_ms": round((time.perf_counter() - t0) * 1e3, 1)}
            if probe is not None:
                info["probe"] = probe_report(probe)
                if abs(info["probe"]["margin"]) < PROBE_MARGIN_WARN:
                    warnings.warn(
                        f"the light probe's mean luminance {probe.mean:.4f} is within "
                        f"{PROBE_MARGIN_WARN:.0%} of the {SCENE_ENV_MIN_MEAN} threshold between "
                        "the scene bake and the studio lights; another adapter (8-bit blending "
                        "rounds differently, the CPU adapters most) can light this scene the "
                        "other way",
                        RuntimeWarning,
                        3,
                    )
            if probe is None:
                info["failed"] = "black"
            elif probe.mean < SCENE_ENV_MIN_MEAN:
                info["failed"] = "dark"
                info["mean"] = probe.mean
            else:
                atlas = r.prefilter_cube(faces)
                return Lighting(
                    "scene",
                    env_intensity=scene_intensity(probe.mean),
                    env_rotation=(0.0, 0.0, 0.0),
                    key_dir=tuple(probe.dir.tolist()),
                    key_intensity=SCENE_KEY_INTENSITY,
                    source=atlas,
                    info=info,
                )
            self.info["bake"] = info
        return Lighting(
            "studio",
            env_intensity=STUDIO_ENV_INTENSITY,
            env_rotation=ENV_ROTATION_ZUP,
            key_dir=STUDIO_KEY_DIR,
            key_intensity=STUDIO_KEY_INTENSITY,
            source=r.prefilter_studio(),
        )

    def render(self, state: dict) -> dict[str, np.ndarray]:
        """A render_state -> {camera name: (H, W, 3) uint8}: its cameras, or
        the scene_init's when it names none."""
        if not state.get("cameras"):
            state = {**state, "cameras": self.scene_init["cameras"]}
        return self.renderer.render(state)

    @property
    def color_format(self) -> str:
        return self.renderer.color_format

    @property
    def last_timing(self) -> dict:
        return self.renderer.last_timing

    def close(self) -> None:
        self.renderer.close()


def read_splat_ply_any(data, max_sh_degree: int = 3) -> Splats:
    from .splat_data import read_splat_ply

    return read_splat_ply(
        data if isinstance(data, (bytes, bytearray, memoryview)) else Path(data), max_sh_degree
    )
