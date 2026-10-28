"""GPU memory plan of D's splat pass: how the data is split into buffers, and
what a scene of N splats with C cameras needs.

Pure Python (no wgpu), so the layout rules can be tested anywhere and a host can
refuse a scene before allocating anything. ``Budget.total`` is the bytes of the
buffers and textures that ``render_wgsl.SplatRenderer`` creates (the tests check
the two against each other). It is not the device memory the process takes: the
driver's share and the allocator's blocks come on top: 1.9, 1.7 and 3.1 times
the buffer bytes for the round-2 room, 1600k and 400k scenes on an RTX 4090. ``Budget.device_estimate`` adds them
with constants measured there; ``check_budget`` compares that estimate with a
memory limit.

Every storage binding stays at or below the binding limit (WebGPU default:
128 MiB; lavapipe has exactly that) and every buffer at or below the buffer
limit (default 256 MiB):

* splat data is cut into *source chunks* of whole workgroups (multiples of 256
  splats), each chunk its own geometry buffer (48 B/splat) and SH buffer (f16,
  ``sh_words`` u32 words per splat);
* cameras are processed in *batches*; a batch of B cameras shares one set of
  sort and compaction buffers (one segment of N entries per camera), so B is at
  most binding_limit / (4 N);
* the sorted records (32 B each) are cut into *record chunks*; each record
  chunk is one binding and one indirect draw. Cameras reuse the records one
  after the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

WG = 256
SORT_TILE = 1024
RADIX = 256
MAX_CAMS = 16
MAX_REC_CHUNKS = 16
ST_WORDS = 200
GEOM_BYTES = 48  # 3 x vec4f: mean + opacity | cov xx xy xz yy | cov yz zz
REC_BYTES = 32
CAMERA_BYTES = 96
MAX_REC_CHUNK = 1 << 21  # caps the shared quad index buffer at 48 MiB
MAX_GROUPS_PER_DISPATCH = 65535  # WebGPU default maxComputeWorkgroupsPerDimension
UNIFORM_SLOT = 256  # dynamic uniform offsets are 256-byte aligned

MiB = 1 << 20

# Device memory beyond the buffers, per process (nvidia-smi), RTX 4090, driver
# 580.178, wgpu-native 29.0.1.1 (Vulkan; buffers are suballocated with
# gpu-allocator 0.28), 2026-10-28:
# - a device alone takes 211 MiB, and that already holds the first 112 MiB of
#   device-local buffers;
# - past those, device-local memory grows in blocks of 256 MiB, and a buffer
#   lies within one block;
# - mapped readback buffers are system memory and do not count;
# - rendering adds up to 2 MiB (16 cameras), FRAME_SLACK covers it.
# On 13 scene and camera configurations (3 to 16 cameras, 0.4M to 1.6M splats,
# SH degree 0 to 3, up to 1280x720) device_estimate without FRAME_SLACK was 0 to
# 2 MiB below the measured memory. Other drivers and backends differ (lavapipe and
# SwiftShader use system RAM); M-D3 measures each adapter class it supports.
DEVICE_BASE = 211 * MiB
DEVICE_FIRST_ROOM = 112 * MiB
DEVICE_BLOCK = 256 * MiB
FRAME_SLACK = 4 * MiB

# The WebGPU default limits this pipeline is designed for (and requests).
WEBGPU_DEFAULT_LIMITS = {
    "max-storage-buffer-binding-size": 128 * MiB,
    "max-buffer-size": 256 * MiB,
    "max-compute-invocations-per-workgroup": 256,
    "max-compute-workgroup-size-x": 256,
    "max-compute-workgroup-storage-size": 16384,
    "max-storage-buffers-per-shader-stage": 8,
    "max-bind-groups": 4,
    "max-compute-workgroups-per-dimension": 65535,
}


class SplatMemoryError(RuntimeError):
    """The scene does not fit the GPU: the message lists what needs how much."""


def sh_coeffs(sh_degree: int) -> int:
    return (sh_degree + 1) ** 2


def sh_words(sh_degree: int) -> int:
    """u32 words of packed f16 SH per splat (3 channels per coefficient)."""
    return (3 * sh_coeffs(sh_degree) + 1) // 2


def readback_row_bytes(width: int) -> int:
    return (width * 4 + 255) // 256 * 256


@dataclass(frozen=True)
class Layout:
    n_splats: int
    sh_degree: int
    binding_limit: int
    buffer_limit: int
    source_chunk: int  # splats per source chunk (multiple of 256)
    n_source_chunks: int
    rec_chunk: int  # records per record chunk
    n_rec_chunks: int
    capacity: int  # visible splats one camera can hold (= n_splats)
    tiles_cap: int  # sort tiles at full capacity
    n_groups: int  # cull workgroups over all source chunks
    max_batch: int  # cameras whose sort buffers fit one binding

    def source_ranges(self) -> list[tuple[int, int]]:
        """(start, length) of each source chunk."""
        out = []
        for k in range(self.n_source_chunks):
            start = k * self.source_chunk
            out.append((start, min(self.source_chunk, self.n_splats - start)))
        return out

    def rec_ranges(self) -> list[tuple[int, int]]:
        """(first sorted position, capacity) of each record chunk."""
        out = []
        for r in range(self.n_rec_chunks):
            start = r * self.rec_chunk
            out.append((start, min(self.rec_chunk, self.capacity - start)))
        return out


def plan_layout(
    n_splats: int,
    sh_degree: int,
    binding_limit: int = WEBGPU_DEFAULT_LIMITS["max-storage-buffer-binding-size"],
    buffer_limit: int = WEBGPU_DEFAULT_LIMITS["max-buffer-size"],
) -> Layout:
    """Split a scene into bindings that fit the limits; raise SplatMemoryError if
    some array cannot be split (one camera's key and flag arrays are one binding
    each, which caps a scene at binding_limit / 4 splats)."""
    if n_splats < 1:
        raise ValueError("a splat scene needs at least one splat")
    if not 0 <= sh_degree <= 3:
        raise ValueError(f"SH degree must be 0..3, got {sh_degree}")
    limit = min(binding_limit, buffer_limit)
    per_splat = max(GEOM_BYTES, 4 * sh_words(sh_degree))
    source = limit // per_splat // WG * WG
    source = min(source, MAX_GROUPS_PER_DISPATCH * WG)
    if source < WG:
        raise SplatMemoryError(f"storage binding limit {limit} B is too small for {WG} splats")
    rec_chunk = min(n_splats, limit // REC_BYTES, MAX_REC_CHUNK)
    n_rec = math.ceil(n_splats / rec_chunk)
    if 4 * n_splats > limit or n_rec > MAX_REC_CHUNKS:
        cap = min(limit // 4, MAX_REC_CHUNKS * rec_chunk)
        raise SplatMemoryError(
            f"{n_splats:,} splats is more than one renderer can sort: a camera's key and index "
            f"arrays are single bindings of 4 B per splat, so the limit here is {cap:,} splats. "
            f"Split the scene, or decimate the splats."
        )
    return Layout(
        n_splats=n_splats,
        sh_degree=sh_degree,
        binding_limit=binding_limit,
        buffer_limit=buffer_limit,
        source_chunk=source,
        n_source_chunks=math.ceil(n_splats / source),
        rec_chunk=rec_chunk,
        n_rec_chunks=n_rec,
        capacity=n_splats,
        tiles_cap=math.ceil(n_splats / SORT_TILE),
        n_groups=math.ceil(n_splats / WG),
        max_batch=min(MAX_CAMS, limit // (4 * n_splats)),
    )


def batches(n_cameras: int, batch: int) -> list[tuple[int, int]]:
    """(first camera, cameras) of each batch."""
    return [(c, min(batch, n_cameras - c)) for c in range(0, n_cameras, batch)]


@dataclass
class Budget:
    """Bytes of every buffer and texture the renderer creates, grouped (``total``),
    and an estimate of the device memory that takes (``device_estimate``)."""

    layout: Layout
    n_cameras: int
    batch: int
    parts: dict[str, int] = field(default_factory=dict)
    largest_buffer: int = 0
    # device-local buffers and textures in the order the renderer creates them
    device_buffers: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Buffer and texture bytes. Not the device memory: see device_estimate."""
        return sum(self.parts.values())

    @property
    def device_estimate(self) -> int:
        """Device memory of the rendering process, estimated: the device's base
        (DEVICE_BASE, which holds the first DEVICE_FIRST_ROOM bytes of buffers)
        plus one DEVICE_BLOCK for every further allocator block the device-local
        buffers fill, packed first-fit in creation order with each buffer inside
        one block. Mapped readback buffers are system memory and not counted.
        Constants from an RTX 4090 (see above): for the round-2 room, 1600k
        and 400k scenes with 3 cameras it gives 727, 471 and 215 MiB; the
        measured memory was 723, 467 and 211 MiB."""
        free: list[int] = [DEVICE_FIRST_ROOM]
        for size in self.device_buffers:
            for i, room in enumerate(free):
                if size <= room:
                    free[i] = room - size
                    break
            else:
                free.append(max(0, DEVICE_BLOCK - size))
        return DEVICE_BASE + FRAME_SLACK + (len(free) - 1) * DEVICE_BLOCK

    def describe(self) -> str:
        lines = [
            f"{self.layout.n_splats:,} splats (SH degree {self.layout.sh_degree}), "
            f"{self.n_cameras} camera(s) in batches of {self.batch}: buffers and textures "
            f"{self.total / MiB:.1f} MiB, device memory about {self.device_estimate / MiB:.0f} MiB"
        ]
        for name, b in self.parts.items():
            lines.append(f"  {name}: {b / MiB:.1f} MiB")
        return "\n".join(lines)


def per_batch_buffers(layout: Layout, batch: int) -> list[int]:
    """Sizes of a batch's buffers in creation order: flags, groups, keys_a, vals_a,
    keys_b, vals_b, histograms, digit totals, state."""
    n = layout.capacity
    return [
        4 * batch * layout.n_splats,
        4 * batch * layout.n_groups,
        *([4 * batch * n] * 4),
        4 * batch * RADIX * layout.tiles_cap,
        4 * batch * RADIX,
        4 * batch * ST_WORDS,
    ]


def per_batch_bytes(layout: Layout, batch: int) -> dict[str, int]:
    """The sort and compaction buffers one batch of cameras shares."""
    n = layout.capacity
    return {
        "flags": 4 * batch * layout.n_splats,
        "groups": 4 * batch * layout.n_groups,
        "keys_values": 4 * 4 * batch * n,  # keys and values, two copies each (sort ping-pong)
        "histogram": 4 * batch * RADIX * layout.tiles_cap + 4 * batch * RADIX,
        "state": 4 * batch * ST_WORDS,
    }


def gpu_budget(
    n_splats: int,
    sh_degree: int,
    camera_sizes: list[tuple[int, int]],
    *,
    batch: int | None = None,
    sort_batch: int | None = None,
    mesh_vertices: int = 0,
    mesh_indices: int = 0,
    mesh_owners: int = 0,
    depth_readback: bool = False,
    binding_limit: int = WEBGPU_DEFAULT_LIMITS["max-storage-buffer-binding-size"],
    buffer_limit: int = WEBGPU_DEFAULT_LIMITS["max-buffer-size"],
    color_bytes: int = 4,
    resolve: bool = False,
) -> Budget:
    """What ``SplatRenderer`` allocates for this scene (camera_sizes: (w, h) each).
    batch: cameras per batch (None: as many as one binding holds). sort_batch:
    the sort buffers are already allocated for this many cameras (a renderer
    whose camera set changed keeps them). color_bytes: bytes per pixel of the
    colour target (4 for rgba8unorm, 8 for rgba16float, 16 for rgba32float);
    resolve: an rgba8unorm texture per camera receives the 8-bit resolve."""
    layout = plan_layout(n_splats, sh_degree, binding_limit, buffer_limit)
    n_cam = len(camera_sizes)
    if n_cam > MAX_CAMS:
        raise ValueError(f"{n_cam} cameras; a renderer draws at most {MAX_CAMS}")
    b_cap = layout.max_batch if batch is None else min(int(batch), layout.max_batch)
    b_cam = max(1, min(b_cap, n_cam or 1)) if sort_batch is None else int(sort_batch)
    b = Budget(layout=layout, n_cameras=n_cam, batch=b_cam)
    n_batches = max(1, len(batches(n_cam, b_cam)))
    n_dispatch = (
        n_batches * layout.n_source_chunks
        + 4
        + max(n_cam, 1) * layout.n_source_chunks * layout.n_rec_chunks
    )
    # device-local allocations, in SplatRenderer's order (_build, then _make_cameras)
    dev = b.device_buffers
    dev += [UNIFORM_SLOT, CAMERA_BYTES * MAX_CAMS]
    for _, ln in layout.source_ranges():
        dev += [GEOM_BYTES * ln, 4 * sh_words(sh_degree) * ln]
    dev += per_batch_buffers(layout, b_cam)
    dev += [REC_BYTES * ln for _, ln in layout.rec_ranges()]
    dev.append(6 * 4 * layout.rec_chunk)
    if mesh_vertices:
        dev += [32 * mesh_vertices, 4 * mesh_indices, 64 * max(1, mesh_owners)]
    dev += [color_bytes * w * h for w, h in camera_sizes]  # colour target
    dev += [4 * w * h for w, h in camera_sizes]  # depth32float
    if resolve:
        dev += [4 * w * h for w, h in camera_sizes]  # rgba8unorm resolve
    dev += [UNIFORM_SLOT * n_dispatch, 16 * n_batches]
    b.device_buffers = [max(16, (x + 3) // 4 * 4) for x in dev]

    b.parts["splat geometry"] = GEOM_BYTES * n_splats
    b.parts["splat SH (f16)"] = 4 * sh_words(sh_degree) * n_splats
    b.parts["sort and compaction (per batch)"] = sum(per_batch_bytes(layout, b_cam).values())
    b.parts["sorted records"] = REC_BYTES * layout.capacity
    b.parts["quad index buffer"] = 6 * 4 * layout.rec_chunk
    targets = 0
    readback = 0
    for w, h in camera_sizes:
        targets += (color_bytes + 4 + (4 if resolve else 0)) * w * h  # colour, depth, resolve
        readback += readback_row_bytes(w) * h * (2 if depth_readback else 1)
    b.parts["camera targets"] = targets
    b.parts["readback"] = readback
    if mesh_vertices:
        b.parts["meshes"] = 32 * mesh_vertices + 4 * mesh_indices + 64 * max(1, mesh_owners)
    b.parts["uniforms"] = UNIFORM_SLOT * (1 + n_dispatch) + CAMERA_BYTES * MAX_CAMS + 16 * n_batches
    b.largest_buffer = max(
        4 * sh_words(sh_degree) * min(n_splats, layout.source_chunk),
        GEOM_BYTES * min(n_splats, layout.source_chunk),
        REC_BYTES * layout.rec_chunk,
        4 * b_cam * n_splats,
        6 * 4 * layout.rec_chunk,
    )
    return b


def check_budget(budget: Budget, memory_limit: int | None, adapter: str = "the adapter") -> None:
    """Raise SplatMemoryError with the breakdown and the fixes when the scene's
    estimated device memory (Budget.device_estimate) exceeds ``memory_limit``
    bytes (None: no up-front limit)."""
    if memory_limit is None or budget.device_estimate <= memory_limit:
        return
    raise SplatMemoryError(memory_error_message(budget, f"{memory_limit / MiB:.0f} MiB", adapter))


def memory_error_message(budget: Budget, limit_text: str, adapter: str) -> str:
    return (
        f"The splat scene does not fit the GPU memory available on {adapter} ({limit_text}).\n"
        f"{budget.describe()}\n"
        "Fixes:\n"
        "  1. Render fewer cameras, or smaller ones; or pass a smaller batch (cameras that\n"
        "     share sort memory), which saves 20 B per splat per camera.\n"
        "  2. Lower the SH degree of the scene (degree 0 needs 8 B of SH per splat instead of 96 B).\n"
        "  3. Decimate the splat scene in the editor.\n"
        "  4. Pass images=False if this run does not need camera images."
    )
