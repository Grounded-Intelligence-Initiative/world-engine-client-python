# D's splat pipeline (`wgsl/1`)

This directory is the only WGSL source of renderer D (design doc 16 §9). The
Python host (`world_engine/renderer/wgsl/`, which world-engine-client ships a
copy of together with these files), the browser host (M-D4) and a Node+Dawn
test host all load these files unchanged. The mesh pass here (`mesh_flat.wgsl`)
is a stand-in until M-D2's PBR shader.

## Modules

`pipeline.json` names the pipeline version and lists each module as a list of
files. A module's source is the files joined in that order; `common.wgsl` comes
first and declares no bindings.

| module | entry points |
|---|---|
| `preprocess` | `cull`, `scan_groups`, `compact`, `project_sorted` |
| `sort` | `sort_histogram`, `sort_scan`, `sort_scatter` |
| `splat` | `vs_splat`, `fs_splat` |
| `mesh_flat` | `vs_mesh`, `fs_mesh` |
| `resolve` | `vs_resolve`, `fs_resolve` |

## Limits

Hosts request the WebGPU default limits and nothing more, on every adapter, so
the buffers are split the same way everywhere. That means 256 invocations per
workgroup, 16 KiB of workgroup memory, 8 storage buffers per stage, 4 bind
groups, storage bindings of at most 128 MiB and buffers of at most 256 MiB. No
optional feature is needed: no subgroups, no `shader-f16` (SH is f16 packed in
u32 and read with `unpack2x16float`). One is used where an adapter has it: on a
CPU adapter, `float32-blendable` for an `rgba32float` colour target (see below).

## Data

| array | layout |
|---|---|
| geometry | per source chunk, 3 `vec4f` per splat: mean.xyz and opacity; covariance xx, xy, xz, yy; yz, zz, 0, 0 (world space) |
| SH | per source chunk, `Params.sh_words` u32 per splat: f16 pairs, coefficient-major (k, channel), low half first; k = 0 is the DC term |
| `cams` | uniform `array<Camera, 16>`; rows of world-to-camera rotation (x right, y down, z forward), centre, fx fy cx cy, width height near far |
| `Params` | scene constants; the defaults are Spark 2.1.0's `SparkRenderer` defaults |
| `Dispatch` | per-dispatch constants, 256-byte slots, selected with a dynamic offset |
| flags, groups, keys, values, histograms, state | per camera batch, one segment per camera of the batch |
| records | per record chunk, 32 bytes per visible splat, in back-to-front order |

`state` holds `ST_WORDS` u32 per camera: the visible count, the indirect
arguments of each record chunk's `project_sorted` dispatch and its
`drawIndexedIndirect`. `sort_args` holds 4 u32 per batch: the dispatch size of
the sort (tiles of the largest segment, 1, cameras in the batch).

## Splitting into bindings

`world_engine/renderer/wgsl/budget.py` is the reference for these rules; its
`plan_layout` and `gpu_budget` give the numbers for a scene. `Budget.total` is
the bytes of the buffers and textures; the device memory of the process is
more (the driver's base and the allocator's 256 MiB blocks), which
`Budget.device_estimate` estimates with constants measured on an RTX 4090.

- Source chunks: the splats are cut into chunks of whole workgroups (multiples of
  256) so that a chunk's geometry and SH each fit one binding. The 1.5M-splat SH3
  room is two chunks under 128 MiB.
- Camera batches: a batch's keys, values and flags are one binding of 4 B per
  splat per camera, so a batch holds at most binding limit / (4 N) cameras.
- Record chunks: the sorted records are cut into chunks of at most 2^21 records
  (and one binding); each chunk is one indirect draw.

## Upload

Upload large arrays through one reusable staging buffer, a piece at a time
(the Python host: 16 MiB), and wait for each copy before the next. Creating a
buffer with its data, or one `writeBuffer` of the whole array, keeps a second
copy of everything in mapped memory until the copies complete.

## Per frame

Write `cams`, reset each batch's `sort_args` to (0, 1, cameras, 0). Then, per
batch, in one compute pass:

1. `cull`, once per source chunk, workgroups (ceil(len / 256), 1, cameras);
2. `scan_groups`, workgroups (1, 1, cameras);
3. `compact`, once per source chunk, as `cull`;
4. four radix passes, `Dispatch.shift` 0, 8, 16, 24: `sort_histogram` and
   `sort_scatter` dispatched indirectly from `sort_args`, `sort_scan` with
   workgroups (256, 1, cameras); the pairs end in keys_a / vals_a.

Then per camera: a compute pass of `project_sorted`, once per (source chunk,
record chunk), dispatched indirectly from the camera's state segment; a render
pass (mesh pass, then one `drawIndexedIndirect` per record chunk with the shared
index buffer `4q, 4q+1, 4q+2, 4q+2, 4q+1, 4q+3`); the `resolve` pass, one
full-target triangle that reads the colour target texel by texel (an
`unfilterable-float` texture binding) into an `rgba8unorm` texture; a copy of
that texture into the shared readback buffer, rows aligned to 256 bytes. Map
once at the end.

The splat pipeline: no culling, depth test `less-equal` against the mesh depth,
no depth write, premultiplied over blending (`one`, `one-minus-src-alpha`).
Clear colour `#0b0d10`, as the capture page.

## Colour target

The mesh and splat passes draw into a target with more than 8 bits per channel:
`rgba16float` on a GPU, `rgba32float` on a CPU adapter whose device has
`float32-blendable` (else `rgb10a2unorm`; lavapipe blends `rgba16float` with a
bias of -1.1/255). `resolve.wgsl` rounds to 8 bits once, half to even with
`round()`, and writes exact multiples of 1/255, so an adapter's own float to
unorm conversion does not enter. An 8-bit target rounds every blended layer to
8 bits, and adapters round differently: on the 1.5M-splat room the RTX 4090's
8-bit result is 1.07/255 darker than exact blending and differs from
SwiftShader's by 1.77/255 in the splat region; with these targets the
adapters differ by 0.10/255. On an RTX 4090 `rgba16float` costs 1.17 times the
8-bit draw pass (0.19 ms for three 256² cameras of the room); `rgb10a2unorm`
1.45 and `rgba32float` 6.4 times. A host may draw into `rgba8unorm` directly
and skip the resolve (`color_format`), for measurements.

## Determinism

Compacted positions come from prefix sums and the sort ranks from prefix sums
and stable splits, so the order never depends on scheduling. The two atomics
are order-free: digit counts in workgroup memory, and the maximum in
`sort_args`. Two runs on one adapter give the same bytes; the sort gives the
same order on every adapter for the same keys.

## Matching Spark 2.1.0

Projection, quad and alpha follow Spark's `splatVertex.glsl` and
`splatFragment.glsl`: EWA covariance plus 0.3 px² with the opacity compensated,
an eigenvector-aligned quad of sqrt(8) standard deviations capped at 512 px,
centres beyond 1.4 times the NDC square culled, alpha below 0.5/255 culled and
discarded, no alpha cap (into an 8-bit target the fragment discard changes no
pixel: below 0.5/255 a fragment moves a channel by less than half a step).
Colour is SH up to degree 3 with Spark's constants, the base colour clamped
before the view-dependent part is added, kept in 8 bits.
Splats are sorted by distance to the camera centre (Spark's `sortRadial`).
Colours are display values, blended without tone mapping or sRGB conversion.

Spark also packs splats in 8-bit fields; the renderer keeps the PLY's precision.
`Splats.spark_packed()` reproduces that packing for comparisons.

Hosts drop splats with a non-finite position, scale, rotation, opacity or SH
coefficient before upload (`Splats.finite_mask`): WGSL lets an implementation
assume there is no NaN or Inf.

## Pitfalls

- lavapipe (Mesa 25.1, LLVM 15) stores the first lane's value when one
  invocation writes its own value to an address the compiler can prove uniform,
  such as `if (lid == 255u) { out[g] = incl; }`. Kernels store workgroup totals
  through a workgroup-memory read (`wg_scan_total`) instead.
- A buffer used for indirect arguments must not be bound writable in the same
  dispatch: `preprocess.wgsl` declares the state twice, `state_rw` for
  `scan_groups` and read-only `state` for `project_sorted`.
- Kernels with barriers never return early; work past a segment's end is masked.
