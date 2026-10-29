# D's pipeline (`wgsl/1`)

This directory is the only WGSL source of renderer D (design doc 16 §9). The
Python host (`world_engine/renderer/wgsl/`, which world-engine-client ships a
copy of together with these files), the browser host (`../wgslRenderer.ts`,
the editor's Policy View, M-D4) and a Node+Dawn test host all load these files
unchanged. M-D1 made the splat pass; M-D2 the
mesh pass (three.js's MeshStandardMaterial as the capture page draws it), the
key light's shadow map, and the environment prefilter (three's
PMREMGenerator); M-D3 the colour target above 8 bits and its resolve.

A pipeline version names one drawing: these shaders and what the host feeds
them (lighting, bake, materials, shadow). world-engine-client 0.7.0 is the
first release with `wgsl/1`. `tests/test_render_protocol.py` pins the shaders'
SHA-256 as it ships them, and `tests/test_wgsl_pin.py` the images they and the
Python host draw on Mesa's lavapipe. Drawing differently after that release
means a new version in `pipeline.json`, with the engine still offering this
one to the clients that have it.

## Modules

`pipeline.json` names the pipeline version and lists each module as a list of
files. A module's source is the files joined in that order; `common.wgsl` comes
first and declares no bindings.

| module | files | entry points |
|---|---|---|
| `preprocess` | common, preprocess | `cull`, `scan_groups`, `compact`, `project_sorted` |
| `sort` | common, sort | `sort_histogram`, `sort_scan`, `sort_scatter` |
| `splat` | common, splat | `vs_splat`, `fs_splat` |
| `mesh` | common, cubeuv, material, mesh | `vs_mesh`, `fs_mesh` |
| `shadow` | material, shadow | `vs_shadow` (depth only), `vs_shadow_masked` with `fs_shadow_masked` |
| `pmrem` | cubeuv, pmrem | `vs_plane` with `fs_cube`, `fs_equirect`, `fs_blur`, `fs_room` |
| `resolve` | resolve | `vs_resolve`, `fs_resolve` |

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

## Meshes

The mesh pass draws three's MeshStandardMaterial (three 0.180) the way the
capture page (`frontend/src/sceneRenderer.ts`) sets its meshes up; `mesh.wgsl`
names the shader chunks it follows. Lighting is computed in three's view space
(x right, y up, z towards the viewer), so the screen-derivative terms
(flat shading, derivative tangents, geometryRoughness) come out as in GLSL;
WGSL's `dpdy` runs down the framebuffer, so the shader negates it. Output:
Neutral tone mapping (exposure 1), then the sRGB transfer function (three's
`pow(c, 0.41666)`), written to the camera's colour target (see Colour target),
which the splats then blend over.

Vertex buffer, one interleaved stream of 80 bytes: position (f32x3, location 0),
normal (f32x3, 1; zero for flat-shaded meshes), uv (f32x2, 2), uv1 (f32x2, 3),
colour (f32x4, 4; linear, ones without COLOR_0), tangent (f32x4, 5; zero without
TANGENT), owner index (u32, 6). The Python host puts the owner's scale in the
vertices (normals divided by it, triangles reversed for a mirroring scale, as
three flips the front face) and the owner array holds the rest of the world
matrix. The browser host keeps the scale in the owner's matrix, so an editor
drag that scales an object does not rebuild the vertex buffer: the inverse
transpose in the owner array divides the normals by it, and the triangles of a
mirrored owner are reversed as above. Both give the gym's image
(`tests/test_editor_policy_view.py`).

Bindings of `mesh`: group 0: `cams` (the splat pass's camera array), the
`Dispatch` slot of the camera (dynamic offset, `D.cam`), `owners` (storage: per
owner the model matrix and the inverse transpose of its 3x3), `MeshFrame`
(uniform: key light direction and colour, environment intensity, rotation and
atlas geometry, the shadow matrix and parameters, flags), the environment atlas
(rgba16float) with a linear clamp sampler, the shadow map (depth32float).
Group 1, one bind group per material (declared in `material.wgsl`, which the
`mesh` and `shadow` modules share): the `Material` uniform (base colour,
opacity, emissive, metalness, roughness, AO intensity, alpha test, normal scale,
flags, uv channel per map, three's uv matrix per map) and six textures with
their samplers (map, normal, roughness, metalness, AO, emissive; map and
emissive are rgba8unorm-srgb, the rest rgba8unorm; unused slots get a white
1x1). Hosts cull back faces (front face counter-clockwise) except for
double-sided materials, depth compare less-equal; a material from a glTF BLEND
mode is drawn opaque without depth writes (the capture page forces GLB
materials opaque, GLTFLoader's depthWrite stays false).

Draw order is three's opaque order, which decides what shows where a material
without depth writes overlaps another, and which coplanar surface wins:
materials in the order they were made (scene_init's objects in order, the
proxy boxes among them, then the robot links, then the end-effector box; a
GLB's materials in order of first use), and the draws of a material without
depth writes one primitive at a time, per camera, by the depth of the
primitive's centre along the view direction, nearest first (three's z; WEM1
records the centre GLTFLoader's bounding sphere has; the browser host scales
the centre by the owner's scale of that frame, as the Python host's baked
vertices are). The index buffer is laid out for the shadow pass instead
(casters by culling mode, then material), so the camera pass draws each
material's range out of that order.

Shadow map (`shadow.wgsl`): one per frame, before the cameras, shared by all of
them. three's DirectionalLightShadow with the editor's quality-tier settings:
1024^2, PCF over 17 nearest taps with radius 6 texels, bias -0.0005, normal
bias 0.02, the orthographic camera fitted to the meshes' bounds as
`editorScene._fitShadowCamera` does. Depth is GL's 0.5 ndc + 0.5 in a
depth32float target, rows in GL order (the vertex shader flips y), casters'
front faces culled. Flat-shaded meshes never receive the shadow: three's
shadow coordinate for a mesh without normals is NaN. A caster with a map and an
alpha test (glTF MASK) is drawn with `vs_shadow_masked` / `fs_shadow_masked`,
its material's bind group in group 1 (`material.wgsl`) and a vertex buffer
layout with uv and uv1: three's depth material takes the caster's map and
alphaTest, so the map's alpha alone (at the map's uv channel and transform)
below alphaTest leaves a hole in the shadow.

## Environment

The meshes are lit by a CubeUV atlas (three's PMREM layout; `cubeuv.wgsl` reads
it like `cube_uv_reflection_fragment.glsl.js`) times an intensity, rotated by
three's envMapRotation, plus one directional key light. `pmrem.wgsl` makes the
atlas the way PMREMGenerator does, each pass a viewport of 3 x 2 faces drawn
with `vs_plane`: the source into the top mip (`fs_cube` for a cube map,
`fs_equirect` for an HDRI, `fs_room` for three's RoomEnvironment ray cast at the
pixel centres), then two half blurs per mip (`fs_blur`, latitudinal then
longitudinal through a ping-pong atlas, three's weights, sample counts and pole
axes). The host picks the set-up as the capture page does
(`world_engine/renderer/wgsl/lighting.py`): a lighting asset (WELI1, the atlas
prefiltered once) when the scene has one; else the HDRI; else the bake at the
light probe; the studio room when the probe is black or dark or there are no
splats. An HDRI goes up as rgba16float the way three's HDRLoader and EXRLoader
fill HalfFloatType data (clamped to 65504, the mantissa truncated:
`lighting.half_float_like_three`), with three's flipY (row 0 at v = 1).

The bake is the capture page's: the splats alone on six 90-degree 128^2 faces
from the probe point, colours decoded with `pow(c, 2.2)` (`Params.encode_linear`)
and blended over a clear colour of (1, 1, 1) / 255 in the host's colour target
(the page's cube target has 8 bits; the host rounds to 8 bits once, at the
resolve, as for the cameras, so the adapters bake alike), the splats sized as
the page's bake renderer sizes them: in Spark's pixels of the 1280 x 720 canvas
(`Camera.rot0.w`, `rot1.w` give the scale) and capped at 256 px (the host writes
`Params.max_pixel_radius` for the bake and puts it back after), the faces read
back for the light probe and mipmapped (2x2 averages) before the prefilter.

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

Before the batches, when there are meshes: write the owners, the frame
constants and the scrolled materials (a belt's maps: three's uv matrix with the
offset `offset0 - L g travel`, mod 1, design doc 14 §7.2), and draw the shadow
map. Then per camera: a compute pass of `project_sorted`, once per (source
chunk, record chunk), dispatched indirectly from the camera's state segment; a
render pass into the camera's colour target (the mesh draws in three's order,
see Meshes, then one `drawIndexedIndirect` per record chunk with the shared
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

## Browser host

The editor's host (M-D4) is TypeScript on the browser's WebGPU, a port of the
Python host that gives the same bytes on the same GPU:

| file | does | Python host |
|---|---|---|
| `wgslSources.ts` | imports this directory raw at build time and joins each module's files | `load_pipeline` |
| `wgslBudget.ts` | the bindings split | `budget.py` |
| `wgslSplats.ts` | the PLY to the geometry and SH arrays, rounded as numpy rounds (float32 products, f16 round to nearest even); in a worker | `splat_data.py` |
| `gltfScene.ts`, `wgslMeshScene.ts` | the mesh arrays from what three's GLTFLoader parsed, so an object imported a moment ago draws | `wem1.py`, `mesh_data.py` |
| `wgslLighting.ts`, `wgslMeshPass.ts` | the environment, the shadow map and the mesh pass | `lighting.py`, `mesh_pass.py` |
| `wgslRenderer.ts` | the passes, one camera per batch, into the colour target and its resolve | `render_wgsl.py` |
| `wgslWorkers.ts`, `wgslSplatsWorker.ts`, `wgslMipsWorker.ts` | the scan's preparation and texture mip chains off the main thread, the same code as on it | |
| `wgslHost.ts` | all of the above as one chunk, which the editor loads only once it has WebGPU | |
| `wgslPipelineId.ts` | the pipeline version this host speaks, which the editor compares with the server's `features.gym_renderer` without loading the shaders | |

The editor builds the scene again whenever an item comes, goes or is renamed,
a scale mirrors, or the scan's Align changes, so the host keeps what does not
change and spreads the rest over frames: each object's GLB is read once (by
its GLTFLoader scene) and the scene is put together from those parts; each
image is decoded once (by its three Source) and mipmapped in a worker; a
texture stays on the GPU across mesh passes until no object uses it, and new
ones are uploaded one per frame; the scan's bytes go to the splat worker once,
a 16 MiB piece a frame, and a new placement is one message; splat buffers are
written about 8 MiB per frame. None of this changes a byte: the worker runs
the main thread's code. While a scene is being built the editor draws its
views with Spark, never with D's stale scene. A mechanism's parts come from the
editor's items as `scene_init` has them (`../../editor/policyParts.ts`): a
part's GLB at its part frame times its offset times its scale, grey boxes of
its collider for a part without one, belts whose maps scroll by the state's
travel.

The mesh arrays must be what WEM1 gives the Python host for the same GLB:
vertex records word for word (GLTFLoader's world matrix applied per primitive,
normals by its inverse transpose, -0 written as the +0 the Python host's
matrix products give), indices, draw ranges, material uniforms and each map
slot's image (`tests/test_editor_policy_meshes.py`). A host that builds its
arrays some other way must pass that test.

The browser host asks for a `high-performance` adapter with the default
limits and refuses a fallback adapter (SwiftShader): the editor keeps Spark
then. A test switch lets it draw on the fallback adapter, to cover the CPU
class. It picks the colour target by adapter class as the Python host does
(`defaultColorFormat`, `deviceFeatures` in `../wgslRenderer.ts`): `rgba16float`
and no optional feature on a GPU; `float32-blendable` and `rgba32float` on a
CPU adapter that has it, else `rgb10a2unorm`. The bake draws into the same
target. A lost device or an uncaptured error stops it the way a refused
adapter does. Each camera's resolve (`rgba8unorm`) is copied into its canvas
(`rgba8unorm`, alpha mode `opaque`, usage `RENDER_ATTACHMENT | COPY_SRC |
COPY_DST`), so the canvas holds the bytes a readback gives; nothing is read
back per frame.

The bake's sizes do not follow the editor's canvas: both hosts size the bake's
splats as Spark does on the capture page's 1280 x 720 canvas and cap them at
256 px (`BAKE_RENDER_SIZE`, `BAKE_MAX_PIXEL_RADIUS` in `wgslLighting.ts`).

The images both hosts must agree on are in `tests/data/<golden>/ref_wgsl*`
(made by `scripts/wgsl/make_preview_refs.py` with the Python host): the
render_state of the golden scenes, the same scene scaled unevenly and mirrored,
the scan with nothing else, and an HDRI. Their `meta.json` records the Python
host's colour target and the bake's light probe, which the editor's tests check
too. `tests/test_editor_policy_view.py` holds the editor to them for the state
the gym renders and for the editor's own state of a layout. A pipeline change
that moves the Python host's images must move this host's in the same commit,
and the references are made again with it.

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
discarded, no alpha cap (in Spark's 8-bit target such a fragment would move a
channel by less than half a step).
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

- Half-float render targets round writes towards zero on the RTX 4090 and on
  lavapipe, so a constant environment drifts down by a few half-float steps over
  the blur passes (three's PMREM on WebGL does the same).
- Spark sizes the bake's splats by its renderSize (the canvas, 1280 x 720 in the
  headless page), not by the cube target, and the page bakes through a second
  SparkRenderer with maxPixelRadius 256: SceneEnvironmentBaker reuses the
  scene's renderer only when its enableLod matches the splat mesh's LoD mode,
  and Spark 2.1.0 defaults enableLod to true. With the scene renderer's 512 px
  cap the bake's large splats come out twice too big.

- lavapipe (Mesa 25.1, LLVM 15) stores the first lane's value when one
  invocation writes its own value to an address the compiler can prove uniform,
  such as `if (lid == 255u) { out[g] = incl; }`. Kernels store workgroup totals
  through a workgroup-memory read (`wg_scan_total`) instead.
- A buffer used for indirect arguments must not be bound writable in the same
  dispatch: `preprocess.wgsl` declares the state twice, `state_rw` for
  `scan_groups` and read-only `state` for `project_sorted`.
- Kernels with barriers never return early; work past a segment's end is masked.
- D3D12 compiles these shaders with FXC (wgpu's default; DXC only when a host
  asks for it), from the HLSL naga makes of them. FXC cannot write a private
  vector or array at an index it does not know: it unrolls the loop around the
  write to make the index constant, and fails (`X3500: array reference cannot
  be used as an l-value`, `X3511: forced to unroll loop`). naga zero-initialises
  workgroup memory from one invocation, and FXC's time for that grows faster
  than the variable: keep each workgroup array small (`sort.wgsl` splits its
  1024 words into four variables) and inside a struct, which FXC stores 16
  bytes at a time. With naga 29.0.4 and d3dcompiler_47 (HLSL Shader Compiler
  10.1), 2026-10-28, the 21 entry points compile in 5.5 s, 3.0 s of it
  `sort_scatter`, which took 25 s when its 11 KiB were five word arrays.
