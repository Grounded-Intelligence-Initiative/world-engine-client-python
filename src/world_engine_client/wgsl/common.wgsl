// D's splat pipeline, shared part (pipeline "wgsl/1", design doc 16 §9.2).
//
// Every module of the pipeline is this file followed by one stage file
// (pipeline.json lists the modules). Hosts concatenate the text as is; there is
// no #include. Nothing here declares a binding, so the stage files own their
// bind group layouts. Sizes stay inside the WebGPU default limits: 256
// invocations per workgroup, 16 KiB of workgroup memory, storage bindings of at
// most 128 MiB (hosts split larger arrays into chunks, see README.md).
//
// Cameras are processed in batches. Within a batch, camera `seg` (0-based)
// owns segment `seg` of every per-camera array (flags, workgroup counts,
// keys, values, sort histograms, state); batched kernels take `seg` from
// workgroup_id.z, per-camera kernels from Dispatch.seg.

const WG: u32 = 256u;            // invocations per workgroup, every kernel
const SORT_ITEMS: u32 = 4u;      // keys per invocation in the radix sort
const SORT_TILE: u32 = 1024u;    // WG * SORT_ITEMS keys per sort workgroup
const RADIX: u32 = 256u;         // 8-bit digits; 4 passes sort a 32-bit key
const MAX_CAMS: u32 = 16u;       // cameras per renderer (uniform array size)
const MAX_REC_CHUNKS: u32 = 16u; // record chunks per camera (state buffer layout)

// state segment, one per camera of a batch (u32 words), written by scan_groups.
const ST_COUNT: u32 = 0u;        // visible splats of the camera
const ST_PROJECT_ARGS: u32 = 8u; // + 4 * r: dispatch args of project_sorted for record chunk r
const ST_DRAW_ARGS: u32 = 72u;   // + 8 * r: drawIndexedIndirect args for record chunk r
const ST_WORDS: u32 = 200u;

const SH_C0: f32 = 0.28209479177387814;

// One camera, image convention: x right, y down, z forward (OpenCV).
struct Camera {
  rot0: vec4f,   // rows of the world -> camera rotation (xyz)
  rot1: vec4f,
  rot2: vec4f,
  pos: vec4f,    // camera centre in the world (xyz)
  intr: vec4f,   // fx, fy, cx, cy in pixels
  view: vec4f,   // width, height, near, far
};

// Scene-wide constants. Defaults are Spark 2.1.0's SparkRenderer defaults,
// the renderer the capture page and the editor use.
struct Params {
  n_splats: u32,
  n_coeffs: u32,          // SH coefficients per colour channel: 1, 4, 9 or 16
  sh_words: u32,          // u32 words of packed f16 SH per splat
  n_groups: u32,          // ceil(n_splats / WG), over all source chunks
  max_std_dev: f32,       // quad half-size in standard deviations (Spark: sqrt(8))
  max_pixel_radius: f32,  // quad half-size cap in pixels (Spark: 512)
  min_pixel_radius: f32,  // splats smaller than this on both axes are culled (Spark: 0)
  min_alpha: f32,         // cull / discard below this alpha (Spark: 0.5 / 255)
  blur: f32,              // px^2 added to the 2D covariance, alpha compensated (Spark: 0.3)
  pre_blur: f32,          // px^2 added before the compensation (Spark: 0)
  clip_xy: f32,           // centre must lie within clip_xy times the NDC square (Spark: 1.4)
  rec_chunk: u32,         // records per record chunk
  n_rec_chunks: u32,      // record chunks per camera slot
  tiles_cap: u32,         // ceil(capacity / SORT_TILE): row stride of the sort histogram
  _pad0: u32,
  _pad1: u32,
  sh_rot0: vec4f,         // rows of world -> splat-object rotation, for SH view directions
  sh_rot1: vec4f,
  sh_rot2: vec4f,
};

// Per-dispatch constants, selected with a dynamic uniform offset.
struct Dispatch {
  start: u32,       // first splat of the source chunk
  len: u32,         // splats in the source chunk
  group0: u32,      // start / WG: first cull workgroup of the chunk
  rec_start: u32,   // first sorted position of the record chunk
  shift: u32,       // sort pass: digit = (key >> shift) & 255
  cam0: u32,        // batched kernels: camera index of segment 0 of the batch
  cam: u32,         // per-camera kernels and draws: camera index
  seg: u32,         // per-camera kernels and draws: segment of that camera in its batch
  args: u32,        // batched kernels: word offset of the batch's sort dispatch args
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

// One visible splat after projection, 32 bytes, in back-to-front order.
struct Rec {
  center: vec2f,    // pixels
  axis1: vec2f,     // major axis: unit eigenvector * half-size (pixels)
  scale2: f32,      // half-size along the minor axis (pixels)
  depth: f32,       // [0, 1] depth of the centre, same mapping as the mesh pass
  color: u32,       // rgb as unorm8 (Spark keeps 8 bits per channel), a unused
  alpha: f32,       // opacity times the blur compensation
};

struct Proj {
  ok: bool,
  center: vec2f,
  axis1: vec2f,
  scale2: f32,
  depth: f32,
  alpha: f32,
};

// [0, 1] depth of a point at distance tz in front of the camera; the mesh pass
// writes the same value, so splats depth-test against meshes.
fn ndc_depth(tz: f32, near: f32, far: f32) -> f32 {
  return far * (tz - near) / (tz * (far - near));
}

// Sort key: farther splats get smaller keys, so an ascending sort is
// back-to-front. Spark sorts by distance to the camera centre (sortRadial).
fn sort_key(dist: f32) -> u32 {
  return 0xffffffffu - bitcast<u32>(dist);
}

// Spark's projection (splatVertex.glsl of Spark 2.1.0): EWA 2D covariance with
// the blur added and the opacity compensated, an eigenvector-aligned quad of
// max_std_dev standard deviations, capped at max_pixel_radius. Culls like
// Spark: behind or outside near/far, centre beyond clip_xy, alpha below
// min_alpha before and after the compensation. Image space is y-down, so the
// minor-axis direction is mirrored relative to Spark's NDC; the quad covers
// the same pixels.
fn project_splat(cam: Camera, P: Params, mean: vec3f, c0: vec3f, c1: vec3f, opacity: f32) -> Proj {
  var p: Proj;
  p.ok = false;
  if (opacity < P.min_alpha) {
    return p;
  }
  let d = mean - cam.pos.xyz;
  let t = vec3f(dot(cam.rot0.xyz, d), dot(cam.rot1.xyz, d), dot(cam.rot2.xyz, d));
  let near = cam.view.z;
  let far = cam.view.w;
  if (!(t.z > near && t.z < far)) {
    return p;
  }
  let fx = cam.intr.x;
  let fy = cam.intr.y;
  let iz = 1.0 / t.z;
  let u = fx * t.x * iz + cam.intr.z;
  let v = fy * t.y * iz + cam.intr.w;
  if (abs(u - cam.intr.z) > P.clip_xy * 0.5 * cam.view.x || abs(v - cam.intr.w) > P.clip_xy * 0.5 * cam.view.y) {
    return p;
  }
  // rows of J * W (camera Jacobian times world -> camera rotation)
  let j0 = (fx * iz) * cam.rot0.xyz - (fx * t.x * iz * iz) * cam.rot2.xyz;
  let j1 = (fy * iz) * cam.rot1.xyz - (fy * t.y * iz * iz) * cam.rot2.xyz;
  // world covariance: c0 = (xx, xy, xz), c1 = (yy, yz, zz)
  let s0 = vec3f(c0.x, c0.y, c0.z);
  let s1 = vec3f(c0.y, c1.x, c1.y);
  let s2 = vec3f(c0.z, c1.y, c1.z);
  let sj0 = vec3f(dot(s0, j0), dot(s1, j0), dot(s2, j0));
  let sj1 = vec3f(dot(s0, j1), dot(s1, j1), dot(s2, j1));
  var a = dot(j0, sj0) + P.pre_blur;
  let b = dot(j0, sj1);
  var e = dot(j1, sj1) + P.pre_blur;
  let det_orig = a * e - b * b;
  a = a + P.blur;
  e = e + P.blur;
  let det = a * e - b * b;
  let alpha = opacity * sqrt(max(0.0, det_orig / det));
  if (alpha < P.min_alpha) {
    return p;
  }
  let mid = 0.5 * (a + e);
  let delta = sqrt(max(0.0, mid * mid - det));
  let eig1 = mid + delta;
  let eig2 = mid - delta;
  var ev1 = vec2f(0.0, 1.0);
  if (abs(b) > 0.001) {
    ev1 = normalize(vec2f(b, eig1 - a));
  } else if (a >= e) {
    ev1 = vec2f(1.0, 0.0);
  }
  let scale1 = min(P.max_pixel_radius, P.max_std_dev * sqrt(eig1));
  let scale2 = min(P.max_pixel_radius, P.max_std_dev * sqrt(max(eig2, 0.0)));
  if (scale1 < P.min_pixel_radius && scale2 < P.min_pixel_radius) {
    return p;
  }
  p.ok = true;
  p.center = vec2f(u, v);
  p.axis1 = ev1 * scale1;
  p.scale2 = scale2;
  p.depth = ndc_depth(t.z, near, far);
  p.alpha = alpha;
  return p;
}

// Workgroup prefix sums (Hillis-Steele over the 256 invocations). Call from
// uniform control flow; every invocation must call it.
var<workgroup> wg_scan_buf: array<u32, 256>;

fn wg_inclusive_scan(lid: u32, value: u32) -> u32 {
  wg_scan_buf[lid] = value;
  workgroupBarrier();
  var acc = value;
  for (var off = 1u; off < WG; off = off << 1u) {
    var add = 0u;
    if (lid >= off) {
      add = wg_scan_buf[lid - off];
    }
    workgroupBarrier();
    acc = acc + add;
    wg_scan_buf[lid] = acc;
    workgroupBarrier();
  }
  return acc;
}

// Total of the last wg_inclusive_scan. Callers that scan again must put a
// workgroupBarrier() between this read and the next scan.
//
// Store a workgroup's total through this read, not from one invocation's own
// scan result: lavapipe (Mesa 25.1, LLVM 15) stores the value of the first
// lane of the 8-wide vector when one invocation stores its own value to an
// address the compiler can prove uniform, e.g. `if (lid == 255u) { out[g] = incl; }`
// writes the result of invocation 248. A value read from workgroup memory at a
// uniform address is uniform, so that store is right everywhere.
fn wg_scan_total() -> u32 {
  return wg_scan_buf[WG - 1u];
}
