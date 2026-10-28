// Cull, compact, project (design doc 16 §9.2 step 3).
//
//   cull            batched, per source chunk: visibility flag per (camera, splat), visible
//                   count per workgroup
//   scan_groups     batched, one workgroup per camera: exclusive prefix of the workgroup
//                   counts, the visible count, the indirect arguments of the camera's
//                   project dispatches and draws, and the batch's sort dispatch size
//   compact         batched, per source chunk: visible splats in splat-index order ->
//                   (key, splat index)
//   project_sorted  per camera, source chunk and record chunk, after the sort: one record
//                   per sorted position, SH colour included
//
// Positions come from prefix sums, never from atomic counters, so the compacted order is
// the splat-index order on every run and every adapter. The one atomic, the sort
// dispatch size, is a maximum, which does not depend on order.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> cams: array<Camera, MAX_CAMS>;
@group(0) @binding(2) var<uniform> D: Dispatch;

@group(1) @binding(0) var<storage, read> geom: array<vec4f>;          // source chunk, 3 per splat
@group(1) @binding(1) var<storage, read> sh: array<u32>;              // source chunk, P.sh_words per splat
@group(1) @binding(2) var<storage, read_write> flags: array<u32>;     // segment * n_splats + splat
@group(1) @binding(3) var<storage, read_write> groups: array<u32>;    // segment * n_groups + workgroup
@group(1) @binding(4) var<storage, read_write> keys: array<u32>;      // segment * n_splats + position
@group(1) @binding(5) var<storage, read_write> vals: array<u32>;
@group(1) @binding(6) var<storage, read_write> state_rw: array<u32>;  // segment * ST_WORDS + ...
@group(1) @binding(7) var<storage, read> sorted_vals: array<u32>;     // project_sorted: sorted splat indices
@group(1) @binding(8) var<storage, read> state: array<u32>;           // project_sorted: counts and its args
@group(1) @binding(9) var<storage, read_write> recs: array<Rec>;      // project_sorted: this record chunk
@group(1) @binding(10) var<storage, read_write> sort_args: array<atomic<u32>>;   // scan_groups

// geometry of splat `local` of the bound chunk: mean.xyz, opacity | xx xy xz yy | yz zz - -
fn load_projection(cam: Camera, local: u32) -> Proj {
  let g0 = geom[3u * local];
  let g1 = geom[3u * local + 1u];
  let g2 = geom[3u * local + 2u];
  return project_splat(cam, P, g0.xyz, g1.xyz, vec3f(g1.w, g2.x, g2.y), g0.w);
}

@compute @workgroup_size(256)
fn cull(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let seg = wid.z;
  let local = gid.x;
  var f = 0u;
  if (local < D.len) {
    if (load_projection(cams[D.cam0 + seg], local).ok) {
      f = 1u;
    }
    flags[seg * P.n_splats + D.start + local] = f;
  }
  _ = wg_inclusive_scan(lid.x, f);
  if (lid.x == 0u) {
    groups[seg * P.n_groups + D.group0 + wid.x] = wg_scan_total();
  }
}

@compute @workgroup_size(256)
fn scan_groups(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let seg = wid.z;
  let row = seg * P.n_groups;
  // each invocation owns a contiguous run of workgroup counts
  let n = P.n_groups;
  let per = (n + WG - 1u) / WG;
  let begin = min(lid.x * per, n);
  let end = min(begin + per, n);
  var sum = 0u;
  for (var i = begin; i < end; i = i + 1u) {
    sum = sum + groups[row + i];
  }
  let incl = wg_inclusive_scan(lid.x, sum);
  var run = incl - sum;
  for (var i = begin; i < end; i = i + 1u) {
    let c = groups[row + i];
    groups[row + i] = run;
    run = run + c;
  }
  if (lid.x == 0u) {
    let count = wg_scan_total();
    let st = seg * ST_WORDS;
    state_rw[st + ST_COUNT] = count;
    atomicMax(&sort_args[D.args], (count + SORT_TILE - 1u) / SORT_TILE);
    for (var r = 0u; r < MAX_REC_CHUNKS; r = r + 1u) {
      var n_r = 0u;
      let first = r * P.rec_chunk;
      if (r < P.n_rec_chunks && count > first) {
        n_r = min(count - first, P.rec_chunk);
      }
      let pa = st + ST_PROJECT_ARGS + 4u * r;
      state_rw[pa] = (n_r + WG - 1u) / WG;
      state_rw[pa + 1u] = 1u;
      state_rw[pa + 2u] = 1u;
      state_rw[pa + 3u] = 0u;
      let da = st + ST_DRAW_ARGS + 8u * r;
      state_rw[da] = 6u * n_r;   // index count: 6 indices per quad
      state_rw[da + 1u] = 1u;    // instance count
      state_rw[da + 2u] = 0u;    // first index
      state_rw[da + 3u] = 0u;    // base vertex
      state_rw[da + 4u] = 0u;    // first instance
    }
  }
}

@compute @workgroup_size(256)
fn compact(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u,
           @builtin(workgroup_id) wid: vec3u) {
  let seg = wid.z;
  let local = gid.x;
  var f = 0u;
  if (local < D.len) {
    f = flags[seg * P.n_splats + D.start + local];
  }
  let incl = wg_inclusive_scan(lid.x, f);
  if (f == 1u) {
    let pos = seg * P.n_splats + groups[seg * P.n_groups + D.group0 + wid.x] + incl - 1u;
    let d = geom[3u * local].xyz - cams[D.cam0 + seg].pos.xyz;
    keys[pos] = sort_key(length(d));
    vals[pos] = D.start + local;
  }
}

fn sh_value(base: u32, h: u32) -> f32 {
  let pair = unpack2x16float(sh[base + (h >> 1u)]);
  return select(pair.x, pair.y, (h & 1u) == 1u);
}

fn sh_coeff(base: u32, k: u32) -> vec3f {
  return vec3f(sh_value(base, 3u * k), sh_value(base, 3u * k + 1u), sh_value(base, 3u * k + 2u));
}

// 3DGS colour (degree <= 3), the same basis and constants as Spark's evaluatePackedSH*.
// The base colour is clamped before the view-dependent part is added, as Spark stores
// it in 8 bits first; the sum is clamped again.
fn sh_color(local: u32, dir: vec3f) -> vec3f {
  let base = local * P.sh_words;
  var rgb = clamp(0.5 + SH_C0 * sh_coeff(base, 0u), vec3f(0.0), vec3f(1.0));
  if (P.n_coeffs > 1u) {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    var v = -0.4886025 * y * sh_coeff(base, 1u) + 0.4886025 * z * sh_coeff(base, 2u)
            - 0.4886025 * x * sh_coeff(base, 3u);
    if (P.n_coeffs > 4u) {
      let xx = x * x;
      let yy = y * y;
      let zz = z * z;
      v = v + 1.0925484 * x * y * sh_coeff(base, 4u) - 1.0925484 * y * z * sh_coeff(base, 5u)
            + 0.3153915 * (2.0 * zz - xx - yy) * sh_coeff(base, 6u)
            - 1.0925484 * x * z * sh_coeff(base, 7u) + 0.5462742 * (xx - yy) * sh_coeff(base, 8u);
      if (P.n_coeffs > 9u) {
        v = v - 0.5900436 * y * (3.0 * xx - yy) * sh_coeff(base, 9u)
              + 2.8906114 * x * y * z * sh_coeff(base, 10u)
              - 0.4570458 * y * (4.0 * zz - xx - yy) * sh_coeff(base, 11u)
              + 0.3731763 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeff(base, 12u)
              - 0.4570458 * x * (4.0 * zz - xx - yy) * sh_coeff(base, 13u)
              + 1.4453057 * z * (xx - yy) * sh_coeff(base, 14u)
              - 0.5900436 * x * (xx - 3.0 * yy) * sh_coeff(base, 15u);
      }
    }
    rgb = rgb + v;
  }
  return clamp(rgb, vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(256)
fn project_sorted(@builtin(global_invocation_id) gid: vec3u) {
  let j = D.rec_start + gid.x;
  if (gid.x >= P.rec_chunk || j >= state[D.seg * ST_WORDS + ST_COUNT]) {
    return;
  }
  let g = sorted_vals[D.seg * P.n_splats + j];
  if (g < D.start || g >= D.start + D.len) {
    return;
  }
  let local = g - D.start;
  let cam = cams[D.cam];
  let pr = load_projection(cam, local);
  var r: Rec;
  r.center = pr.center;
  r.axis1 = pr.axis1;
  r.scale2 = pr.scale2;
  r.depth = pr.depth;
  r.alpha = select(0.0, pr.alpha, pr.ok);
  let dw = normalize(geom[3u * local].xyz - cam.pos.xyz);
  let dir = vec3f(dot(P.sh_rot0.xyz, dw), dot(P.sh_rot1.xyz, dw), dot(P.sh_rot2.xyz, dw));
  r.color = pack4x8unorm(vec4f(sh_color(local, dir), 0.0));
  recs[gid.x] = r;
}
