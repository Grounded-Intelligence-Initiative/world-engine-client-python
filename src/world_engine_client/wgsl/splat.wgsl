// Splat draw, one camera (design doc 16 §9.2 step 6).
//
// One non-instanced indexed draw per record chunk, drawIndexedIndirect with the
// arguments scan_groups wrote. The shared index buffer holds 6 indices per quad
// (4q, 4q+1, 4q+2, 4q+2, 4q+1, 4q+3), so vertex_index / 4 is the sorted record
// and vertex_index % 4 the corner. Records are back-to-front; blending is
// premultiplied "over", depth-tested against the mesh pass, no depth write.
// Colours go to an rgba8unorm target as they are: Spark's splat colours skip
// tone mapping and blend in display (sRGB-encoded) values.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> cams: array<Camera, MAX_CAMS>;
@group(0) @binding(2) var<storage, read> recs: array<Rec>;   // this record chunk
@group(0) @binding(3) var<uniform> D: Dispatch;              // D.cam: the camera drawn

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,                            // corner * max_std_dev
  @location(1) @interpolate(flat) color: vec4f,
};

// Corner corner_id (0..3: x from bit 0, y from bit 1) of the quad of sorted record `rec`.
// vs_splat is the product entry point; the draw A/B bench (scripts/wgsl/draw_ab.wgsl)
// adds instanced and 6-vertex entry points over the same function.
fn splat_vertex(rec: u32, corner_id: u32) -> VOut {
  let r = recs[rec];
  let corner = vec2f(f32(corner_id & 1u) * 2.0 - 1.0, f32((corner_id >> 1u) & 1u) * 2.0 - 1.0);
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);   // outside the clip volume
  out.uv = vec2f(0.0);
  out.color = vec4f(0.0);
  let cam = cams[D.cam];
  let s1 = length(r.axis1);
  if (r.alpha <= 0.0 || s1 <= 0.0) {
    return out;
  }
  let e1 = r.axis1 / s1;
  let axis2 = vec2f(e1.y, -e1.x) * r.scale2;
  let px = r.center + corner.x * r.axis1 + corner.y * axis2;
  out.pos = vec4f(px.x / cam.view.x * 2.0 - 1.0, 1.0 - px.y / cam.view.y * 2.0, r.depth, 1.0);
  out.uv = corner * P.max_std_dev;
  out.color = vec4f(unpack4x8unorm(r.color).rgb, r.alpha);
  return out;
}

@vertex
fn vs_splat(@builtin(vertex_index) vid: u32) -> VOut {
  return splat_vertex(vid >> 2u, vid & 3u);
}

@fragment
fn fs_splat(in: VOut) -> @location(0) vec4f {
  let z2 = dot(in.uv, in.uv);
  if (z2 > P.max_std_dev * P.max_std_dev) {
    discard;
  }
  let a = in.color.a * exp(-0.5 * z2);
  if (a < P.min_alpha) {
    discard;
  }
  return vec4f(in.color.rgb * a, a);
}
