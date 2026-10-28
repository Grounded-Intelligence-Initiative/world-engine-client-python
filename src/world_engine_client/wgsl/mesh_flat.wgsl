// Placeholder mesh pass until M-D2's mesh.wgsl (PBR, IBL, shadows): flat
// Lambert plus ambient, the D prototype's mesh shader. It exists so the splat
// pass has something to depth-test against and hosts can produce whole
// observation images while M-D1 is measured. Opaque, writes depth with the
// same mapping as the splat centres (ndc_depth). Hosts cull back faces with
// front face = counter-clockwise, as three.js does for a FrontSide material.

@group(0) @binding(0) var<uniform> P: Params;          // unused; shared layout with splat.wgsl
@group(0) @binding(1) var<uniform> cams: array<Camera, MAX_CAMS>;
@group(0) @binding(2) var<storage, read> owners: array<mat4x4f>;   // body / link world transforms
@group(0) @binding(3) var<uniform> D: Dispatch;                     // D.cam: the camera drawn

struct MIn {
  @location(0) p: vec3f,
  @location(1) nrm: vec3f,
  @location(2) col: vec4f,
  @location(3) owner: u32,
};

struct MOut {
  @builtin(position) pos: vec4f,
  @location(0) nrm: vec3f,
  @location(1) col: vec4f,
};

@vertex
fn vs_mesh(v: MIn) -> MOut {
  let cam = cams[D.cam];
  let m = owners[v.owner];
  let w = (m * vec4f(v.p, 1.0)).xyz;
  let d = w - cam.pos.xyz;
  let t = vec3f(dot(cam.rot0.xyz, d), dot(cam.rot1.xyz, d), dot(cam.rot2.xyz, d));
  let near = cam.view.z;
  let far = cam.view.w;
  let xn = (cam.intr.x * t.x / t.z + cam.intr.z) / cam.view.x * 2.0 - 1.0;
  let yn = 1.0 - (cam.intr.y * t.y / t.z + cam.intr.w) / cam.view.y * 2.0;
  var o: MOut;
  // clip-space w = t.z, so z / w = ndc_depth(t.z, near, far)
  o.pos = vec4f(xn * t.z, yn * t.z, far * (t.z - near) / (far - near), t.z);
  o.nrm = (m * vec4f(v.nrm, 0.0)).xyz;
  o.col = v.col;
  return o;
}

@fragment
fn fs_mesh(i: MOut) -> @location(0) vec4f {
  let l = normalize(vec3f(2.0, -2.0, 4.0));
  let n = normalize(i.nrm);
  let k = 0.38 + 0.62 * max(0.0, dot(n, l));
  return vec4f(i.col.rgb * k, 1.0);
}
