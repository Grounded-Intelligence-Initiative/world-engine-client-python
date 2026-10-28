// Environment prefilter (pipeline "wgsl/1", design doc 16 §9.2 step 8): three's
// PMREMGenerator (three 0.180) in WGSL, writing the CubeUV atlas the mesh pass
// reads (cubeuv.wgsl). Used when a scene comes without a prefiltered lighting
// asset: the host renders the splats into a cube map at the light probe (the
// capture page's bake), or loads an HDRI, or draws three's RoomEnvironment
// (the studio fallback), and these passes do the rest the way three does:
//
// * fs_cube, fs_equirect: the source into the atlas's top mip (_textureToCubeUV
//   with the cube or equirectangular material; lod plane 0 with its one-texel
//   border);
// * fs_room: RoomEnvironment drawn straight into the top mip, as
//   _sceneToCubeUV does (90-degree face cameras over the whole face, no border);
// * fs_blur: one half of the separable spherical Gaussian (_halfBlur): along
//   latitudes about the pole axis, then along longitudes, from one mip into
//   the next, through a ping-pong atlas.
//
// A pass draws vs_plane (36 vertices: the six face quads of _createPlanes)
// into a viewport of 3 x 2 faces. Each pass's constants are one 256-byte slot
// of a uniform buffer, picked with a dynamic offset. WebGL puts NDC y = -1 at
// the bottom of the viewport, which is the first row in memory; WebGPU puts it
// at the top, so vs_plane flips y and the atlas rows come out in three's order.

struct Pmrem {
  geo: CubeUV,             // atlas being read (texel sizes, max mip)
  plane: vec4f,            // x: border texel 1 / (size - 2) of the plane drawn, y: face size of that mip
  blur: vec4f,             // x: dTheta, y: samples, z: latitudinal (1) or longitudinal (0), w: mipInt read
  pole: vec4f,             // pole axis
  weights: array<vec4f, 5>,
  room: vec4f,             // fs_room: x: face size (pixels), y: near plane
};

@group(0) @binding(0) var<uniform> PM: Pmrem;
@group(0) @binding(1) var src_atlas: texture_2d<f32>;
@group(0) @binding(2) var src_cube: texture_cube<f32>;
@group(0) @binding(3) var src_equirect: texture_2d<f32>;
@group(0) @binding(4) var src_smp: sampler;

struct PlaneOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) face: u32,
};

@vertex
fn vs_plane(@builtin(vertex_index) vi: u32) -> PlaneOut {
  let face = vi / 6u;
  let k = vi % 6u;
  let x = f32(face % 3u) * 2.0 / 3.0 - 1.0;
  let y = select(-1.0, 0.0, face > 2u);
  // _createPlanes: (x, y), (x + 2/3, y), (x + 2/3, y + 1), (x, y), (x + 2/3, y + 1), (x, y + 1)
  var dx = array<f32, 6>(0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0, 2.0 / 3.0, 0.0);
  var dy = array<f32, 6>(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);
  let lo = -PM.plane.x;
  let hi = 1.0 + PM.plane.x;
  var out: PlaneOut;
  out.pos = vec4f(x + dx[k], -(y + dy[k]), 0.0, 1.0);
  out.uv = vec2f(select(lo, hi, dx[k] > 0.0), select(lo, hi, dy[k] > 0.0));
  out.face = face;
  return out;
}

// _getCommonVertexShader getDirection: PMREM's face convention (right-handed).
fn pmrem_direction(uv_in: vec2f, face: u32) -> vec3f {
  let uv = 2.0 * uv_in - 1.0;
  var d = vec3f(uv, 1.0);
  if (face == 0u) {
    d = d.zyx;
  } else if (face == 1u) {
    d = d.xzy;
    d = vec3f(-d.x, d.y, -d.z);
  } else if (face == 2u) {
    d = vec3f(-d.x, d.y, d.z);
  } else if (face == 3u) {
    d = d.zyx;
    d = vec3f(-d.x, d.y, -d.z);
  } else if (face == 4u) {
    d = d.xzy;
    d = vec3f(-d.x, -d.y, d.z);
  } else {
    d = vec3f(d.x, d.y, -d.z);
  }
  return d;
}

// _getCubemapMaterial (flipEnvMap = 1: the source is a rendered cube map)
@fragment
fn fs_cube(i: PlaneOut) -> @location(0) vec4f {
  let d = pmrem_direction(i.uv, i.face);
  return textureSample(src_cube, src_smp, d);
}

// _getEquirectMaterial: equirectUv of the normalised direction
@fragment
fn fs_equirect(i: PlaneOut) -> @location(0) vec4f {
  let d = normalize(pmrem_direction(i.uv, i.face));
  let u = atan2(d.z, d.x) * 0.15915494309189535 + 0.5;
  let v = asin(clamp(d.y, -1.0, 1.0)) * 0.3183098861837907 + 0.5;
  return vec4f(textureSample(src_equirect, src_smp, vec2f(u, v)).rgb, 1.0);
}

fn blur_sample(theta: f32, axis: vec3f, out_dir: vec3f) -> vec3f {
  let ct = cos(theta);
  let d = out_dir * ct + cross(axis, out_dir) * sin(theta) + axis * dot(axis, out_dir) * (1.0 - ct);
  return cubeuv_bilinear(src_atlas, src_smp, PM.geo, d, PM.blur.w);
}

fn blur_weight(k: u32) -> f32 {
  let v = PM.weights[k / 4u];
  let j = k % 4u;
  return select(select(select(v.w, v.z, j == 2u), v.y, j == 1u), v.x, j == 0u);
}

// _getBlurShader (SphericalGaussianBlur)
@fragment
fn fs_blur(i: PlaneOut) -> @location(0) vec4f {
  let out_dir = pmrem_direction(i.uv, i.face);
  var axis = PM.pole.xyz;
  if (PM.blur.z < 0.5) {
    axis = cross(PM.pole.xyz, out_dir);
  }
  if (all(axis == vec3f(0.0))) {
    axis = vec3f(out_dir.z, 0.0, -out_dir.x);
  }
  axis = normalize(axis);
  var c = blur_weight(0u) * blur_sample(0.0, axis, out_dir);
  let samples = u32(PM.blur.y);
  for (var k = 1u; k < 20u; k = k + 1u) {
    if (k >= samples) {
      break;
    }
    let theta = PM.blur.x * f32(k);
    c = c + blur_weight(k) * blur_sample(-1.0 * theta, axis, out_dir);
    c = c + blur_weight(k) * blur_sample(theta, axis, out_dir);
  }
  return vec4f(c, 1.0);
}

// ---- RoomEnvironment (three/examples/jsm/environments/RoomEnvironment.js), the
// capture page's studio fallback, ray cast instead of rasterised (the same
// nearest surface at every pixel centre). Boxes: centre, half size, rotation
// about +y (the scene's instanced boxes), and what they are: the room (seen from
// inside), a furniture box, or an emissive area light (Lambert, black colour,
// emissive white times the intensity). Room and furniture are white
// MeshStandardMaterial (roughness 1, metalness 0) lit by one point light
// (900 cd at (0.418, 16.199, 0.300), distance 28, decay 2, no shadows).

struct RoomBox {
  c: vec3f,
  h: vec3f,
  ry: f32,
  kind: f32,   // 0 room, 1 box, > 1: area light with this emissive intensity
};

fn room_box(i: u32) -> RoomBox {
  var b: RoomBox;
  switch i {
    case 0u: { b = RoomBox(vec3f(-0.757, 13.219, 0.717), vec3f(31.713, 28.305, 28.591) * 0.5, 0.0, 0.0); }
    case 1u: { b = RoomBox(vec3f(-10.906, 2.009, 1.846), vec3f(2.328, 7.905, 4.651) * 0.5, -0.195, 1.0); }
    case 2u: { b = RoomBox(vec3f(-5.607, -0.754, -0.758), vec3f(1.970, 1.534, 3.955) * 0.5, 0.994, 1.0); }
    case 3u: { b = RoomBox(vec3f(6.167, 0.857, 7.803), vec3f(3.927, 6.285, 3.687) * 0.5, 0.561, 1.0); }
    case 4u: { b = RoomBox(vec3f(-2.017, 0.018, 6.124), vec3f(2.002, 4.566, 2.064) * 0.5, 0.333, 1.0); }
    case 5u: { b = RoomBox(vec3f(2.291, -0.756, -2.621), vec3f(1.546, 1.552, 1.496) * 0.5, -0.286, 1.0); }
    case 6u: { b = RoomBox(vec3f(-2.193, -0.369, -5.547), vec3f(3.875, 3.487, 2.986) * 0.5, 0.516, 1.0); }
    case 7u: { b = RoomBox(vec3f(-16.116, 14.37, 8.208), vec3f(0.1, 2.428, 2.739) * 0.5, 0.0, 50.0); }
    case 8u: { b = RoomBox(vec3f(-16.109, 18.021, -8.207), vec3f(0.1, 2.425, 2.751) * 0.5, 0.0, 50.0); }
    case 9u: { b = RoomBox(vec3f(14.904, 12.198, -1.832), vec3f(0.15, 4.265, 6.331) * 0.5, 0.0, 17.0); }
    case 10u: { b = RoomBox(vec3f(-0.462, 8.89, 14.520), vec3f(4.38, 5.441, 0.088) * 0.5, 0.0, 43.0); }
    case 11u: { b = RoomBox(vec3f(3.235, 11.486, -12.541), vec3f(2.5, 2.0, 0.1) * 0.5, 0.0, 20.0); }
    default: { b = RoomBox(vec3f(0.0, 20.0, 0.0), vec3f(1.0, 0.1, 1.0) * 0.5, 0.0, 100.0); }
  }
  return b;
}

// Ray (origin, unit direction) against one box: (t, normal); t < 0 when missed.
// Inside the room box the far face is the one seen (BackSide).
fn room_hit(b: RoomBox, o: vec3f, d: vec3f) -> vec4f {
  // to the box frame: rotation about y by -ry (three's Euler (0, ry, 0))
  let cr = cos(b.ry);
  let sr = sin(b.ry);
  let po = o - b.c;
  let lo = vec3f(cr * po.x - sr * po.z, po.y, sr * po.x + cr * po.z);
  let ld = vec3f(cr * d.x - sr * d.z, d.y, sr * d.x + cr * d.z);
  let inv = 1.0 / ld;
  let t0 = (-b.h - lo) * inv;
  let t1 = (b.h - lo) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tn = max(max(tmin.x, tmin.y), tmin.z);
  let tf = min(min(tmax.x, tmax.y), tmax.z);
  if (tn > tf || tf <= 0.0) {
    return vec4f(-1.0, 0.0, 0.0, 0.0);
  }
  var t = tn;
  var nl = vec3f(0.0);
  if (b.kind == 0.0) {
    // inside: the exit face, normal pointing into the room (BackSide flips it)
    t = tf;
    if (tf == tmax.x) {
      nl = vec3f(-sign(ld.x), 0.0, 0.0);
    } else if (tf == tmax.y) {
      nl = vec3f(0.0, -sign(ld.y), 0.0);
    } else {
      nl = vec3f(0.0, 0.0, -sign(ld.z));
    }
  } else {
    if (tn <= 0.0) {
      return vec4f(-1.0, 0.0, 0.0, 0.0);
    }
    if (tn == tmin.x) {
      nl = vec3f(-sign(ld.x), 0.0, 0.0);
    } else if (tn == tmin.y) {
      nl = vec3f(0.0, -sign(ld.y), 0.0);
    } else {
      nl = vec3f(0.0, 0.0, -sign(ld.z));
    }
  }
  // back to the world frame
  let nw = vec3f(cr * nl.x + sr * nl.z, nl.y, -sr * nl.x + cr * nl.z);
  return vec4f(t, nw);
}

fn room_brdf_ggx(l: vec3f, v: vec3f, n: vec3f) -> f32 {
  // roughness 1, F0 0.04, F90 1 (BRDF_GGX of lights_physical_pars_fragment)
  let alpha = 1.0;
  let h = normalize(l + v);
  let dot_nl = saturate(dot(n, l));
  let dot_nv = saturate(dot(n, v));
  let dot_nh = saturate(dot(n, h));
  let dot_vh = saturate(dot(v, h));
  let fresnel = exp2((-5.55473 * dot_vh - 6.98316) * dot_vh);
  let fr = 0.04 * (1.0 - fresnel) + fresnel;
  let a2 = alpha * alpha;
  let gv = dot_nl * sqrt(a2 + (1.0 - a2) * dot_nv * dot_nv);
  let gl = dot_nv * sqrt(a2 + (1.0 - a2) * dot_nl * dot_nl);
  let vis = 0.5 / max(gv + gl, 1e-6);
  let denom = dot_nh * dot_nh * (a2 - 1.0) + 1.0;
  let dd = 0.3183098861837907 * a2 / (denom * denom);
  return fr * vis * dd;
}

// _sceneToCubeUV's camera for PMREM face `face`: 90 degrees over the whole face.
fn room_view_dir(face: u32, px: vec2f) -> vec3f {
  var fwd = vec3f(1.0, 0.0, 0.0);
  var up = vec3f(0.0, 1.0, 0.0);
  switch face {
    case 0u: { fwd = vec3f(1.0, 0.0, 0.0); up = vec3f(0.0, 1.0, 0.0); }
    case 1u: { fwd = vec3f(0.0, 1.0, 0.0); up = vec3f(0.0, 0.0, -1.0); }
    case 2u: { fwd = vec3f(0.0, 0.0, 1.0); up = vec3f(0.0, 1.0, 0.0); }
    case 3u: { fwd = vec3f(-1.0, 0.0, 0.0); up = vec3f(0.0, 1.0, 0.0); }
    case 4u: { fwd = vec3f(0.0, -1.0, 0.0); up = vec3f(0.0, 0.0, 1.0); }
    default: { fwd = vec3f(0.0, 0.0, -1.0); up = vec3f(0.0, 1.0, 0.0); }
  }
  // Matrix4.lookAt(eye, target, up): z = eye - target, x = up x z, y = z x x
  let z = -fwd;
  let x = normalize(cross(up, z));
  let y = cross(z, x);
  // px: (column, GL row) of the pixel within the face, both from 0 at the GL bottom-left
  let ndc = (px + 0.5) / PM.room.x * 2.0 - 1.0;
  return normalize(ndc.x * x + ndc.y * y + fwd);
}

@fragment
fn fs_room(i: PlaneOut) -> @location(0) vec4f {
  let size = PM.room.x;
  let frag = i.pos;  // framebuffer position; the room pass's viewport starts at (0, 0)
  // the face's pixel: viewport-relative column, and GL row (memory row within the face)
  let col = frag.x - f32(i.face % 3u) * size - 0.5;
  let row = frag.y - select(0.0, size, i.face > 2u) - 0.5;
  let d = room_view_dir(i.face, vec2f(col, row));
  var best = vec4f(1e30, 0.0, 0.0, 0.0);
  var kind = -1.0;
  for (var k = 0u; k < 13u; k = k + 1u) {
    let b = room_box(k);
    let h = room_hit(b, vec3f(0.0), d);
    if (h.x > PM.room.y && h.x < best.x) {
      best = h;
      kind = b.kind;
    }
  }
  if (kind > 1.0) {
    return vec4f(vec3f(kind), 1.0);
  }
  if (kind < 0.0) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let p = d * best.x;
  let n = best.yzw;
  let lv = vec3f(0.418, 16.199, 0.300) - p;
  let dist = length(lv);
  let l = lv / dist;
  var atten = 1.0 / max(pow(dist, 2.0), 0.01);
  let cut = saturate(1.0 - pow(dist / 28.0, 4.0));
  atten = atten * cut * cut;
  let dot_nl = saturate(dot(n, l));
  let irradiance = dot_nl * 900.0 * atten;
  let v = -d;
  let c = irradiance * 0.3183098861837907 + irradiance * room_brdf_ggx(l, v, n);
  return vec4f(vec3f(c), 1.0);
}
