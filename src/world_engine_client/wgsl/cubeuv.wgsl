// CubeUV environment lookups (pipeline "wgsl/1", design doc 16 §9.2 step 5),
// shared by the mesh pass and the environment prefilter.
//
// The environment is three.js's PMREM layout (three 0.180,
// src/extras/PMREMGenerator.js), so a prefiltered environment made by three
// (the capture page's bake, a lighting asset) and one made by these shaders
// read the same way: one rgba16float atlas, 3 x max(2^max_mip, 112) wide and
// 4 x 2^max_mip high, holding the mips 2^max_mip ... 16 and six more 16^2
// levels for the roughest lobes, each mip a 3 x 2 grid of faces with a one-texel
// border. These functions are cube_uv_reflection_fragment.glsl.js, line for
// line; the face indexing is PMREM's (right-handed), not the GL cube map's.

const CUBEUV_MIN_MIP: f32 = 4.0;
const CUBEUV_MIN_TILE: f32 = 16.0;

// Atlas geometry: texel width, texel height, max mip (three's CUBEUV_TEXEL_WIDTH,
// CUBEUV_TEXEL_HEIGHT, CUBEUV_MAX_MIP defines, here per environment).
struct CubeUV {
  texel_w: f32,
  texel_h: f32,
  max_mip: f32,
  _pad: f32,
};

fn cubeuv_face(direction: vec3f) -> f32 {
  let a = abs(direction);
  var face = -1.0;
  if (a.x > a.z) {
    if (a.x > a.y) {
      face = select(3.0, 0.0, direction.x > 0.0);
    } else {
      face = select(4.0, 1.0, direction.y > 0.0);
    }
  } else {
    if (a.z > a.y) {
      face = select(5.0, 2.0, direction.z > 0.0);
    } else {
      face = select(4.0, 1.0, direction.y > 0.0);
    }
  }
  return face;
}

fn cubeuv_uv(direction: vec3f, face: f32) -> vec2f {
  var uv: vec2f;
  if (face == 0.0) {
    uv = vec2f(direction.z, direction.y) / abs(direction.x);
  } else if (face == 1.0) {
    uv = vec2f(-direction.x, -direction.z) / abs(direction.y);
  } else if (face == 2.0) {
    uv = vec2f(-direction.x, direction.y) / abs(direction.z);
  } else if (face == 3.0) {
    uv = vec2f(-direction.z, direction.y) / abs(direction.x);
  } else if (face == 4.0) {
    uv = vec2f(-direction.x, direction.z) / abs(direction.y);
  } else {
    uv = vec2f(direction.x, direction.y) / abs(direction.z);
  }
  return 0.5 * (uv + 1.0);
}

// Bilinear sample of one mip (texture coordinates in texels, then normalised).
fn cubeuv_bilinear(atlas: texture_2d<f32>, smp: sampler, g: CubeUV, direction: vec3f, mip_int_in: f32) -> vec3f {
  var face = cubeuv_face(direction);
  let filter_int = max(CUBEUV_MIN_MIP - mip_int_in, 0.0);
  let mip_int = max(mip_int_in, CUBEUV_MIN_MIP);
  let face_size = exp2(mip_int);
  var uv = cubeuv_uv(direction, face) * (face_size - 2.0) + 1.0;
  if (face > 2.0) {
    uv.y = uv.y + face_size;
    face = face - 3.0;
  }
  uv.x = uv.x + face * face_size;
  uv.x = uv.x + filter_int * 3.0 * CUBEUV_MIN_TILE;
  uv.y = uv.y + 4.0 * (exp2(g.max_mip) - face_size);
  uv = uv * vec2f(g.texel_w, g.texel_h);
  return textureSampleLevel(atlas, smp, uv, 0.0).rgb;
}

const CUBEUV_R0: f32 = 1.0;
const CUBEUV_M0: f32 = -2.0;
const CUBEUV_R1: f32 = 0.8;
const CUBEUV_M1: f32 = -1.0;
const CUBEUV_R4: f32 = 0.4;
const CUBEUV_M4: f32 = 2.0;
const CUBEUV_R5: f32 = 0.305;
const CUBEUV_M5: f32 = 3.0;
const CUBEUV_R6: f32 = 0.21;
const CUBEUV_M6: f32 = 4.0;

fn cubeuv_roughness_to_mip(roughness: f32) -> f32 {
  var mip = 0.0;
  if (roughness >= CUBEUV_R1) {
    mip = (CUBEUV_R0 - roughness) * (CUBEUV_M1 - CUBEUV_M0) / (CUBEUV_R0 - CUBEUV_R1) + CUBEUV_M0;
  } else if (roughness >= CUBEUV_R4) {
    mip = (CUBEUV_R1 - roughness) * (CUBEUV_M4 - CUBEUV_M1) / (CUBEUV_R1 - CUBEUV_R4) + CUBEUV_M1;
  } else if (roughness >= CUBEUV_R5) {
    mip = (CUBEUV_R4 - roughness) * (CUBEUV_M5 - CUBEUV_M4) / (CUBEUV_R4 - CUBEUV_R5) + CUBEUV_M4;
  } else if (roughness >= CUBEUV_R6) {
    mip = (CUBEUV_R5 - roughness) * (CUBEUV_M6 - CUBEUV_M5) / (CUBEUV_R5 - CUBEUV_R6) + CUBEUV_M5;
  } else {
    mip = -2.0 * log2(1.16 * roughness);
  }
  return mip;
}

fn texture_cube_uv(atlas: texture_2d<f32>, smp: sampler, g: CubeUV, sample_dir: vec3f, roughness: f32) -> vec3f {
  let mip = clamp(cubeuv_roughness_to_mip(roughness), CUBEUV_M0, g.max_mip);
  let mip_f = fract(mip);
  let mip_int = floor(mip);
  let color0 = cubeuv_bilinear(atlas, smp, g, sample_dir, mip_int);
  if (mip_f == 0.0) {
    return color0;
  }
  let color1 = cubeuv_bilinear(atlas, smp, g, sample_dir, mip_int + 1.0);
  return mix(color0, color1, mip_f);
}
