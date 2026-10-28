// One material of the mesh pass (pipeline "wgsl/1"): three's
// MeshStandardMaterial parameters and maps as mesh.wgsl and the shadow pass's
// alpha-tested casters (shadow.wgsl) read them, in bind group 1. Hosts make
// one bind group per material: the Material uniform, then the six maps (map,
// normal, roughness, metalness, AO, emissive) and their samplers.

// material flags (Material.flags.x)
const MF_MAP: u32 = 1u;
const MF_NORMAL_MAP: u32 = 2u;
const MF_ROUGHNESS_MAP: u32 = 4u;
const MF_METALNESS_MAP: u32 = 8u;
const MF_AO_MAP: u32 = 16u;
const MF_EMISSIVE_MAP: u32 = 32u;
const MF_VERTEX_RGB: u32 = 64u;
const MF_VERTEX_RGBA: u32 = 128u;
const MF_FLAT: u32 = 256u;
const MF_DOUBLE_SIDED: u32 = 512u;
const MF_TANGENT: u32 = 1024u;
const MF_UNLIT: u32 = 2048u;
const MF_RECEIVE_SHADOW: u32 = 4096u;

// map slots: map, normal, roughness, metalness, ao, emissive
struct Material {
  color: vec4f,          // linear rgb, opacity
  emissive: vec4f,       // linear rgb (times emissiveIntensity)
  params: vec4f,         // metalness, roughness, aoMapIntensity, alphaTest
  normal_scale: vec4f,   // xy (y already negated for derivative tangents, as GLTFLoader)
  flags: vec4u,          // x: MF_*, y: uv channel of map slot k in bit k (0: uv, 1: uv1)
  tr: array<vec4f, 12>,  // per map slot two rows of three's texture matrix (uv' = row . (u, v, 1))
};

@group(1) @binding(0) var<uniform> M: Material;
@group(1) @binding(1) var t_map: texture_2d<f32>;
@group(1) @binding(2) var t_normal: texture_2d<f32>;
@group(1) @binding(3) var t_rough: texture_2d<f32>;
@group(1) @binding(4) var t_metal: texture_2d<f32>;
@group(1) @binding(5) var t_ao: texture_2d<f32>;
@group(1) @binding(6) var t_emissive: texture_2d<f32>;
@group(1) @binding(7) var s_map: sampler;
@group(1) @binding(8) var s_normal: sampler;
@group(1) @binding(9) var s_rough: sampler;
@group(1) @binding(10) var s_metal: sampler;
@group(1) @binding(11) var s_ao: sampler;
@group(1) @binding(12) var s_emissive: sampler;

fn uv_for(slot: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  var uv = uv0;
  if (((M.flags.y >> slot) & 1u) == 1u) {
    uv = uv1;
  }
  let r0 = M.tr[2u * slot];
  let r1 = M.tr[2u * slot + 1u];
  return vec2f(dot(r0.xyz, vec3f(uv, 1.0)), dot(r1.xyz, vec3f(uv, 1.0)));
}

fn has(bit: u32) -> bool {
  return (M.flags.x & bit) != 0u;
}
