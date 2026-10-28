// Key-light shadow map (pipeline "wgsl/1", design doc 16 §9.2 step 5): the
// depth of every shadow-casting mesh seen from the directional light, one map a
// frame shared by all cameras. three's DirectionalLightShadow: an orthographic
// camera at the light looking at its target (up +y), depth in GL's
// 0.5 * ndc + 0.5 (orthographic: linear in distance). three packs that depth
// into RGBA8; here it is a depth32float target, read with textureLoad by the
// mesh pass (nearest, as three's NearestFilter map). Hosts cull front faces of
// single-sided materials (three renders a FrontSide caster's back faces) and
// nothing for double-sided ones, depth compare less-equal, cleared to 1.
//
// A caster with a map and an alpha test (glTF MASK: leaves, fences, cut-out
// labels) is drawn with vs_shadow_masked / fs_shadow_masked and its material's
// bind group (material.wgsl), as WebGLShadowMap.getDepthMaterial gives three's
// depth material the caster's map and alphaTest: the map's alpha alone (no base
// colour, opacity or vertex colour; MeshDepthMaterial with RGBA packing keeps
// diffuseColor's alpha at 1 before the map) at the map's uv channel and
// transform, discarded below alphaTest.

struct ShadowFrame {
  view_proj: mat4x4f,    // world -> NDC of the light camera (GL convention, z in [-1, 1])
};

struct ShadowOwner {
  model: mat4x4f,
  normal: mat4x4f,
};

@group(0) @binding(0) var<uniform> SF: ShadowFrame;
@group(0) @binding(1) var<storage, read> shadow_owners: array<ShadowOwner>;

fn shadow_position(p: vec3f, owner: u32) -> vec4f {
  let world = shadow_owners[owner].model * vec4f(p, 1.0);
  let c = SF.view_proj * world;
  // y flipped so the map's rows are in GL order (row 0 at v = 0), the order the
  // mesh pass indexes them in.
  return vec4f(c.x, -c.y, 0.5 * c.z + 0.5 * c.w, c.w);
}

@vertex
fn vs_shadow(@location(0) p: vec3f, @location(6) owner: u32) -> @builtin(position) vec4f {
  return shadow_position(p, owner);
}

struct VOutShadowMasked {
  @builtin(position) pos: vec4f,
  @location(0) uv0: vec2f,
  @location(1) uv1: vec2f,
};

@vertex
fn vs_shadow_masked(@location(0) p: vec3f, @location(2) uv0: vec2f, @location(3) uv1: vec2f,
                    @location(6) owner: u32) -> VOutShadowMasked {
  var out: VOutShadowMasked;
  out.pos = shadow_position(p, owner);
  out.uv0 = uv0;
  out.uv1 = uv1;
  return out;
}

@fragment
fn fs_shadow_masked(i: VOutShadowMasked) {
  let a = textureSample(t_map, s_map, uv_for(0u, i.uv0, i.uv1)).a;
  if (a < M.params.w) {
    discard;
  }
}
