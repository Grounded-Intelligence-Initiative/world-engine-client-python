// Mesh pass (pipeline "wgsl/1", design doc 16 §9.2 step 5): three.js's
// MeshStandardMaterial as the capture page draws it, plus the key light's
// shadow map.
//
// The shading is three 0.180's meshphysical.glsl.js for a MeshStandardMaterial
// (no PHYSICAL features), chunk by chunk: base colour times map times vertex
// colour, alpha test, roughness (map .g) and metalness (map .b), flat shading
// from screen derivatives when a mesh has no normals, normal maps with a
// tangent attribute or derivative tangents (getTangentFrame), emissive (map
// .rgb), roughness raised by the normal's screen-space change
// (geometryRoughness), one directional light (GGX + Lambert, PCF shadow),
// image-based light from a CubeUV environment (cubeuv.wgsl) with
// Fdez-Aguera multiple scattering, ambient occlusion (map .r, also specular
// occlusion), then Neutral tone mapping and the sRGB transfer function, written
// to rgba8unorm the way three writes its canvas. Lighting is computed in three's
// view space (x right, y up, z towards the viewer), so the derivative terms
// match: WGSL's dpdy runs down the framebuffer, GLSL's dFdy up, hence dfdy_gl.
// KHR_materials_unlit (MeshBasicMaterial) is the colour, tone-mapped.
//
// Depth uses the splats' mapping (ndc_depth), so splats depth-test against the
// meshes. Hosts cull back faces (front face counter-clockwise, three's
// FrontSide) except for double-sided materials, which draw both and flip the
// normal on back faces. The key light's depth map is drawn by shadow.wgsl.
// The material (bind group 1, Material, uv_for, has) is material.wgsl, which
// the module puts before this file.

const PI: f32 = 3.141592653589793;
const RECIPROCAL_PI: f32 = 0.3183098861837907;
const EPSILON: f32 = 1e-6;

struct Owner {
  model: mat4x4f,        // object -> world (pose, local offset, scale)
  normal: mat4x4f,       // inverse transpose of model's 3x3 (in the upper 3x3)
};

struct MeshFrame {
  light_dir: vec4f,      // world direction towards the light, w = 1 when the light is on
  light_color: vec4f,    // colour times intensity
  env: vec4f,            // x: environment intensity, y: 1 when there is an environment
  env_rot0: vec4f,       // rows of three's envMapRotation (world -> environment lookup)
  env_rot1: vec4f,
  env_rot2: vec4f,
  cubeuv: CubeUV,
  shadow_m: mat4x4f,     // world -> (u, v, depth, 1) of the shadow map (three's shadow.matrix)
  shadow: vec4f,         // bias, normalBias, radius (texels), map size
  flags: vec4u,          // x: shadows on, y: tone mapping (1 Neutral, 0 none), z: output sRGB (1) or linear (0)
  exposure: vec4f,       // x: toneMappingExposure
};

@group(0) @binding(0) var<uniform> cams: array<Camera, MAX_CAMS>;
@group(0) @binding(1) var<uniform> D: Dispatch;                 // D.cam: the camera drawn
@group(0) @binding(2) var<storage, read> owners: array<Owner>;
@group(0) @binding(3) var<uniform> F: MeshFrame;
@group(0) @binding(4) var env_atlas: texture_2d<f32>;
@group(0) @binding(5) var env_smp: sampler;
@group(0) @binding(6) var shadow_map: texture_depth_2d;

struct VIn {
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @location(2) uv0: vec2f,
  @location(3) uv1: vec2f,
  @location(4) color: vec4f,
  @location(5) tangent: vec4f,
  @location(6) owner: u32,
};

struct VOutMesh {
  @builtin(position) pos: vec4f,
  @location(0) view_pos: vec3f,      // three's vViewPosition (= -mvPosition)
  @location(1) normal: vec3f,        // vNormal (view space)
  @location(2) uv0: vec2f,
  @location(3) uv1: vec2f,
  @location(4) color: vec4f,
  @location(5) tangent: vec3f,       // vTangent (view space)
  @location(6) bitangent: vec3f,     // vBitangent
  @location(7) shadow_coord: vec4f,  // vDirectionalShadowCoord
};

// three's view space of camera `cam`: x right, y up, z towards the viewer.
fn view_dir_of(cam: Camera, w: vec3f) -> vec3f {
  return vec3f(dot(cam.rot0.xyz, w), -dot(cam.rot1.xyz, w), -dot(cam.rot2.xyz, w));
}

fn world_dir_of(cam: Camera, v: vec3f) -> vec3f {
  // inverse of view_dir_of (the rotation is orthonormal)
  return cam.rot0.xyz * v.x - cam.rot1.xyz * v.y - cam.rot2.xyz * v.z;
}

fn clip_of(cam: Camera, world: vec3f) -> vec4f {
  let d = world - cam.pos.xyz;
  let t = vec3f(dot(cam.rot0.xyz, d), dot(cam.rot1.xyz, d), dot(cam.rot2.xyz, d));
  let near = cam.view.z;
  let far = cam.view.w;
  let xn = (cam.intr.x * t.x / t.z + cam.intr.z) / cam.view.x * 2.0 - 1.0;
  let yn = 1.0 - (cam.intr.y * t.y / t.z + cam.intr.w) / cam.view.y * 2.0;
  // clip-space w = t.z, so z / w = ndc_depth(t.z, near, far)
  return vec4f(xn * t.z, yn * t.z, far * (t.z - near) / (far - near), t.z);
}

@vertex
fn vs_mesh(v: VIn) -> VOutMesh {
  let cam = cams[D.cam];
  let o = owners[v.owner];
  let world = (o.model * vec4f(v.p, 1.0)).xyz;
  let nm = mat3x3f(o.normal[0].xyz, o.normal[1].xyz, o.normal[2].xyz);
  let nw = nm * v.n;                                 // world normal, unnormalised
  var out: VOutMesh;
  out.pos = clip_of(cam, world);
  let d = world - cam.pos.xyz;
  out.view_pos = -view_dir_of(cam, d);
  let nv = view_dir_of(cam, nw);
  out.normal = vec3f(0.0, 0.0, 1.0);  // flat-shaded meshes have no normals (unused then)
  if (dot(nv, nv) > 0.0) {
    out.normal = normalize(nv);
  }
  out.uv0 = v.uv0;
  out.uv1 = v.uv1;
  out.color = v.color;
  let tv = view_dir_of(cam, (o.model * vec4f(v.tangent.xyz, 0.0)).xyz);
  out.tangent = vec3f(0.0);
  out.bitangent = vec3f(0.0);
  if (dot(v.tangent.xyz, v.tangent.xyz) > 0.0) {
    out.tangent = normalize(tv);
    out.bitangent = normalize(cross(out.normal, out.tangent) * v.tangent.w);
  }
  // shadowmap_vertex: offset along the world normal by normalBias. A mesh
  // without normals (flat-shaded) has a zero normal attribute in three, whose
  // normalize() is NaN, so its shadow coordinate fails every test: it never
  // receives a shadow. The fragment shader reproduces that with MF_FLAT.
  var swn = vec3f(0.0);
  if (dot(nw, nw) > 0.0) {
    swn = normalize(nw);
  }
  out.shadow_coord = F.shadow_m * vec4f(world + swn * F.shadow.y, 1.0);
  return out;
}

// three's dFdy (towards window +y, up); WGSL's dpdy goes down the framebuffer.
fn dfdy_gl3(x: vec3f) -> vec3f {
  return -dpdyFine(x);
}

fn dfdy_gl2(x: vec2f) -> vec2f {
  return -dpdyFine(x);
}

fn pow2(x: f32) -> f32 {
  return x * x;
}

fn f_schlick(f0: vec3f, f90: f32, dot_vh: f32) -> vec3f {
  let fresnel = exp2((-5.55473 * dot_vh - 6.98316) * dot_vh);
  return f0 * (1.0 - fresnel) + (f90 * fresnel);
}

fn v_ggx_smith_correlated(alpha: f32, dot_nl: f32, dot_nv: f32) -> f32 {
  let a2 = pow2(alpha);
  let gv = dot_nl * sqrt(a2 + (1.0 - a2) * pow2(dot_nv));
  let gl = dot_nv * sqrt(a2 + (1.0 - a2) * pow2(dot_nl));
  return 0.5 / max(gv + gl, EPSILON);
}

fn d_ggx(alpha: f32, dot_nh: f32) -> f32 {
  let a2 = pow2(alpha);
  let denom = pow2(dot_nh) * (a2 - 1.0) + 1.0;
  return RECIPROCAL_PI * a2 / pow2(denom);
}

fn brdf_ggx(l: vec3f, v: vec3f, n: vec3f, f0: vec3f, f90: f32, roughness: f32) -> vec3f {
  let alpha = pow2(roughness);
  let h = normalize(l + v);
  let dot_nl = saturate(dot(n, l));
  let dot_nv = saturate(dot(n, v));
  let dot_nh = saturate(dot(n, h));
  let dot_vh = saturate(dot(v, h));
  let fr = f_schlick(f0, f90, dot_vh);
  let vis = v_ggx_smith_correlated(alpha, dot_nl, dot_nv);
  let dd = d_ggx(alpha, dot_nh);
  return fr * (vis * dd);
}

fn dfg_approx(n: vec3f, v: vec3f, roughness: f32) -> vec2f {
  let dot_nv = saturate(dot(n, v));
  let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * dot_nv)) * r.x + r.y;
  return vec2f(-1.04, 1.04) * a004 + r.zw;
}

// shadowmap_pars_fragment: 17 nearest taps of the packed depth, step(compare, depth).
fn shadow_tap(uv: vec2f, compare: f32) -> f32 {
  let size = i32(F.shadow.w);
  let ij = clamp(vec2i(floor(uv * F.shadow.w)), vec2i(0), vec2i(size - 1));
  let depth = textureLoad(shadow_map, ij, 0);
  return step(compare, depth);
}

fn get_shadow(coord_in: vec4f) -> f32 {
  var c = coord_in.xyz / coord_in.w;
  c.z = c.z + F.shadow.x;
  let in_frustum = c.x >= 0.0 && c.x <= 1.0 && c.y >= 0.0 && c.y <= 1.0;
  if (!(in_frustum && c.z <= 1.0)) {
    return 1.0;
  }
  let ts = 1.0 / F.shadow.w;
  let dx0 = -ts * F.shadow.z;
  let dy0 = -ts * F.shadow.z;
  let dx1 = ts * F.shadow.z;
  let dy1 = ts * F.shadow.z;
  let dx2 = dx0 / 2.0;
  let dy2 = dy0 / 2.0;
  let dx3 = dx1 / 2.0;
  let dy3 = dy1 / 2.0;
  let s = shadow_tap(c.xy + vec2f(dx0, dy0), c.z)
    + shadow_tap(c.xy + vec2f(0.0, dy0), c.z)
    + shadow_tap(c.xy + vec2f(dx1, dy0), c.z)
    + shadow_tap(c.xy + vec2f(dx2, dy2), c.z)
    + shadow_tap(c.xy + vec2f(0.0, dy2), c.z)
    + shadow_tap(c.xy + vec2f(dx3, dy2), c.z)
    + shadow_tap(c.xy + vec2f(dx0, 0.0), c.z)
    + shadow_tap(c.xy + vec2f(dx2, 0.0), c.z)
    + shadow_tap(c.xy, c.z)
    + shadow_tap(c.xy + vec2f(dx3, 0.0), c.z)
    + shadow_tap(c.xy + vec2f(dx1, 0.0), c.z)
    + shadow_tap(c.xy + vec2f(dx2, dy3), c.z)
    + shadow_tap(c.xy + vec2f(0.0, dy3), c.z)
    + shadow_tap(c.xy + vec2f(dx3, dy3), c.z)
    + shadow_tap(c.xy + vec2f(dx0, dy1), c.z)
    + shadow_tap(c.xy + vec2f(0.0, dy1), c.z)
    + shadow_tap(c.xy + vec2f(dx1, dy1), c.z);
  return s * (1.0 / 17.0);
}

fn env_lookup(world_dir: vec3f, roughness: f32) -> vec3f {
  let d = vec3f(dot(F.env_rot0.xyz, world_dir), dot(F.env_rot1.xyz, world_dir), dot(F.env_rot2.xyz, world_dir));
  return texture_cube_uv(env_atlas, env_smp, F.cubeuv, d, roughness);
}

fn neutral_tone_mapping(c_in: vec3f) -> vec3f {
  let start_compression = 0.8 - 0.04;
  let desaturation = 0.15;
  var color = c_in * F.exposure.x;
  let x = min(color.r, min(color.g, color.b));
  var offset = 0.04;
  if (x < 0.08) {
    offset = x - 6.25 * x * x;
  }
  color = color - offset;
  let peak = max(color.r, max(color.g, color.b));
  if (peak < start_compression) {
    return color;
  }
  let d = 1.0 - start_compression;
  let new_peak = 1.0 - d * d / (peak + d - start_compression);
  color = color * (new_peak / peak);
  let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
  return mix(color, vec3f(new_peak), g);
}

fn srgb_oetf(c: vec3f) -> vec3f {
  return select(pow(c, vec3f(0.41666)) * 1.055 - vec3f(0.055), c * 12.92, c <= vec3f(0.0031308));
}

fn output_color(c: vec3f) -> vec4f {
  var o = c;
  if (F.flags.y == 1u) {
    o = neutral_tone_mapping(o);
  }
  if (F.flags.z == 1u) {
    o = srgb_oetf(o);
  }
  // Rounded to 8 bits here, as the capture page's 8-bit canvas stores a mesh
  // before the splats blend over it: in a colour target above 8 bits (README,
  // Colour target) the mesh is then the same k / 255 on every adapter, and the
  // resolve gives k back (an rgba16float write rounds by less than half a step).
  // Left to the target, a half float rounds it once and the resolve again.
  return vec4f(round(saturate(o) * 255.0) / 255.0, 1.0);
}

@fragment
fn fs_mesh(i: VOutMesh, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let cam = cams[D.cam];
  // ---- every texture sample and derivative first (uniform control flow)
  let s_map_v = textureSample(t_map, s_map, uv_for(0u, i.uv0, i.uv1));
  let uv_n = uv_for(1u, i.uv0, i.uv1);
  let s_normal_v = textureSample(t_normal, s_normal, uv_n);
  let s_rough_v = textureSample(t_rough, s_rough, uv_for(2u, i.uv0, i.uv1));
  let s_metal_v = textureSample(t_metal, s_metal, uv_for(3u, i.uv0, i.uv1));
  let s_ao_v = textureSample(t_ao, s_ao, uv_for(4u, i.uv0, i.uv1));
  let s_em_v = textureSample(t_emissive, s_emissive, uv_for(5u, i.uv0, i.uv1));
  let fdx = dpdxFine(i.view_pos);
  let fdy = dfdy_gl3(i.view_pos);
  let st0 = dpdxFine(uv_n);
  let st1 = dfdy_gl2(uv_n);
  let n_interp = normalize(i.normal);
  let nflat = normalize(cross(fdx, fdy));
  // geometryRoughness (lights_physical_fragment) needs the normal's derivatives;
  // take both candidates here, before any discard. The sign of a derivative
  // does not matter there (abs), so the double-sided flip does not either.
  let d_interp = max(abs(dpdxFine(n_interp)), abs(dfdy_gl3(n_interp)));
  let d_flat = max(abs(dpdxFine(nflat)), abs(dfdy_gl3(nflat)));

  // ---- map_fragment, color_fragment, alphatest
  var diffuse_color = M.color;
  if (has(MF_MAP)) {
    diffuse_color = diffuse_color * s_map_v;
  }
  if (has(MF_VERTEX_RGBA)) {
    diffuse_color = diffuse_color * i.color;
  } else if (has(MF_VERTEX_RGB)) {
    diffuse_color = vec4f(diffuse_color.rgb * i.color.rgb, diffuse_color.a);
  }
  if (M.params.w > 0.0 && diffuse_color.a < M.params.w) {
    discard;
  }
  if (has(MF_UNLIT)) {
    // meshbasic: indirectDiffuse = 1, times the diffuse colour, AO applies
    var basic = diffuse_color.rgb;
    if (has(MF_AO_MAP)) {
      basic = basic * ((s_ao_v.r - 1.0) * M.params.z + 1.0);
    }
    return output_color(basic);
  }
  var roughness_factor = M.params.y;
  if (has(MF_ROUGHNESS_MAP)) {
    roughness_factor = roughness_factor * s_rough_v.g;
  }
  var metalness_factor = M.params.x;
  if (has(MF_METALNESS_MAP)) {
    metalness_factor = metalness_factor * s_metal_v.b;
  }

  // ---- normal_fragment_begin / _maps
  let face_direction = select(-1.0, 1.0, front);
  var normal = n_interp;
  if (has(MF_FLAT)) {
    normal = nflat;
  } else if (has(MF_DOUBLE_SIDED)) {
    normal = normal * face_direction;
  }
  if (has(MF_NORMAL_MAP)) {
    var t: vec3f;
    var b: vec3f;
    if (has(MF_TANGENT)) {
      t = normalize(i.tangent);
      b = normalize(i.bitangent);
    } else {
      // getTangentFrame( -vViewPosition, normal, uv )
      let q0 = -fdx;
      let q1 = -fdy;
      let q1perp = cross(q1, normal);
      let q0perp = cross(normal, q0);
      let tt = q1perp * st0.x + q0perp * st1.x;
      let bb = q1perp * st0.y + q0perp * st1.y;
      let det = max(dot(tt, tt), dot(bb, bb));
      var sc = 0.0;
      if (det != 0.0) {
        sc = inverseSqrt(det);
      }
      t = tt * sc;
      b = bb * sc;
    }
    if (has(MF_DOUBLE_SIDED) && !has(MF_FLAT)) {
      t = t * face_direction;
      b = b * face_direction;
    }
    var map_n = s_normal_v.xyz * 2.0 - 1.0;
    map_n = vec3f(map_n.xy * M.normal_scale.xy, map_n.z);
    normal = normalize(mat3x3f(t, b, normal) * map_n);
  }

  // ---- emissivemap_fragment
  var emissive = M.emissive.rgb;
  if (has(MF_EMISSIVE_MAP)) {
    emissive = emissive * s_em_v.rgb;
  }

  // ---- lights_physical_fragment
  let diffuse = diffuse_color.rgb * (1.0 - metalness_factor);
  let dxy = select(d_interp, d_flat, has(MF_FLAT));
  let geometry_roughness = max(max(dxy.x, dxy.y), dxy.z);
  let roughness = min(max(roughness_factor, 0.0525) + geometry_roughness, 1.0);
  let specular_color = mix(vec3f(0.04), diffuse_color.rgb, metalness_factor);
  let specular_f90 = 1.0;

  // ---- lights_fragment_begin (view space)
  let view_dir = normalize(i.view_pos);
  var direct_diffuse = vec3f(0.0);
  var direct_specular = vec3f(0.0);
  if (F.light_dir.w > 0.5) {
    let l = normalize(view_dir_of(cam, F.light_dir.xyz));
    var light_color = F.light_color.rgb;
    if (F.flags.x == 1u && has(MF_RECEIVE_SHADOW) && !has(MF_FLAT)) {
      light_color = light_color * get_shadow(i.shadow_coord);
    }
    let dot_nl = saturate(dot(normal, l));
    let irradiance = dot_nl * light_color;
    direct_specular = irradiance * brdf_ggx(l, view_dir, normal, specular_color, specular_f90, roughness);
    direct_diffuse = irradiance * RECIPROCAL_PI * diffuse;
  }

  // ---- lights_fragment_maps / _end: image-based light
  var indirect_diffuse = vec3f(0.0);
  var indirect_specular = vec3f(0.0);
  if (F.env.y > 0.5) {
    let world_normal = normalize(world_dir_of(cam, normal));
    let ibl_irradiance = PI * env_lookup(world_normal, 1.0) * F.env.x;
    var reflect_vec = reflect(-view_dir, normal);
    reflect_vec = normalize(mix(reflect_vec, normal, roughness * roughness));
    let radiance = env_lookup(normalize(world_dir_of(cam, reflect_vec)), roughness) * F.env.x;
    // computeMultiscattering
    let fab = dfg_approx(normal, view_dir, roughness);
    let fss_ess = specular_color * fab.x + specular_f90 * fab.y;
    let ess = fab.x + fab.y;
    let ems = 1.0 - ess;
    let favg = specular_color + (1.0 - specular_color) * 0.047619;
    let fms = fss_ess * favg / (1.0 - ems * favg);
    let single_scattering = fss_ess;
    let multi_scattering = fms * ems;
    let total_scattering = single_scattering + multi_scattering;
    let diffuse_ms = diffuse * (1.0 - max(max(total_scattering.r, total_scattering.g), total_scattering.b));
    let cos_weighted_irradiance = ibl_irradiance * RECIPROCAL_PI;
    indirect_specular = radiance * single_scattering + multi_scattering * cos_weighted_irradiance;
    indirect_diffuse = diffuse_ms * cos_weighted_irradiance;
  }

  // ---- aomap_fragment
  if (has(MF_AO_MAP)) {
    let ao = (s_ao_v.r - 1.0) * M.params.z + 1.0;
    indirect_diffuse = indirect_diffuse * ao;
    if (F.env.y > 0.5) {
      let dot_nv = saturate(dot(normal, view_dir));
      let spec_ao = saturate(pow(dot_nv + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
      indirect_specular = indirect_specular * spec_ao;
    }
  }

  let outgoing = direct_diffuse + indirect_diffuse + direct_specular + indirect_specular + emissive;
  return output_color(outgoing);
}
