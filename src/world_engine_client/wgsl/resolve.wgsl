// Colour target -> rgba8unorm, once per camera after its draws (design doc 16
// §9.2 step 7).
//
// The mesh and splat passes draw into a colour target with more precision than
// 8 bits (rgba16float on a GPU, rgba32float on a CPU adapter with
// float32-blendable), so every adapter blends the back-to-front splats nearly
// exactly instead of rounding each layer to 8 bits in its own way. This pass
// draws one full-target triangle that reads the target texel by texel and
// writes it to an rgba8unorm texture, which the host copies to the readback
// buffer: the image is rounded to 8 bits once, at the end. The shader rounds
// (round() is half to even) and writes an exact multiple of 1/255, so how an
// adapter converts float to unorm8 does not matter.

@group(0) @binding(0) var colour: texture_2d<f32>;   // the camera's colour target

@vertex
fn vs_resolve(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(p[i], 0.0, 1.0);
}

@fragment
fn fs_resolve(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = textureLoad(colour, vec2i(pos.xy), 0);
  return vec4f(round(clamp(c.rgb, vec3f(0.0), vec3f(1.0)) * 255.0) / 255.0, 1.0);
}
