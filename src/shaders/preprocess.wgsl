// =============================================================================
// preprocess.wgsl — Gaussian preprocessing compute shader
//
// Dispatch: ceil(numGaussians / 256) workgroups × 256 threads
//
// Pass 2b-ii: position transform + frustum cull + opacity cull.
//   Right-handed view space: objects in front have pos_view.z < 0.
//   clip.w = -pos_view.z  (positive for visible points).
// =============================================================================

// -----------------------------------------------------------------------------
// Bind group 0
// -----------------------------------------------------------------------------

struct Uniforms {
  view_matrix:      mat4x4<f32>, // offset   0  (64 bytes)
  proj_matrix:      mat4x4<f32>, // offset  64  (64 bytes)
  camera_pos:       vec3<f32>,   // offset 128  (align 16 ✓)
  tan_half_fov_x:   f32,         // offset 140
  tan_half_fov_y:   f32,         // offset 144
  focal_x:          f32,         // offset 148
  focal_y:          f32,         // offset 152
  scale_modifier:   f32,         // offset 156
  screen_size:      vec2<u32>,   // offset 160  (align 8 ✓)
  num_gaussians:    u32,         // offset 168
  tile_size:        u32,         // offset 172
}                                // total: 176 bytes

// Input: one entry per Gaussian, packed by buffers.ts (64 bytes each).
// vec3<f32> has AlignOf=16, so sh_dc at offset 52 would be misaligned.
// Using three separate f32 fields keeps the layout identical to the CPU side.
struct GaussianInput {
  position:  vec3<f32>, // offset  0  (align 16 ✓)
  _pad0:     f32,       // offset 12
  log_scale: vec3<f32>, // offset 16  (align 16 ✓)
  _pad1:     f32,       // offset 28
  rotation:  vec4<f32>, // offset 32  (align 16 ✓)  wxyz
  opacity:   f32,       // offset 48
  sh_dc_r:   f32,       // offset 52
  sh_dc_g:   f32,       // offset 56
  sh_dc_b:   f32,       // offset 60
}                       // total: 64 bytes

// Output: one entry per Gaussian, read by the rasterizer.
struct GaussData {
  id:                i32,       // offset  0
  radii:             i32,       // offset  4   0 = culled
  depth:             f32,       // offset  8
  tiles_touched:     u32,       // offset 12
  cum_tiles_touched: u32,       // offset 16
  _pad0:             u32,       // offset 20   (align uv to 8)
  uv:                vec2<f32>, // offset 24   (align 8 ✓)  screen-space center [0,1]
  conic:             vec3<f32>, // offset 32   (align 16 ✓) [C/det, -B/det, A/det]
  _pad1:             f32,       // offset 44
  color:             vec3<f32>, // offset 48   (align 16 ✓) pre-evaluated RGB
  opacity:           f32,       // offset 60   sigmoid(logit)
}                               // total: 64 bytes

@group(0) @binding(0) var<uniform>            uniforms:    Uniforms;
@group(0) @binding(1) var<storage, read>      gaussians:   array<GaussianInput>;
@group(0) @binding(2) var<storage, read_write> gauss_data: array<GaussData>;

// -----------------------------------------------------------------------------
// Entry point (skeleton — all outputs zeroed, radii = 0 = culled)
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

@compute @workgroup_size(256, 1, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uniforms.num_gaussians) { return; }

  var out: GaussData;
  out.id    = i32(idx);
  out.radii = 0; // culled by default

  let gauss = gaussians[idx];

  // ---- Transform to view space --------------------------------------------
  let pos_view = uniforms.view_matrix * vec4<f32>(gauss.position, 1.0);

  // Right-handed: z < 0 means in front of camera. Cull if too close/behind.
  if (pos_view.z > -0.2) {
    gauss_data[idx] = out;
    return;
  }

  // ---- Opacity cull -------------------------------------------------------
  let alpha = sigmoid(gauss.opacity);
  if (alpha < 0.03) {
    gauss_data[idx] = out;
    return;
  }

  // ---- Clip space + perspective divide ------------------------------------
  let pos_clip = uniforms.proj_matrix * pos_view;
  let pos_ndc  = pos_clip.xyz / pos_clip.w; // clip.w = -pos_view.z > 0

  // ---- Frustum cull (with a small border for partially-visible splats) ----
  if (abs(pos_ndc.x) > 1.1 || abs(pos_ndc.y) > 1.1) {
    gauss_data[idx] = out;
    return;
  }

  // ---- Visible — fill basic output fields ---------------------------------
  // depth: positive, smaller = closer to camera
  out.depth   = -pos_view.z;
  // uv: NDC [-1,1] → screen [0,1]; y is flipped (NDC +y = up, screen +y = down)
  out.uv      = vec2<f32>(pos_ndc.x * 0.5 + 0.5, -pos_ndc.y * 0.5 + 0.5);
  out.opacity = alpha;
  out.radii   = 1; // placeholder — real pixel radius computed in 2b-iii

  gauss_data[idx] = out;
}
