// =============================================================================
// preprocess.wgsl — Gaussian preprocessing compute shader
//
// Dispatch: ceil(numGaussians / 256) workgroups × 256 threads
//
// Pass 2b-iii: 3D covariance → EWA projection → conic + radius + tile count.
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
  uv:                vec2<f32>, // offset 24   (align 8 ✓)  screen-space [0,1]
  conic:             vec3<f32>, // offset 32   (align 16 ✓) [C/det, -B/det, A/det]
  _pad1:             f32,       // offset 44
  color:             vec3<f32>, // offset 48   (align 16 ✓) pre-evaluated RGB
  opacity:           f32,       // offset 60   sigmoid(logit)
}                               // total: 64 bytes

@group(0) @binding(0) var<uniform>             uniforms:   Uniforms;
@group(0) @binding(1) var<storage, read>       gaussians:  array<GaussianInput>;
@group(0) @binding(2) var<storage, read_write> gauss_data: array<GaussData>;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

// Quaternion (wxyz stored as vec4.xyzw) → rotation matrix (column-major).
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
  let w = q.x; let x = q.y; let y = q.z; let z = q.w;
  return mat3x3<f32>(
    // col 0                          col 1                          col 2
    1.0 - 2.0*(y*y + z*z),    2.0*(x*y + w*z),           2.0*(x*z - w*y),
    2.0*(x*y - w*z),           1.0 - 2.0*(x*x + z*z),    2.0*(y*z + w*x),
    2.0*(x*z + w*y),           2.0*(y*z - w*x),           1.0 - 2.0*(x*x + y*y),
  );
}

// Build 3D covariance Σ = Mᵀ·M  where M = diag(exp(logScale)) · R.
fn build_sigma(log_scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
  let s = exp(log_scale);
  let R = quat_to_mat3(rotation);
  // Scale each column of R by the corresponding scale factor → M = S·R
  let M = mat3x3<f32>(s * R[0], s * R[1], s * R[2]);
  return transpose(M) * M;
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

  // Right-handed: z < 0 means in front of camera. Cull if behind / too close.
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

  // ---- Frustum cull (with margin for splats straddling the border) ---------
  if (abs(pos_ndc.x) > 1.1 || abs(pos_ndc.y) > 1.1) {
    gauss_data[idx] = out;
    return;
  }

  // ---- 3D covariance from scale + quaternion ------------------------------
  let sigma = build_sigma(gauss.log_scale, gauss.rotation);

  // ---- Perspective Jacobian (EWA splatting, Zwicker 2002 eq. 29) ----------
  // J maps view-space differentials to screen-space differentials.
  // Column-major: col0=(fx/z,0,0), col1=(0,−fy/z,0), col2=(−fx·x/z²,fy·y/z²,0)
  let z  = pos_view.z;           // negative
  let xv = pos_view.x;
  let yv = pos_view.y;
  let fx = uniforms.focal_x;
  let fy = uniforms.focal_y;
  let J = mat3x3<f32>(
     fx / z,        0.0,  0.0,  // col 0
     0.0,      -fy / z,   0.0,  // col 1
    -fx * xv / (z * z),  fy * yv / (z * z),  0.0,  // col 2
  );

  // W = upper-left 3×3 of view matrix (rotation part)
  let W = mat3x3<f32>(
    uniforms.view_matrix[0].xyz,
    uniforms.view_matrix[1].xyz,
    uniforms.view_matrix[2].xyz,
  );

  let T      = W * J;
  let cov2d  = transpose(T) * sigma * T;

  // ---- Regularise and extract 2D covariance elements ----------------------
  // cov2d[col][row]: A=(0,0), B=(1,0) off-diag, C=(1,1)
  let A = cov2d[0][0] + 0.3;
  let B = cov2d[1][0];
  let C = cov2d[1][1] + 0.3;

  // ---- Inverse conic [C/det, -B/det, A/det] --------------------------------
  let det = A * C - B * B;
  if (det == 0.0) {
    gauss_data[idx] = out;
    return;
  }
  out.conic = vec3<f32>(C / det, -B / det, A / det);

  // ---- Screen-space radius (pixels) ----------------------------------------
  let mid        = (A + C) * 0.5;
  let lambda_max = mid + sqrt(max(0.1, mid * mid - det));
  let radius_px  = ceil(3.0 * sqrt(lambda_max));

  // Screen-size cull: splat too small to contribute
  if (radius_px < 0.5) {
    gauss_data[idx] = out;
    return;
  }

  out.radii = i32(radius_px);
  out.depth = -pos_view.z;
  out.uv    = vec2<f32>(pos_ndc.x * 0.5 + 0.5, -pos_ndc.y * 0.5 + 0.5);
  out.opacity = alpha;

  // ---- Tile overlap count (used by Phase 3 prefix sum) --------------------
  let px    = out.uv * vec2<f32>(uniforms.screen_size);
  let ts    = f32(uniforms.tile_size);
  let ntx   = i32((uniforms.screen_size.x + uniforms.tile_size - 1u) / uniforms.tile_size);
  let nty   = i32((uniforms.screen_size.y + uniforms.tile_size - 1u) / uniforms.tile_size);
  let t_min = max(vec2<i32>(0),        vec2<i32>((px - radius_px) / ts));
  let t_max = min(vec2<i32>(ntx, nty), vec2<i32>((px + radius_px) / ts) + vec2<i32>(1));
  let n_tiles = max(0, (t_max.x - t_min.x) * (t_max.y - t_min.y));
  out.tiles_touched = u32(n_tiles);

  gauss_data[idx] = out;
}
