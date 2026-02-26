// ---------------------------------------------------------------------------
// Uniform buffer layout — matches the WGSL struct exactly
//
// struct Uniforms {
//   view_matrix:      mat4x4<f32>,  // offset   0  (64 bytes)
//   proj_matrix:      mat4x4<f32>,  // offset  64  (64 bytes)
//   camera_pos:       vec3<f32>,    // offset 128  (12 bytes, align 16)
//   tan_half_fov_x:   f32,          // offset 140
//   tan_half_fov_y:   f32,          // offset 144
//   focal_x:          f32,          // offset 148
//   focal_y:          f32,          // offset 152
//   scale_modifier:   f32,          // offset 156
//   screen_size:      vec2<u32>,    // offset 160  (8 bytes, align 8)
//   num_gaussians:    u32,          // offset 168
//   tile_size:        u32,          // offset 172
// }                                 // total: 176 bytes
// ---------------------------------------------------------------------------

export const UNIFORMS_SIZE = 176; // bytes
export const TILE_SIZE = 16;

export interface UniformData {
  viewMatrix: Float32Array; // 16 floats, column-major
  projMatrix: Float32Array; // 16 floats, column-major
  cameraPos: [number, number, number];
  tanHalfFovX: number;
  tanHalfFovY: number;
  focalX: number; // focal_x = width  / (2 * tanHalfFovX)
  focalY: number; // focal_y = height / (2 * tanHalfFovY)
  scaleModifier: number; // global splat scale multiplier (1.0 = no change)
  screenWidth: number;
  screenHeight: number;
  numGaussians: number;
}

export function createUniformBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'uniforms',
    size: UNIFORMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function updateUniforms(device: GPUDevice, buffer: GPUBuffer, d: UniformData): void {
  const ab = new ArrayBuffer(UNIFORMS_SIZE);
  const f32 = new Float32Array(ab);
  const u32 = new Uint32Array(ab);

  // offset   0 — view_matrix (16 f32)
  f32.set(d.viewMatrix, 0);

  // offset  64 — proj_matrix (16 f32, starting at f32 index 16)
  f32.set(d.projMatrix, 16);

  // offset 128 — camera_pos (f32 index 32–34)
  f32[32] = d.cameraPos[0];
  f32[33] = d.cameraPos[1];
  f32[34] = d.cameraPos[2];

  // offset 140 — tan_half_fov_x (f32 index 35)
  f32[35] = d.tanHalfFovX;

  // offset 144 — tan_half_fov_y (f32 index 36)
  f32[36] = d.tanHalfFovY;

  // offset 148 — focal_x (f32 index 37)
  f32[37] = d.focalX;

  // offset 152 — focal_y (f32 index 38)
  f32[38] = d.focalY;

  // offset 156 — scale_modifier (f32 index 39)
  f32[39] = d.scaleModifier;

  // offset 160 — screen_size (u32 index 40, 41)
  u32[40] = d.screenWidth;
  u32[41] = d.screenHeight;

  // offset 168 — num_gaussians (u32 index 42)
  u32[42] = d.numGaussians;

  // offset 172 — tile_size (u32 index 43)
  u32[43] = TILE_SIZE;

  device.queue.writeBuffer(buffer, 0, ab);
}

// ---------------------------------------------------------------------------
// Minimal mat4 helpers (column-major Float32Array, 16 elements)
// ---------------------------------------------------------------------------

/** 4×4 identity matrix. */
export function mat4Identity(): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/**
 * Perspective projection matrix for WebGPU clip space (z ∈ [0, 1]).
 * Right-handed view space (camera looks down −Z).
 *
 * @param fovYRad  Vertical field of view in radians
 * @param aspect   Width / height
 * @param near     Near clip plane distance (positive)
 * @param far      Far clip plane distance (positive)
 */
export function mat4Perspective(
  fovYRad: number,
  aspect: number,
  near: number,
  far: number,
): Float32Array {
  const f = 1.0 / Math.tan(fovYRad / 2);
  const ri = 1.0 / (far - near);
  // clip.w = -pos_view.z  (positive for objects in front, right-handed)
  // NDC.z maps near→0, far→1
  // prettier-ignore
  return new Float32Array([
    f / aspect,  0,   0,                 0,   // col 0
    0,           f,   0,                 0,   // col 1
    0,           0,  -far * ri,         -1,   // col 2  ← -1 makes clip.w = -z (positive)
    0,           0,  -near * far * ri,   0,   // col 3
  ]);
}

/**
 * Look-at view matrix (right-handed, camera looks toward `center`).
 *
 * @param eye     Camera world position
 * @param center  Point the camera looks at
 * @param up      World up vector (usually [0,1,0])
 */
export function mat4LookAt(
  eye: [number, number, number],
  center: [number, number, number],
  up: [number, number, number],
): Float32Array {
  // Forward = normalize(center - eye)
  let fx = center[0] - eye[0];
  let fy = center[1] - eye[1];
  let fz = center[2] - eye[2];
  const fl = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
  fx /= fl;
  fy /= fl;
  fz /= fl;

  // Right = normalize(forward × up)
  let rx = fy * up[2] - fz * up[1];
  let ry = fz * up[0] - fx * up[2];
  let rz = fx * up[1] - fy * up[0];
  const rl = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
  rx /= rl;
  ry /= rl;
  rz /= rl;

  // Up = right × forward  (re-orthogonalized)
  const ux = ry * fz - rz * fy;
  const uy = rz * fx - rx * fz;
  const uz = rx * fy - ry * fx;

  // prettier-ignore
  return new Float32Array([
    rx,  ux, -fx,  0,   // col 0
    ry,  uy, -fy,  0,   // col 1
    rz,  uz, -fz,  0,   // col 2
    -(rx*eye[0] + ry*eye[1] + rz*eye[2]),   // col 3
    -(ux*eye[0] + uy*eye[1] + uz*eye[2]),
     (fx*eye[0] + fy*eye[1] + fz*eye[2]),
    1,
  ]);
}
