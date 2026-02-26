import type { GaussianSceneData } from '../loader/types.ts';

// ---------------------------------------------------------------------------
// GPU input buffer layout (per Gaussian, DC-only = 64 bytes)
//
//   offset  0: position  vec3<f32> + 4-byte pad  → 16 bytes
//   offset 16: log_scale vec3<f32> + 4-byte pad  → 16 bytes
//   offset 32: rotation  vec4<f32>               → 16 bytes  (wxyz)
//   offset 48: opacity   f32                     →  4 bytes
//   offset 52: sh_dc     vec3<f32>               → 12 bytes
//   ──────────────────────────────────────────────── 64 bytes total
//
// Higher SH bands (sh_rest) are appended after offset 64 in Phase 7.
// ---------------------------------------------------------------------------

export const GAUSSIAN_STRIDE = 64; // bytes per Gaussian in the GPU buffer

/**
 * Pack CPU-side GaussianSceneData into a flat Float32Array ready for
 * writeBuffer(). For DC-only files shRest is empty; higher-SH support
 * will be added in Phase 7.
 */
export function packGaussianBuffer(data: GaussianSceneData): Float32Array {
  const { numGaussians: N, positions, logScales, rotations, opacities, shDC } = data;
  const FLOATS = GAUSSIAN_STRIDE / 4; // 16 float32s per Gaussian
  const packed = new Float32Array(N * FLOATS);

  for (let i = 0; i < N; i++) {
    const base = i * FLOATS;

    // offset  0 (f32 index  0): position xyz + pad
    packed[base + 0] = positions[i * 3 + 0]!;
    packed[base + 1] = positions[i * 3 + 1]!;
    packed[base + 2] = positions[i * 3 + 2]!;
    packed[base + 3] = 0; // padding

    // offset 16 (f32 index  4): log_scale xyz + pad
    packed[base + 4] = logScales[i * 3 + 0]!;
    packed[base + 5] = logScales[i * 3 + 1]!;
    packed[base + 6] = logScales[i * 3 + 2]!;
    packed[base + 7] = 0; // padding

    // offset 32 (f32 index  8): rotation wxyz
    packed[base +  8] = rotations[i * 4 + 0]!;
    packed[base +  9] = rotations[i * 4 + 1]!;
    packed[base + 10] = rotations[i * 4 + 2]!;
    packed[base + 11] = rotations[i * 4 + 3]!;

    // offset 48 (f32 index 12): opacity logit
    packed[base + 12] = opacities[i]!;

    // offset 52 (f32 index 13): shDC rgb
    packed[base + 13] = shDC[i * 3 + 0]!;
    packed[base + 14] = shDC[i * 3 + 1]!;
    packed[base + 15] = shDC[i * 3 + 2]!;
  }

  return packed;
}

/**
 * Allocate a STORAGE | COPY_DST GPUBuffer and upload the packed Gaussian data.
 * The returned buffer is the input to the preprocessing compute shader.
 */
export function createGaussianBuffer(device: GPUDevice, data: GaussianSceneData): GPUBuffer {
  const packed = packGaussianBuffer(data);
  const gpuBuffer = device.createBuffer({
    label: 'gaussian-input',
    size: packed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gpuBuffer, 0, packed.buffer as ArrayBuffer, packed.byteOffset, packed.byteLength);
  return gpuBuffer;
}
