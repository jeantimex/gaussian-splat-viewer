import shaderCode from '../shaders/preprocess.wgsl?raw';

// ---------------------------------------------------------------------------
// Buffer sizes
// ---------------------------------------------------------------------------

export const GAUSS_DATA_STRIDE = 64; // bytes per GaussData entry

/**
 * Holds all GPU resources for the preprocessing compute pass.
 * Re-created when the scene changes (different numGaussians).
 */
export interface PreprocessPass {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  gaussDataBuffer: GPUBuffer; // output: array<GaussData>
  numGaussians: number;
}

/**
 * Compile the preprocessing pipeline and allocate the GaussData output buffer.
 *
 * @param device          WebGPU device
 * @param uniformBuffer   The 176-byte uniform buffer (from gpu/uniforms.ts)
 * @param gaussianBuffer  The packed input buffer (from gpu/buffers.ts)
 * @param numGaussians    Number of Gaussians in the scene
 */
export function createPreprocessPass(
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  gaussianBuffer: GPUBuffer,
  numGaussians: number,
): PreprocessPass {
  // ---- Shader module -------------------------------------------------------
  const shaderModule = device.createShaderModule({
    label: 'preprocess',
    code: shaderCode,
  });

  // ---- Output buffer -------------------------------------------------------
  const gaussDataBuffer = device.createBuffer({
    label: 'gauss-data',
    size: numGaussians * GAUSS_DATA_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // ---- Bind group layout (explicit, so we can inspect it) ------------------
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'preprocess-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const bindGroup = device.createBindGroup({
    label: 'preprocess-bg',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gaussianBuffer } },
      { binding: 2, resource: { buffer: gaussDataBuffer } },
    ],
  });

  // ---- Pipeline ------------------------------------------------------------
  const pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: 'preprocess' },
  });

  return { pipeline, bindGroup, gaussDataBuffer, numGaussians };
}

/**
 * Encode the preprocessing dispatch into `encoder`.
 * Call this once per frame before the sort and rasterize passes.
 */
export function encodePreprocessPass(encoder: GPUCommandEncoder, pass: PreprocessPass): void {
  const computePass = encoder.beginComputePass({ label: 'preprocess' });
  computePass.setPipeline(pass.pipeline);
  computePass.setBindGroup(0, pass.bindGroup);
  computePass.dispatchWorkgroups(Math.ceil(pass.numGaussians / 256));
  computePass.end();
}
