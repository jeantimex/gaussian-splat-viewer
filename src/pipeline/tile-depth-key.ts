import shaderCode from '../shaders/tile-depth-key.wgsl?raw';
import { TILE_SIZE } from '../gpu/uniforms.ts';

// ---------------------------------------------------------------------------
// Holds all GPU resources for the tile-depth key expansion pass.
// Re-created whenever the scene or screen size changes.
// ---------------------------------------------------------------------------
export interface TileKeyPass {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  keysHiBuffer: GPUBuffer; // u32 tile indices  — length: totalIntersections
  keysLoBuffer: GPUBuffer; // u32 depth bits    — length: totalIntersections
  valuesBuffer: GPUBuffer; // u32 gaussian idx  — length: totalIntersections
  numGaussians: number;
  totalIntersections: number;
}

/**
 * Compile the tile-depth-key pipeline and allocate output buffers.
 *
 * @param device              WebGPU device
 * @param gaussDataBuffer     Output of preprocess pass (array<GaussData>)
 * @param cumTilesBuffer      Output of prefix-sum pass (exclusive prefix sums, array<u32>)
 * @param numGaussians        Total Gaussian count
 * @param totalIntersections  Grand total tile-Gaussian intersections (from prefix sum)
 * @param screenWidth         Canvas pixel width
 * @param screenHeight        Canvas pixel height
 */
export function createTileKeyPass(
  device: GPUDevice,
  gaussDataBuffer: GPUBuffer,
  cumTilesBuffer: GPUBuffer,
  numGaussians: number,
  totalIntersections: number,
  screenWidth: number,
  screenHeight: number,
): TileKeyPass {
  const shaderModule = device.createShaderModule({ label: 'tile-depth-key', code: shaderCode });

  // Uniform buffer: screen_width, screen_height, tile_size, _pad
  const uniformBuf = device.createBuffer({
    label: 'tile-key-uniforms',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([screenWidth, screenHeight, TILE_SIZE, 0]));

  // Three parallel u32 arrays, one entry per tile-Gaussian intersection.
  const entrySize = Math.max(totalIntersections, 1) * 4;
  const keysHiBuffer = device.createBuffer({
    label: 'tile-keys-hi',
    size: entrySize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const keysLoBuffer = device.createBuffer({
    label: 'tile-keys-lo',
    size: entrySize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const valuesBuffer = device.createBuffer({
    label: 'tile-values',
    size: entrySize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const bgl = device.createBindGroupLayout({
    label: 'tile-key-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const bindGroup = device.createBindGroup({
    label: 'tile-key-bg',
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: gaussDataBuffer } },
      { binding: 2, resource: { buffer: cumTilesBuffer } },
      { binding: 3, resource: { buffer: keysHiBuffer } },
      { binding: 4, resource: { buffer: keysLoBuffer } },
      { binding: 5, resource: { buffer: valuesBuffer } },
    ],
  });

  const pipeline = device.createComputePipeline({
    label: 'tile-depth-key',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module: shaderModule, entryPoint: 'tile_depth_key' },
  });

  return { pipeline, bindGroup, keysHiBuffer, keysLoBuffer, valuesBuffer, numGaussians, totalIntersections };
}

/**
 * Encode the tile-depth-key dispatch into `encoder`.
 * No-op when totalIntersections is 0 (all Gaussians culled).
 */
export function encodeTileKeyPass(encoder: GPUCommandEncoder, pass: TileKeyPass): void {
  if (pass.totalIntersections === 0) return;
  const cp = encoder.beginComputePass({ label: 'tile-depth-key' });
  cp.setPipeline(pass.pipeline);
  cp.setBindGroup(0, pass.bindGroup);
  cp.dispatchWorkgroups(Math.ceil(pass.numGaussians / 256));
  cp.end();
}
