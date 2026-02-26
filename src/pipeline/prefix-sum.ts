import shaderCode from '../shaders/prefix-sum.wgsl?raw';

// ---------------------------------------------------------------------------
// Hierarchical exclusive prefix sum on a flat u32 array.
//
// Works for N up to 256³ = 16,777,216 elements (more than enough for splats).
//
// Algorithm (5 passes for N > 256, 2 passes for N ≤ 256):
//
//   B0 = ceil(N  / 256)   — workgroups for the input array
//   B1 = ceil(B0 / 256)   — workgroups for the first block-totals array
//
//   Pass 1: local_scan(input   → cum,     block_A)   dispatch B0
//   Pass 2: local_scan(block_A → block_B, block_C)   dispatch B1
//   Pass 3: local_scan(block_C → block_D, _dummy)    dispatch 1   (≤256 items)
//   Pass 4: add_blocks(block_B, offset=block_D)      dispatch B1  → block_B correct
//   Pass 5: add_blocks(cum,     offset=block_B)      dispatch B0  → cum correct
//
// When B0 == 1 (N ≤ 256) only Pass 1 is needed.
// When B1 == 1 (N ≤ 65536) Passes 1+2+5 suffice (pass 5 uses block_A as offset).
// ---------------------------------------------------------------------------

/** Cached shader module and pipeline pair (one per device). */
interface PrefixSumPipelines {
  localScan: GPUComputePipeline;
  addBlocks: GPUComputePipeline;
  bgl: GPUBindGroupLayout;
}

let cachedDevice: GPUDevice | null = null;
let cachedPipelines: PrefixSumPipelines | null = null;

function getPipelines(device: GPUDevice): PrefixSumPipelines {
  if (device === cachedDevice && cachedPipelines) return cachedPipelines;

  const module = device.createShaderModule({ label: 'prefix-sum', code: shaderCode });

  const bgl = device.createBindGroupLayout({
    label: 'prefix-sum-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  const localScan = device.createComputePipeline({
    label: 'prefix-sum/local_scan',
    layout,
    compute: { module, entryPoint: 'local_scan' },
  });

  const addBlocks = device.createComputePipeline({
    label: 'prefix-sum/add_blocks',
    layout,
    compute: { module, entryPoint: 'add_blocks' },
  });

  cachedDevice = device;
  cachedPipelines = { localScan, addBlocks, bgl };
  return cachedPipelines;
}

// ---------------------------------------------------------------------------
// Helper: allocate a temporary u32 storage buffer
// ---------------------------------------------------------------------------
function tmpBuf(device: GPUDevice, numElements: number, label: string): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(numElements, 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

// ---------------------------------------------------------------------------
// Helper: create a 4-byte uniform buffer containing num_items
// ---------------------------------------------------------------------------
function uniformBuf(device: GPUDevice, numItems: number): GPUBuffer {
  const buf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, new Uint32Array([numItems]));
  return buf;
}

// ---------------------------------------------------------------------------
// Helper: encode one local_scan or add_blocks dispatch
// ---------------------------------------------------------------------------
function encodePass(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bgl: GPUBindGroupLayout,
  device: GPUDevice,
  numItems: number,
  input: GPUBuffer,
  output: GPUBuffer,
  blocks: GPUBuffer,
  label: string,
): void {
  const uniforms = uniformBuf(device, numItems);
  const bg = device.createBindGroup({
    label,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniforms } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: blocks } },
    ],
  });

  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(numItems / 256));
  pass.end();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run an exclusive prefix sum on `inputBuf` (flat array of N u32 values).
 * The result is written back into `inputBuf` in-place.
 * Returns the total sum (sum of all input elements) via an async GPU readback.
 *
 * @param device     WebGPU device
 * @param inputBuf   STORAGE | COPY_SRC | COPY_DST buffer of N u32
 * @param N          Number of elements
 * @returns          Promise resolving to the total sum of the original input
 */
export async function runPrefixSum(
  device: GPUDevice,
  inputBuf: GPUBuffer,
  N: number,
): Promise<number> {
  const { localScan, addBlocks, bgl } = getPipelines(device);

  const B0 = Math.ceil(N / 256); // workgroups at level 0
  const B1 = Math.ceil(B0 / 256); // workgroups at level 1

  // Allocate intermediate buffers
  const cum = tmpBuf(device, N, 'ps-cum'); // level-0 output (becomes final result)
  const blockA = tmpBuf(device, B0, 'ps-block-a'); // level-0 block totals
  const blockB = tmpBuf(device, B0, 'ps-block-b'); // level-0 block prefix sums (corrected)
  const blockC = tmpBuf(device, B1, 'ps-block-c'); // level-1 block totals
  const blockD = tmpBuf(device, B1, 'ps-block-d'); // level-1 block prefix sums

  const encoder = device.createCommandEncoder({ label: 'prefix-sum' });

  if (B0 === 1) {
    // N ≤ 256: single pass writes directly into cum
    encodePass(encoder, localScan, bgl, device, N, inputBuf, cum, blockA, 'ps-L0');
  } else if (B1 === 1) {
    // 256 < N ≤ 65536: two scan passes + one add pass
    encodePass(encoder, localScan, bgl, device, N, inputBuf, cum, blockA, 'ps-L0');
    encodePass(encoder, localScan, bgl, device, B0, blockA, blockB, blockC, 'ps-L1');
    encodePass(encoder, addBlocks, bgl, device, N, inputBuf, cum, blockB, 'ps-add0');
  } else {
    // N > 65536: full 5-pass hierarchy
    encodePass(encoder, localScan, bgl, device, N, inputBuf, cum, blockA, 'ps-L0');
    encodePass(encoder, localScan, bgl, device, B0, blockA, blockB, blockC, 'ps-L1');
    encodePass(encoder, localScan, bgl, device, B1, blockC, blockD, blockA, 'ps-L2');
    // Pass 4: add level-2 offsets into level-1 sums
    encodePass(encoder, addBlocks, bgl, device, B0, blockC, blockB, blockD, 'ps-add1');
    // Pass 5: add corrected level-1 offsets into level-0 output
    encodePass(encoder, addBlocks, bgl, device, N, inputBuf, cum, blockB, 'ps-add0');
  }

  // Copy result back into inputBuf (in-place semantics)
  encoder.copyBufferToBuffer(cum, 0, inputBuf, 0, N * 4);

  // Read the total: it's stored in blockA[0] (the sum of all level-0 blocks after pass 1)
  // For the corrected total we read blockA[0] which is block 0's total — not right.
  // The global total = last element of the prefix sum + last element of the original.
  // Easier: for N≤256 it's in blockA[0]; for larger N the full total is in blockC[0] or blockD[0].
  // Simplest reliable approach: read blockA[0] — which contains block 0's sum — then... no.
  // Actually: blockA[B0-1] = sum of elements [(B0-1)*256 .. N-1].
  // The total = cum[N-1] + original[N-1], but we've already overwritten inputBuf.
  // Store total in a dedicated 4-byte buffer read from the last slot of the highest block array.
  const totalReadBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // The grand total is always the sum of all block_A entries.
  // After pass 2, blockB is the prefix-sum of blockA; blockC[0] = sum(blockA).
  // After pass 3, blockD is prefix-sum of blockC; and blockA[0] (reused as dummy) = sum(blockC).
  // Easiest: read blockC[0] which = sum of all blockA = total for N>256,
  //          or blockA[0] = total for N≤256.
  if (B0 === 1) {
    encoder.copyBufferToBuffer(blockA, 0, totalReadBuf, 0, 4);
  } else if (B1 === 1) {
    encoder.copyBufferToBuffer(blockC, 0, totalReadBuf, 0, 4);
  } else {
    // blockA was reused as the dummy "blocks" buffer for pass 3, so blockA[0] = sum(blockC) = grand total
    encoder.copyBufferToBuffer(blockA, 0, totalReadBuf, 0, 4);
  }

  device.queue.submit([encoder.finish()]);

  // Read total
  await totalReadBuf.mapAsync(GPUMapMode.READ);
  const total = new Uint32Array(totalReadBuf.getMappedRange())[0]!;
  totalReadBuf.unmap();
  totalReadBuf.destroy();

  // Clean up temporaries
  cum.destroy();
  blockA.destroy();
  blockB.destroy();
  blockC.destroy();
  blockD.destroy();

  return total;
}
