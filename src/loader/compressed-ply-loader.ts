import type { GaussianSceneData } from './types.ts';

// ---------------------------------------------------------------------------
// SuperSplat Compressed PLY decoder
//
// Binary layout (after end_header):
//   [Chunk table]  nChunks × 18 × float32  (72 bytes / chunk)
//   [Vertex data]  nVerts  ×  4 × uint32   (16 bytes / Gaussian)
//
// Chunk table (18 floats per chunk, 256 Gaussians per chunk):
//   [0-2]  min_xyz    [3-5]  max_xyz
//   [6-8]  min_scale  [9-11] max_scale   (log-space)
//   [12-14] min_rgb   [15-17] max_rgb
//
// Vertex packing (4 uint32s per Gaussian):
//   packed_position: X[31..21] Y[20..11] Z[10..0]       11+10+11 unorm
//   packed_rotation: mode[31..30] A[29..20] B[19..10] C[9..0]  smallest-three quat
//   packed_scale:    SX[31..21] SY[20..11] SZ[10..0]    11+10+11 unorm (log-space)
//   packed_color:    R[31..24] G[23..16] B[15..8] A[7..0]  8+8+8+8
// ---------------------------------------------------------------------------

const GAUSSIANS_PER_CHUNK = 256;
const CHUNK_FLOATS = 18;
const CHUNK_BYTES = CHUNK_FLOATS * 4; // 72 bytes/chunk
const VERTEX_BYTES = 16;              // 4 × uint32 per Gaussian

const SH_C0 = 0.28209479177387814;

/** Interpret the low `bits` bits of `val` as an unsigned normalized float in [0, 1]. */
function unorm(val: number, bits: number): number {
  const mask = (1 << bits) - 1;
  return (val & mask) / mask;
}

export function parseCompressedPLY(buffer: ArrayBuffer): GaussianSceneData {
  // ---- Header ---------------------------------------------------------------
  const headerBytes = new Uint8Array(buffer, 0, Math.min(65536, buffer.byteLength));
  const text = new TextDecoder().decode(headerBytes);

  const marker = 'end_header\n';
  const markerIdx = text.indexOf(marker);
  if (markerIdx === -1) throw new Error('Compressed PLY: missing end_header');

  const headerText = text.slice(0, markerIdx);
  const dataOffset = new TextEncoder().encode(text.slice(0, markerIdx + marker.length)).length;

  const chunkMatch = headerText.match(/element chunk (\d+)/);
  if (!chunkMatch) throw new Error('Compressed PLY: missing element chunk');
  const nChunks = parseInt(chunkMatch[1]!);

  const vertexMatch = headerText.match(/element vertex (\d+)/);
  if (!vertexMatch) throw new Error('Compressed PLY: missing element vertex');
  const N = parseInt(vertexMatch[1]!);

  // ---- DataViews ------------------------------------------------------------
  const chunkDataOffset  = dataOffset;
  const vertexDataOffset = dataOffset + nChunks * CHUNK_BYTES;

  const chunkView  = new DataView(buffer, chunkDataOffset,  nChunks * CHUNK_BYTES);
  const vertexView = new DataView(buffer, vertexDataOffset, N * VERTEX_BYTES);

  // ---- Output arrays --------------------------------------------------------
  const positions = new Float32Array(N * 3);
  const logScales = new Float32Array(N * 3);
  const rotations = new Float32Array(N * 4);
  const opacities = new Float32Array(N);
  const shDC      = new Float32Array(N * 3);

  // ---- Decode ---------------------------------------------------------------
  // Cache current chunk bounds to avoid re-reading for every Gaussian
  let curChunk = -1;
  let minX = 0, maxX = 0, minY = 0, maxY = 0, minZ = 0, maxZ = 0;
  let minSX = 0, maxSX = 0, minSY = 0, maxSY = 0, minSZ = 0, maxSZ = 0;
  let minR = 0, maxR = 0, minG = 0, maxG = 0, minB = 0, maxB = 0;

  for (let i = 0; i < N; i++) {
    const chunkIdx = Math.floor(i / GAUSSIANS_PER_CHUNK);

    if (chunkIdx !== curChunk) {
      curChunk = chunkIdx;
      const co = chunkIdx * CHUNK_BYTES;
      minX  = chunkView.getFloat32(co +  0, true);
      minY  = chunkView.getFloat32(co +  4, true);
      minZ  = chunkView.getFloat32(co +  8, true);
      maxX  = chunkView.getFloat32(co + 12, true);
      maxY  = chunkView.getFloat32(co + 16, true);
      maxZ  = chunkView.getFloat32(co + 20, true);
      minSX = chunkView.getFloat32(co + 24, true);
      minSY = chunkView.getFloat32(co + 28, true);
      minSZ = chunkView.getFloat32(co + 32, true);
      maxSX = chunkView.getFloat32(co + 36, true);
      maxSY = chunkView.getFloat32(co + 40, true);
      maxSZ = chunkView.getFloat32(co + 44, true);
      minR  = chunkView.getFloat32(co + 48, true);
      minG  = chunkView.getFloat32(co + 52, true);
      minB  = chunkView.getFloat32(co + 56, true);
      maxR  = chunkView.getFloat32(co + 60, true);
      maxG  = chunkView.getFloat32(co + 64, true);
      maxB  = chunkView.getFloat32(co + 68, true);
    }

    const vo = i * VERTEX_BYTES;
    const pp = vertexView.getUint32(vo + 0,  true); // packed_position
    const pr = vertexView.getUint32(vo + 4,  true); // packed_rotation
    const ps = vertexView.getUint32(vo + 8,  true); // packed_scale
    const pc = vertexView.getUint32(vo + 12, true); // packed_color

    // Position: X[31..21] Y[20..11] Z[10..0]  (11+10+11 unorm)
    positions[i * 3 + 0] = minX + unorm(pp >>> 21, 11) * (maxX - minX);
    positions[i * 3 + 1] = minY + unorm(pp >>> 11, 10) * (maxY - minY);
    positions[i * 3 + 2] = minZ + unorm(pp,        11) * (maxZ - minZ);

    // Scale: SX[31..21] SY[20..11] SZ[10..0]  (11+10+11 unorm, already log-space)
    logScales[i * 3 + 0] = minSX + unorm(ps >>> 21, 11) * (maxSX - minSX);
    logScales[i * 3 + 1] = minSY + unorm(ps >>> 11, 10) * (maxSY - minSY);
    logScales[i * 3 + 2] = minSZ + unorm(ps,        11) * (maxSZ - minSZ);

    // Rotation: mode[31..30] A[29..20] B[19..10] C[9..0]  (smallest-three quaternion)
    const mode = pr >>> 30;
    const a = (unorm(pr >>> 20, 10) - 0.5) * Math.SQRT2;
    const b = (unorm(pr >>> 10, 10) - 0.5) * Math.SQRT2;
    const c = (unorm(pr,        10) - 0.5) * Math.SQRT2;
    const m = Math.sqrt(Math.max(0, 1 - a * a - b * b - c * c));

    // mode = index of the omitted (largest) component; reconstruct wxyz
    let rw: number, rx: number, ry: number, rz: number;
    switch (mode) {
      case 0: rw = m; rx = a; ry = b; rz = c; break;
      case 1: rw = a; rx = m; ry = b; rz = c; break;
      case 2: rw = a; rx = b; ry = m; rz = c; break;
      default: rw = a; rx = b; ry = c; rz = m; break;
    }
    rotations[i * 4 + 0] = rw;
    rotations[i * 4 + 1] = rx;
    rotations[i * 4 + 2] = ry;
    rotations[i * 4 + 3] = rz;

    // Color: R[31..24] G[23..16] B[15..8] A[7..0]  (8+8+8+8)
    const r  = minR + ((pc >>> 24) / 255)         * (maxR - minR);
    const g  = minG + (((pc >>> 16) & 0xFF) / 255) * (maxG - minG);
    const b_ = minB + (((pc >>>  8) & 0xFF) / 255) * (maxB - minB);
    shDC[i * 3 + 0] = (r  - 0.5) / SH_C0;
    shDC[i * 3 + 1] = (g  - 0.5) / SH_C0;
    shDC[i * 3 + 2] = (b_ - 0.5) / SH_C0;

    // Opacity: raw uint8 → logit(α)
    const alpha = (pc & 0xFF) / 255;
    opacities[i] = Math.log(Math.max(alpha, 1e-10) / Math.max(1 - alpha, 1e-10));
  }

  return {
    kind: 'gaussian',
    numGaussians: N,
    positions,
    logScales,
    rotations,
    opacities,
    shDC,
    shRest: new Float32Array(0),
    shDegree: 0,
  };
}
