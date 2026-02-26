import type { GaussianSceneData } from './types.ts';

// ---------------------------------------------------------------------------
// antimatter15 .splat binary loader
//
// Fixed 32 bytes per Gaussian, no header:
//   [0..11]   float32 × 3   position (x, y, z)
//   [12..23]  float32 × 3   scale (linear, exp already applied → log needed)
//   [24..27]  uint8 × 4     RGBA color (DC term baked in; A = opacity 0-255)
//   [28..31]  uint8 × 4     quaternion wxyz, packed as round(q * 128 + 128)
// ---------------------------------------------------------------------------

const SPLAT_ROW = 32;
const SH_C0 = 0.28209479177387814;

export function parseSplat(buffer: ArrayBuffer): GaussianSceneData {
  if (buffer.byteLength % SPLAT_ROW !== 0) {
    throw new Error(`.splat: byteLength ${buffer.byteLength} is not a multiple of 32`);
  }
  const N = buffer.byteLength / SPLAT_ROW;
  const view = new DataView(buffer);

  const positions = new Float32Array(N * 3);
  const logScales = new Float32Array(N * 3);
  const rotations = new Float32Array(N * 4);
  const opacities = new Float32Array(N);
  const shDC      = new Float32Array(N * 3);

  for (let i = 0; i < N; i++) {
    const base = i * SPLAT_ROW;

    // Position
    positions[i * 3 + 0] = view.getFloat32(base + 0,  true);
    positions[i * 3 + 1] = view.getFloat32(base + 4,  true);
    positions[i * 3 + 2] = view.getFloat32(base + 8,  true);

    // Scale: stored as exp(logScale) — take log to get log-space
    const sx = view.getFloat32(base + 12, true);
    const sy = view.getFloat32(base + 16, true);
    const sz = view.getFloat32(base + 20, true);
    logScales[i * 3 + 0] = Math.log(Math.max(sx, 1e-10));
    logScales[i * 3 + 1] = Math.log(Math.max(sy, 1e-10));
    logScales[i * 3 + 2] = Math.log(Math.max(sz, 1e-10));

    // Color: uint8 RGB in [0,255] → linear [0,1] → inverse SH DC
    const cr = view.getUint8(base + 24) / 255;
    const cg = view.getUint8(base + 25) / 255;
    const cb = view.getUint8(base + 26) / 255;
    shDC[i * 3 + 0] = (cr - 0.5) / SH_C0;
    shDC[i * 3 + 1] = (cg - 0.5) / SH_C0;
    shDC[i * 3 + 2] = (cb - 0.5) / SH_C0;

    // Opacity: uint8 in [0,255] → alpha → logit
    const alpha = view.getUint8(base + 27) / 255;
    opacities[i] = Math.log(Math.max(alpha, 1e-10) / Math.max(1 - alpha, 1e-10));

    // Quaternion: uint8 packed as round(q * 128 + 128), order wxyz
    const qw = (view.getUint8(base + 28) - 128) / 128;
    const qx = (view.getUint8(base + 29) - 128) / 128;
    const qy = (view.getUint8(base + 30) - 128) / 128;
    const qz = (view.getUint8(base + 31) - 128) / 128;
    const qlen = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) || 1;
    rotations[i * 4 + 0] = qw / qlen;
    rotations[i * 4 + 1] = qx / qlen;
    rotations[i * 4 + 2] = qy / qlen;
    rotations[i * 4 + 3] = qz / qlen;
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
