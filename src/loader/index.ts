import type { SceneData } from './types.ts';
import { detectPLYSubtype, parseGaussianPLY, parsePointCloudPLY } from './ply-parser.ts';
import { parseCompressedPLY } from './compressed-ply-loader.ts';
import { parseSplat } from './splat-loader.ts';

// ---------------------------------------------------------------------------
// Format detection + dispatch
//
// Detection order:
//   1. PLY magic bytes (0x70 0x6C 0x79 0x0A = "ply\n")
//      → inspect header for element chunk / f_dc_0 to pick sub-parser
//   2. gzip magic (0x1F 0x8B) → SPZ [Phase 7]
//   3. "KSpl" ASCII magic → KSPLAT [Phase 7]
//   4. Fallback: .splat binary (fixed 32-byte stride, no header)
// ---------------------------------------------------------------------------

export async function loadScene(file: File): Promise<SceneData> {
  const buffer = await file.arrayBuffer();
  const u8 = new Uint8Array(buffer, 0, Math.min(4, buffer.byteLength));

  // PLY family
  if (u8[0] === 0x70 && u8[1] === 0x6c && u8[2] === 0x79 && u8[3] === 0x0a) {
    const headerText = new TextDecoder().decode(
      new Uint8Array(buffer, 0, Math.min(4096, buffer.byteLength)),
    );
    const subtype = detectPLYSubtype(headerText);
    if (subtype === 'compressed') return parseCompressedPLY(buffer);
    if (subtype === 'gaussian') return parseGaussianPLY(buffer);
    return parsePointCloudPLY(buffer);
  }

  // SPZ (gzip magic) — Phase 7
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    throw new Error('SPZ format is not yet supported (Phase 7).');
  }

  // KSPLAT magic "KSpl" — Phase 7
  if (u8[0] === 0x4b && u8[1] === 0x53 && u8[2] === 0x70 && u8[3] === 0x6c) {
    throw new Error('KSPLAT format is not yet supported (Phase 7).');
  }

  // Fallback: .splat binary
  return parseSplat(buffer);
}

export type { SceneData } from './types.ts';
