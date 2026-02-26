// =============================================================================
// tile-depth-key.wgsl — Expand each visible Gaussian into per-tile sort keys.
//
// Dispatch: ceil(numGaussians / 256) workgroups × 256 threads
//
// Each visible Gaussian (radii > 0) writes one entry per tile it overlaps:
//   keys_hi[offset + k] = tile_index                (u32, for upper sort key)
//   keys_lo[offset + k] = bitcast<u32>(depth)       (u32, positive float → correct order)
//   values[offset + k]  = gaussian_index            (u32)
//
// offset = cum_tiles[idx]  (exclusive prefix sum from Phase 3a).
// =============================================================================

struct TileKeyUniforms {
  screen_width:  u32,
  screen_height: u32,
  tile_size:     u32,
  _pad:          u32,
}

// Mirror of GaussData in preprocess.wgsl (same layout).
struct GaussData {
  id:                i32,
  radii:             i32,
  depth:             f32,
  tiles_touched:     u32,
  cum_tiles_touched: u32,
  _pad0:             u32,
  uv:                vec2<f32>,
  conic:             vec3<f32>,
  _pad1:             f32,
  color:             vec3<f32>,
  opacity:           f32,
}

@group(0) @binding(0) var<uniform>             tk_uniforms: TileKeyUniforms;
@group(0) @binding(1) var<storage, read>       gauss_data:  array<GaussData>;
@group(0) @binding(2) var<storage, read>       cum_tiles:   array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_hi:     array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_lo:     array<u32>;
@group(0) @binding(5) var<storage, read_write> values:      array<u32>;

@compute @workgroup_size(256, 1, 1)
fn tile_depth_key(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&gauss_data)) { return; }

  let g = gauss_data[idx];
  if (g.radii == 0) { return; }

  let ntx = (tk_uniforms.screen_width  + tk_uniforms.tile_size - 1u) / tk_uniforms.tile_size;
  let nty = (tk_uniforms.screen_height + tk_uniforms.tile_size - 1u) / tk_uniforms.tile_size;
  let ts  = f32(tk_uniforms.tile_size);
  let r   = f32(g.radii);
  let px  = g.uv * vec2<f32>(f32(tk_uniforms.screen_width), f32(tk_uniforms.screen_height));

  let t_min = max(vec2<i32>(0),                    vec2<i32>((px - r) / ts));
  let t_max = min(vec2<i32>(i32(ntx), i32(nty)),  vec2<i32>((px + r) / ts) + vec2<i32>(1));

  // Positive float depth → bitcast preserves sort order for u32 radix sort.
  let depth_bits = bitcast<u32>(g.depth);
  var offset     = cum_tiles[idx];

  for (var ty = t_min.y; ty < t_max.y; ty++) {
    for (var tx = t_min.x; tx < t_max.x; tx++) {
      let tile_id     = u32(ty) * ntx + u32(tx);
      keys_hi[offset] = tile_id;
      keys_lo[offset] = depth_bits;
      values[offset]  = idx;
      offset++;
    }
  }
}
