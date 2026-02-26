// =============================================================================
// prefix-sum.wgsl — Generic exclusive prefix sum on a flat u32 array.
//
// Two entry points share one bind group layout:
//   @binding(0)  ps_uniforms  — { num_items: u32 }
//   @binding(1)  ps_input     — read-only u32 source
//   @binding(2)  ps_output    — read-write u32 destination (also used in-place)
//   @binding(3)  ps_blocks    — read-write u32 per-workgroup totals / offsets
//
// local_scan  — dispatch ceil(N/256) workgroups
//   Thread 0 in each workgroup runs a sequential exclusive scan over the
//   256 shared-memory slots, then all threads write their result.
//   Block total written to ps_blocks[workgroup_id].
//
// add_blocks  — dispatch ceil(N/256) workgroups
//   Adds ps_blocks[workgroup_id] to ps_output[i] in-place.
//   (ps_input is bound for layout compatibility but not read.)
// =============================================================================

struct PrefixSumUniforms {
  num_items: u32,
}

@group(0) @binding(0) var<uniform>             ps_uniforms: PrefixSumUniforms;
@group(0) @binding(1) var<storage, read>       ps_input:    array<u32>;
@group(0) @binding(2) var<storage, read_write> ps_output:   array<u32>;
@group(0) @binding(3) var<storage, read_write> ps_blocks:   array<u32>;

var<workgroup> wg_data: array<u32, 256>;

// ---------------------------------------------------------------------------
// local_scan
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn local_scan(
  @builtin(global_invocation_id) gid:  vec3<u32>,
  @builtin(local_invocation_id)  lid:  vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) {
  let i       = gid.x;
  let local_i = lid.x;

  // Load element — 0 if out of bounds
  wg_data[local_i] = select(0u, ps_input[i], i < ps_uniforms.num_items);
  workgroupBarrier();

  // Thread 0: sequential exclusive scan + capture block total
  if (local_i == 0u) {
    var running = 0u;
    for (var j = 0u; j < 256u; j++) {
      let val    = wg_data[j];
      wg_data[j] = running; // exclusive: store prefix before adding val
      running   += val;
    }
    ps_blocks[wgid.x] = running;
  }
  workgroupBarrier();

  // All threads write their local exclusive prefix sum
  if (i < ps_uniforms.num_items) {
    ps_output[i] = wg_data[local_i];
  }
}

// ---------------------------------------------------------------------------
// add_blocks — adds block offset to each element in-place
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn add_blocks(
  @builtin(global_invocation_id) gid:  vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) {
  let i = gid.x;
  if (i < ps_uniforms.num_items) {
    ps_output[i] += ps_blocks[wgid.x];
  }
}
