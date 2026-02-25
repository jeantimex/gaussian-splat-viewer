# Gaussian Splat Viewer — Implementation Plan (WebGPU)

## 1. Background: What is 3D Gaussian Splatting?

3D Gaussian Splatting (Kerbl et al., 2023) represents a scene as a set of **oriented 3D Gaussians**. Each
Gaussian encodes:

- A 3D position (mean)
- A 3D covariance (ellipsoid shape, factored as scale + rotation)
- An opacity
- A view-dependent color via Spherical Harmonics (SH) coefficients

Rendering is performed by projecting each Gaussian into screen space as an oriented 2D Gaussian "splat",
then alpha-compositing them back-to-front (painter's algorithm). The key challenge is doing this efficiently
for millions of Gaussians in real time.

---

## 2. Reference Implementation Analysis

### 2.1 WebGL Reference (`antimatter15/splat`)

**Architecture**: Single HTML file, pure JS + WebGL 2.0, no dependencies.

**Pipeline**:

1. Fetch `.splat` binary (or convert `.ply` → `.splat` in a Web Worker)
2. Web Worker: depth-sort Gaussians every frame (16-bit counting sort, O(n))
3. Upload sorted index buffer and Gaussian texture to GPU
4. GPU vertex shader: project 3D Gaussian center, compute 2D covariance via Jacobian,
   decompose into eigen-axes, emit oriented quad
5. GPU fragment shader: evaluate Gaussian falloff `exp(-(x² + y²))`, multiply by opacity
6. GL blend: `ONE_MINUS_DST_ALPHA / ONE` for back-to-front premultiplied alpha compositing

**Data format** — `.splat` binary, 32 bytes per Gaussian:

```
[0..11]  Float32 × 3    position x, y, z
[12..23] Float32 × 3    scale sx, sy, sz  (exp already applied)
[24..27] Uint8 × 4      RGBA color (SH DC term baked in)
[28..31] Uint8 × 4      quaternion (packed as q/128+128 → [0,255])
```

**Covariance computation** (done in Web Worker once, stored in RGBA32UI texture as half-float16 pairs):

```
S = diag(scale)
R = quaternion_to_rotation_matrix(q)
M = S * R
Sigma_3D = Mᵀ * M          (symmetric 3×3, 6 unique values)
```

**2D projection** (done per-frame in vertex shader):

```
J = perspective_jacobian(fx, fy, x_cam, y_cam, z_cam)
Cov_2D = Jᵀ · Sigma_3D · J    (upper-left 2×2 block)
[λ1, λ2, v1, v2] = eigen2x2(Cov_2D)
quad size = sqrt(2*λi) along each eigen-axis
```

**Sorting**: CPU counting sort in Web Worker; posts `depthIndex: Uint32Array` each frame.
Skips re-sort if camera dot product change < 0.01.

**Limitations**:

- Sorting is on CPU → bottleneck at ~3M splats
- Only SH degree-0 (DC color), no view-dependent effects
- No compute shaders

---

### 2.2 WebGPU Reference (`gaussian-splatting-webgpu`)

**Architecture**: TypeScript + WGSL, multiple compute passes, GPU radix sort.

**Full Pipeline** (all GPU except PLY parsing):

```
[CPU] Load & parse PLY
        ↓
[CPU] Pack binary (position + log-scale + quaternion + opacity-logit + SH coefficients)
        ↓
[GPU Compute Pass 1] Gaussian Preprocessor  (gauss.wgsl)
   - Frustum cull (z < 0.2, xy outside [-1.1, 1.1] NDC)
   - Opacity cull (sigmoid(logit) < 0.03)
   - 3D covariance from scale/rotation
   - Project to 2D covariance (EWA splatting, Zwicker 2002)
   - Inverse conic computation (for per-pixel evaluation)
   - Compute screen-space radius and bounding box
   - Count how many 16×16 tiles each Gaussian overlaps (atomic adds)
   - Evaluate SH (degree 0+1 currently) → RGB color
        ↓
[GPU Compute Pass 2] Prefix Sum on tile intersection counts
   - Converts per-tile intersection counts to cumulative offsets
        ↓
[GPU Compute Pass 3] Per-Gaussian prefix sum (cum_tiles_touched)
        ↓
[GPU Compute Pass 4] Tile-Depth Key Generation  (tileDepthKey.wgsl)
   - For each Gaussian × tile pair, emit:
     key = (tile_index << 16) | normalized_depth_uint16
     value = gaussian_id
        ↓
[GPU Compute] Radix Sort (webgpu-radix-sort, 32-bit keys)
   - Sorts (key, gaussian_id) pairs by tile then by depth within tile
        ↓
[GPU Compute Pass 5] Tile Rasterizer  (rasterize.wgsl)
   - One workgroup per 16×16 tile
   - Each thread covers 4×2 = 8 pixels
   - Iterates over sorted Gaussians for its tile
   - Per-pixel: power = -0.5*(A·dx²+ C·dy² + 2B·dx·dy), alpha = opacity*exp(power)
   - Alpha-over blend: color += gauss_color * alpha * transparency; transparency *= (1 - alpha)
   - Early exit when max_transparency < 1/255
   - Writes rgba8unorm to storage texture
        ↓
[GPU Render Pass] Fullscreen Quad (screen.wgsl)
   - Samples the rasterized texture and blits to canvas
```

**GaussData struct** output from preprocessor:

```wgsl
struct GaussData {
  id:              i32,
  radii:           i32,       // 0 = culled
  depth:           f32,
  tiles_touched:   u32,
  cum_tiles_touched: u32,
  uv:              vec2<f32>, // screen-space center [0,1]
  conic:           vec3<f32>, // inverse 2D covariance [C/det, -B/det, A/det]
  color:           vec3<f32>,
  opacity:         f32,
}
```

**Key differences from WebGL version**:

- Sorting entirely on GPU (radix sort)
- Tile-based deferred rendering (each tile only touches its own Gaussians)
- Compute-shader rasterizer instead of GPU raster pipeline
- Stores PLY data in full precision (SH coefficients)
- 16×16 tile granularity

---

## 3. Our Implementation Plan

We will build a **WebGPU-native** Gaussian Splat Viewer inspired by both references, optimized for
clarity, modularity, and correctness. The architecture closely follows the WebGPU reference but is
implemented cleanly from scratch in TypeScript + WGSL.

---

### 3.1 Project Structure

```
src/
├── main.ts                     Entry point, ties everything together
├── renderer.ts                 Top-level WebGPU orchestrator
├── camera.ts                   Camera model + mouse/keyboard controls
│
├── loader/
│   ├── index.ts                Format detection + dispatch to correct parser
│   ├── types.ts                SceneData discriminated union (gaussian | pointcloud)
│   ├── ply-parser.ts           PLY → Gaussian (has f_dc_0) or PointCloud (xyz+rgb)
│   ├── splat-loader.ts         Load .splat binary (32 B/Gaussian, DC only)
│   ├── xyz-loader.ts           Plain-text point cloud (x y z [r g b])
│   ├── pcd-loader.ts           PCL PCD format (ASCII + binary) [Phase 7]
│   ├── las-loader.ts           LAS/LAZ LiDAR format [Phase 8]
│   ├── spz-loader.ts           Load .spz (gzip + fixed-point quantization) [Phase 7]
│   ├── ksplat-loader.ts        Load .ksplat chunk format [Phase 7]
│   └── camera-loader.ts        Parse cameras.json + auto-fit fallback
│
├── gpu/
│   ├── gpu-context.ts          Adapter, device, canvas context setup
│   ├── buffers.ts              Buffer allocation helpers
│   ├── uniforms.ts             Uniform struct layout + upload
│   └── tile-cache.ts           GPU slot allocator + LRU eviction [Phase 7]
│
├── pipeline/
│   ├── preprocess.ts           Orchestrates the 4 compute pre-passes
│   ├── sort.ts                 GPU radix sort wrapper
│   ├── rasterize.ts            Compute rasterizer pass
│   ├── tile-manager.ts         Frustum-based tile selection + load queue [Phase 7]
│   └── octree-traversal.ts     SSE-driven octree node traversal [Phase 8]
│
└── shaders/
    ├── preprocess.wgsl         Gaussian preprocessing compute shader
    ├── prefix-sum.wgsl         Tile intersection prefix sum
    ├── tile-depth-key.wgsl     Tile×depth key generation
    ├── rasterize.wgsl          Tile-based alpha compositing (Gaussian)
    ├── point-rasterize.wgsl    Atomic depth+color framebuffer (point cloud) [Phase 8]
    └── screen.wgsl             Fullscreen blit vertex+fragment
```

---

### 3.2 Step-by-Step Implementation

---

#### Step 1 — PLY Loader (`loader/ply-parser.ts`)

**Goal**: Parse a standard `.ply` file and return a flat `Float32Array` + metadata.

**Algorithm**:

1. Detect `ply\n` magic bytes
2. Parse ASCII header line-by-line:
   - Extract `element vertex N`
   - For each `property <type> <name>`, record byte offset and accessor type
3. Locate `end_header\n` and treat everything after as binary vertex data
4. For each vertex, read fields using `DataView`:
   - `x`, `y`, `z` → position (Float32)
   - `scale_0..2` → log-scale (Float32)
   - `rot_0..3` → quaternion (Float32)
   - `opacity` → logit (Float32)
   - `f_dc_0..2` → SH DC coefficients (Float32)
   - `f_rest_0..N` → higher SH coefficients if present

**Output struct per Gaussian** (will be packed into GPU buffer):

```typescript
interface GaussianData {
  position:   [number, number, number];   // xyz
  logScale:   [number, number, number];   // log-space scales
  rotation:   [number, number, number, number]; // quaternion wxyz
  opacityLogit: number;
  shDC:       [number, number, number];   // degree-0 SH (DC color)
  shRest:     number[];                   // higher-order SH coeffs
}
```

**Notes**:

- All fields stored at full Float32 precision
- SH degree determined from rest count: `degree = sqrt(nRestCoeffs/3 + 1) - 1`
- Support progressive loading: parse in chunks, yield partial arrays

---

#### Step 2 — GPU Buffer Layout

**Input buffer** (one large `GPUBuffer`, `STORAGE | COPY_DST`):

```
Per Gaussian (stride = 4 bytes aligned):
  position:    vec3<f32>   offset 0   (12 bytes + 4 pad = 16)
  log_scale:   vec3<f32>   offset 16  (12 bytes + 4 pad = 16)
  rotation:    vec4<f32>   offset 32  (16 bytes)
  opacity:     f32         offset 48  (4 bytes)
  sh_dc:       vec3<f32>   offset 52  (12 bytes)  ← DC color
  sh_rest:     array<f32>  offset 64  (48*3*4 bytes for degree-3)
```

Use `@align(16)` WGSL structs to match WebGPU alignment requirements.

**Preprocessed output buffer** — one `GaussData` per Gaussian (same as WebGPU reference).

**Tile-depth key buffer** — worst-case: `numGaussians × maxTilesPerSplat` entries, pre-allocated.

**Intersection offsets buffer** — one `u32` per tile = `ceil(W/16) × ceil(H/16)` entries.

---

#### Step 3 — Uniform Buffer (`gpu/uniforms.ts`)

Updated each frame before any compute pass:

```wgsl
struct Uniforms {
  view_matrix:      mat4x4<f32>,   // world-to-camera
  proj_matrix:      mat4x4<f32>,   // camera-to-clip
  camera_pos:       vec3<f32>,     // world position
  tan_half_fov_x:   f32,
  tan_half_fov_y:   f32,
  focal_x:          f32,
  focal_y:          f32,
  scale_modifier:   f32,
  screen_size:      vec2<u32>,     // canvas width, height
  num_gaussians:    u32,
  tile_size:        u32,           // 16
}
```

---

#### Step 4 — Preprocessing Compute Shader (`shaders/preprocess.wgsl`)

**Dispatch**: `ceil(numGaussians / 256)` workgroups, 256 threads each.

**Per-Gaussian operations**:

1. **Load** position, scale, rotation, opacity, SH coefficients from input buffer

2. **Transform to view/clip space**:

   ```wgsl
   let pos_view = uniforms.view_matrix * vec4(pos, 1.0);
   let pos_clip = uniforms.proj_matrix * pos_view;
   let pos_ndc  = pos_clip.xyz / pos_clip.w;
   ```

3. **Frustum cull**: discard if `z_view < 0.2` or `abs(pos_ndc.xy) > 1.1`

4. **Opacity cull**: `sigmoid(opacity_logit) < 0.03`

5. **3D covariance from scale + quaternion**:

   ```wgsl
   let S = mat3x3(exp(log_scale.x), 0, 0,   0, exp(log_scale.y), 0,   0, 0, exp(log_scale.z));
   let R = quat_to_mat3(rotation);
   let M = S * R;
   let sigma = transpose(M) * M;   // symmetric 3×3
   ```

6. **Perspective Jacobian** (EWA splatting, eq. 29):

   ```wgsl
   let z = pos_view.z;
   let J = mat3x3(
     focal_x / z,  0,           -focal_x * pos_view.x / (z*z),
     0,           -focal_y / z,  focal_y * pos_view.y / (z*z),
     0,            0,            0,
   );
   let W = mat3x3(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
   let T = W * J;
   let cov2d_mat = transpose(T) * sigma * T;
   ```

7. **Regularize and extract 2D covariance**:

   ```wgsl
   let A = cov2d_mat[0][0] + 0.3;
   let B = cov2d_mat[0][1];
   let C = cov2d_mat[1][1] + 0.3;
   ```

8. **Inverse conic** (for per-pixel evaluation in rasterizer):

   ```wgsl
   let det = A * C - B * B;
   let conic = vec3(C / det, -B / det, A / det);
   ```

9. **Screen-space radius** (for tile overlap):

   ```wgsl
   let mid = (A + C) / 2.0;
   let lambda_max = mid + sqrt(max(0.1, mid*mid - det));
   let radius_px = ceil(3.0 * sqrt(lambda_max));
   ```

10. **Count tile overlaps** (atomic increment per tile):

    ```wgsl
    let tile_min = max(vec2i(0), (uv_px - radius_px) / 16);
    let tile_max = min(tile_bounds, (uv_px + radius_px) / 16 + 1);
    let n_tiles = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    atomicAdd(&intersection_offsets[tile_y * num_tiles_x + tile_x], 1u);
    ```

11. **SH evaluation** (view-direction → RGB):

    ```wgsl
    let dir = normalize(pos - uniforms.camera_pos);
    var color = SH_C0 * sh_dc;                    // degree-0
    color += SH_C1 * (-dir.y*sh[1] + dir.z*sh[2] - dir.x*sh[3]);  // degree-1
    // degrees 2, 3 when needed
    color = clamp(color + 0.5, vec3(0.0), vec3(1.0));
    ```

12. **Write to GaussData output buffer**

---

#### Step 5 — Prefix Sum (`shaders/prefix-sum.wgsl`)

Converts per-tile intersection counts to cumulative offsets so tile-depth key generation knows
where to write each Gaussian's keys.

Two-phase scan:

- Phase 1: workgroup-local prefix sum (parallel scan within 256 elements)
- Phase 2: add workgroup-level offsets back

Also compute per-Gaussian `cum_tiles_touched` with a separate scan over the `tiles_touched` array.

---

#### Step 6 — Tile-Depth Key Generation (`shaders/tile-depth-key.wgsl`)

**Dispatch**: `ceil(numGaussians / 256)` workgroups.

For each visible Gaussian, emit one key per overlapping tile:

```wgsl
// Normalize depth to uint16
let depth_uint = u32(clamp((depth - min_depth) / (max_depth - min_depth), 0.0, 1.0) * 65535.0);

// Pack tile index (high 16 bits) + depth (low 16 bits)
let key = (tile_index << 16u) | depth_uint;

// Write to correct position in global key array using cum_tiles_touched
keys[cum_tiles_touched + local_tile_offset] = key;
values[cum_tiles_touched + local_tile_offset] = gaussian_id;
```

---

#### Step 7 — GPU Radix Sort (`pipeline/sort.ts`)

Use the **`webgpu-radix-sort`** npm package (same as reference).

```typescript
import RadixSortKernel from 'webgpu-radix-sort';

const sorter = new RadixSortKernel(device, {
  count: MAX_INTERSECTIONS,   // pre-allocated worst case
  bit_count: 32,
  workgroup_size: { x: 16, y: 16 },
});
```

Sorts `(key, gaussian_id)` pairs in-place, giving us Gaussians ordered by tile then by depth.

---

#### Step 8 — Tile Rasterizer (`shaders/rasterize.wgsl`)

**Dispatch**: `(ceil(W/16), ceil(H/16), 1)` workgroups, `(4, 8, 1)` threads per workgroup.

Each thread covers 4×2 = 8 pixels. Each workgroup covers a 16×16 tile.

**Algorithm** per pixel `(px, py)`:

```wgsl
var T = 1.0;        // accumulated transparency
var C = vec3(0.0);  // accumulated color

// Iterate over sorted Gaussians for this tile (from back to front)
for (let i = tile_start; i < tile_end; i++) {
  let g = gauss_data[sorted_ids[i]];

  // Evaluate 2D Gaussian
  let d = vec2(g.uv * screen_size) - vec2(px, py);
  let power = -0.5 * (g.conic.x*d.x*d.x + g.conic.z*d.y*d.y + 2.0*g.conic.y*d.x*d.y);
  if (power > 0.0) { continue; }

  let alpha = min(g.opacity * exp(power), 0.99);
  if (alpha < 1.0/255.0) { continue; }

  C += g.color * (alpha * T);
  T *= (1.0 - alpha);

  // Early exit: all pixels in thread fully covered
  if (T < 1.0/255.0) { break; }
}

// Write to output texture
textureStore(output_texture, vec2i(px, py), vec4(C, 1.0 - T));
```

Output: `rgba8unorm` storage texture.

---

#### Step 9 — Screen Blit (`shaders/screen.wgsl`)

Minimal fullscreen triangle pass:

**Vertex shader**: Generate 3-vertex fullscreen triangle from `vertex_index` (no VBO needed).

**Fragment shader**: `textureSample(color_buffer, sampler, uv)`.

This copies the compute rasterizer's output to the swap chain.

---

#### Step 10 — Camera (`camera.ts`)

**Camera state**:

- `position: vec3` — world position
- `rotation: mat3` — orientation (columns = right, up, forward)
- `focalX, focalY` — pixel focal lengths derived from FOV + resolution

**Matrices** (computed per frame):

```typescript
// View matrix (world → camera)
viewMatrix = [...rotation col0, 0,
              ...rotation col1, 0,
              ...rotation col2, 0,
              -dot(R_col0,pos), -dot(R_col1,pos), -dot(R_col2,pos), 1]

// Projection matrix (camera → clip)
projMatrix = perspectiveMatrix(fovX, fovY, zNear, zFar)
```

**Controls** (mouse + keyboard):

- Left drag: orbit (rotate around origin or a focal point)
- Right drag / middle drag: pan
- Scroll: dolly in/out
- WASD / arrow keys: translate
- QE: roll

---

#### Step 10a — Initial Camera Position

**Do `.ply` or `.splat` files contain camera information?**

**No.** Neither format embeds any camera data:

- `.splat` — only Gaussian geometry (position, scale, rotation, color, opacity). 32 bytes/Gaussian flat binary.
- `.ply` — only Gaussian vertex properties. No camera fields in the header.

Camera information is produced separately by the training pipeline (COLMAP + 3DGS) and saved as
a standalone `cameras.json` file in the output directory.

---

**`cameras.json` format** (standard 3DGS training output):

```json
[
  {
    "id": 0,
    "img_name": "00001",
    "width": 1959,
    "height": 1090,
    "position": [-3.009, -0.111, -3.753],
    "rotation": [
      [ 0.876,  0.069,  0.477],
      [-0.047,  0.997, -0.058],
      [-0.480,  0.028,  0.877]
    ],
    "fx": 1159.59,
    "fy": 1164.66
  },
  ...
]
```

- `position` — camera's world-space position (3-vector)
- `rotation` — camera-to-world rotation as row-major 3×3 matrix
- `fx`, `fy` — focal lengths in pixels (used to derive FOV)
- `width`, `height` — original training image resolution (used for FOV computation)

Both reference implementations use this identical format.

---

**Initial camera resolution strategy** — three tiers, tried in order:

**Tier 1: Load `cameras.json`** *(best — uses actual training viewpoints)*

When a `cameras.json` is drag-dropped or fetched alongside the scene, use `cameras[0]` as the
initial view. Expose the full list as camera presets the user can cycle through.

```typescript
// camera.ts
function cameraFromJSON(entry: CameraJSON, canvasW: number, canvasH: number): Camera {
  // Reconstruct view matrix from position + rotation
  const R = entry.rotation;               // 3×3 row-major
  const t = entry.position;               // [x, y, z]

  // camToWorld (column-major, WebGPU convention):
  const viewMatrix = [
    R[0][0], R[1][0], R[2][0], 0,
    R[0][1], R[1][1], R[2][1], 0,
    R[0][2], R[1][2], R[2][2], 0,
    -(t[0]*R[0][0] + t[1]*R[1][0] + t[2]*R[2][0]),
    -(t[0]*R[0][1] + t[1]*R[1][1] + t[2]*R[2][1]),
    -(t[0]*R[0][2] + t[1]*R[1][2] + t[2]*R[2][2]),
    1,
  ];

  // Derive FOV from training focal lengths
  // Use canvas aspect; scale fx/fy to canvas resolution
  const scaleX = canvasW / entry.width;
  const scaleY = canvasH / entry.height;
  const focalX = entry.fx * scaleX;
  const focalY = entry.fy * scaleY;

  return new Camera(viewMatrix, focalX, focalY, canvasW, canvasH);
}
```

**Tier 2: Auto-fit from scene bounding box** *(fallback when no cameras.json)*

After uploading Gaussians to the GPU, read back (or compute on CPU during parsing) the
axis-aligned bounding box of all Gaussian positions. Place the camera outside the bounding
sphere looking at the centroid:

```typescript
function autoFitCamera(positions: Float32Array, canvasW: number, canvasH: number): Camera {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < positions.length; i += 3) {
    minX = Math.min(minX, positions[i]);
    minY = Math.min(minY, positions[i + 1]);
    minZ = Math.min(minZ, positions[i + 2]);
    maxX = Math.max(maxX, positions[i]);
    maxY = Math.max(maxY, positions[i + 1]);
    maxZ = Math.max(maxZ, positions[i + 2]);
  }

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;

  const radius = Math.sqrt(
    (maxX - minX) ** 2 + (maxY - minY) ** 2 + (maxZ - minZ) ** 2,
  ) / 2;

  // Place camera at centroid + 2× radius along +Z, looking toward centroid
  const camPos = [cx, cy, cz + radius * 2];

  // Identity rotation (looking down -Z in camera space = +Z in world)
  const viewMatrix = [
    1, 0,  0, 0,
    0, 1,  0, 0,
    0, 0,  1, 0,
    -camPos[0], -camPos[1], -camPos[2], 1,
  ];

  // Use a standard 60° horizontal FOV
  const fovX = Math.PI / 3;
  const fovY = fovX * (canvasH / canvasW);
  const focalX = canvasW / (2 * Math.tan(fovX / 2));
  const focalY = canvasH / (2 * Math.tan(fovY / 2));

  return new Camera(viewMatrix, focalX, focalY, canvasW, canvasH);
}
```

This is computed during PLY parsing (positions are available on CPU before GPU upload) so
there is no GPU readback stall.

**Tier 3: Hardcoded sensible default** *(last resort)*

If no scene is loaded yet, use a default position that matches the WebGPU reference's
`Camera.default()` so the empty viewer doesn't show a blank screen:

```typescript
// Looking at origin from [0, 0, 3] — neutral starting position
const DEFAULT_VIEW_MATRIX = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, -3, 1,
];
```

---

**Implementation plan**:

```
loader/
  camera-loader.ts     parse cameras.json → Camera[]
                       autoFitCamera(positions) → Camera
camera.ts             Camera class + all three init tiers
```

**UI**:

- Drop-zone accepts both `.ply`/`.splat` and `.json` simultaneously (drag multiple files)
- If a `cameras.json` is loaded, show a camera index selector (prev/next buttons)
- URL hash encodes current view matrix so scenes are shareable (same as WebGL reference)

---

#### Step 11 — Renderer Orchestration (`renderer.ts`)

**Per-frame loop**:

```typescript
async function frame() {
  // 1. Update uniforms from camera
  uploadUniforms(device, uniformBuffer, camera);

  // 2. Preprocess compute passes
  const encoder = device.createCommandEncoder();
  runPreprocess(encoder);      // gauss.wgsl
  runPrefixSum(encoder);       // prefix-sum.wgsl
  runTileDepthKeys(encoder);   // tile-depth-key.wgsl

  // 3. GPU radix sort
  sorter.sort(encoder, keyBuffer, valueBuffer, numIntersections);

  // 4. Rasterize compute pass
  runRasterize(encoder);       // rasterize.wgsl

  // 5. Screen blit render pass
  runScreenBlit(encoder, context.getCurrentTexture().createView());

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}
```

**Dirty flag**: skip redraw if camera unchanged and no data updates.

---

### 3.3 Bind Group Layout

| Bind Group | Bindings | Used By |
|---|---|---|
| `bg_preprocess` | uniforms, input_gaussians, gauss_data_out, tile_keys, intersection_offsets | preprocess + tile-key shaders |
| `bg_sort` | key_buffer, value_buffer | radix sort |
| `bg_rasterize` | uniforms, gauss_data, sorted_ids, intersection_offsets, output_texture | rasterize.wgsl |
| `bg_screen` | sampler, output_texture (read) | screen.wgsl |

---

### 3.4 Buffer Sizes (for 3M Gaussians at 1920×1080)

| Buffer | Size | Notes |
|---|---|---|
| Input Gaussians | ~600 MB | 200 bytes/gaussian × 3M (with full SH) |
| Input Gaussians (DC only) | ~48 MB | 16 bytes/gaussian |
| GaussData output | ~112 MB | ~37 bytes × 3M |
| Tile-depth keys | ~112 MB | 8 bytes × 14M max intersections |
| Intersection offsets | ~30 KB | 4 bytes × 7680 tiles (1920×1080/16²) |
| Uniform buffer | 256 bytes | |
| Output texture | ~8 MB | rgba8unorm 1920×1080 |

In practice, start with DC-only color (~48 MB input) for initial development.

---

### 3.5 Development Phases

#### Phase 1 — PLY Loader + Data Upload

- [ ] Implement `ply-parser.ts`: parse header, extract vertex fields
- [ ] Pack parsed data into GPU input buffer (position + scale + rotation + opacity + SH DC)
- [ ] Verify by logging first 5 Gaussian positions

#### Phase 2 — Preprocessing Shader

- [ ] Implement `preprocess.wgsl`: 3D covariance, 2D projection, conic
- [ ] Implement frustum + opacity culling
- [ ] Implement tile intersection counting (atomic adds)
- [ ] Test by reading back `GaussData` and checking radius/UV values

#### Phase 3 — Prefix Sum + Tile-Depth Keys

- [ ] Implement `prefix-sum.wgsl` for intersection offsets
- [ ] Implement `tile-depth-key.wgsl`
- [ ] Verify keys are correctly ordered by reading them back

#### Phase 4 — GPU Radix Sort

- [ ] Install `webgpu-radix-sort`
- [ ] Wire up sort on key/value buffers
- [ ] Verify sorted order

#### Phase 5 — Tile Rasterizer

- [ ] Implement `rasterize.wgsl`: per-tile Gaussian accumulation
- [ ] Write output to `rgba8unorm` storage texture
- [ ] Test with a few known Gaussians

#### Phase 6 — Screen Blit + Camera

- [ ] Implement `screen.wgsl` fullscreen blit
- [ ] Implement camera with orbit controls
- [ ] End-to-end test: load a `.ply` file and see splats

#### Phase 7 — Polish

- [ ] Add full SH evaluation (degrees 2 and 3)
- [ ] Add progress indicator for file loading
- [ ] Add scale modifier UI
- [ ] Performance profiling with timestamp queries
- [ ] Support `.splat` binary format as alternative input

---

### 3.6 Key Mathematical References

**3D Covariance from scale + rotation**:

```
Σ = (S·R)ᵀ · (S·R)   where S = diag(exp(log_scale)), R = quat_to_mat3(q)
```

**2D Covariance via EWA splatting** (Zwicker et al. 2002, eq. 29-31):

```
J = ∂(x_screen, y_screen) / ∂(x_cam, y_cam, z_cam)   [perspective Jacobian]
W = upper-left 3×3 of view matrix
T = W · J
Cov_2D = Tᵀ · Σ · T    [take upper-left 2×2]
```

**Inverse conic** (used in rasterizer for fast per-pixel evaluation):

```
Given Cov_2D = [[A, B], [B, C]]:
det = A·C - B²
conic = [C/det, -B/det, A/det]
```

**Per-pixel Gaussian evaluation**:

```
power = -0.5 · (conic.x·dx² + conic.z·dy² + 2·conic.y·dx·dy)
alpha = opacity · exp(power)
```

**Alpha-over compositing** (back-to-front):

```
C_out += C_splat · alpha · T_accumulated
T_accumulated *= (1 - alpha)
```

**SH evaluation** (degree 0+1):

```
C0 = 0.28209479177387814
C1 = 0.4886025119029199
color = C0·dc + C1·(-y·sh1 + z·sh2 - x·sh3) + 0.5
```

---

### 3.7 Supported File Formats

The 3DGS ecosystem has several formats at different points of the quality / file-size / complexity
spectrum. We support them in two tiers.

---

#### Tier 1 — Must Support

| Format | Ext | Size/Gaussian | SH | Notes |
|--------|-----|---------------|----|-------|
| **PLY** | `.ply` | 64–200 B | 0–3 | Training output, full precision, the "source of truth" |
| **SPLAT** | `.splat` | 28 B | 0 (DC only) | antimatter15 binary; no header; most common web format today |

**PLY detection**: magic bytes `ply\n` (0x70 0x6C 0x79 0x0A).

**PLY binary layout** (per vertex, from training pipeline):

```
x, y, z               float32 × 3   position
scale_0..2            float32 × 3   log-space scale
rot_0..3              float32 × 4   quaternion
opacity               float32       logit-encoded
f_dc_0..2             float32 × 3   SH degree-0 color
f_rest_0..N           float32 × M   higher-order SH (optional; M = 9×3 for degree-2, 16×3 for degree-3)
```

SH degree is inferred from the header: `degree = sqrt(nRestCoeffs / 3 + 1) - 1`.

**SPLAT detection**: no magic — identified by not being PLY and having `byteLength % 32 === 0`.

**SPLAT binary layout** (fixed 32 bytes per Gaussian):

```
[0..11]   float32 × 3   position (x, y, z)
[12..23]  float32 × 3   scale (exp already applied)
[24..27]  uint8 × 4     RGBA color (DC term baked in, opacity in alpha)
[28..31]  uint8 × 4     quaternion (packed as round(q * 128 + 128), clamped to [0,255])
```

> Note: Some sources cite 28 bytes; the actual implementation uses `rowLength = 3*4 + 3*4 + 4 + 4 = 32`.

---

#### Tier 2 — Should Support (Phase 7)

| Format | Ext | Size/Gaussian | SH | Notes |
|--------|-----|---------------|----|-------|
| **SPZ** | `.spz` | ~6 B compressed | 0–3 | Niantic/Scaniverse; gzipped column layout; ~10× smaller than PLY; MIT license; now part of glTF `KHR_gaussian_splatting_compression_spz` standard |
| **KSPLAT** | `.ksplat` | ~20–40 B | 0–3 | mkkellogg's Three.js format; block-based chunks; uint16 positions, float16 scale/rotation; common in the Three.js/PlayCanvas ecosystem |

**Why SPZ matters**: Niantic open-sourced SPZ in 2024 (MIT). In 2025, the Khronos Group adopted it as
the official glTF compression extension (`KHR_gaussian_splatting_compression_spz`). A 250 MB PLY file
becomes ~25 MB as SPZ — critical for mobile and bandwidth-constrained delivery. The format is
gzip-compressed column-major data with fixed-point quantization.

**SPZ format overview**:

```
[0..15]   header (16 bytes): magic, version, numPoints, shDegree, flags
[16..]    gzip-compressed payload:
            positions:  uint24 × 3 per Gaussian  (fixed-point in scene bounding box)
            scales:     uint8 × 3 per Gaussian   (log-space, quantized)
            rotations:  uint8 × 4 per Gaussian   (normalized quaternion)
            alphas:     uint8 per Gaussian        (opacity)
            colors:     uint8 × 3 per Gaussian   (DC SH)
            sh_rest:    uint8 × M per Gaussian   (higher-order SH, if shDegree > 0)
```

SPZ loading requires decompressing the gzip stream. Use the browser's native
`DecompressionStream('gzip')` API — no library needed.

**KSPLAT format overview** (mkkellogg/GaussianSplats3D):

```
Header chunk: magic ("KSPLAT"), version, maxSHBand, blockSize
Data chunks: each chunk stores a block of Gaussians with:
  - Center:    float16 × 3 (quantized to chunk bounds)
  - Scale:     float16 × 3
  - Rotation:  uint8 × 4
  - Color:     uint8 × 4
  - SH bands:  uint8 × M (optional, compressed to uint8)
```

---

#### Tier 3 — Future / Out of Scope

| Format | Notes |
|--------|-------|
| **glTF/GLB** with `KHR_gaussian_splatting` | Full glTF container; SPZ payload embedded in a buffer view; not prioritized for initial release |
| **Compressed PLY** | Some tools output PLY with quantized fields; rare in practice |
| **SOG / Parquet** | niche converter ecosystem formats |

---

#### Format Loading Architecture

All loaders share the same output contract — a `SceneData` struct — so the rest of the pipeline
is format-agnostic:

```typescript
// loader/types.ts
interface SceneData {
  numGaussians: number;
  // CPU-side parsed data (uploaded to GPU input buffer)
  positions:     Float32Array;   // xyz
  logScales:     Float32Array;   // log-space sx,sy,sz
  rotations:     Float32Array;   // quaternion w,x,y,z
  opacities:     Float32Array;   // raw logit
  shDC:          Float32Array;   // DC color r,g,b
  shRest:        Float32Array;   // higher-order SH coefficients (may be empty)
  shDegree:      number;         // 0, 1, 2, or 3
}
```

Format detection and dispatch:

```typescript
// loader/index.ts
async function loadScene(file: File): Promise<SceneData> {
  const buf = await file.arrayBuffer();
  const header = new Uint8Array(buf, 0, 4);

  if (header[0] === 0x70 && header[1] === 0x6C && header[2] === 0x79 && header[3] === 0x0A) {
    return parsePLY(buf);            // "ply\n"
  }
  if (header[0] === 0x1F && header[1] === 0x8B) {
    return parseSPZ(buf);            // gzip magic bytes (SPZ)
  }
  if (new TextDecoder().decode(header) === 'KSpl') {
    return parseKSplat(buf);         // KSPLAT magic
  }
  // Fallback: assume .splat binary
  return parseSplat(buf);
}
```

---

### 3.8 Dependencies

| Package | Purpose |
|---|---|
| `vite` | Dev server + bundler |
| `typescript` | Type safety |
| `@webgpu/types` | WebGPU TypeScript declarations |
| `webgpu-radix-sort` | GPU radix sort (same as reference) |

SPZ decompression uses the browser-native `DecompressionStream` API — no extra dependency needed.

---

### 3.9 Point Cloud Support

Point clouds are collections of 3D positions with optional per-point attributes (color, normals,
intensity). Unlike Gaussian splats they carry no covariance / SH — they are just points in space.
Supporting them turns the viewer into a general-purpose 3D point data tool.

---

#### 3.9.1 Supported Point Cloud Formats

| Format | Ext | Magic | Notes |
|--------|-----|-------|-------|
| **PLY point cloud** | `.ply` | `ply\n` | Same magic as Gaussian PLY — distinguished by header properties |
| **XYZ** | `.xyz`, `.txt` | none | Plain text, one point per line |
| **PCD** | `.pcd` | `# .PCD` | PCL library format; ASCII, binary, binary\_compressed |
| **LAS** | `.las` | `LASF` | LiDAR industry standard; int32 coords with scale/offset |
| **LAZ** | `.laz` | `LASF` | Lossless-compressed LAS (LASzip); 7–25% of LAS size |

**PLY disambiguation** — a PLY file is a Gaussian splat if its header contains `f_dc_0`; otherwise
it is a point cloud:

```typescript
// loader/ply-parser.ts
function isGaussianPLY(header: string): boolean {
  return header.includes('f_dc_0');
}
// → true  → parse as Gaussian splat (existing path)
// → false → parse as point cloud (new path)
```

**XYZ detection** — no magic bytes; identify by file extension or by successfully parsing the
first line as 3–6 space-separated numbers:

```
x y z                  (positions only)
x y z r g b            (positions + RGB 0–255)
x y z r g b nx ny nz   (positions + color + normals)
```

**PCD header** example:

```
# .PCD v0.7
FIELDS x y z rgb
SIZE   4 4 4 4
TYPE   F F F F
COUNT  1 1 1 1
WIDTH  100000
HEIGHT 1
POINTS 100000
DATA binary
<binary body>
```

Supported DATA modes: `ascii`, `binary`. `binary_compressed` (LZ4) deferred to Phase 8.

**LAS header** (first 227 bytes, fixed):

```
[0..3]    char[4]   "LASF"                 file signature
[94..97]  uint32    point data offset
[98..101] uint32    number of variable length records
[104]     uint8     point data format (0–10)
[105..106] uint16   point data record length
[107..110] uint32   legacy point count
[131..138] float64  x scale factor
[139..146] float64  y scale factor
[147..154] float64  z scale factor
[155..162] float64  x offset
...
```

Per-point coordinates are `int32`; actual value = `int32 * scale + offset`. RGB optional (formats 2, 3, 5, 7, 8, 10).

LAZ is the same header structure as LAS — decompression requires the [LASzip](https://github.com/LASzip/LASzip)
algorithm (or the JS port `laz-perf`). Add `laz-perf` as an optional dependency for Phase 8.

---

#### 3.9.2 Shared Output Contract

Both Gaussian splats and point clouds share the top-level `SceneData` discriminated union:

```typescript
// loader/types.ts
type SceneData = GaussianSceneData | PointCloudSceneData;

interface GaussianSceneData {
  kind: 'gaussian';
  numPoints: number;
  positions:  Float32Array;   // xyz
  logScales:  Float32Array;   // log sx sy sz
  rotations:  Float32Array;   // quat w x y z
  opacities:  Float32Array;   // logit
  shDC:       Float32Array;   // DC color r g b
  shRest:     Float32Array;   // higher SH (may be empty)
  shDegree:   number;
}

interface PointCloudSceneData {
  kind: 'pointcloud';
  numPoints: number;
  positions:  Float32Array;   // xyz
  colors:     Uint8Array;     // r g b per point (0–255); empty = no color
  normals:    Float32Array;   // nx ny nz per point; empty = no normals
  intensity:  Float32Array;   // scalar intensity; empty = not present
}
```

---

#### 3.9.3 Rendering Strategy

Two strategies, chosen based on point count and scene type:

**Strategy A — Isotropic Gaussian Conversion** *(≤ 2M points)*

Convert each point cloud point into a tiny spherical Gaussian and push it through the entire
existing Gaussian pipeline unchanged. No new render code needed.

```typescript
function pointCloudToGaussians(pc: PointCloudSceneData): GaussianSceneData {
  const N = pc.numPoints;
  const logScales  = new Float32Array(N * 3).fill(Math.log(POINT_WORLD_RADIUS)); // e.g. log(0.005)
  const rotations  = new Float32Array(N * 4);
  const opacities  = new Float32Array(N).fill(0);   // logit(1) ≈ infinity → clamp to 10
  const shDC       = new Float32Array(N * 3);

  for (let i = 0; i < N; i++) {
    // Identity quaternion: w=1, x=y=z=0
    rotations[i * 4 + 0] = 1;

    // Convert uint8 RGB to SH DC coefficients: dc = (color/255 - 0.5) / SH_C0
    const SH_C0 = 0.28209479177387814;
    shDC[i * 3 + 0] = (pc.colors[i * 3 + 0] / 255 - 0.5) / SH_C0;
    shDC[i * 3 + 1] = (pc.colors[i * 3 + 1] / 255 - 0.5) / SH_C0;
    shDC[i * 3 + 2] = (pc.colors[i * 3 + 2] / 255 - 0.5) / SH_C0;

    opacities[i] = 10;  // logit(≈1.0) → fully opaque
  }

  return { kind: 'gaussian', numPoints: N,
           positions: pc.positions, logScales, rotations, opacities, shDC,
           shRest: new Float32Array(0), shDegree: 0 };
}
```

Benefits: complete for free — sorting, culling, tile rasterizer all just work.
`POINT_WORLD_RADIUS` is user-adjustable (e.g. 0.002–0.05) to control visual point size.

**Strategy B — Compute Point Rasterizer** *(> 2M points or LiDAR data)*

Based on Schütz et al. 2021 *"Rendering Point Clouds with Compute Shaders"*. Achieves
60 FPS with **2 billion points** on commodity hardware by:

- Grouping ~10k points per compute workgroup (batch-level frustum culling)
- Software rasterization via **atomic compare-and-swap** on a `u32` depth+color buffer
- No sort pass needed — depth buffer handles visibility

```wgsl
// shaders/point-rasterize.wgsl
@group(0) @binding(0) var<storage, read>       points:   array<PointData>;
@group(0) @binding(1) var<storage, read_write> fb:       array<atomic<u32>>; // packed depth+color
@group(0) @binding(2) var<uniform>             uniforms: Uniforms;

const BATCH_SIZE = 128u;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= uniforms.num_points) { return; }

  let pos_world = vec4(points[idx].position, 1.0);
  let pos_clip  = uniforms.proj_matrix * uniforms.view_matrix * pos_world;
  if (pos_clip.w <= 0.0) { return; }

  let ndc = pos_clip.xyz / pos_clip.w;
  if (any(abs(ndc.xy) > vec2(1.0))) { return; }   // frustum cull

  let px = u32((ndc.x * 0.5 + 0.5) * f32(uniforms.screen_size.x));
  let py = u32((1.0 - (ndc.y * 0.5 + 0.5)) * f32(uniforms.screen_size.y));
  let pixel_idx = py * uniforms.screen_size.x + px;

  // Pack depth (upper 20 bits) + color (lower 12 bits) into u32
  let depth_uint = u32(saturate(pos_clip.z / pos_clip.w) * 0xFFFFF);
  let color_uint = pack_rgb4(points[idx].color);
  let value = (depth_uint << 12u) | color_uint;

  // Atomic min: closest point wins
  atomicMin(&fb[pixel_idx], value);
}
```

A second pass resolves the framebuffer to an rgba8unorm texture for the screen blit.
No sort, no tile prefix sum — the pipeline reduces to: **preprocess → compute rasterize → blit**.

---

#### 3.9.4 Strategy Selection Logic

```typescript
// renderer.ts
function selectPointCloudStrategy(data: PointCloudSceneData): 'gaussian' | 'compute' {
  if (data.numPoints <= 2_000_000) return 'gaussian';  // reuse Gaussian pipeline
  return 'compute';                                      // dedicated point rasterizer
}
```

The threshold is configurable. Users can force either mode via a UI toggle.

---

#### 3.9.5 LOD for Large Point Clouds

For Strategy B (compute rasterizer) with > 10M points, a voxel-grid LOD ensures distant regions
are not over-sampled:

- Assign each point an LOD level 0–3 at load time based on spatial voxel membership:
  - LOD 0: one representative point per 1m³ voxel (coarsest)
  - LOD 1: one per 0.5m³
  - LOD 2: one per 0.25m³
  - LOD 3: all points (full detail)
- At render time, compute screen-space projected voxel size; render only LOD levels whose
  projected voxel covers ≥ 1 pixel
- This is computed CPU-side once at load time; costs O(N log N) for sorting into voxel grid

---

#### 3.9.6 Point Cloud Feature Checklist

**Phase 6 (with Strategy A — Gaussian conversion):**

- [ ] `loader/types.ts` — `PointCloudSceneData` interface + discriminated union
- [ ] PLY point cloud detection (header lacks `f_dc_0`)
- [ ] PLY point cloud parser (xyz + rgb fields)
- [ ] XYZ text parser
- [ ] `pointCloudToGaussians()` conversion — reuses full Gaussian pipeline
- [ ] `POINT_WORLD_RADIUS` UI slider (point size control)

**Phase 7 (PCD + LAS):**

- [ ] PCD parser (ASCII + binary modes)
- [ ] LAS parser (formats 0–10, int32 → float64 with scale/offset)

**Phase 8 (Strategy B — compute rasterizer + LOD):**

- [ ] `shaders/point-rasterize.wgsl` — atomic depth+color framebuffer
- [ ] Framebuffer resolve pass
- [ ] Strategy selection logic (auto switch at 2M threshold)
- [ ] Voxel-grid LOD assignment at load time
- [ ] LAZ support via `laz-perf` npm package
- [ ] PCD `binary_compressed` mode

---

### 3.10 Progressive Loading

#### 3.10.1 Problem Statement

A monolithic 3DGS scene for a single room might be 50–200 MB. A city-block scene (CityGaussian,
VastGaussian) can be 2–20 GB; a LiDAR survey can be 100 GB+. Waiting for a full download before
rendering is completely impractical at these scales. We need to show something useful within seconds
and continue refining as data arrives.

There are three fundamentally different strategies, applicable to different file sizes and scene types.
They are ordered from simplest to implement to most powerful:

---

#### 3.10.2 Strategy 1 — Importance-Ordered Single-File Streaming *(Baseline)*

**When to use**: Single-file scenes up to ~2 GB. No special server required.

**How it works**: The file (`.ply` or `.splat`) is pre-sorted by **Gaussian importance**
`exp(sx + sy + sz) × sigmoid(opacity)` before hosting. The most visually important Gaussians
sit at the start of the file. Fetch via `ReadableStream`, render each chunk as it arrives.

```
Time 0s:   Fetch begins
Time 0.5s: First 50k Gaussians arrive → upload to GPU → render (blurry but recognisable)
Time 1.5s: 500k Gaussians → render (most major features visible)
Time 5s:   Full 3M Gaussians loaded → full quality
```

**File layout** (pre-processed once):

```
[Gaussian 0]  importance = 0.98   ← most prominent splat
[Gaussian 1]  importance = 0.97
...
[Gaussian N]  importance = 0.001  ← nearly invisible micro-splat
```

**Client-side implementation**:

```typescript
// loader/ply-parser.ts
async function* streamPLY(url: string, chunkSize = 50_000): AsyncGenerator<SceneData> {
  const res  = await fetch(url);
  const reader = res.body!.getReader();
  const header = await readPLYHeader(reader);          // reads until end_header
  const stride = computeStride(header.properties);

  let buffer = new Uint8Array(0);

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer = concat(buffer, value);

    const completeSplats = Math.floor(buffer.byteLength / stride);
    if (completeSplats >= chunkSize) {
      yield parsePLYChunk(buffer.subarray(0, completeSplats * stride), header);
      buffer = buffer.subarray(completeSplats * stride);
    }
  }
  if (buffer.byteLength >= stride) {
    yield parsePLYChunk(buffer, header);  // final partial chunk
  }
}

// renderer.ts — consume the generator
for await (const chunk of streamPLY(url)) {
  appendGaussiansToGPU(chunk);
  renderer.setNumGaussians(totalLoaded);
  // pipeline runs with whatever is loaded — no waiting
}
```

**Server requirement**: only a static file server. `Content-Length` header recommended for
progress bar. No range-request support needed.

**Limitations**: the whole file must eventually download; no spatial selectivity. Fine for
scenes up to ~2 GB, but not for city-scale.

---

#### 3.10.3 Strategy 2 — Spatial Tile Manifest *(Recommended for Large Gaussian Scenes)*

**When to use**: Large single-floor or city-block Gaussian splat scenes (2–20 GB total) that
are pre-tiled during training (VastGaussian / CityGaussian style), or any scene you are willing
to preprocess.

**How it works**: The scene is partitioned into a regular **spatial grid of tiles**. A small
JSON manifest describes every tile's bounding box and URL. At runtime the viewer loads only the
tiles whose bounding boxes intersect the camera frustum and are within a distance budget.
Distant/occluded tiles are evicted from the GPU tile cache.

**Manifest format** (`scene.json`):

```json
{
  "version": "1.0",
  "bounds": {
    "min": [-150.0, -5.0, -120.0],
    "max": [ 150.0, 30.0,  120.0]
  },
  "tileSize": [20.0, 35.0, 20.0],
  "tiles": [
    {
      "id": "t_0_0_0",
      "url": "tiles/t_0_0_0.splat",
      "bounds": { "min": [-150,-5,-120], "max": [-130,30,-100] },
      "numPoints": 124800,
      "lodLevel": 0,
      "byteSize": 3993600
    },
    ...
  ]
}
```

**Runtime tile selection** — evaluated every frame (cheap — just AABB vs frustum):

```typescript
// pipeline/tile-manager.ts

interface TileRecord {
  id:          string;
  bounds:      AABB;
  url:         string;
  numPoints:   number;
  state:       'unloaded' | 'fetching' | 'resident' | 'evicted';
  gpuOffset:   number;       // byte offset in the shared GPU tile buffer
  lastUsedFrame: number;     // for LRU eviction
  priority:    number;       // distance-weighted frustum score
}

function selectVisibleTiles(
  tiles: TileRecord[],
  camera: Camera,
  frustum: Frustum,
  maxResidentBytes: number,   // GPU tile cache budget
): TileRecord[] {
  return tiles
    .filter(t => frustum.intersectsAABB(t.bounds))
    .map(t => ({ ...t, priority: 1 / distanceSq(camera.position, t.bounds.center) }))
    .sort((a, b) => b.priority - a.priority)
    .slice(0, MAX_RESIDENT_TILES);
}
```

**GPU tile cache** (see §3.10.6 for full detail):

- One large pre-allocated `GPUBuffer` (`STORAGE | COPY_DST`), e.g., 512 MB
- Divided into fixed-size **slots** of e.g. 2 MB each (256 slots total)
- A tile occupies one or more contiguous slots
- LRU eviction when cache is full: evict the tile with the smallest `lastUsedFrame`
- The preprocessing uniform includes the per-tile `(gpuOffset, numPoints)` pair so the
  preprocess shader knows which tiles to process this frame

**Loading pipeline** (per tile, in a Web Worker):

```
1. fetch(tile.url)
2. Parse .splat binary (Worker thread)
3. postMessage({ tileId, gpuData: Float32Array }, [gpuData.buffer])  ← zero-copy transfer
4. Main thread: device.queue.writeBuffer(tileBuffer, gpuOffset, gpuData)
5. Mark tile as 'resident'
```

Tiles load concurrently (Promise.all up to N in-flight at a time).

**Preprocessing**: use a Python or CLI tool to split a large `.ply` into tiles + write `scene.json`.
Each tile is importance-sorted within itself so Strategy 1 streaming still works per tile.

---

#### 3.10.4 Strategy 3 — Octree LOD Streaming *(For Point Clouds and City-Scale Gaussian Scenes)*

**When to use**: Massive point clouds > 50M points (LiDAR surveys, aerial photogrammetry) or
city-scale Gaussian scenes where per-tile resolution also needs to vary with distance.

This is a Potree/COPC-style approach.

**Core idea**: Build a **sparse octree** over the scene. Each node at depth `d` stores a
**spatially uniform subsample** of the points in its volume, at resolution `spacing = baseSpacing / 2^d`.
Loading starts at the root (coarsest, fewest points) and descends the tree as the user zooms in.

```
Depth 0 (root):  1 point per 8m³  →  ~10k points, whole scene visible  ← loaded immediately
Depth 1:         1 point per 4m³  →  ~80k points
Depth 2:         1 point per 2m³  →  ~640k points
Depth 3:         1 point per 1m³  →  ~5M points
Depth 4:         all points       →  ~40M points  ← only loaded when camera is very close
```

**Screen-Space Error (SSE) traversal** — decides which nodes to load/render per frame:

```typescript
function computeSSE(node: OctreeNode, camera: Camera, screenHeight: number): number {
  // Project the node's bounding sphere onto the screen
  const dist     = vec3.distance(camera.position, node.bounds.center);
  const radius   = node.bounds.radius;
  // Projected radius in pixels:
  const projectedRadius = (camera.focalY * radius) / dist;
  // Screen-space error = projected radius / node spacing
  // When SSE > threshold (e.g. 1.5 px/point-spacing), descend to children
  return projectedRadius / node.spacing;
}

function traverseOctree(
  node: OctreeNode,
  camera: Camera,
  frustum: Frustum,
  loadQueue: OctreeNode[],
  renderList: OctreeNode[],
  sseThreshold = 1.5,
): void {
  if (!frustum.intersectsSphere(node.bounds)) return;  // frustum cull entire subtree

  const sse = computeSSE(node, camera, canvas.height);

  if (node.state === 'unloaded') {
    loadQueue.push(node);   // enqueue for async fetch; render parent in the meantime
    return;
  }

  renderList.push(node);    // render this node (always — even while children load)

  if (sse > sseThreshold && node.children.length > 0) {
    for (const child of node.children) {
      traverseOctree(child, camera, frustum, loadQueue, renderList, sseThreshold);
    }
  }
}
```

**Key property**: a parent node is always rendered while its children load. The scene is never
blank — it just gets progressively sharper as more data arrives.

**COPC for LiDAR** — a single `.copc.laz` file that embeds the octree index in its VLR.
Enables HTTP range requests to fetch specific octree node byte ranges without downloading the
full file:

```typescript
// loader/copc-loader.ts

// 1. Fetch just the VLR (first ~64KB) to get the octree page offsets
const vlrBuffer = await fetchRange(url, 0, 65536);
const index     = parseCOPCIndex(vlrBuffer);   // OctreeNode[] with {offset, byteSize}

// 2. On each traversal, fetch only the nodes we need
async function loadCOPCNode(node: OctreeNode): Promise<PointCloudSceneData> {
  const chunk = await fetchRange(url, node.offset, node.offset + node.byteSize);
  return decompressLAZChunk(chunk);   // uses laz-perf
}

// Thin HTTP range fetch helper
async function fetchRange(url: string, start: number, end: number): Promise<ArrayBuffer> {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
  return res.arrayBuffer();
}
```

This means a 10 GB COPC file can be streamed interactively with only a few MB of actual
HTTP traffic per frame.

**Preprocessing** — generate the octree structure:

- Point clouds: use `PotreeConverter` or `untwine` to produce Potree 2.0 format or COPC.
- Gaussian scenes: write a custom splitter that assigns each Gaussian to the finest octree node
  it fits in at its natural scale (e.g., a Gaussian with radius 0.1m belongs to the depth-3 node
  with spacing 0.25m).

---

#### 3.10.5 HTTP Range Requests

Both Strategy 1 (importance-sorted) and Strategy 3 (COPC) benefit from HTTP range requests when
the server supports them (any S3-compatible storage, nginx with default config, CDN).

Detect support at startup:

```typescript
async function supportsRangeRequests(url: string): Promise<boolean> {
  const res = await fetch(url, { method: 'HEAD' });
  return res.headers.get('Accept-Ranges') === 'bytes';
}
```

**Use cases**:

| Scenario | Range request pattern |
|---|---|
| Importance-sorted PLY, quick preview | `bytes=0–<first 10%>` then background-fetch remainder |
| COPC octree nodes | `bytes=<nodeOffset>–<nodeOffset+nodeSize>` per node |
| Tile manifest tiles | Full tile fetch (tiles are already small enough) |
| Splat with known Gaussian count | `bytes=0–<firstNSplats * 32>` for instant preview |

**Fallback**: if range requests are not supported, fall back to Strategy 1 full-file streaming.

---

#### 3.10.6 GPU Tile Cache Management

All three strategies ultimately put point data into a shared GPU buffer. Managing that buffer
efficiently is critical.

**Design**: a single large `GPUBuffer` pre-allocated at startup, divided into fixed-size **slots**.

```typescript
// gpu/tile-cache.ts

const SLOT_SIZE_BYTES = 2 * 1024 * 1024;   // 2 MB per slot
const NUM_SLOTS       = 256;                // 512 MB total cache
const CACHE_BYTES     = SLOT_SIZE_BYTES * NUM_SLOTS;

interface CacheSlot {
  tileId:       string | null;
  inUse:        boolean;
  lastUsedFrame: number;
}

class GPUTileCache {
  private buffer: GPUBuffer;
  private slots:  CacheSlot[];
  private frame = 0;

  constructor(device: GPUDevice) {
    this.buffer = device.createBuffer({
      size:  CACHE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.slots = Array.from({ length: NUM_SLOTS }, () =>
      ({ tileId: null, inUse: false, lastUsedFrame: 0 })
    );
  }

  // Find a free slot, evicting LRU if cache is full
  allocate(tileId: string, numSlots: number): number {
    const free = this.slots.findIndex(s => !s.inUse);
    if (free !== -1) {
      this.markUsed(free, tileId);
      return free * SLOT_SIZE_BYTES;
    }
    // LRU eviction
    const lru = this.slots
      .filter(s => s.inUse)
      .sort((a, b) => a.lastUsedFrame - b.lastUsedFrame)[0]!;
    const lruIdx = this.slots.indexOf(lru);
    lru.inUse = false;
    lru.tileId = null;
    this.markUsed(lruIdx, tileId);
    return lruIdx * SLOT_SIZE_BYTES;
  }

  touch(tileId: string) {
    const slot = this.slots.find(s => s.tileId === tileId);
    if (slot) slot.lastUsedFrame = this.frame;
  }

  tickFrame() { this.frame++; }
}
```

**The GPU preprocessor** needs to know the active tile list each frame — pass it as a uniform
array of `(gpuByteOffset, numPoints)` pairs:

```wgsl
struct TileInfo {
  byte_offset: u32,
  num_points:  u32,
}

@group(0) @binding(5) var<storage, read> tiles: array<TileInfo>;

// In the preprocess compute shader main():
let tile_idx = /* determined from global_invocation_id */;
let tile     = tiles[tile_idx];
let local_id = global_id.x - tile.byte_offset / GAUSSIAN_STRIDE;
if (local_id >= tile.num_points) { return; }
// ... process Gaussian at (tile.byte_offset + local_id * GAUSSIAN_STRIDE)
```

---

#### 3.10.7 Strategy Comparison

| | Strategy 1 | Strategy 2 | Strategy 3 |
|---|---|---|---|
| **Name** | Importance streaming | Spatial tile manifest | Octree LOD |
| **Best for** | Single-file splat scenes | Pre-tiled city-scale Gaussian scenes | LiDAR / 50M+ point clouds |
| **Server requirement** | Static file (any CDN) | Static files per tile | Static files or range-request server |
| **Preprocessing** | Sort by importance | Split into grid + write manifest | Build octree (PotreeConverter / untwine) |
| **First-render latency** | ~0.5–2 s | ~0.1–0.5 s (load nearest tiles first) | ~0.1 s (root node only) |
| **Max scene size** | ~2 GB practical | Unlimited (disk-limited) | Unlimited |
| **Spatial selectivity** | None — whole file streams | Yes — load only frustum tiles | Yes — load only visible octree nodes |
| **Distance-based detail** | No | Partially (tile LOD levels) | Yes — full SSE-driven LOD |
| **Implementation complexity** | Low | Medium | High |
| **Phase** | Phase 1 | Phase 7 | Phase 8 |

---

#### 3.10.8 Implementation Checklist

**Phase 1 (Strategy 1 — always implement)**

- [ ] Async generator `streamPLY()` / `streamSplat()` — yields chunks as network data arrives
- [ ] Renderer accepts incremental `appendGaussiansToGPU()` call
- [ ] Progress bar (bytes received / Content-Length)
- [ ] Web Worker for chunk parsing so main thread stays responsive

**Phase 7 (Strategy 2 — spatial tile manifest)**

- [ ] `scene.json` manifest format parser (`loader/tile-manifest.ts`)
- [ ] `TileManager` — frustum test, priority sort, load queue, LRU eviction
- [ ] `GPUTileCache` — slot allocator with LRU eviction (`gpu/tile-cache.ts`)
- [ ] Per-tile `TileInfo` uniform array fed to preprocess shader
- [ ] Preprocessing CLI tool: split large `.ply` into grid tiles + write `scene.json`
- [ ] UI: loading indicator per-tile (map-style overlay showing which tiles are resident)

**Phase 8 (Strategy 3 — octree LOD)**

- [ ] `OctreeNode` data structure + SSE traversal (`pipeline/octree-traversal.ts`)
- [ ] COPC loader: VLR parse + per-node HTTP range fetch (`loader/copc-loader.ts`)
- [ ] Potree 2.0 node format reader (for pre-converted datasets)
- [ ] `laz-perf` integration for LAZ chunk decompression
- [ ] Prefetch heuristic: start loading nodes likely to enter SSE threshold based on camera velocity

---

### 3.12 Test Assets

- `train.splat` from the original antimatter15 demo (small, well-known scene)
- Any `.ply` from official 3DGS outputs (garden, bicycle, room, stump, etc.)
- A Scaniverse-exported `.spz` file for SPZ loader testing
- Any standard PLY point cloud (e.g. Stanford Bunny, Dragon from The Stanford 3D Scanning Repository)
- An XYZ file from open LiDAR datasets (e.g. USGS 3DEP, OpenTopography)
- A LAS/LAZ file from any aerial survey dataset
- A `.copc.laz` file (any COPC-format LiDAR dataset) for octree streaming tests
- For unit testing compute shaders: hand-craft a 1–10 point/Gaussian buffer and verify output

---

## 4. Performance Improvement Plan

Large scenes (3–6M Gaussians) stress every part of the pipeline. The optimizations below are grouped
by category and ordered roughly by impact. Each section explains **what** to do, **why** it helps,
and **how** to implement it.

---

### 4.1 Loading & Streaming

#### 4.1.1 Streaming PLY/Splat Over the Network

**Problem**: A 3M-Gaussian PLY file is 500–800 MB. Waiting for a full download before rendering
gives a terrible UX.

**Solution**: Fetch with a `ReadableStream` and render partial data as it arrives.

```typescript
// loader/ply-parser.ts
const response = await fetch(url);
const reader = response.body!.getReader();
let received = 0;

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  // Append chunk to staging buffer
  stagingBuffer.set(value, received);
  received += value.length;

  // Upload complete Gaussians parsed so far
  const completeGaussians = Math.floor(received / PLY_STRIDE);
  if (completeGaussians > lastUploaded + CHUNK_SIZE) {
    uploadGaussiansToGPU(stagingBuffer, lastUploaded, completeGaussians);
    lastUploaded = completeGaussians;
    renderer.setNumGaussians(completeGaussians);
  }
}
```

**Key choices**:

- Upload in chunks of ~50k Gaussians so the viewer is usable within 1–2 seconds
- Keep a pre-allocated `ArrayBuffer` sized to the full file (`Content-Length` header) to avoid
  repeated allocations
- Show a progress bar with `received / totalBytes`

#### 4.1.2 Parse in a Web Worker

**Problem**: Parsing 3M Gaussians with `DataView` takes ~200–500 ms on the main thread, causing
a visible freeze.

**Solution**: Move all PLY parsing and binary packing into a `Worker`. Use `SharedArrayBuffer`
(or `postMessage` with `Transferable` ownership) for zero-copy hand-off to the main thread.

```
Main thread:  fetch chunks → postMessage({ chunk }) → Worker
Worker:       parse chunk → pack into Float32Array → postMessage({ gpuBuffer }, [gpuBuffer.buffer])
Main thread:  device.queue.writeBuffer(inputGpuBuffer, offset, gpuBuffer)
```

**Benefits**:

- Main thread stays responsive (camera, UI)
- Parsing and GPU upload overlap with prior frame's rendering

#### 4.1.3 Pre-sort by Importance at Load Time

**Problem**: A 6M-Gaussian scene often has many low-opacity, tiny Gaussians that contribute
almost nothing visually. Processing all of them costs full pipeline bandwidth.

**Solution**: Sort Gaussians by importance once, CPU-side, during loading:

```typescript
// Importance ≈ product of scales × sigmoid(opacity_logit)
const importance = Math.exp(logScaleX + logScaleY + logScaleZ)
                 * (1 / (1 + Math.exp(-opacityLogit)));
```

The top-N most important Gaussians are uploaded first (and optionally the remainder capped).
This matches the `convert.py` sorting in the WebGL reference. For interactive use, capping at
~3M Gaussians retains near-full visual quality.

---

### 4.2 GPU-Side Culling (Preprocessor)

Culling happens in the preprocessing compute shader and determines how many Gaussians reach the
sort + rasterize stages. **Every Gaussian culled here saves one sort key + all tile rasterization
work.**

#### 4.2.1 Frustum Culling

Cull Gaussians whose projected center (plus a generous margin for the splat radius) falls outside
the view frustum. This is already in the basic plan; the margin is critical to avoid popping:

```wgsl
// View-space near-plane cull
if (pos_view.z < uniforms.z_near * 0.8) { mark_culled(); return; }

// NDC-space extent cull — use padded bounds to account for splat radius
let limit = 1.3;  // 30% margin beyond screen edge
if (abs(pos_ndc.x) > limit || abs(pos_ndc.y) > limit) { mark_culled(); return; }
```

For large Gaussians the center may be outside the frustum while the splat itself is visible.
The 1.3 NDC limit is a conservative approximation; a tighter bound would project the
ellipse corners.

#### 4.2.2 Opacity Culling

```wgsl
let opacity = 1.0 / (1.0 + exp(-opacity_logit));
if (opacity < 0.004) { mark_culled(); return; }  // below 1/255 × 0.99 ≈ invisible
```

In a typical trained scene ~5–15% of Gaussians are effectively transparent and can be
discarded with no visual impact.

#### 4.2.3 Screen-Space Size Culling

After projecting the 2D covariance, compute the screen-space radius. A Gaussian whose splat
covers less than half a pixel contributes sub-threshold alpha and can be skipped:

```wgsl
// lambda_max from 2D covariance eigenvalue (computed during conic step)
let radius_px = 3.0 * sqrt(lambda_max);
if (radius_px < 0.5) { mark_culled(); return; }
```

This is extremely effective in scenes viewed from a distance — easily 20–40% of Gaussians
vanish at typical viewing distances.

#### 4.2.4 Tile Count Cap

A single extremely large (blurry) Gaussian can touch thousands of tiles, dominating the key
buffer and sort. Cap the radius before generating tile intersections:

```wgsl
let radius_clamped = min(radius_px, f32(MAX_SPLAT_RADIUS_PX));  // e.g. 1024px max
```

This prevents degenerate cases from blowing up sort buffer size.

---

### 4.3 Stream Compaction: Sort Only Visible Gaussians

**This is the single biggest performance win beyond the reference implementation.**

**Problem**: If frustum + opacity + size culling removes 50% of Gaussians, the reference still
generates tile-depth keys for all N Gaussians (checking `radii == 0` in the key shader).
The sort operates on a buffer pre-allocated for all N Gaussians.

**Solution**: After the preprocessing pass, use a GPU **stream compaction** (prefix sum on the
visibility flags) to produce a compact array of only the visible Gaussian indices.
All subsequent passes (key generation, sort) work on `numVisible << numTotal`.

```
[Preprocessing pass]
  Output: visible_flags[i] = 1 if Gaussian i passed all culls, else 0
  Output: GaussData[i]     (only valid for visible Gaussians)

[Compact pass — new prefix-sum.wgsl entry]
  Input:  visible_flags[0..N]
  Output: compact_ids[0..numVisible]   (list of visible Gaussian indices)
  Output: numVisible (atomic counter or readback)

[Tile-depth key pass]
  Iterate over compact_ids[0..numVisible] instead of all N Gaussians
  → Key buffer is numVisible × avgTilesPerSplat, not N × avgTilesPerSplat

[Radix sort]
  count = numVisible × avgTilesPerSplat   (much smaller)
```

**Expected gains**:

- At a 50% cull rate, sort input is halved → sort time roughly halved
- At 70% cull rate (distant/cluttered scenes), sort time drops to 30%
- Key buffer allocation can be sized to `numVisible × maxTiles` (smaller)

**Implementation**:

```wgsl
// In prefix-sum.wgsl, add a compaction pass
@group(0) @binding(0) var<storage, read>       visible_flags: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix:        array<u32>;
@group(0) @binding(2) var<storage, read_write> compact_ids:   array<u32>;

// After prefix sum is complete:
if (visible_flags[gid] == 1u) {
  compact_ids[prefix[gid]] = gid;
}
```

---

### 4.4 Sort Optimizations

#### 4.4.1 Sort Only When Camera Moves

The sort result from the previous frame is still nearly correct if the camera barely moved.
Re-using the previous frame's order causes minor transparency artifacts at Gaussian boundaries,
but they're imperceptible below a velocity threshold.

```typescript
// In renderer.ts, before submitting the sort
const viewDelta = dot(prevViewDir, currViewDir);
if (viewDelta > 0.9998 && translationDelta < 0.001) {
  skipSort = true;   // reuse previous sort
}
```

The WebGL reference uses a similar threshold (`dot product > 0.99`).

#### 4.4.2 Depth Key Precision

The tile-depth key packs tile index in the upper 16 bits and depth in the lower 16 bits.
The depth is normalized over the **visible** depth range `[min_visible_depth, max_visible_depth]`
rather than the full `[z_near, z_far]` range. This maximizes sort resolution for the actually
visible depth range and avoids depth aliasing in deep scenes.

```wgsl
// In tile-depth-key.wgsl
// min/max depth computed in preprocessor via atomicMin/atomicMax on a separate buffer
let depth_normalized = (depth - uniforms.min_visible_depth)
                     / (uniforms.max_visible_depth - uniforms.min_visible_depth);
let depth_uint = u32(clamp(depth_normalized, 0.0, 1.0) * 65535.0);
```

#### 4.4.3 Reduce Radix Sort Bit Width

If `numVisible × avgTiles < 16M` (i.e., the compacted key count fits in 24 bits for the
tile×depth key), use a 32-bit sort but only run enough passes to cover the actual key range.
The `webgpu-radix-sort` library supports configuring `bit_count`.

---

### 4.5 Rasterizer Optimizations

#### 4.5.1 Workgroup-Local Shared Memory Prefetch

The tile rasterizer's inner loop fetches `GaussData` entries by scattered indices. On many GPU
architectures, loading from a global storage buffer with non-sequential access is slow.
Prefetch each Gaussian's data into workgroup shared memory before the per-pixel loop:

```wgsl
var<workgroup> shared_gauss: array<GaussData, 64>;   // load up to 64 per batch

// Batch loop
for (var batch_start = tile_start; batch_start < tile_end; batch_start += 64u) {
  // Cooperatively load batch into shared memory (all 32 threads help)
  let load_idx = batch_start + local_invocation_index;
  if (load_idx < tile_end) {
    shared_gauss[local_invocation_index] = gauss_data[sorted_ids[load_idx]];
  }
  workgroupBarrier();

  // Per-pixel evaluation using shared_gauss[0..min(64, remaining)]
  for (var j = 0u; j < min(64u, tile_end - batch_start); j++) {
    let g = shared_gauss[j];
    // ... evaluate
  }
  workgroupBarrier();
}
```

This converts scattered reads into a cooperative sequential load, which is much more
cache-friendly.

#### 4.5.2 Early Exit with Workgroup Reduction

The reference's early exit checks per-thread transparency. Improve it with a workgroup-wide
check so the entire workgroup exits when **all** pixels in the tile are saturated:

```wgsl
var<workgroup> any_active: atomic<u32>;

// At start of each Gaussian iteration:
atomicStore(&any_active, 0u);
workgroupBarrier();

// After per-pixel update:
if (T > 1.0/255.0) { atomicAdd(&any_active, 1u); }
workgroupBarrier();
if (atomicLoad(&any_active) == 0u) { break; }  // entire tile is saturated
```

#### 4.5.3 Avoid Storing Culled GaussData Slots

After compaction, the `GaussData` output buffer only needs entries for `numVisible` Gaussians.
Index it by compact position (`compact_ids[i]`) rather than original Gaussian index.
This improves cache hit rate during rasterization because visible Gaussian data is contiguous.

---

### 4.6 Memory Bandwidth Reduction

#### 4.6.1 Use f16 for GaussData Fields in the Rasterizer

The rasterizer reads `GaussData` for every Gaussian touching every tile. Halving the struct
size directly halves bandwidth in the inner loop.

Fields that can be safely stored as `f16`:

- `uv: vec2<f16>` — sub-pixel screen position (sufficient at 0.01px precision)
- `conic: vec3<f16>` — inverse covariance values (range is typically [−10, 10])
- `color: vec3<f16>` — RGB (range [0, 1])
- `opacity: f16`

Fields that must stay `f32`:

- `depth` — used for sort key normalization, needs precision
- `tiles_touched`, `cum_tiles_touched` — integer counters

WebGPU supports `f16` in WGSL with the `shader-f16` device feature. Check at init:

```typescript
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
});
```

**Fallback**: If `shader-f16` is unavailable, pack two f16 values into one u32 manually.

#### 4.6.2 Compact Input Buffer (DC-only vs Full SH)

Full SH degree-3 costs 200 bytes/Gaussian. A DC-only layout costs 16 bytes/Gaussian — 12.5×
smaller, massively reducing the preprocessing pass bandwidth:

| Layout | Bytes/Gaussian | 3M total |
|--------|---------------|---------|
| Full (degree-3 SH) | 200 B | 600 MB |
| Degree-1 SH only | 64 B | 192 MB |
| DC only (degree-0) | 16 B | 48 MB |

Strategy: detect available SH degree from PLY header. Use the appropriate WGSL struct variant
(specialization constant or pipeline variant per SH degree).

---

### 4.7 Level of Detail (LOD)

#### 4.7.1 Distance-Based SH Degree Reduction

Spherical Harmonics give view-dependent color highlights. At distance, the contribution of
degree 2 and 3 SH is invisible. Evaluate fewer bands for distant Gaussians:

```wgsl
let dist = length(pos - uniforms.camera_pos);
let sh_degree: i32;
if      (dist < 2.0)  { sh_degree = 3; }
else if (dist < 5.0)  { sh_degree = 2; }
else if (dist < 10.0) { sh_degree = 1; }
else                  { sh_degree = 0; }
```

This reduces ALU cost in the preprocessing compute shader for the majority of Gaussians in a
large scene.

#### 4.7.2 Importance-Ranked Cap

At load time, sort Gaussians by importance (see §4.1.3) and expose a `maxGaussians` slider.
The renderer only uploads and processes the top-N. Set a sensible default:

| Device class | Default cap |
|---|---|
| Desktop GPU (4GB+) | 5M |
| Integrated GPU | 1M |
| Mobile GPU | 500K |

Detect device tier from `GPUAdapterInfo.architecture` or simply from total VRAM via a
heuristic based on buffer allocation success.

#### 4.7.3 Screen-Coverage LOD

Batch Gaussians into spatial cells (octree or flat grid). When an entire cell projects to
fewer than N×N pixels on screen, replace it with a single representative Gaussian (centroid
position, blended color, combined opacity). This is analogous to mipmap reduction and prevents
micro-splat overdraw in distant regions.

**Implementation complexity is high** — treat this as a Phase 7 stretch goal.

---

### 4.8 Temporal & Frame-Level Optimizations

#### 4.8.1 Render-on-Demand (Dirty Flag)

Only re-run the full pipeline when the camera or scene data changes. The current plan already
mentions a dirty flag; the key is that all 5 GPU passes are skipped, not just the rasterizer:

```typescript
if (!cameraDirty && !dataDirty) {
  // Blit the last frame's output texture directly — ~0.1ms
  runScreenBlit(encoder, lastOutputTexture);
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
  return;
}
```

#### 4.8.2 Resolution Scaling During Camera Motion

While the camera is actively moving, render at 50–75% resolution and scale up with bilinear
filtering for the screen blit. This cuts rasterizer cost (proportional to pixel count) by up to
4× during panning/orbiting, then snap back to full resolution when the camera stops.

```typescript
const scale = cameraMoveVelocity > VELOCITY_THRESHOLD ? 0.5 : 1.0;
const renderW = Math.ceil(canvas.width * scale);
const renderH = Math.ceil(canvas.height * scale);
// Resize output texture if scale changed, re-dispatch rasterizer at renderW×renderH
```

#### 4.8.3 Asynchronous GPU Readback for Visible Count

After the preprocessing pass, `numVisible` (the result of stream compaction) must be known on
the CPU to size the sort dispatch and the key buffer. Use `mapAsync` with a small staging buffer
rather than blocking the CPU:

```typescript
// Kick off the read at the end of frame N
device.queue.submit([encoder.finish()]);
visibleCountBuffer.mapAsync(GPUMapMode.READ).then(() => {
  numVisible = new Uint32Array(visibleCountBuffer.getMappedRange())[0];
  visibleCountBuffer.unmap();
  // Used for frame N+1 dispatch sizing
});
```

This means frame N+1 uses frame N's visible count (one frame latency), which is fine since
scene composition doesn't change dramatically between consecutive frames.

---

### 4.9 Profiling Strategy

Use WebGPU **timestamp queries** to measure each pass independently:

```typescript
const querySet = device.createQuerySet({ type: 'timestamp', count: 12 });
// Pairs: [start, end] for each of 6 passes
```

| Pass | Expected share of frame time |
|------|------------------------------|
| Preprocessing (gauss.wgsl) | 10–20% |
| Prefix sum | 2–5% |
| Tile-depth key gen | 3–8% |
| Radix sort | 30–50% |
| Tile rasterizer | 20–35% |
| Screen blit | <1% |

**Typical bottleneck**: radix sort dominates at high Gaussian counts. Stream compaction
(§4.3) is the primary lever to reduce it. Rasterizer cost grows with tile overdraw depth.

**Tooling**:

- Chrome DevTools GPU tab for overall frame time
- `timestamp-query` feature for per-pass breakdown
- `console.time` around `mapAsync` calls to measure CPU stalls

---

### 4.10 Performance Checklist by Phase

#### Phase 1–2 (Correctness first)

- [x] Frustum culling (z_near + NDC bounds)
- [x] Opacity culling (sigmoid < threshold)
- [x] Screen-space size culling (radius < 0.5px)

#### Phase 3–4 (Sort efficiency)

- [ ] Stream compaction: compact visible list before sort
- [ ] Sort dirty flag: skip sort when camera barely moves
- [ ] Depth normalization over visible depth range only

#### Phase 5–6 (Rasterizer efficiency)

- [ ] Workgroup shared memory prefetch for GaussData
- [ ] Workgroup-wide early exit via `atomicAdd`
- [ ] Resolution scaling during camera motion

#### Phase 7 (Memory & bandwidth)

- [ ] f16 GaussData fields (with `shader-f16` feature check)
- [ ] DC-only vs full-SH input buffer variants
- [ ] Distance-based SH degree LOD

#### Phase 8 (Scale)

- [ ] Importance-based Gaussian cap with UI slider
- [ ] Async visible count readback for dispatch sizing
- [ ] Timestamp query profiling UI overlay

---

## 5. Summary of Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Sorting | GPU radix sort | WebGL reference does CPU counting sort (bottleneck at 3M+); GPU sort scales better |
| Rasterization | Compute shader, tile-based | Avoids raster pipeline limitations; natural fit for alpha accumulation |
| Tile size | 16×16 | Same as reference; balances workgroup size with cache locality |
| SH degree | Start degree-0, add 1-3 later | Simpler to debug; DC color is sufficient for a working viewer |
| Input format | PLY + .splat (Tier 1); .spz + .ksplat (Tier 2) | PLY = training source of truth; .splat = common web format; .spz = 10× compressed, glTF standard |
| Camera | Orbit + WASD | Standard viewer interaction model |
| Color output | `rgba8unorm` storage texture | Sufficient precision; cheap write in compute, cheap read in blit |
| Loading | Streaming + Web Worker | Avoid main-thread freeze; render partial data within 1-2s of fetch start |
| Culling | Frustum + opacity + size (GPU) | Cull in preprocessor; every Gaussian culled here saves sort + raster work |
| Stream compaction | GPU prefix-sum visible list | Sort only culled-surviving Gaussians; 50% cull → ~50% sort time reduction |
| GaussData precision | f16 fields where safe | Halves rasterizer bandwidth; guarded by `shader-f16` feature check |
| Dirty rendering | Skip all passes if camera static | Renders at near 0 cost for static views (interactive presentations) |
| Resolution scaling | 50% during camera movement | Cuts raster cost 4× during navigation; imperceptible at high velocity |
| Point clouds | Gaussian conversion (≤2M) + compute rasterizer (>2M) | Gaussian path reuses whole pipeline for free; compute path scales to billions of points |
