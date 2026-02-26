import './style.css';
import { loadScene } from './loader/index.ts';
import type { SceneData } from './loader/index.ts';
import { createGaussianBuffer, GAUSSIAN_STRIDE } from './gpu/buffers.ts';
import {
  createUniformBuffer,
  updateUniforms,
  mat4Perspective,
  mat4LookAt,
  UNIFORMS_SIZE,
} from './gpu/uniforms.ts';
import {
  createPreprocessPass,
  encodePreprocessPass,
  type PreprocessPass,
} from './pipeline/preprocess.ts';

// ---------------------------------------------------------------------------
// WebGPU init
// ---------------------------------------------------------------------------

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

let gpuDevice: GPUDevice | null = null;
let canvasFormat: GPUTextureFormat = 'bgra8unorm';
let gpuContext: GPUCanvasContext | null = null;
let uniformBuffer: GPUBuffer | null = null;
let preprocessPass: PreprocessPass | null = null;

async function initWebGPU() {
  if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser.');

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found.');

  gpuDevice = await adapter.requestDevice();
  gpuContext = canvas.getContext('webgpu') as GPUCanvasContext;
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  gpuContext.configure({ device: gpuDevice, format: canvasFormat });
}

function resize() {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  gpuContext?.configure({ device: gpuDevice!, format: canvasFormat });
}

// Clear-loop to confirm WebGPU is alive
function frame() {
  if (!gpuDevice || !gpuContext) return;
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: gpuContext.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Verification overlay
// ---------------------------------------------------------------------------

const overlay = document.getElementById('overlay')!;
const dropHint = document.getElementById('drop-hint')!;

function showStatus(msg: string) {
  overlay.textContent = msg;
  overlay.className = 'info';
}

function showError(msg: string) {
  overlay.textContent = msg;
  overlay.className = 'error';
}

function showSceneInfo(scene: SceneData, parseMs: number, bufferBytes: number) {
  let html = '';

  if (scene.kind === 'gaussian') {
    const p = scene.positions;
    const fmt = (v: number) => v.toFixed(4);
    html += `<b>Format:</b> Gaussian Splat\n`;
    html += `<b>Gaussians:</b> ${scene.numGaussians.toLocaleString()}\n`;
    html += `<b>SH degree:</b> ${scene.shDegree}\n`;
    html += `<b>Parse time:</b> ${parseMs.toFixed(1)} ms\n`;
    html += `<b>GPU buffer:</b> ${(bufferBytes / 1024 / 1024).toFixed(1)} MB\n`;
    html += `<b>Uniform buffer:</b> ${UNIFORMS_SIZE} bytes\n`;
    html += `<b>Preprocess shader:</b> ${preprocessPass ? '✓ compiled + dispatched' : '—'}\n`;
    html += `<b>First positions:</b>\n`;
    for (let i = 0; i < Math.min(3, scene.numGaussians); i++) {
      html += `  [${i}] (${fmt(p[i * 3]!)}, ${fmt(p[i * 3 + 1]!)}, ${fmt(p[i * 3 + 2]!)})\n`;
    }
  } else {
    const p = scene.positions;
    const fmt = (v: number) => v.toFixed(4);
    html += `<b>Format:</b> Point Cloud\n`;
    html += `<b>Points:</b> ${scene.numPoints.toLocaleString()}\n`;
    html += `<b>Has colors:</b> ${scene.colors.length > 0}\n`;
    html += `<b>Has normals:</b> ${scene.normals.length > 0}\n`;
    html += `<b>Parse time:</b> ${parseMs.toFixed(1)} ms\n`;
    html += `<b>First positions:</b>\n`;
    for (let i = 0; i < Math.min(3, scene.numPoints); i++) {
      html += `  [${i}] (${fmt(p[i * 3]!)}, ${fmt(p[i * 3 + 1]!)}, ${fmt(p[i * 3 + 2]!)})\n`;
    }
  }

  overlay.innerHTML = html;
  overlay.className = 'info';
}

// ---------------------------------------------------------------------------
// File handling
// ---------------------------------------------------------------------------

async function handleFile(file: File) {
  dropHint.style.display = 'none';
  showStatus(`Loading ${file.name} …`);

  try {
    const t0 = performance.now();
    const scene = await loadScene(file);
    const parseMs = performance.now() - t0;

    let bufferBytes = 0;
    if (scene.kind === 'gaussian' && gpuDevice && uniformBuffer) {
      const gaussianBuffer = createGaussianBuffer(gpuDevice, scene);
      bufferBytes = scene.numGaussians * GAUSSIAN_STRIDE;

      // Update uniforms with correct Gaussian count
      uploadPlaceholderCamera(scene.numGaussians);

      // Build and run the preprocess pass
      preprocessPass?.gaussDataBuffer.destroy();
      preprocessPass = createPreprocessPass(
        gpuDevice,
        uniformBuffer,
        gaussianBuffer,
        scene.numGaussians,
      );

      const encoder = gpuDevice.createCommandEncoder();
      encodePreprocessPass(encoder, preprocessPass);
      gpuDevice.queue.submit([encoder.finish()]);
    }

    showSceneInfo(scene, parseMs, bufferBytes);
  } catch (err) {
    showError(`Error: ${err instanceof Error ? err.message : String(err)}`);
    console.error(err);
  }
}

// ---------------------------------------------------------------------------
// Drag-and-drop
// ---------------------------------------------------------------------------

function setupDragDrop() {
  const prevent = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  window.addEventListener('dragover', prevent);
  window.addEventListener('dragenter', prevent);
  window.addEventListener('dragleave', prevent);

  window.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });
}

// ---------------------------------------------------------------------------
// Camera helpers
// ---------------------------------------------------------------------------

/** Upload a simple look-at camera centred on the origin. Replaced in Phase 6. */
function uploadPlaceholderCamera(numGaussians = 0) {
  if (!gpuDevice || !uniformBuffer) return;
  const w = canvas.width || 1280;
  const h = canvas.height || 720;
  const fovY = (60 * Math.PI) / 180;
  const tanHalfFovY = Math.tan(fovY / 2);
  const tanHalfFovX = tanHalfFovY * (w / h);

  updateUniforms(gpuDevice, uniformBuffer, {
    viewMatrix: mat4LookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]),
    projMatrix: mat4Perspective(fovY, w / h, 0.1, 100),
    cameraPos: [0, 0, 5],
    tanHalfFovX,
    tanHalfFovY,
    focalX: w / (2 * tanHalfFovX),
    focalY: h / (2 * tanHalfFovY),
    scaleModifier: 1.0,
    screenWidth: w,
    screenHeight: h,
    numGaussians,
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function init() {
  try {
    await initWebGPU();
  } catch (err) {
    showError(`WebGPU init failed: ${err instanceof Error ? err.message : String(err)}`);
    return;
  }

  resize();
  window.addEventListener('resize', resize);

  // Allocate uniform buffer and upload a placeholder camera
  uniformBuffer = createUniformBuffer(gpuDevice!);
  uploadPlaceholderCamera();

  setupDragDrop();
  requestAnimationFrame(frame);
}

init().catch(console.error);
