import './style.css';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

async function init() {
  if (!navigator.gpu) {
    alert('WebGPU is not supported in this browser.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    alert('No WebGPU adapter found.');
    return;
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu') as GPUCanvasContext;

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format });

  // Resize canvas to match display size
  function resize() {
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;
    context.configure({ device, format });
  }

  resize();
  window.addEventListener('resize', resize);

  // Clear to a dark background to confirm WebGPU is working
  function frame() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

init().catch(console.error);
