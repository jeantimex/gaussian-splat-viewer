import type { GaussianSceneData, PointCloudSceneData } from './types.ts';

// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------

interface PropInfo {
  name: string;
  type: string;
  offset: number; // byte offset within one vertex row
}

interface PLYHeader {
  numVertices: number;
  stride: number;
  props: PropInfo[];
  dataOffset: number; // byte offset in the full buffer where vertex data begins
}

const TYPE_SIZES: Record<string, number> = {
  double: 8,
  float: 4,
  int: 4,
  uint: 4,
  short: 2,
  ushort: 2,
  char: 1,
  uchar: 1,
};

function parsePLYHeader(buffer: ArrayBuffer): PLYHeader {
  // Read up to 64 KB — enough for any PLY header
  const headerBytes = new Uint8Array(buffer, 0, Math.min(65536, buffer.byteLength));
  const text = new TextDecoder().decode(headerBytes);

  const marker = 'end_header\n';
  const markerIdx = text.indexOf(marker);
  if (markerIdx === -1) throw new Error('PLY: missing end_header');

  const header = text.slice(0, markerIdx);
  const dataOffset = new TextEncoder().encode(text.slice(0, markerIdx + marker.length)).length;

  const vertexMatch = header.match(/element vertex (\d+)/);
  if (!vertexMatch) throw new Error('PLY: missing element vertex count');
  const numVertices = parseInt(vertexMatch[1]!);

  // Only read properties that belong to the vertex element
  const vertexSectionStart = header.indexOf('element vertex');
  const afterVertex = header.slice(vertexSectionStart);
  // Stop at the next element or end
  const nextElement = afterVertex.indexOf('\nelement ', 1);
  const vertexSection = nextElement === -1 ? afterVertex : afterVertex.slice(0, nextElement + 1);

  const props: PropInfo[] = [];
  let offset = 0;
  for (const line of vertexSection.split('\n')) {
    const m = line.trim().match(/^property (\S+) (\S+)$/);
    if (!m) continue;
    const [, type, name] = m as [string, string, string];
    props.push({ name, type, offset });
    offset += TYPE_SIZES[type] ?? 4;
  }

  return { numVertices, stride: offset, props, dataOffset };
}

function makeReader(
  view: DataView,
  stride: number,
  prop: PropInfo | undefined,
): (i: number) => number {
  if (!prop) return () => 0;
  const { offset, type } = prop;
  switch (type) {
    case 'float':
      return (i) => view.getFloat32(i * stride + offset, true);
    case 'double':
      return (i) => view.getFloat64(i * stride + offset, true);
    case 'int':
      return (i) => view.getInt32(i * stride + offset, true);
    case 'uint':
      return (i) => view.getUint32(i * stride + offset, true);
    case 'short':
      return (i) => view.getInt16(i * stride + offset, true);
    case 'ushort':
      return (i) => view.getUint16(i * stride + offset, true);
    case 'uchar':
      return (i) => view.getUint8(i * stride + offset);
    default:
      return () => 0;
  }
}

// ---------------------------------------------------------------------------
// Subtype detection
// ---------------------------------------------------------------------------

export function detectPLYSubtype(headerText: string): 'gaussian' | 'compressed' | 'pointcloud' {
  if (headerText.includes('element chunk') && headerText.includes('packed_position')) {
    return 'compressed';
  }
  if (headerText.includes('f_dc_0')) return 'gaussian';
  return 'pointcloud';
}

// ---------------------------------------------------------------------------
// Standard 3DGS Gaussian PLY
// ---------------------------------------------------------------------------

export function parseGaussianPLY(buffer: ArrayBuffer): GaussianSceneData {
  const hdr = parsePLYHeader(buffer);
  const { numVertices: N, stride, props, dataOffset } = hdr;

  const view = new DataView(buffer, dataOffset);
  const find = (name: string) => props.find((p) => p.name === name);

  // Build fast per-property readers
  const rx = makeReader(view, stride, find('x'));
  const ry = makeReader(view, stride, find('y'));
  const rz = makeReader(view, stride, find('z'));
  const rs0 = makeReader(view, stride, find('scale_0'));
  const rs1 = makeReader(view, stride, find('scale_1'));
  const rs2 = makeReader(view, stride, find('scale_2'));
  const rr0 = makeReader(view, stride, find('rot_0'));
  const rr1 = makeReader(view, stride, find('rot_1'));
  const rr2 = makeReader(view, stride, find('rot_2'));
  const rr3 = makeReader(view, stride, find('rot_3'));
  const rop = makeReader(view, stride, find('opacity'));
  const rdc0 = makeReader(view, stride, find('f_dc_0'));
  const rdc1 = makeReader(view, stride, find('f_dc_1'));
  const rdc2 = makeReader(view, stride, find('f_dc_2'));

  // Determine SH degree from f_rest_* count
  const restProps = props.filter((p) => p.name.startsWith('f_rest_'));
  const nRestPerChannel = restProps.length / 3;
  const shDegree = Math.round(Math.sqrt(nRestPerChannel + 1) - 1);

  // Build rest readers
  const restReaders = restProps.map((p) => makeReader(view, stride, p));

  const positions = new Float32Array(N * 3);
  const logScales = new Float32Array(N * 3);
  const rotations = new Float32Array(N * 4);
  const opacities = new Float32Array(N);
  const shDC = new Float32Array(N * 3);
  const shRest = new Float32Array(restProps.length > 0 ? N * restProps.length : 0);

  for (let i = 0; i < N; i++) {
    positions[i * 3 + 0] = rx(i);
    positions[i * 3 + 1] = ry(i);
    positions[i * 3 + 2] = rz(i);

    logScales[i * 3 + 0] = rs0(i);
    logScales[i * 3 + 1] = rs1(i);
    logScales[i * 3 + 2] = rs2(i);

    // 3DGS quaternion order: rot_0=w, rot_1=x, rot_2=y, rot_3=z
    rotations[i * 4 + 0] = rr0(i);
    rotations[i * 4 + 1] = rr1(i);
    rotations[i * 4 + 2] = rr2(i);
    rotations[i * 4 + 3] = rr3(i);

    opacities[i] = rop(i);

    shDC[i * 3 + 0] = rdc0(i);
    shDC[i * 3 + 1] = rdc1(i);
    shDC[i * 3 + 2] = rdc2(i);

    for (let j = 0; j < restProps.length; j++) {
      shRest[i * restProps.length + j] = restReaders[j]!(i);
    }
  }

  return {
    kind: 'gaussian',
    numGaussians: N,
    positions,
    logScales,
    rotations,
    opacities,
    shDC,
    shRest,
    shDegree,
  };
}

// ---------------------------------------------------------------------------
// Point-cloud PLY  (no Gaussian properties)
// ---------------------------------------------------------------------------

export function parsePointCloudPLY(buffer: ArrayBuffer): PointCloudSceneData {
  const hdr = parsePLYHeader(buffer);
  const { numVertices: N, stride, props, dataOffset } = hdr;

  const view = new DataView(buffer, dataOffset);
  const find = (name: string) => props.find((p) => p.name === name);

  const rx = makeReader(view, stride, find('x'));
  const ry = makeReader(view, stride, find('y'));
  const rz = makeReader(view, stride, find('z'));

  // Color — accept 'red/green/blue' or 'r/g/b'
  const rr = makeReader(view, stride, find('red') ?? find('r'));
  const rg = makeReader(view, stride, find('green') ?? find('g'));
  const rb = makeReader(view, stride, find('blue') ?? find('b'));
  const hasColor = !!(find('red') ?? find('r'));

  const rnx = makeReader(view, stride, find('nx'));
  const rny = makeReader(view, stride, find('ny'));
  const rnz = makeReader(view, stride, find('nz'));
  const hasNormals = !!find('nx');

  const positions = new Float32Array(N * 3);
  const colors = hasColor ? new Uint8Array(N * 3) : new Uint8Array(0);
  const normals = hasNormals ? new Float32Array(N * 3) : new Float32Array(0);

  for (let i = 0; i < N; i++) {
    positions[i * 3 + 0] = rx(i);
    positions[i * 3 + 1] = ry(i);
    positions[i * 3 + 2] = rz(i);

    if (hasColor) {
      colors[i * 3 + 0] = rr(i);
      colors[i * 3 + 1] = rg(i);
      colors[i * 3 + 2] = rb(i);
    }
    if (hasNormals) {
      normals[i * 3 + 0] = rnx(i);
      normals[i * 3 + 1] = rny(i);
      normals[i * 3 + 2] = rnz(i);
    }
  }

  return { kind: 'pointcloud', numPoints: N, positions, colors, normals };
}
