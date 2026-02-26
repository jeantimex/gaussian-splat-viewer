export interface GaussianSceneData {
  kind: 'gaussian';
  numGaussians: number;
  positions:  Float32Array; // xyz  × N  (flat)
  logScales:  Float32Array; // xyz  × N  (log-space)
  rotations:  Float32Array; // wxyz × N  (quaternion)
  opacities:  Float32Array; // logit × N
  shDC:       Float32Array; // rgb  × N  (DC SH coefficients)
  shRest:     Float32Array; // higher SH coefficients, empty for DC-only formats
  shDegree:   number;       // 0, 1, 2, or 3
}

export interface PointCloudSceneData {
  kind: 'pointcloud';
  numPoints: number;
  positions: Float32Array; // xyz × N
  colors:    Uint8Array;   // rgb 0–255 × N  (empty if no color)
  normals:   Float32Array; // xyz × N        (empty if no normals)
}

export type SceneData = GaussianSceneData | PointCloudSceneData;
