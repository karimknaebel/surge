import { FloatType } from "three";
import { EXRLoader } from "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/loaders/EXRLoader.js";

export async function loadPointmap(base, id, signal, onProgress = () => {}) {
  const transfers = ["intrinsics.json", "exr", "png"].map((extension) => ({ extension, loaded: 0, total: null }));
  function reportProgress() {
    onProgress(
      transfers.reduce((sum, transfer) => sum + transfer.loaded, 0),
      transfers.every((transfer) => transfer.total !== null)
        ? transfers.reduce((sum, transfer) => sum + transfer.total, 0) : null,
    );
  }
  const files = await Promise.all(transfers.map(async (transfer) => {
    const response = await fetch(`${base}${id}.${transfer.extension}`, { signal });
    if (!response.ok) throw new Error(`${id}.${transfer.extension}: HTTP ${response.status}`);
    if (!response.headers.get("Content-Encoding")) {
      transfer.total = Number(response.headers.get("Content-Length")) || null;
    }
    reportProgress();
    const reader = response.body.getReader();
    const chunks = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      transfer.loaded += value.byteLength;
      reportProgress();
    }
    transfer.total = transfer.loaded;
    reportProgress();
    const buffer = new Uint8Array(transfer.loaded);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return buffer.buffer;
  }));
  signal.throwIfAborted();
  const [[fx, , cx], [, fy, cy]] = JSON.parse(new TextDecoder().decode(files[0]));
  if (![fx, fy, cx, cy].every(Number.isFinite) || fx <= 0 || fy <= 0) {
    throw new Error("Invalid camera intrinsics");
  }
  const xyz = new EXRLoader().setDataType(FloatType).parse(files[1]);
  const canvas = await decodeImage(new Blob([files[2]], { type: "image/png" }));
  signal.throwIfAborted();
  const { width, height } = xyz;
  if (canvas.width !== width || canvas.height !== height) {
    throw new Error("Point map and RGB dimensions do not match");
  }
  const rgba = canvas.getContext("2d").getImageData(0, 0, width, height).data;
  const positions = new Float32Array(width * height * 3);
  const colors = new Uint8Array(width * height * 3);
  const depths = new Float32Array(width * height);
  const bounds = [[Infinity, Infinity, Infinity], [-Infinity, -Infinity, -Infinity]];
  let count = 0;
  for (let i = 0; i < width * height; i++) {
    // EXRLoader returns bottom-up RGBA; the image grid is top-down.
    const offset = ((height - 1 - Math.floor(i / width)) * width + i % width) * 4;
    const x = xyz.data[offset], y = xyz.data[offset + 1], z = xyz.data[offset + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    positions[count * 3] = x;
    positions[count * 3 + 1] = y;
    positions[count * 3 + 2] = z;
    depths[count] = z;
    bounds[0][0] = Math.min(bounds[0][0], x);
    bounds[0][1] = Math.min(bounds[0][1], y);
    bounds[0][2] = Math.min(bounds[0][2], z);
    bounds[1][0] = Math.max(bounds[1][0], x);
    bounds[1][1] = Math.max(bounds[1][1], y);
    bounds[1][2] = Math.max(bounds[1][2], z);
    colors[count * 3] = rgba[i * 4];
    colors[count * 3 + 1] = rgba[i * 4 + 1];
    colors[count * 3 + 2] = rgba[i * 4 + 2];
    count++;
  }
  if (!count) throw new Error("Point map has no finite points");
  return {
    metadata: {
      count,
      resolution: [width, height],
      intrinsics: { fx: fx * width, fy: fy * height, cx: cx * width, cy: cy * height },
      targetDepth: median(depths.subarray(0, count)),
      bounds,
    },
    positions: positions.subarray(0, count * 3),
    colors: colors.subarray(0, count * 3),
    image: canvas,
  };
}

function median(values) {
  const middle = values.length >>> 1;
  let left = 0, right = values.length - 1;
  while (left < right) {
    const pivot = values[(left + right) >>> 1];
    let i = left, j = right;
    while (i <= j) {
      while (values[i] < pivot) i++;
      while (values[j] > pivot) j--;
      if (i <= j) {
        const value = values[i];
        values[i++] = values[j];
        values[j--] = value;
      }
    }
    if (middle <= j) right = j;
    else if (middle >= i) left = i;
    else break;
  }
  if (values.length % 2) return values[middle];
  let lower = -Infinity;
  for (let i = 0; i < middle; i++) lower = Math.max(lower, values[i]);
  return (lower + values[middle]) / 2;
}

async function decodeImage(blob) {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvas;
}
