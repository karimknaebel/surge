const viewer = document.getElementById("pointmap-viewer");
const stage = viewer.querySelector(".pointmap-stage");
const status = viewer.querySelector(".pointmap-status");
const statusText = viewer.querySelector(".pointmap-status-text");
const progress = viewer.querySelector(".pointmap-progress");
const download = viewer.querySelector(".pointmap-download");
const downloadSize = viewer.querySelector(".pointmap-download-size");
const downloadPercent = viewer.querySelector(".pointmap-download-percent");
const label = viewer.querySelector(".pointmap-label");
const picker = document.getElementById("pointmap-picker");
const frustumButton = document.getElementById("pointmap-frustum");
const resetButton = document.getElementById("pointmap-reset");
const cameraButton = document.getElementById("pointmap-camera");
const fullscreenButton = document.getElementById("pointmap-fullscreen");
const previewBase = new URL("../previews/", new URL(viewer.dataset.assetBase, document.baseURI));
const examples = [
  { id: "chairs", label: "Chairs", width: 928, height: 768 },
  { id: "umic_building", label: "UMIC Building", width: 720, height: 976 },
  { id: "kitchen", label: "Kitchen", width: 976, height: 720 },
  { id: "maitreya", label: "Maitreya", width: 976, height: 720 },
  { id: "office_chair", label: "Office Chair", width: 720, height: 976 },
  { id: "oranges", label: "Oranges", width: 1024, height: 688 },
  { id: "sandwich", label: "Sandwich", width: 976, height: 720 },
  { id: "staircase", label: "Staircase", width: 1136, height: 624 },
  { id: "volterra", label: "Volterra", width: 1136, height: 624 },
];
let activeExample = null;
let loading = null;
let rendererReady = null;

for (const example of examples) {
  const button = document.createElement("button");
  button.className = "pointmap-option";
  button.type = "button";
  button.dataset.example = example.id;
  button.setAttribute("aria-label", example.label);
  button.setAttribute("aria-pressed", "false");
  button.innerHTML = `<img src="${previewBase}${example.id}.webp" width="${example.width}" height="${example.height}" alt="" loading="lazy" draggable="false" />`;
  button.addEventListener("click", () => loadExample(example));
  picker.append(button);
}

loadExample(examples[0]);

async function loadExample(example) {
  if (activeExample === example) return;
  activeExample = example;
  loading?.abort();
  const request = new AbortController();
  loading = request;
  statusText.textContent = "Loading point map…";
  progress.removeAttribute("value");
  progress.removeAttribute("aria-valuetext");
  downloadSize.textContent = "0.0 MB downloaded";
  downloadPercent.textContent = "";
  download.hidden = false;
  status.hidden = false;
  stage.setAttribute("aria-busy", "true");
  for (const button of picker.children) {
    const selected = button.dataset.example === example.id;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  try {
    rendererReady ??= createViewer().catch((error) => {
      rendererReady = null;
      throw error;
    });
    const [display, { metadata, positions, colors, image }] = await Promise.all([
      rendererReady,
      import("./pointmap-data.js?v=20261028-intrinsics").then(({ loadPointmap }) =>
        loadPointmap(viewer.dataset.assetBase, example.id, request.signal, (loaded, total) => {
          if (request.signal.aborted) return;
          if (total) {
            progress.value = Math.min(loaded / total, 1);
            downloadSize.textContent = `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
            downloadPercent.textContent = `${Math.floor(progress.value * 100)}%`;
          } else {
            progress.removeAttribute("value");
            downloadSize.textContent = `${(loaded / 1e6).toFixed(1)} MB downloaded`;
            downloadPercent.textContent = "";
          }
          progress.setAttribute("aria-valuetext", `${downloadSize.textContent} ${downloadPercent.textContent}`.trim());
        })),
    ]);
    if (request.signal.aborted) return;
    display(metadata, positions, colors, image);
    label.textContent = `${metadata.count.toLocaleString()} points`;
    label.hidden = false;
    status.hidden = true;
    stage.setAttribute("aria-busy", "false");
  } catch (error) {
    if (request.signal.aborted) return;
    request.abort();
    console.error("Point map could not be loaded:", error);
    statusText.textContent = "Could not load this point map. Select an example to try again.";
    download.hidden = true;
    stage.setAttribute("aria-busy", "false");
    activeExample = null;
    for (const button of picker.children) {
      button.classList.remove("is-active");
      button.setAttribute("aria-pressed", "false");
    }
  }
}

async function createViewer() {
  const [THREE, { OrbitControls }] = await Promise.all([
    import("three"),
    import("https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/controls/OrbitControls.js"),
  ]);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x1c1c1c);
  renderer.domElement.setAttribute("aria-label", "Interactive point map. Drag to orbit, scroll to zoom, right-drag to pan. Double-click the camera frustum for its perspective.");
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera();
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.rotateSpeed = matchMedia("(pointer: coarse)").matches ? 0.5 : 1;
  const material = new THREE.ShaderMaterial({
    uniforms: { pointSize: { value: 2 * renderer.getPixelRatio() } },
    vertexShader: `
      attribute vec3 color;
      uniform float pointSize;
      varying vec3 pointColor;
      void main() {
        pointColor = color;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = pointSize;
      }
    `,
    fragmentShader: `
      varying vec3 pointColor;
      void main() {
        if (distance(gl_PointCoord, vec2(0.5)) > 0.5) discard;
        // The recording stores sRGB bytes, already in the canvas output color space.
        gl_FragColor = vec4(pointColor, 1.0);
      }
    `,
  });
  let points = null;
  let metadata = null;
  let frustum = null;
  let transitionFrame = 0;
  let hasView = false;

  function render() {
    if (frustum) frustum.visible = frustumButton.getAttribute("aria-pressed") === "true" &&
      camera.position.length() > metadata.targetDepth * 0.01;
    renderer.render(scene, camera);
  }

  function resize() {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    if (metadata) {
      const [imageWidth, imageHeight] = metadata.resolution;
      const scale = Math.min(width / imageWidth, height / imageHeight);
      camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(height / (2 * metadata.intrinsics.fy * scale)));
      camera.updateProjectionMatrix();
      camera.projectionMatrix.elements[0] = 2 * metadata.intrinsics.fx * scale / width;
      camera.projectionMatrix.elements[8] = (imageWidth - 2 * metadata.intrinsics.cx) * scale / width;
      camera.projectionMatrix.elements[9] = (2 * metadata.intrinsics.cy - imageHeight) * scale / height;
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    }
    render();
  }

  function stopTransition() {
    cancelAnimationFrame(transitionFrame);
    transitionFrame = 0;
  }

  function moveTo(position) {
    stopTransition();
    const startPosition = camera.position.clone();
    const startTarget = controls.target.clone();
    const target = new THREE.Vector3(0, 0, -metadata.targetDepth);
    const started = performance.now();
    const duration = hasView ? 500 : 0;
    hasView = true;

    function animate(now) {
      const progress = duration ? Math.min((now - started) / duration, 1) : 1;
      const eased = progress * progress * (3 - 2 * progress);
      camera.position.lerpVectors(startPosition, position, eased);
      camera.up.set(0, 1, 0);
      controls.target.lerpVectors(startTarget, target, eased);
      controls.update();
      render();
      transitionFrame = progress < 1 ? requestAnimationFrame(animate) : 0;
    }

    animate(started);
  }

  function reset() {
    moveTo(new THREE.Vector3(
      Math.sin(THREE.MathUtils.degToRad(24)) * metadata.targetDepth * 1.45,
      Math.tan(THREE.MathUtils.degToRad(6)) * metadata.targetDepth * 1.45,
      (Math.cos(THREE.MathUtils.degToRad(24)) * 1.45 - 1) * metadata.targetDepth,
    ));
  }

  function cameraView() {
    moveTo(new THREE.Vector3());
  }

  renderer.domElement.addEventListener("dblclick", (event) => {
    if (!points) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x / rect.width * 2 - 1, 1 - y / rect.height * 2), camera);
    raycaster.params.Line.threshold = 12 * camera.position.length() / (rect.height * camera.projectionMatrix.elements[5]);
    if (frustum.visible && raycaster.intersectObject(frustum).length) {
      cameraView();
    }
  });

  controls.addEventListener("change", render);
  controls.addEventListener("start", stopTransition);
  new ResizeObserver(resize).observe(stage);
  resetButton.addEventListener("click", reset);
  cameraButton.addEventListener("click", cameraView);
  frustumButton.addEventListener("click", () => {
    const visible = frustumButton.getAttribute("aria-pressed") !== "true";
    frustumButton.setAttribute("aria-pressed", String(visible));
    frustumButton.title = visible ? "Hide camera frustum" : "Show camera frustum";
    render();
  });
  fullscreenButton.hidden = !document.fullscreenEnabled;
  fullscreenButton.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement === viewer) await document.exitFullscreen();
      else await viewer.requestFullscreen();
    } catch (error) {
      console.error("Fullscreen could not be opened:", error);
    }
  });
  document.addEventListener("fullscreenchange", () => {
    const expanded = document.fullscreenElement === viewer;
    fullscreenButton.title = expanded ? "Exit fullscreen" : "Fullscreen";
    fullscreenButton.setAttribute("aria-label", fullscreenButton.title);
    fullscreenButton.querySelector("i").classList.toggle("fa-expand", !expanded);
    fullscreenButton.querySelector("i").classList.toggle("fa-compress", expanded);
  });

  return (nextMetadata, positions, colors, image) => {
    metadata = nextMetadata;
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
      scene.remove(frustum);
      for (const object of frustum.children) {
        object.geometry.dispose();
        object.material.map?.dispose();
        object.material.dispose();
      }
    }
    // Rerun uses right/down/forward; Three.js uses right/up/back.
    for (let i = 0; i < positions.length; i += 3) {
      positions[i + 1] *= -1;
      positions[i + 2] *= -1;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    geometry.computeBoundingSphere();
    points = new THREE.Points(geometry, material);
    scene.add(points);
    frustum = new THREE.Group();
    const [width, height] = metadata.resolution;
    const depth = metadata.targetDepth * 0.2;
    const corners = [[0, 0], [width, 0], [width, height], [0, height]].map(([x, y]) => new THREE.Vector3(
      (x - metadata.intrinsics.cx) * depth / metadata.intrinsics.fx,
      -(y - metadata.intrinsics.cy) * depth / metadata.intrinsics.fy,
      -depth,
    ));
    frustum.add(new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(corners.flatMap((corner, i) => [
        new THREE.Vector3(), corner, corner, corners[(i + 1) % 4],
      ])),
      new THREE.LineBasicMaterial({ color: 0x72c5ff }),
    ));
    const imagePlane = new THREE.BufferGeometry().setFromPoints(corners);
    imagePlane.setIndex([0, 1, 2, 0, 2, 3]);
    imagePlane.setAttribute("uv", new THREE.Float32BufferAttribute([0, 1, 1, 1, 1, 0, 0, 0], 2));
    const texture = new THREE.CanvasTexture(image);
    texture.colorSpace = THREE.SRGBColorSpace;
    frustum.add(new THREE.Mesh(imagePlane, new THREE.MeshBasicMaterial({
      map: texture,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.4,
      depthWrite: false,
    })));
    scene.add(frustum);
    camera.near = metadata.targetDepth * 0.001;
    camera.far = Math.max(...metadata.bounds.flat().map(Math.abs), metadata.targetDepth) * 100;
    controls.minDistance = metadata.targetDepth * 0.01;
    controls.maxDistance = camera.far * 0.5;
    reset();
    resize();
    frustumButton.disabled = false;
    resetButton.disabled = false;
    cameraButton.disabled = false;
    fullscreenButton.disabled = false;
  };
}
