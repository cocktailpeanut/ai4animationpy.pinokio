import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const THEME_KEY = "ai4animation-theme";
const STORAGE_FALLBACK_THEME = "dark";
const FRAME_DELAY_MS = 1000 / 30;
const SESSION_REPLACED_CLOSE_CODE = 4001;
const SESSION_REPLACED_MESSAGE = "This demo is now active in another window.";
const CLIENT_ID_KEY = "ai4animation-viewer-client-id";
const CLIENT_ID = (() => {
  const existing = sessionStorage.getItem(CLIENT_ID_KEY);
  if (existing) return existing;
  const value = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  sessionStorage.setItem(CLIENT_ID_KEY, value);
  return value;
})();
const CLIENT_STARTED_AT = Date.now();

const dom = {
  backButton: document.getElementById("back-button"),
  themeButton: document.getElementById("theme-button"),
  landingView: document.getElementById("landing-view"),
  viewerView: document.getElementById("viewer-view"),
  viewerLayout: document.getElementById("viewer-layout"),
  heroTitle: document.getElementById("hero-title"),
  heroCopy: document.getElementById("hero-copy"),
  heroUsecase: document.getElementById("hero-usecase"),
  heroHighlights: document.getElementById("hero-highlights"),
  sectionCaption: document.getElementById("section-caption"),
  demoGrid: document.getElementById("demo-grid"),
  viewport: document.getElementById("viewport"),
  demoCategory: document.getElementById("demo-category"),
  demoTitle: document.getElementById("demo-title"),
  demoDescription: document.getElementById("demo-description"),
  demoMeta: document.getElementById("demo-meta"),
  connectionBadge: document.getElementById("connection-badge"),
  zoomOutButton: document.getElementById("zoom-out-button"),
  recenterButton: document.getElementById("recenter-button"),
  zoomInButton: document.getElementById("zoom-in-button"),
  panelButton: document.getElementById("panel-button"),
  playbackButton: document.getElementById("playback-button"),
  speedRange: document.getElementById("speed-range"),
  speedValue: document.getElementById("speed-value"),
  mirrorToggle: document.getElementById("mirror-toggle"),
  skeletonToggle: document.getElementById("skeleton-toggle"),
  demoControls: document.getElementById("demo-controls"),
  toast: document.getElementById("toast"),
};

const state = {
  manifest: null,
  currentDemo: null,
  ws: null,
  wsInputTimer: null,
  framePrev: null,
  frameCurr: null,
  framePrevAt: 0,
  frameCurrAt: 0,
  entityNames: [],
  entityCount: 0,
  paused: false,
  speed: 1,
  mirror: false,
  showSkeleton: false,
  sliderValues: {},
  boneMap: {},
  skinnedMeshes: [],
  skeletonPairs: [],
  markerMeshes: new Map(),
  modelRoot: null,
  currentRootPos: new THREE.Vector3(),
  cameraFollowsRoot: false,
  hasSnappedToFrame: false,
  panelCollapsed: false,
  liveBounds: null,
};

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
camera.position.set(0.9, 1.6, 4.8);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
dom.viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enableRotate = true;
controls.enableZoom = true;
controls.enablePan = false;
controls.minDistance = 1.8;
controls.maxDistance = 12;
controls.rotateSpeed = 0.9;
controls.zoomSpeed = 0.95;
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.DOLLY,
  RIGHT: THREE.MOUSE.PAN,
};
controls.touches = {
  ONE: THREE.TOUCH.ROTATE,
  TWO: THREE.TOUCH.DOLLY_PAN,
};
controls.target.set(0, 1.1, 0);
renderer.domElement.style.touchAction = "none";

const ambient = new THREE.AmbientLight(0xffffff, 0.85);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
keyLight.position.set(4.5, 7.5, 5.5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 28;
keyLight.shadow.camera.left = -8;
keyLight.shadow.camera.right = 8;
keyLight.shadow.camera.top = 8;
keyLight.shadow.camera.bottom = -8;
scene.add(keyLight);

const fillLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.55);
scene.add(fillLight);

const groundMaterial = new THREE.MeshStandardMaterial({
  color: 0xe8e8ea,
  roughness: 0.92,
  metalness: 0.02,
});

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(80, 80),
  groundMaterial,
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
ground.position.y = -0.0001;
scene.add(ground);

const grid = new THREE.GridHelper(60, 60, 0xc8c8cd, 0xe0e0e4);
grid.position.y = 0.002;
scene.add(grid);

const skeletonGeometry = new THREE.BufferGeometry();
skeletonGeometry.setAttribute(
  "position",
  new THREE.BufferAttribute(new Float32Array(4096 * 3), 3),
);
const skeletonMaterial = new THREE.LineBasicMaterial({
  color: 0x0a69ff,
  transparent: true,
  opacity: 0.72,
  depthTest: false,
});
const skeletonLines = new THREE.LineSegments(skeletonGeometry, skeletonMaterial);
skeletonLines.frustumCulled = false;
skeletonLines.visible = false;
scene.add(skeletonLines);

const loader = new GLTFLoader();

function frameCameraToModel(root) {
  root.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) {
    return;
  }

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.01);
  const fitDistance = maxDim / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5)));
  const distance = fitDistance * 1.7;
  const offset = new THREE.Vector3(0, distance * 0.14, distance * 1.08);

  controls.target.copy(center);
  camera.position.copy(center).add(offset);
  camera.near = Math.max(0.01, distance / 100);
  camera.far = Math.max(100, distance * 20);
  camera.updateProjectionMatrix();
  controls.minDistance = Math.max(0.4, distance * 0.45);
  controls.maxDistance = Math.max(6, distance * 4.5);
  controls.update();
  state.currentRootPos.copy(center);
}

function fitCameraToCurrentMesh() {
  const target = getCameraTargetPoint();
  const maxDim = state.liveBounds ? state.liveBounds.maxDim : 1.8;
  const distance = Math.max(3.4, maxDim * 2.9);
  const offset = new THREE.Vector3(0, distance * 0.14, distance * 1.08);
  controls.target.copy(target);
  camera.position.copy(target).add(offset);
  controls.minDistance = Math.max(0.75, maxDim * 0.7);
  controls.maxDistance = Math.max(8, maxDim * 7.5);
  camera.near = Math.max(0.01, distance / 140);
  camera.far = Math.max(120, distance * 24);
  camera.updateProjectionMatrix();
  controls.update();
}

function getCameraTargetPoint() {
  return state.currentRootPos.clone();
}

function recenterCamera() {
  fitCameraToCurrentMesh();
}

function zoomBy(multiplier) {
  const offset = camera.position.clone().sub(controls.target);
  const nextDistance = THREE.MathUtils.clamp(
    offset.length() * multiplier,
    controls.minDistance,
    controls.maxDistance,
  );
  offset.setLength(nextDistance);
  camera.position.copy(controls.target).add(offset);
  controls.update();
}

function orbitBy(deltaTheta, deltaPhi = 0) {
  const offset = camera.position.clone().sub(controls.target);
  const spherical = new THREE.Spherical().setFromVector3(offset);
  spherical.theta += deltaTheta;
  spherical.phi = THREE.MathUtils.clamp(
    spherical.phi + deltaPhi,
    0.12,
    Math.PI - 0.12,
  );
  offset.setFromSpherical(spherical);
  camera.position.copy(controls.target).add(offset);
  controls.update();
}

function setPanelCollapsed(collapsed) {
  state.panelCollapsed = collapsed;
  dom.viewerLayout?.classList.toggle("panel-collapsed", collapsed);
  if (dom.panelButton) {
    dom.panelButton.textContent = collapsed ? "Show panel" : "Hide panel";
  }
  resize();
}

function getRouteDemoId() {
  return new URLSearchParams(window.location.search).get("demo");
}

function setRouteDemoId(demoId) {
  const url = new URL(window.location.href);
  if (demoId) {
    url.searchParams.set("demo", demoId);
  } else {
    url.searchParams.delete("demo");
  }
  window.history.pushState({}, "", url);
  renderRoute();
}

function loadTheme() {
  return window.localStorage.getItem(THEME_KEY) || STORAGE_FALLBACK_THEME;
}

function cycleTheme() {
  const current = loadTheme();
  const next = current === "dark" ? "light" : "dark";
  window.localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  dom.themeButton.textContent = theme === "dark" ? "Light" : "Dark";

  if (theme === "dark") {
    scene.background = new THREE.Color(0x09090b);
    groundMaterial.color.set(0x111115);
    grid.material.color.set(0x28282d);
    grid.material.vertexColors = false;
    skeletonMaterial.color.set(0x78a8ff);
  } else {
    scene.background = new THREE.Color(0xf6f6f3);
    groundMaterial.color.set(0xebebe8);
    grid.material.color.set(0xdadadf);
    grid.material.vertexColors = false;
    skeletonMaterial.color.set(0x0a69ff);
  }
}

function resize() {
  const width = dom.viewport.clientWidth || window.innerWidth;
  const height = dom.viewport.clientHeight || window.innerHeight * 0.72;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function showToast(message) {
  if (!message) {
    return;
  }
  dom.toast.textContent = message;
  dom.toast.classList.remove("hidden");
  window.clearTimeout(showToast.timeoutId);
  showToast.timeoutId = window.setTimeout(() => {
    dom.toast.classList.add("hidden");
  }, 3600);
}

function setConnectionState(kind, text) {
  dom.connectionBadge.textContent = text;
  dom.connectionBadge.className = `badge ${kind}`;
}

function renderManifestChrome() {
  const manifest = state.manifest;
  const categories = new Set(manifest.demos.map((demo) => demo.category));
  const highlights = manifest.highlights || [
    `${manifest.demos.length} demos`,
    `${categories.size} modes`,
    "Local only",
  ];

  dom.heroTitle.textContent = manifest.title || "Motion Playground";
  dom.heroCopy.textContent = manifest.subtitle || "A calm local browser surface for trying assets.";
  dom.heroUsecase.textContent = manifest.useCase || "Best for quick browsing when you want to see how an asset feels.";
  dom.sectionCaption.textContent = manifest.sectionCaption || "Pick a demo and start moving around.";

  dom.heroHighlights.innerHTML = "";
  for (const label of highlights) {
    const pill = document.createElement("span");
    pill.className = "highlight-pill";
    pill.textContent = label;
    dom.heroHighlights.appendChild(pill);
  }
}

function renderLanding() {
  dom.landingView.classList.remove("hidden");
  dom.viewerView.classList.add("hidden");
  dom.backButton.classList.add("hidden");
  cleanupViewerSession();
  document.title = state.manifest?.title || "Motion Playground";

  dom.demoGrid.innerHTML = "";
  state.manifest.demos.forEach((demo, index) => {
    const card = document.createElement("a");
    card.className = "card";
    if (index === 0) {
      card.classList.add("featured");
    }
    card.href = demo.launchPath || `/?demo=${demo.id}`;
    card.innerHTML = `
      <div class="card-head">
        <span class="card-meta">${demo.category}</span>
        <span class="card-meta">${demo.assetLabel || "Local preview"}</span>
      </div>
      <div>
        <h2 class="card-title">${demo.title}</h2>
        <p class="card-copy">${demo.description}</p>
      </div>
      <div class="card-footer">
        <span class="card-pill">${demo.surfaceLabel || (demo.supportsMirror ? "Mirror ready" : "Open viewer")}</span>
        <span class="card-link">Open demo</span>
      </div>
    `;
    card.addEventListener("click", (event) => {
      event.preventDefault();
      if (demo.launchPath) {
        window.location.href = demo.launchPath;
        return;
      }
      setRouteDemoId(demo.id);
    });
    dom.demoGrid.appendChild(card);
  });
}

function renderViewer(demo) {
  dom.landingView.classList.add("hidden");
  dom.viewerView.classList.remove("hidden");
  dom.backButton.classList.remove("hidden");
  document.title = `${demo.title} • ${state.manifest?.title || "Motion Playground"}`;

  dom.demoCategory.textContent = demo.category;
  dom.demoTitle.textContent = demo.title;
  dom.demoDescription.textContent = demo.description;
  dom.demoMeta.innerHTML = `
    <div class="meta-item">
      <span class="meta-label">Asset</span>
      <span class="meta-value">${demo.assetLabel || "Local scene"}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Surface</span>
      <span class="meta-value">${demo.surfaceLabel || "Playback"}</span>
    </div>
  `;

  state.paused = false;
  state.speed = 1;
  state.mirror = false;
  state.showSkeleton = false;
  state.sliderValues = {};
  state.hasSnappedToFrame = false;

  dom.playbackButton.textContent = "Pause";
  dom.speedRange.value = "1";
  dom.speedValue.textContent = "1.00x";
  dom.mirrorToggle.checked = false;
  dom.mirrorToggle.disabled = !demo.supportsMirror;
  dom.skeletonToggle.checked = false;
  const mirrorLabel = dom.mirrorToggle.parentElement?.querySelector("span");
  if (mirrorLabel) {
    mirrorLabel.textContent = "Off";
  }
  const skeletonLabel = dom.skeletonToggle.parentElement?.querySelector("span");
  if (skeletonLabel) {
    skeletonLabel.textContent = "Off";
  }

  dom.demoControls.innerHTML = "";
  dom.demoControls.classList.toggle("hidden", !(demo.sliders && demo.sliders.length));

  if (demo.sliders && demo.sliders.length) {
    for (const slider of demo.sliders) {
      state.sliderValues[slider.id] = slider.default;
      const field = document.createElement("label");
      field.className = "slider-field";
      field.innerHTML = `
        <div class="slider-head">
          <span class="label">${slider.label}</span>
          <span class="value" data-slider-value="${slider.id}">${Number(slider.default).toFixed(2)}</span>
        </div>
        <input
          class="range"
          type="range"
          min="${slider.min}"
          max="${slider.max}"
          step="${slider.step}"
          value="${slider.default}"
          data-slider-id="${slider.id}"
        >
      `;
      dom.demoControls.appendChild(field);
    }
  }

  setPanelCollapsed(false);
}

function matrixFromFlat(flat, offset = 0) {
  const matrix = new THREE.Matrix4();
  matrix.set(
    flat[offset + 0], flat[offset + 1], flat[offset + 2], flat[offset + 3],
    flat[offset + 4], flat[offset + 5], flat[offset + 6], flat[offset + 7],
    flat[offset + 8], flat[offset + 9], flat[offset + 10], flat[offset + 11],
    flat[offset + 12], flat[offset + 13], flat[offset + 14], flat[offset + 15],
  );
  return matrix;
}

function parseFrame(message) {
  const entityMatrices = [];
  for (let index = 0; index < state.entityCount; index += 1) {
    entityMatrices.push(matrixFromFlat(message.entityMatrices, index * 16));
  }
  return {
    rootMatrix: matrixFromFlat(message.root),
    entityMatrices,
    markers: message.markers || [],
  };
}

const _prevPos = new THREE.Vector3();
const _prevQuat = new THREE.Quaternion();
const _prevScale = new THREE.Vector3();
const _nextPos = new THREE.Vector3();
const _nextQuat = new THREE.Quaternion();
const _nextScale = new THREE.Vector3();
const _lerpPos = new THREE.Vector3();
const _lerpQuat = new THREE.Quaternion();
const _lerpScale = new THREE.Vector3();
const _matrixTarget = new THREE.Vector3();

function interpolateTransform(a, b, alpha) {
  a.decompose(_prevPos, _prevQuat, _prevScale);
  b.decompose(_nextPos, _nextQuat, _nextScale);
  _lerpPos.lerpVectors(_prevPos, _nextPos, alpha);
  _lerpQuat.slerpQuaternions(_prevQuat, _nextQuat, alpha);
  _lerpScale.lerpVectors(_prevScale, _nextScale, alpha);
}

function applyFrame(alpha) {
  if (!state.frameCurr) {
    return;
  }

  if (!state.framePrev) {
    state.frameCurr.rootMatrix.decompose(_matrixTarget, _nextQuat, _nextScale);
    state.currentRootPos.copy(_matrixTarget);
    for (let index = 0; index < state.entityCount; index += 1) {
      const bone = state.boneMap[state.entityNames[index]];
      if (bone) {
        bone.matrixWorld.copy(state.frameCurr.entityMatrices[index]);
      }
    }
  } else {
    interpolateTransform(state.framePrev.rootMatrix, state.frameCurr.rootMatrix, alpha);
    state.currentRootPos.copy(_lerpPos);
    for (let index = 0; index < state.entityCount; index += 1) {
      const bone = state.boneMap[state.entityNames[index]];
      if (!bone) {
        continue;
      }
      interpolateTransform(
        state.framePrev.entityMatrices[index],
        state.frameCurr.entityMatrices[index],
        alpha,
      );
      bone.matrixWorld.compose(_lerpPos, _lerpQuat, _lerpScale);
    }
  }

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let index = 0; index < state.entityCount; index += 1) {
    const matrix = state.frameCurr.entityMatrices[index];
    const position = matrix.elements;
    const x = position[12];
    const y = position[13];
    const z = position[14];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      continue;
    }
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  if (Number.isFinite(minX)) {
    const center = new THREE.Vector3(
      (minX + maxX) * 0.5,
      (minY + maxY) * 0.5,
      (minZ + maxZ) * 0.5,
    );
    const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.6);
    state.liveBounds = { center, maxDim };
  }

  for (const skinnedMesh of state.skinnedMeshes) {
    skinnedMesh.skeleton.update();
  }

  if (!state.hasSnappedToFrame) {
    recenterCamera();
    state.hasSnappedToFrame = true;
  }
  updateMarkers();
  updateSkeleton();
}

function updateMarkers() {
  const markers = state.frameCurr ? state.frameCurr.markers : [];
  const liveIds = new Set();

  for (const marker of markers) {
    liveIds.add(marker.id);
    let mesh = state.markerMeshes.get(marker.id);
    if (!mesh) {
      const material = new THREE.MeshStandardMaterial({
        color: marker.tone === "accent" ? 0x0a69ff : 0xff7a59,
        emissive: marker.tone === "accent" ? 0x0a69ff : 0xff7a59,
        emissiveIntensity: 0.18,
      });
      mesh = new THREE.Mesh(new THREE.SphereGeometry(0.06, 24, 24), material);
      mesh.castShadow = true;
      scene.add(mesh);
      state.markerMeshes.set(marker.id, mesh);
    }
    mesh.visible = true;
    mesh.position.set(marker.position[0], marker.position[1], marker.position[2]);
  }

  for (const [id, mesh] of state.markerMeshes.entries()) {
    if (!liveIds.has(id)) {
      mesh.visible = false;
    }
  }
}

function updateSkeleton() {
  skeletonLines.visible = state.showSkeleton;
  if (!state.showSkeleton) {
    return;
  }

  const positions = skeletonGeometry.getAttribute("position");
  let pairCount = 0;
  for (const [childName, parentName] of state.skeletonPairs) {
    const child = state.boneMap[childName];
    const parent = state.boneMap[parentName];
    if (!child || !parent) {
      continue;
    }
    positions.setXYZ(
      pairCount * 2,
      child.matrixWorld.elements[12],
      child.matrixWorld.elements[13],
      child.matrixWorld.elements[14],
    );
    positions.setXYZ(
      pairCount * 2 + 1,
      parent.matrixWorld.elements[12],
      parent.matrixWorld.elements[13],
      parent.matrixWorld.elements[14],
    );
    pairCount += 1;
  }
  positions.needsUpdate = true;
  skeletonGeometry.setDrawRange(0, pairCount * 2);
}

function buildSkeletonPairs() {
  const entityNameSet = new Set(state.entityNames);
  state.skeletonPairs = [];
  for (const name of state.entityNames) {
    const bone = state.boneMap[name];
    if (!bone || !bone.parent || !bone.parent.isBone) {
      continue;
    }
    if (!entityNameSet.has(bone.parent.name)) {
      continue;
    }
    state.skeletonPairs.push([name, bone.parent.name]);
  }
}

async function loadModel(modelPath) {
  if (state.modelRoot) {
    scene.remove(state.modelRoot);
  }
  state.modelRoot = null;
  state.boneMap = {};
  state.skinnedMeshes = [];

  const gltf = await loader.loadAsync(modelPath);
  const root = gltf.scene;
  root.updateMatrixWorld(true);
  root.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
    if (child.isSkinnedMesh) {
      child.frustumCulled = false;
      state.skinnedMeshes.push(child);
    }
    if (child.isBone) {
      child.updateMatrix();
      child.matrixAutoUpdate = false;
      child.matrixWorldAutoUpdate = false;
      state.boneMap[child.name] = child;
    }
  });
  scene.add(root);
  state.modelRoot = root;
  buildSkeletonPairs();
  frameCameraToModel(root);
}

function cleanupViewerSession() {
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }
  if (state.wsInputTimer) {
    window.clearInterval(state.wsInputTimer);
    state.wsInputTimer = null;
  }
  state.framePrev = null;
  state.frameCurr = null;
  state.framePrevAt = 0;
  state.frameCurrAt = 0;
  state.entityNames = [];
  state.entityCount = 0;
  state.currentDemo = null;
  state.hasSnappedToFrame = false;
  state.liveBounds = null;
  state.boneMap = {};
  state.skinnedMeshes = [];
  state.skeletonPairs = [];
  if (state.modelRoot) {
    scene.remove(state.modelRoot);
    state.modelRoot = null;
  }
  skeletonLines.visible = false;
  skeletonGeometry.setDrawRange(0, 0);
  state.markerMeshes.forEach((mesh) => scene.remove(mesh));
  state.markerMeshes.clear();
}

function getControlPayload() {
  return {
    type: "input",
    paused: state.paused,
    speed: state.speed,
    mirror: state.mirror,
    sliders: state.sliderValues,
  };
}

function openSocket(demoId) {
  if (state.ws && (state.ws.readyState === WebSocket.CONNECTING || state.ws.readyState === WebSocket.OPEN)) return;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const params = new URLSearchParams({
    client_id: CLIENT_ID,
    started_at: `${CLIENT_STARTED_AT}`,
  });
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/${demoId}?${params}`);

  socket.addEventListener("open", () => {
    setConnectionState("live", "Live");
    socket.send(JSON.stringify(getControlPayload()));
    state.wsInputTimer = window.setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(getControlPayload()));
      }
    }, 120);
  });

  socket.addEventListener("message", async (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "busy") {
        setConnectionState("busy", "Busy");
        showToast(message.message);
        return;
      }
      if (message.type === "replaced") {
        setConnectionState("busy", "Moved");
        showToast(message.message);
        return;
      }
      if (message.type === "error") {
        setConnectionState("error", "Error");
        showToast(message.message);
        return;
      }
      if (message.type === "init") {
        state.entityNames = message.entityNames || [];
        state.entityCount = message.entityCount || state.entityNames.length;
        buildSkeletonPairs();
        return;
      }
      if (message.type === "frame") {
        state.framePrev = state.frameCurr;
        state.framePrevAt = state.frameCurrAt;
        state.frameCurr = parseFrame(message);
        state.frameCurrAt = performance.now();
      }
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      setConnectionState("error", "Client error");
      showToast(`Frame error: ${text}`);
    }
  });

  socket.addEventListener("close", (event) => {
    if (state.wsInputTimer) {
      window.clearInterval(state.wsInputTimer);
      state.wsInputTimer = null;
    }
    if (event.code === SESSION_REPLACED_CLOSE_CODE) {
      setConnectionState("busy", "Moved");
      showToast(SESSION_REPLACED_MESSAGE);
      return;
    }
    if (state.currentDemo) {
      const suffix = event.code ? ` ${event.code}` : "";
      setConnectionState("error", `Disconnected${suffix}`);
      if (event.code && event.code !== 1000) {
        showToast(`Socket closed with code ${event.code}`);
      }
    }
  });

  state.ws = socket;
}

async function mountDemo(demo) {
  cleanupViewerSession();
  renderViewer(demo);
  state.currentDemo = demo;
  setConnectionState("busy", "Loading");
  await loadModel(demo.modelPath);
  openSocket(demo.id);
}

async function renderRoute() {
  if (!state.manifest) {
    return;
  }
  const demoId = getRouteDemoId();
  if (!demoId) {
    renderLanding();
    return;
  }

  const demo = state.manifest.demos.find((item) => item.id === demoId);
  if (!demo) {
    showToast("That demo is not available.");
    renderLanding();
    return;
  }

  await mountDemo(demo);
}

function bindUi() {
  window.addEventListener("error", (event) => {
    const text = event.error?.message || event.message || "Unknown client error";
    setConnectionState("error", "Client error");
    showToast(text);
  });

  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason);
    setConnectionState("error", "Client error");
    showToast(reason);
  });

  dom.backButton.addEventListener("click", () => {
    setRouteDemoId(null);
  });

  dom.themeButton.addEventListener("click", cycleTheme);
  dom.viewport.addEventListener("dblclick", recenterCamera);
  dom.recenterButton?.addEventListener("click", recenterCamera);
  dom.zoomOutButton?.addEventListener("click", () => zoomBy(1.18));
  dom.zoomInButton?.addEventListener("click", () => zoomBy(0.84));
  dom.panelButton?.addEventListener("click", () => {
    setPanelCollapsed(!state.panelCollapsed);
  });

  dom.playbackButton.addEventListener("click", () => {
    state.paused = !state.paused;
    dom.playbackButton.textContent = state.paused ? "Resume" : "Pause";
    setConnectionState(state.paused ? "busy" : "live", state.paused ? "Paused" : "Live");
  });

  dom.speedRange.addEventListener("input", () => {
    state.speed = Number(dom.speedRange.value);
    dom.speedValue.textContent = `${state.speed.toFixed(2)}x`;
  });

  dom.mirrorToggle.addEventListener("change", () => {
    state.mirror = dom.mirrorToggle.checked;
    const label = dom.mirrorToggle.parentElement?.querySelector("span");
    if (label) {
      label.textContent = state.mirror ? "On" : "Off";
    }
  });

  dom.skeletonToggle.addEventListener("change", () => {
    state.showSkeleton = dom.skeletonToggle.checked;
    const label = dom.skeletonToggle.parentElement?.querySelector("span");
    if (label) {
      label.textContent = state.showSkeleton ? "On" : "Off";
    }
  });

  dom.demoControls.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
      return;
    }
    const sliderId = target.dataset.sliderId;
    if (!sliderId) {
      return;
    }
    state.sliderValues[sliderId] = Number(target.value);
    const valueNode = dom.demoControls.querySelector(`[data-slider-value="${sliderId}"]`);
    if (valueNode) {
      valueNode.textContent = Number(target.value).toFixed(2);
    }
  });

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) {
      return;
    }
    switch (event.key) {
      case "+":
      case "=":
        zoomBy(0.84);
        break;
      case "-":
      case "_":
        zoomBy(1.18);
        break;
      case "ArrowLeft":
        orbitBy(-0.16, 0);
        break;
      case "ArrowRight":
        orbitBy(0.16, 0);
        break;
      case "ArrowUp":
        orbitBy(0, -0.12);
        break;
      case "ArrowDown":
        orbitBy(0, 0.12);
        break;
      case "0":
        recenterCamera();
        break;
      default:
        return;
    }
    event.preventDefault();
  });

  window.addEventListener("resize", resize);
  window.addEventListener("popstate", renderRoute);
}

function animate(now) {
  requestAnimationFrame(animate);
  controls.update();

  if (state.frameCurr) {
    let alpha = 1;
    if (state.framePrev) {
      const span = Math.max(state.frameCurrAt - state.framePrevAt, FRAME_DELAY_MS * 0.5);
      alpha = THREE.MathUtils.clamp(
        (now - FRAME_DELAY_MS - state.framePrevAt) / span,
        0,
        1,
      );
    }
    applyFrame(alpha);
  }

  renderer.render(scene, camera);
}

async function main() {
  applyTheme(loadTheme());
  resize();
  bindUi();
  requestAnimationFrame(animate);

  const response = await fetch("/api/manifest");
  state.manifest = await response.json();
  renderManifestChrome();
  await renderRoute();
}

main().catch((error) => {
  console.error(error);
  showToast("Unable to start the local playground.");
});
