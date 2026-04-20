import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const DEMO = {
    title: "Neural Biped Locomotion",
    modelPath: "/assets/geno/Model.glb",
    wsPath: "/ws-interactive/biped",
    boneNames: [
        "Hips",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        "Spine", "Spine1", "Spine2", "Spine3",
        "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    ],
    bonePairs: [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [12, 13], [13, 14],
        [12, 15], [15, 16], [16, 17], [17, 18],
        [12, 19], [19, 20], [20, 21], [21, 22],
    ],
    contactBoneIndices: [3, 4, 7, 8],
};

const BONE_NAMES = DEMO.boneNames;
const TRAJ_SAMPLES = 16;
const CONTACT_BONE_INDICES = DEMO.contactBoneIndices;
const BONE_PAIRS = DEMO.bonePairs;
const HEARTBEAT_INTERVAL = 10000;
const SERVER_FRAME_MS = 1000 / 30;
const INTERPOLATION_DELAY_MS = SERVER_FRAME_MS;
const INPUT_SEND_INTERVAL_MS = SERVER_FRAME_MS;

let ws = null;
let styleNames = [];
let currentStyleIndex = 0;
let debugEnabled = true;
let meshVisible = true;
let entityNames = [];
let entityCount = 0;
let entityNameToIndex = new Map();
let skeletonPairEntityIndices = [];
let contactEntityIndices = [];
let boneMap = {};
let skinnedMesh = null;
let framePrev = null;
let frameCurr = null;
let framePrevReceivedAt = 0;
let frameCurrReceivedAt = 0;
let sessionState = "connecting";
let sessionRemainingSeconds = 120;
let heartbeatTimer = null;
let timerUpdateInterval = null;
let lastInputSendAt = 0;
let renderFrameCount = 0;
let fpsLastTime = performance.now();
let serverFrameCount = 0;
let serverFpsLastTime = performance.now();
let serverFpsValue = 0;
let avgInferenceMs = null;
const keys = {};
let rightMouseDown = false;
let directionMouseStart = null;
let rightStick = [0, 0];
const DIRECTION_MOMENTUM = 0.01;
let cameraDistance = 5.0;
let cameraPhi = Math.PI / 16;
let cameraTheta = 0;
const CAM_SELF_HEIGHT = 0.9;
const CAM_TARGET_HEIGHT = 0.8;
const CAM_SMOOTHING = 10.0;
let cameraTarget = new THREE.Vector3(0, CAM_TARGET_HEIGHT, 0);
let cameraPos = new THREE.Vector3(0, CAM_SELF_HEIGHT, cameraDistance);
let currentRootPos = new THREE.Vector3();
let lastFrameTime = performance.now();

const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scl = new THREE.Vector3();
const _prevPos = new THREE.Vector3();
const _prevQuat = new THREE.Quaternion();
const _prevScl = new THREE.Vector3();
const _currPos = new THREE.Vector3();
const _currQuat = new THREE.Quaternion();
const _currScl = new THREE.Vector3();
const _interpPos = new THREE.Vector3();
const _interpQuat = new THREE.Quaternion();
const _interpScl = new THREE.Vector3();
const _interpVec = new THREE.Vector3();
const _interpDir = new THREE.Vector3();
const _tmpDir = new THREE.Vector3();

const titleNode = document.getElementById("demo-title");
if (titleNode) titleNode.textContent = DEMO.title;
document.title = DEMO.title;

function rebuildEntityLookup() {
    entityNameToIndex = new Map(entityNames.map((name, idx) => [name, idx]));
    skeletonPairEntityIndices = BONE_PAIRS.map(([a, b]) => [
        entityNameToIndex.get(BONE_NAMES[a]) ?? -1,
        entityNameToIndex.get(BONE_NAMES[b]) ?? -1,
    ]);
    contactEntityIndices = CONTACT_BONE_INDICES.map((idx) => entityNameToIndex.get(BONE_NAMES[idx]) ?? -1);
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x2a2a3e);
scene.fog = new THREE.Fog(0x2a2a3e, 15, 40);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 2, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.getElementById("viewport").appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0x8899bb, 0.8));
const dirLight = new THREE.DirectionalLight(0xffeedd, 2.0);
dirLight.position.set(5, 10, 5);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.near = 0.5;
dirLight.shadow.camera.far = 30;
dirLight.shadow.camera.left = -10;
dirLight.shadow.camera.right = 10;
dirLight.shadow.camera.top = 10;
dirLight.shadow.camera.bottom = -10;
dirLight.shadow.bias = -0.001;
scene.add(dirLight);
scene.add(new THREE.HemisphereLight(0x88aacc, 0x443322, 0.5));

const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.MeshStandardMaterial({ color: 0x3a3a50, roughness: 0.9, metalness: 0.1 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const grid = new THREE.GridHelper(60, 60, 0x555577, 0x444466);
grid.position.y = 0.005;
scene.add(grid);

const debugGroup = new THREE.Group();
debugGroup.renderOrder = 999;
scene.add(debugGroup);

const AXIS_LEN = 0.5;
const axisX = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), AXIS_LEN, 0xff3333, 0.12, 0.06);
const axisY = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), AXIS_LEN, 0x33ff33, 0.12, 0.06);
const axisZ = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), AXIS_LEN, 0x3333ff, 0.12, 0.06);
debugGroup.add(axisX, axisY, axisZ);

const ctrlTrajSpheres = [];
const ctrlTrajArrows = [];
const ctrlTrajGroup = new THREE.Group();
for (let i = 0; i < TRAJ_SAMPLES; i++) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 8), new THREE.MeshBasicMaterial({ color: 0x00cccc }));
    ctrlTrajSpheres.push(s);
    ctrlTrajGroup.add(s);
    const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.25, 0xff8800, 0.08, 0.04);
    ctrlTrajArrows.push(a);
    ctrlTrajGroup.add(a);
}
const ctrlTrajLineGeo = new THREE.BufferGeometry();
ctrlTrajLineGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAJ_SAMPLES * 3), 3));
const ctrlTrajLine = new THREE.Line(ctrlTrajLineGeo, new THREE.LineBasicMaterial({ color: 0x00cccc, depthTest: false }));
ctrlTrajLine.frustumCulled = false;
ctrlTrajGroup.add(ctrlTrajLine);
debugGroup.add(ctrlTrajGroup);

const skelGeo = new THREE.BufferGeometry();
skelGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BONE_PAIRS.length * 2 * 3), 3));
const skelLine = new THREE.LineSegments(skelGeo, new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
skelLine.frustumCulled = false;
debugGroup.add(skelLine);

const contactSpheres = [];
for (let i = 0; i < 4; i++) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8), new THREE.MeshBasicMaterial({ color: 0x00ff00, depthTest: false }));
    s.renderOrder = 999;
    contactSpheres.push(s);
    debugGroup.add(s);
}
let jointSpheres = [];

new GLTFLoader().load(
    DEMO.modelPath,
    (gltf) => {
        const model = gltf.scene;
        model.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
                if (child.isSkinnedMesh) {
                    skinnedMesh = child;
                    child.frustumCulled = false;
                }
            }
            if (child.isBone) {
                boneMap[child.name] = child;
                child.matrixAutoUpdate = false;
                child.matrixWorldAutoUpdate = false;
            }
        });
        scene.add(model);
        connectWebSocket();
    },
    undefined,
    (err) => console.error("Failed to load model:", err)
);

function parseFrame(buffer) {
    const floats = new Float32Array(buffer);
    let offset = 0;
    const rootElements = floats.slice(offset, offset + 16);
    offset += 16;
    const rootMatrix = new THREE.Matrix4();
    rootMatrix.set(
        rootElements[0], rootElements[1], rootElements[2], rootElements[3],
        rootElements[4], rootElements[5], rootElements[6], rootElements[7],
        rootElements[8], rootElements[9], rootElements[10], rootElements[11],
        rootElements[12], rootElements[13], rootElements[14], rootElements[15]
    );

    const entityMatrices = [];
    for (let i = 0; i < entityCount; i++) {
        const e = floats.slice(offset, offset + 16);
        offset += 16;
        const m = new THREE.Matrix4();
        m.set(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15]);
        entityMatrices.push(m);
    }
    const contacts = [floats[offset], floats[offset + 1], floats[offset + 2], floats[offset + 3]];
    offset += 4;
    const simTrajectory = [];
    for (let i = 0; i < TRAJ_SAMPLES; i++) {
        simTrajectory.push(new THREE.Vector3(floats[offset], floats[offset + 1], floats[offset + 2]));
        offset += 3;
    }
    const simTrajectoryDir = [];
    for (let i = 0; i < TRAJ_SAMPLES; i++) {
        simTrajectoryDir.push(new THREE.Vector3(floats[offset], floats[offset + 1], floats[offset + 2]));
        offset += 3;
    }
    const ctrlTrajectory = [];
    for (let i = 0; i < TRAJ_SAMPLES; i++) {
        ctrlTrajectory.push(new THREE.Vector3(floats[offset], floats[offset + 1], floats[offset + 2]));
        offset += 3;
    }
    const ctrlTrajectoryDir = [];
    for (let i = 0; i < TRAJ_SAMPLES; i++) {
        ctrlTrajectoryDir.push(new THREE.Vector3(floats[offset], floats[offset + 1], floats[offset + 2]));
        offset += 3;
    }
    const speed = offset < floats.length ? floats[offset] : 0.0;
    return { rootMatrix, entityMatrices, contacts, simTrajectory, simTrajectoryDir, ctrlTrajectory, ctrlTrajectoryDir, speed };
}

function interpolateTransform(a, b, alpha, outPos, outQuat, outScale) {
    a.decompose(_prevPos, _prevQuat, _prevScl);
    b.decompose(_currPos, _currQuat, _currScl);
    outPos.lerpVectors(_prevPos, _currPos, alpha);
    outQuat.slerpQuaternions(_prevQuat, _currQuat, alpha);
    outScale.lerpVectors(_prevScl, _currScl, alpha);
}

function applyFrame(alpha) {
    if (!frameCurr) return;
    if (!framePrev) {
        frameCurr.rootMatrix.decompose(_pos, _quat, _scl);
        currentRootPos.copy(_pos);
        for (let i = 0; i < entityCount; i++) {
            const bone = boneMap[entityNames[i]];
            if (bone) bone.matrixWorld.copy(frameCurr.entityMatrices[i]);
        }
        return;
    }
    interpolateTransform(framePrev.rootMatrix, frameCurr.rootMatrix, alpha, _pos, _quat, _scl);
    currentRootPos.copy(_pos);
    for (let i = 0; i < entityCount; i++) {
        const bone = boneMap[entityNames[i]];
        if (!bone) continue;
        interpolateTransform(framePrev.entityMatrices[i], frameCurr.entityMatrices[i], alpha, _interpPos, _interpQuat, _interpScl);
        bone.matrixWorld.compose(_interpPos, _interpQuat, _interpScl);
    }
}

function getInterpolationAlpha(now) {
    if (!frameCurr || !framePrev) return 1;
    const span = Math.max(frameCurrReceivedAt - framePrevReceivedAt, SERVER_FRAME_MS * 0.5);
    const renderTime = now - INTERPOLATION_DELAY_MS;
    return THREE.MathUtils.clamp((renderTime - framePrevReceivedAt) / span, 0, 1);
}

function interpolateVector(prevVec, currVec, alpha, outVec) {
    if (!framePrev) {
        outVec.copy(currVec);
        return outVec;
    }
    outVec.lerpVectors(prevVec, currVec, alpha);
    return outVec;
}

function updateDebugLabel() {
    const debugLabel = document.getElementById("debug-label");
    if (debugLabel) debugLabel.textContent = "Debug G";
    const mobile = document.getElementById("debug-btn-mobile");
    if (mobile) {
        mobile.textContent = "Debug";
        mobile.classList.toggle("active", debugEnabled);
    }
}

function setDebugEnabled(value) {
    debugEnabled = value;
    debugGroup.visible = debugEnabled;
    updateDebugLabel();
}

function setInputKeyActive(id, active) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle("active", !!active);
}

function updateControlsHelp(device) {
    const keyboardControls = document.getElementById("keyboard-controls");
    const gamepadControls = document.getElementById("gamepad-controls");
    if (!keyboardControls || !gamepadControls) return;
    const showGamepad = device === "gamepad";
    keyboardControls.classList.toggle("hidden", showGamepad);
    gamepadControls.classList.toggle("hidden", !showGamepad);
}

function updateInputVisualizerKeyboard() {
    const deviceLabel = document.getElementById("input-device-label");
    if (deviceLabel) deviceLabel.textContent = "Input: Keyboard";
    const kbPanel = document.getElementById("keyboard-front-buttons");
    const gpPanel = document.getElementById("gamepad-front-buttons");
    const leftStickCol = document.getElementById("left-stick-column");
    if (kbPanel) kbPanel.classList.remove("hidden");
    if (gpPanel) gpPanel.classList.add("hidden");
    if (leftStickCol) leftStickCol.classList.add("hidden");

    setInputKeyActive("key-w", !!keys["KeyW"]);
    setInputKeyActive("key-a", !!keys["KeyA"]);
    setInputKeyActive("key-s", !!keys["KeyS"]);
    setInputKeyActive("key-d", !!keys["KeyD"]);
    setInputKeyActive("key-q", !!keys["KeyQ"]);
    setInputKeyActive("key-e", !!keys["KeyE"]);
    let fx = 0;
    let fy = 0;
    const rm = Math.hypot(rightStick[0], rightStick[1]);
    if (rm > 1e-6) {
        fx = rightStick[0] / rm;
        fy = rightStick[1] / rm;
    }
    updateFacingStickVisualizer(fx, fy);
    updateControlsHelp("keyboard");
}

const STICK_VIZ_RADIUS = 24;

function updateStickCircleHud(dotId, pointerId, x, y) {
    const dot = document.getElementById(dotId);
    const pointer = document.getElementById(pointerId);
    if (!dot) return;
    const cx = THREE.MathUtils.clamp(x, -1, 1) * STICK_VIZ_RADIUS;
    const cy = -THREE.MathUtils.clamp(y, -1, 1) * STICK_VIZ_RADIUS;
    dot.style.transform = `translate(calc(-50% + ${cx.toFixed(1)}px), calc(-50% + ${cy.toFixed(1)}px))`;
    if (pointer) {
        const idle = Math.abs(cx) < 0.5 && Math.abs(cy) < 0.5;
        pointer.style.opacity = idle ? "0.25" : "1";
        const deg = idle ? -90 : (Math.atan2(cy, cx) * 180) / Math.PI;
        pointer.style.transform = `translate(0, -50%) rotate(${deg.toFixed(1)}deg)`;
    }
}

function updateFacingStickVisualizer(x, y) {
    updateStickCircleHud("right-stick-dot", "right-stick-pointer", x, y);
}

function updateInputVisualizerGamepad(gp) {
    const deviceLabel = document.getElementById("input-device-label");
    if (deviceLabel) deviceLabel.textContent = "Input: Gamepad";
    const kbPanel = document.getElementById("keyboard-front-buttons");
    const gpPanel = document.getElementById("gamepad-front-buttons");
    if (kbPanel) kbPanel.classList.add("hidden");
    if (gpPanel) gpPanel.classList.remove("hidden");
    const leftStickCol = document.getElementById("left-stick-column");
    if (leftStickCol) leftStickCol.classList.remove("hidden");

    setInputKeyActive("pad-lb", !!(gp.buttons[4] && gp.buttons[4].pressed));
    setInputKeyActive("pad-rb", !!(gp.buttons[5] && gp.buttons[5].pressed));
    const [lvx, lvy] = applyDeadzone(gp.axes[0] || 0, -(gp.axes[1] || 0));
    updateStickCircleHud("left-stick-dot", "left-stick-pointer", lvx, lvy);
    const [frx, fry] = applyDeadzone(gp.axes[2] || 0, -(gp.axes[3] || 0));
    updateFacingStickVisualizer(frx, fry);
    updateControlsHelp("gamepad");
}

function updateCamera() {
    const now = performance.now();
    const dt = Math.min((now - lastFrameTime) / 1000, 0.1);
    lastFrameTime = now;
    const blend = 1.0 - Math.exp(-CAM_SMOOTHING * dt);
    const desiredTarget = new THREE.Vector3(currentRootPos.x, currentRootPos.y + CAM_TARGET_HEIGHT, currentRootPos.z);
    const desiredPos = new THREE.Vector3(
        currentRootPos.x + cameraDistance * Math.cos(cameraPhi) * Math.sin(cameraTheta),
        currentRootPos.y + CAM_SELF_HEIGHT + cameraDistance * Math.sin(cameraPhi),
        currentRootPos.z + cameraDistance * Math.cos(cameraPhi) * Math.cos(cameraTheta)
    );
    cameraTarget.lerp(desiredTarget, blend);
    cameraPos.lerp(desiredPos, blend);
    camera.position.copy(cameraPos);
    camera.lookAt(cameraTarget);
    dirLight.position.set(currentRootPos.x + 5, 12, currentRootPos.z + 5);
    dirLight.target.position.copy(currentRootPos);
    dirLight.target.updateMatrixWorld();
}

function setSessionState(state) {
    sessionState = state;
    const timeoutOverlay = document.getElementById("timeout-overlay");
    const sessionTimer = document.getElementById("session-timer");
    if (timeoutOverlay) timeoutOverlay.classList.toggle("hidden", state !== "timeout");
    if (sessionTimer) sessionTimer.classList.toggle("hidden", state !== "active");
    if (timerUpdateInterval) {
        clearInterval(timerUpdateInterval);
        timerUpdateInterval = null;
    }
    if (state === "active") startTimerCountdown();
    if (state === "timeout") {
        stopHeartbeat();
        if (ws) ws.close();
    }
    if (state === "busy") stopHeartbeat();
    if (state === "disconnected") stopHeartbeat();
}

function formatTime(s) {
    return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, "0")}`;
}

function updateTimerDisplay(r) {
    sessionRemainingSeconds = r;
    const timerVal = document.getElementById("timer-value");
    if (timerVal) timerVal.textContent = formatTime(r);
    const c = document.getElementById("session-timer");
    if (!c) return;
    c.classList.remove("warning", "critical");
    if (r <= 30) c.classList.add("critical");
    else if (r <= 60) c.classList.add("warning");
}

function startTimerCountdown() {
    timerUpdateInterval = setInterval(() => {
        sessionRemainingSeconds = Math.max(0, sessionRemainingSeconds - 1);
        updateTimerDisplay(sessionRemainingSeconds);
    }, 1000);
}

function startHeartbeat() {
    stopHeartbeat();
    heartbeatTimer = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "heartbeat" }));
    }, HEARTBEAT_INTERVAL);
}

function stopHeartbeat() {
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

function showBusyOverlay(message) {
    const overlay = document.getElementById("busy-overlay");
    if (overlay) {
        if (message) {
            const p = overlay.querySelector("#busy-box p");
            if (p) p.textContent = message;
        }
        overlay.classList.remove("hidden");
    }
}

function hideBusyOverlay() {
    const overlay = document.getElementById("busy-overlay");
    if (overlay) overlay.classList.add("hidden");
}

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const sid = new URLSearchParams(window.location.search).get("sid") || "";
    ws = new WebSocket(`${protocol}//${window.location.host}${DEMO.wsPath}?sid=${encodeURIComponent(sid)}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
        const dot = document.getElementById("connection-dot");
        const text = document.getElementById("connection-text");
        if (dot) dot.classList.add("connected");
        if (text) text.textContent = "Connected";
        lastInputSendAt = 0;
        startHeartbeat();
    };
    ws.onclose = () => {
        const dot = document.getElementById("connection-dot");
        const text = document.getElementById("connection-text");
        if (dot) dot.classList.remove("connected");
        if (text) text.textContent = "Disconnected";
        stopHeartbeat();
        if (sessionState !== "timeout" && sessionState !== "busy") {
            setSessionState("disconnected");
            setTimeout(connectWebSocket, 3000);
        }
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => {
        if (typeof event.data === "string") {
            const msg = JSON.parse(event.data);
            switch (msg.type) {
                case "init":
                    hideBusyOverlay();
                    styleNames = msg.styles || [];
                    entityNames = msg.entityNames;
                    entityCount = msg.entityCount;
                    rebuildEntityLookup();
                    buildStyleSwitcher();
                    updateStyleDisplay();
                    createJointSpheres();
                    setSessionState("active");
                    if (msg.avgInferenceMs != null) avgInferenceMs = msg.avgInferenceMs;
                    if (msg.remainingSeconds !== undefined) updateTimerDisplay(msg.remainingSeconds);
                    { const el = document.getElementById("perf-display"); if (el) el.classList.remove("hidden"); }
                    break;
                case "time_update":
                    if (msg.remainingSeconds !== undefined) updateTimerDisplay(msg.remainingSeconds);
                    break;
                case "perf_update":
                    if (msg.avgInferenceMs != null) avgInferenceMs = msg.avgInferenceMs;
                    break;
                case "timeout":
                    setSessionState("timeout");
                    break;
                case "busy":
                    setSessionState("busy");
                    showBusyOverlay(msg.message);
                    break;
                case "error":
                    break;
            }
        } else {
            const receivedAt = performance.now();
            framePrev = frameCurr;
            framePrevReceivedAt = frameCurrReceivedAt;
            frameCurr = parseFrame(event.data);
            frameCurrReceivedAt = receivedAt;
            serverFrameCount++;
            if (!framePrev) {
                framePrev = frameCurr;
                framePrevReceivedAt = receivedAt - SERVER_FRAME_MS;
            }
        }
    };
}

function sendInput(timestamp) {
    if (!ws || ws.readyState !== WebSocket.OPEN || sessionState !== "active") return;
    if (timestamp - lastInputSendAt < INPUT_SEND_INTERVAL_MS) return;
    const input = getInput();
    input.guidance_index = currentStyleIndex;
    ws.send(JSON.stringify(input));
    lastInputSendAt = timestamp;
}

function createJointSpheres() {
    for (const s of jointSpheres) debugGroup.remove(s);
    jointSpheres = [];
    for (let i = 0; i < entityCount; i++) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(0.015, 6, 6), new THREE.MeshBasicMaterial({ color: 0xffaa00, depthTest: false }));
        s.renderOrder = 999;
        jointSpheres.push(s);
        debugGroup.add(s);
    }
}

function buildStyleSwitcher() {
    const container = document.getElementById("style-switcher");
    if (container) {
        container.innerHTML = "";
        styleNames.forEach((name, idx) => {
            const btn = document.createElement("button");
            btn.className = "style-btn" + (idx === currentStyleIndex ? " active" : "");
            btn.textContent = name;
            btn.addEventListener("click", () => {
                currentStyleIndex = idx;
                updateStyleDisplay();
            });
            container.appendChild(btn);
        });
    }
    const dropdown = document.getElementById("style-dropdown");
    if (dropdown) {
        dropdown.innerHTML = "";
        styleNames.forEach((name, idx) => {
            const option = document.createElement("option");
            option.value = idx;
            option.textContent = name;
            if (idx === currentStyleIndex) option.selected = true;
            dropdown.appendChild(option);
        });
    }
}

function updateStyleDisplay() {
    const styleDisplay = document.getElementById("style-display");
    if (styleDisplay) styleDisplay.textContent = `Style: ${styleNames[currentStyleIndex] || "—"}`;
    document.querySelectorAll(".style-btn").forEach((btn, idx) => btn.classList.toggle("active", idx === currentStyleIndex));
    const dropdown = document.getElementById("style-dropdown");
    if (dropdown) dropdown.value = currentStyleIndex;
}

const DEADZONE = 0.25;
let gamepadIndex = -1;
let prevL1 = false;
let prevR1 = false;

function ensureGamepadIndex() {
    const pads = navigator.getGamepads();
    if (gamepadIndex >= 0 && pads[gamepadIndex]) return;
    gamepadIndex = -1;
    for (let i = 0; i < pads.length; i++) {
        if (pads[i]) {
            gamepadIndex = i;
            break;
        }
    }
}

function refreshInputDeviceHud() {
    ensureGamepadIndex();
    const gp = gamepadIndex >= 0 ? navigator.getGamepads()[gamepadIndex] : null;
    if (gp) updateInputVisualizerGamepad(gp);
    else updateInputVisualizerKeyboard();
}

window.addEventListener("gamepadconnected", (e) => {
    gamepadIndex = e.gamepad.index;
});
window.addEventListener("gamepaddisconnected", (e) => {
    if (e.gamepad.index === gamepadIndex) {
        gamepadIndex = -1;
        refreshInputDeviceHud();
    }
});

function getActiveGamepad() {
    ensureGamepadIndex();
    if (gamepadIndex < 0) return null;
    return navigator.getGamepads()[gamepadIndex] || null;
}

function applyDeadzone(x, y) {
    const mag = Math.sqrt(x * x + y * y);
    if (mag < DEADZONE) return [0, 0];
    const scale = (mag - DEADZONE) / (1.0 - DEADZONE) / mag;
    return [x * scale, y * scale];
}

function getGamepadInput(gp) {
    updateInputVisualizerGamepad(gp);
    const [lx, ly] = applyDeadzone(gp.axes[0], -gp.axes[1]);
    const [rx, ry] = applyDeadzone(gp.axes[2] || 0, -(gp.axes[3] || 0));
    const c = Math.cos(cameraTheta);
    const s = Math.sin(cameraTheta);
    const left_stick = [lx * c - ly * s, lx * s + ly * c];
    const right_stick_local = [rx, ry];
    const l1 = !!(gp.buttons[4] && gp.buttons[4].pressed);
    const r1 = !!(gp.buttons[5] && gp.buttons[5].pressed);
    if (l1 && !prevL1 && styleNames.length > 0) {
        currentStyleIndex = (currentStyleIndex - 1 + styleNames.length) % styleNames.length;
        updateStyleDisplay();
    }
    if (r1 && !prevR1 && styleNames.length > 0) {
        currentStyleIndex = (currentStyleIndex + 1) % styleNames.length;
        updateStyleDisplay();
    }
    prevL1 = l1;
    prevR1 = r1;
    return { left_stick, right_stick: right_stick_local, speed_toggle: false };
}

document.addEventListener("keydown", (e) => {
    keys[e.code] = true;
    if (e.code === "KeyG") setDebugEnabled(!debugEnabled);
    if (e.code === "KeyM") {
        meshVisible = !meshVisible;
        if (skinnedMesh) skinnedMesh.visible = meshVisible;
    }
    if (e.code === "KeyQ" && styleNames.length > 0) {
        currentStyleIndex = (currentStyleIndex - 1 + styleNames.length) % styleNames.length;
        updateStyleDisplay();
    }
    if (e.code === "KeyE" && styleNames.length > 0) {
        currentStyleIndex = (currentStyleIndex + 1) % styleNames.length;
        updateStyleDisplay();
    }
});
document.addEventListener("keyup", (e) => {
    keys[e.code] = false;
});

function clearKeyboardState() {
    for (const code of Object.keys(keys)) {
        keys[code] = false;
    }
    updateInputVisualizerKeyboard();
}

window.addEventListener("blur", clearKeyboardState);
window.addEventListener("focus", clearKeyboardState);
document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearKeyboardState();
});

const canvas = renderer.domElement;
canvas.addEventListener("mousedown", (e) => {
    if (e.button === 2) {
        rightMouseDown = true;
        directionMouseStart = null;
        e.preventDefault();
    }
});
canvas.addEventListener("mouseup", (e) => {
    if (e.button === 2) {
        rightMouseDown = false;
        directionMouseStart = null;
        rightStick = [0, 0];
    }
});
canvas.addEventListener("mousemove", (e) => {
    if (!rightMouseDown) return;
    const x = e.clientX / window.innerWidth;
    const y = e.clientY / window.innerHeight;
    const pos = [x, y];
    if (!directionMouseStart) directionMouseStart = [...pos];
    else {
        directionMouseStart[0] += (pos[0] - directionMouseStart[0]) * DIRECTION_MOMENTUM;
        directionMouseStart[1] += (pos[1] - directionMouseStart[1]) * DIRECTION_MOMENTUM;
    }
    rightStick = [pos[0] - directionMouseStart[0], directionMouseStart[1] - pos[1]];
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (e) => {
    cameraDistance = Math.max(1.5, Math.min(15, cameraDistance + e.deltaY * 0.005));
    e.preventDefault();
}, { passive: false });

let touchLeftId = null;
let touchRightId = null;
let touchLeftStick = [0, 0];
let touchRightStick = [0, 0];

function getKeyboardInput() {
    updateInputVisualizerKeyboard();
    let lx = 0;
    let ly = 0;
    if (keys["KeyW"] || keys["ArrowUp"]) ly = 1;
    if (keys["KeyS"] || keys["ArrowDown"]) ly = -1;
    if (keys["KeyA"] || keys["ArrowLeft"]) lx = -1;
    if (keys["KeyD"] || keys["ArrowRight"]) lx = 1;
    const c = Math.cos(cameraTheta);
    const s = Math.sin(cameraTheta);
    return {
        left_stick: [lx * c + ly * s, -lx * s + ly * c],
        right_stick: rightStick,
        speed_toggle: false,
    };
}

function getTouchInput() {
    updateInputVisualizerKeyboard();
    updateFacingStickVisualizer(touchRightStick[0], touchRightStick[1]);
    const c = Math.cos(cameraTheta);
    const s = Math.sin(cameraTheta);
    const lx = touchLeftStick[0];
    const ly = touchLeftStick[1];
    return {
        left_stick: [lx * c - ly * s, lx * s + ly * c],
        right_stick: [touchRightStick[0], touchRightStick[1]],
        speed_toggle: false,
    };
}

function getInput() {
    const gp = getActiveGamepad();
    if (gp) return getGamepadInput(gp);
    const hasTouch = touchLeftStick[0] !== 0 || touchLeftStick[1] !== 0 || touchRightStick[0] !== 0 || touchRightStick[1] !== 0;
    return hasTouch ? getTouchInput() : getKeyboardInput();
}

function setupTouchJoystick(joystickEl, knobEl, onMove, onEnd) {
    if (!joystickEl || !knobEl) return;
    const radius = joystickEl.offsetWidth / 2;
    const knobRadius = knobEl.offsetWidth / 2;
    const maxDist = radius - knobRadius;
    function getCenter() {
        const rect = joystickEl.getBoundingClientRect();
        return { x: rect.left + radius, y: rect.top + radius };
    }
    function handleMove(cx, cy, center) {
        let dx = cx - center.x;
        let dy = cy - center.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > maxDist) {
            dx = (dx / dist) * maxDist;
            dy = (dy / dist) * maxDist;
        }
        knobEl.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
        knobEl.classList.add("active");
        onMove(dx / maxDist, -dy / maxDist);
    }
    function handleEnd() {
        knobEl.style.transform = "translate(-50%, -50%)";
        knobEl.classList.remove("active");
        onEnd();
    }
    joystickEl.addEventListener("touchstart", (e) => {
        e.preventDefault();
        const touch = e.changedTouches[0];
        if (joystickEl.id === "joystick-left") touchLeftId = touch.identifier;
        else touchRightId = touch.identifier;
        handleMove(touch.clientX, touch.clientY, getCenter());
    }, { passive: false });
    joystickEl.addEventListener("touchmove", (e) => {
        e.preventDefault();
        const targetId = joystickEl.id === "joystick-left" ? touchLeftId : touchRightId;
        for (const touch of e.changedTouches) {
            if (touch.identifier === targetId) {
                handleMove(touch.clientX, touch.clientY, getCenter());
                break;
            }
        }
    }, { passive: false });
    const endHandler = (e) => {
        const targetId = joystickEl.id === "joystick-left" ? touchLeftId : touchRightId;
        for (const touch of e.changedTouches) {
            if (touch.identifier === targetId) {
                if (joystickEl.id === "joystick-left") touchLeftId = null;
                else touchRightId = null;
                handleEnd();
                break;
            }
        }
    };
    joystickEl.addEventListener("touchend", endHandler);
    joystickEl.addEventListener("touchcancel", endHandler);
}

if ("ontouchstart" in window || navigator.maxTouchPoints > 0) {
    const jl = document.getElementById("joystick-left");
    const kl = document.getElementById("knob-left");
    const jr = document.getElementById("joystick-right");
    const kr = document.getElementById("knob-right");
    const debugBtnMobile = document.getElementById("debug-btn-mobile");
    requestAnimationFrame(() => {
        setupTouchJoystick(jl, kl, (x, y) => { touchLeftStick = [x, y]; }, () => { touchLeftStick = [0, 0]; });
        setupTouchJoystick(jr, kr, (x, y) => { touchRightStick = [x, y]; }, () => { touchRightStick = [0, 0]; });
    });
    if (debugBtnMobile) {
        debugBtnMobile.addEventListener("click", () => setDebugEnabled(!debugEnabled));
    }
    const dropdown = document.getElementById("style-dropdown");
    if (dropdown) {
        dropdown.addEventListener("change", (e) => {
            currentStyleIndex = parseInt(e.target.value, 10);
            updateStyleDisplay();
        });
    }
}

function renderFrame(alpha) {
    if (!frameCurr || entityCount === 0) return;
    applyFrame(alpha);
    if (!debugEnabled) return;

    const ao = _pos.clone();
    ao.y += 0.01;
    axisX.position.copy(ao);
    axisY.position.copy(ao);
    axisZ.position.copy(ao);
    axisX.setDirection(new THREE.Vector3(1, 0, 0).applyQuaternion(_quat));
    axisY.setDirection(new THREE.Vector3(0, 1, 0).applyQuaternion(_quat));
    axisZ.setDirection(new THREE.Vector3(0, 0, 1).applyQuaternion(_quat));

    const skelAttr = skelLine.geometry.getAttribute("position");
    for (let p = 0; p < skeletonPairEntityIndices.length; p++) {
        const idxA = skeletonPairEntityIndices[p][0];
        const idxB = skeletonPairEntityIndices[p][1];
        if (idxA >= 0 && idxB >= 0) {
            const bA = boneMap[entityNames[idxA]];
            const bB = boneMap[entityNames[idxB]];
            if (bA && bB) {
                skelAttr.setXYZ(p * 2, bA.matrixWorld.elements[12], bA.matrixWorld.elements[13], bA.matrixWorld.elements[14]);
                skelAttr.setXYZ(p * 2 + 1, bB.matrixWorld.elements[12], bB.matrixWorld.elements[13], bB.matrixWorld.elements[14]);
            }
        }
    }
    skelAttr.needsUpdate = true;

    for (let i = 0; i < entityCount && i < jointSpheres.length; i++) {
        const bone = boneMap[entityNames[i]];
        if (bone) jointSpheres[i].position.set(bone.matrixWorld.elements[12], bone.matrixWorld.elements[13], bone.matrixWorld.elements[14]);
    }

    const cl = ctrlTrajLine.geometry.getAttribute("position");
    for (let i = 0; i < TRAJ_SAMPLES; i++) {
        const p = interpolateVector(framePrev ? framePrev.ctrlTrajectory[i] : frameCurr.ctrlTrajectory[i], frameCurr.ctrlTrajectory[i], alpha, _interpVec);
        const d = interpolateVector(framePrev ? framePrev.ctrlTrajectoryDir[i] : frameCurr.ctrlTrajectoryDir[i], frameCurr.ctrlTrajectoryDir[i], alpha, _interpDir);
        ctrlTrajSpheres[i].position.set(p.x, p.y + 0.01, p.z);
        cl.setXYZ(i, p.x, p.y + 0.01, p.z);
        ctrlTrajArrows[i].position.set(p.x, p.y + 0.01, p.z);
        if (d.lengthSq() > 0.001) {
            _tmpDir.copy(d).normalize();
            ctrlTrajArrows[i].setDirection(_tmpDir);
        }
    }
    cl.needsUpdate = true;

    for (let i = 0; i < contactEntityIndices.length; i++) {
        const idx = contactEntityIndices[i];
        if (idx >= 0) {
            const bone = boneMap[entityNames[idx]];
            if (bone) {
                contactSpheres[i].position.set(bone.matrixWorld.elements[12], bone.matrixWorld.elements[13], bone.matrixWorld.elements[14]);
                const contact = framePrev ? THREE.MathUtils.lerp(framePrev.contacts[i], frameCurr.contacts[i], alpha) : frameCurr.contacts[i];
                contactSpheres[i].material.color.setRGB(1 - contact, contact, 0);
            }
        }
    }
}

function updateFpsDisplay() {
    const now = performance.now();
    const elapsed = now - fpsLastTime;
    if (elapsed >= 1000) {
        const renderFps = Math.round((renderFrameCount * 1000) / elapsed);
        const fpsNode = document.getElementById("render-fps");
        if (fpsNode) fpsNode.textContent = renderFps;
        renderFrameCount = 0;
        fpsLastTime = now;
    }
    const srvElapsed = now - serverFpsLastTime;
    if (srvElapsed >= 1000) {
        serverFpsValue = Math.round((serverFrameCount * 1000) / srvElapsed);
        serverFrameCount = 0;
        serverFpsLastTime = now;
        const srvNode = document.getElementById("server-fps");
        if (srvNode) {
            srvNode.textContent = serverFpsValue;
            srvNode.className = serverFpsValue < 15 ? "perf-crit" : serverFpsValue < 25 ? "perf-warn" : "";
        }
        const infNode = document.getElementById("inference-time");
        if (infNode && avgInferenceMs != null) {
            const ms = Math.round(avgInferenceMs);
            infNode.textContent = ms;
            infNode.className = ms > 80 ? "perf-crit" : ms > 40 ? "perf-warn" : "";
        }
    }
}

const TARGET_RENDER_FPS = 60;
const MIN_RENDER_DT_MS = 1000 / TARGET_RENDER_FPS;
let lastRenderAt = 0;

function animate(timestamp) {
    requestAnimationFrame(animate);
    const delta = timestamp - lastRenderAt;
    if (delta < MIN_RENDER_DT_MS) return;
    lastRenderAt = timestamp - (delta % MIN_RENDER_DT_MS);
    renderFrameCount++;
    renderFrame(getInterpolationAlpha(timestamp));
    updateCamera();
    updateFpsDisplay();
    renderer.render(scene, camera);
    sendInput(timestamp);
}

setDebugEnabled(true);
refreshInputDeviceHud();
requestAnimationFrame(() => refreshInputDeviceHud());
animate(0);

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
