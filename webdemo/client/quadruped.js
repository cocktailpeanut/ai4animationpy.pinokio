import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const DEMO = {
    title: "Neural Quadruped Locomotion",
    modelPath: "/assets/quadruped/Dog.glb",
    wsPath: "/ws-interactive/quadruped",
    trajectorySamples: 16,
    contactCount: 4,
};

const HEARTBEAT_INTERVAL = 10000;
const SERVER_FRAME_MS = 1000 / 30;
const INTERPOLATION_DELAY_MS = SERVER_FRAME_MS;
const INPUT_SEND_INTERVAL_MS = SERVER_FRAME_MS;
const TARGET_RENDER_FPS = 60;
const MIN_RENDER_DT_MS = 1000 / TARGET_RENDER_FPS;
const INPUT_DEADZONE = 0.05;
const MAX_CHARACTER_SPEED = 4.0;
const IS_TOUCH_MOBILE = window.matchMedia("(hover: none) and (pointer: coarse)").matches;

let ws = null;
let debugEnabled = false;
let meshVisible = true;
let entityNames = [];
let entityCount = 0;
let boneMap = {};
let skinnedMesh = null;
let skeletonPairs = [];
let contactEntityIndices = [];
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
let lastRenderAt = 0;
const keys = {};
let gamepadIndex = -1;
let currentRootPos = new THREE.Vector3();
let cameraDistance = 4.0;
let cameraPhi = Math.PI / 9;
let cameraTheta = 0;
const CAM_SELF_HEIGHT = 0.5;
const CAM_TARGET_HEIGHT = 0.5;
const CAM_SMOOTHING = 10.0;
let cameraTarget = new THREE.Vector3(0, CAM_TARGET_HEIGHT, 0);
let cameraPos = new THREE.Vector3(0, CAM_SELF_HEIGHT, cameraDistance);
let lastFrameTime = performance.now();
let currentSpeed = 0.0;

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
const _tmpDir = new THREE.Vector3();

const titleNode = document.getElementById("demo-title");
if (titleNode) titleNode.textContent = DEMO.title;
document.title = DEMO.title;

const styleSwitcher = document.getElementById("style-switcher");
if (styleSwitcher) styleSwitcher.style.display = "none";
const styleDropdown = document.getElementById("style-dropdown");
if (styleDropdown) styleDropdown.style.display = "none";
const styleDropdownLabel = document.getElementById("style-dropdown-label");
if (styleDropdownLabel) styleDropdownLabel.style.display = "none";

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x2a2a3e);
scene.fog = new THREE.Fog(0x2a2a3e, 15, 40);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 2, 6);

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

const TRAJ_SAMPLES = DEMO.trajectorySamples;
const ctrlTrajSpheres = [];
const ctrlTrajArrows = [];
const ctrlTrajGroup = new THREE.Group();
for (let i = 0; i < TRAJ_SAMPLES; i++) {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 8), new THREE.MeshBasicMaterial({ color: 0x00cccc }));
    ctrlTrajSpheres.push(sphere);
    ctrlTrajGroup.add(sphere);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.25, 0xff8800, 0.08, 0.04);
    ctrlTrajArrows.push(arrow);
    ctrlTrajGroup.add(arrow);
}
const ctrlTrajLineGeo = new THREE.BufferGeometry();
ctrlTrajLineGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAJ_SAMPLES * 3), 3));
const ctrlTrajLine = new THREE.Line(ctrlTrajLineGeo, new THREE.LineBasicMaterial({ color: 0x00cccc, depthTest: false }));
ctrlTrajLine.frustumCulled = false;
ctrlTrajGroup.add(ctrlTrajLine);
debugGroup.add(ctrlTrajGroup);

const simTrajSpheres = [];
const simTrajArrows = [];
const simTrajGroup = new THREE.Group();
for (let i = 0; i < TRAJ_SAMPLES; i++) {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.028, 8, 8), new THREE.MeshBasicMaterial({ color: 0x00ff66 }));
    simTrajSpheres.push(sphere);
    simTrajGroup.add(sphere);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.22, 0x44ff44, 0.08, 0.04);
    simTrajArrows.push(arrow);
    simTrajGroup.add(arrow);
}
const simTrajLineGeo = new THREE.BufferGeometry();
simTrajLineGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAJ_SAMPLES * 3), 3));
const simTrajLine = new THREE.Line(simTrajLineGeo, new THREE.LineBasicMaterial({ color: 0x00ff66, depthTest: false }));
simTrajLine.frustumCulled = false;
simTrajGroup.add(simTrajLine);
debugGroup.add(simTrajGroup);

const skelGeo = new THREE.BufferGeometry();
skelGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(1024 * 3), 3));
const skelLine = new THREE.LineSegments(skelGeo, new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
skelLine.frustumCulled = false;
debugGroup.add(skelLine);

const contactSpheres = [];
for (let i = 0; i < DEMO.contactCount; i++) {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8), new THREE.MeshBasicMaterial({ color: 0x00ff00, depthTest: false }));
    sphere.renderOrder = 999;
    contactSpheres.push(sphere);
    debugGroup.add(sphere);
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

function rebuildSkeletonPairs() {
    const boneSet = new Set(entityNames);
    skeletonPairs = [];
    for (const name of entityNames) {
        const bone = boneMap[name];
        if (!bone || !bone.parent || !bone.parent.isBone) continue;
        if (!boneSet.has(bone.parent.name)) continue;
        skeletonPairs.push([name, bone.parent.name]);
    }
    contactEntityIndices = [
        entityNames.indexOf("LeftHandSite"),
        entityNames.indexOf("RightHandSite"),
        entityNames.indexOf("LeftFootSite"),
        entityNames.indexOf("RightFootSite"),
    ];
}

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

function setDebugEnabled(value) {
    debugEnabled = value;
    debugGroup.visible = debugEnabled;
    const label = document.getElementById("debug-label");
    if (label) label.textContent = "Debug G";
    const mobile = document.getElementById("debug-btn-mobile");
    if (mobile) {
        mobile.textContent = "Debug";
        mobile.classList.toggle("active", debugEnabled);
    }
}

function updateSpeedBar(speedValue) {
    const speed = THREE.MathUtils.clamp(speedValue || 0, 0, MAX_CHARACTER_SPEED);
    const fill = document.getElementById("speed-bar-fill");
    const label = document.getElementById("speed-bar-label");
    if (fill) fill.style.width = `${(speed / MAX_CHARACTER_SPEED) * 100}%`;
    if (label) label.textContent = `Speed: ${speed.toFixed(2)} m/s`;
}

function updateSpeedOverlayPosition() {
    if (IS_TOUCH_MOBILE) return;
    const overlay = document.getElementById("speed-overlay");
    if (!overlay) return;
    const worldPos = currentRootPos.clone();
    worldPos.y += 1.35;
    worldPos.project(camera);
    if (worldPos.z > 1.0) {
        overlay.classList.add("hidden");
        return;
    }
    overlay.classList.remove("hidden");
    const x = (worldPos.x * 0.5 + 0.5) * window.innerWidth;
    const y = (-worldPos.y * 0.5 + 0.5) * window.innerHeight;
    overlay.style.left = `${x}px`;
    overlay.style.top = `${Math.max(80, y)}px`;
}

function setInputKeyActive(id, active) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle("active", !!active);
}

function getButtonPressed(gp, index) {
    return !!(gp.buttons[index] && gp.buttons[index].pressed);
}

function getButtonValue(gp, index) {
    return gp.buttons[index] ? gp.buttons[index].value : 0.0;
}

function getMappedGamepadState(gp) {
    const standard = gp.mapping === "standard";

    const physicalL1 = getButtonPressed(gp, 4);
    const physicalR1 = getButtonPressed(gp, 5);
    let physicalL2 = getButtonPressed(gp, 6);
    let physicalR2 = getButtonPressed(gp, 7);
    let faceBottom = getButtonPressed(gp, 0);

    if (!standard) {
        physicalL2 = physicalL2 || getButtonValue(gp, 6) > 0.4 || ((gp.axes[4] || 0) > 0.5);
        physicalR2 = physicalR2 || getButtonValue(gp, 7) > 0.4 || ((gp.axes[5] || 0) > 0.5);
        faceBottom = faceBottom || getButtonPressed(gp, 1);
    }

    const lxRaw = gp.axes[0] || 0;
    const lyRaw = gp.axes[1] || 0;

    return {
        lb: physicalL2,
        rb: physicalR2,
        l2: physicalL1,
        r2: physicalR1,
        canter: faceBottom,
        lxRaw,
        lyRaw,
    };
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
    if (kbPanel) kbPanel.classList.remove("hidden");
    if (gpPanel) gpPanel.classList.add("hidden");

    setInputKeyActive("key-w", !!keys["KeyW"]);
    setInputKeyActive("key-a", !!keys["KeyA"]);
    setInputKeyActive("key-s", !!keys["KeyS"]);
    setInputKeyActive("key-d", !!keys["KeyD"]);
    setInputKeyActive("key-alt", !!(keys["AltLeft"] || keys["AltRight"]));
    setInputKeyActive("key-ctrl", !!(keys["ControlLeft"] || keys["ControlRight"]));
    setInputKeyActive("key-shift", !!(keys["ShiftLeft"] || keys["ShiftRight"]));
    let lx = 0;
    let ly = 0;
    if (keys["KeyW"] || keys["ArrowUp"]) ly = 1;
    if (keys["KeyS"] || keys["ArrowDown"]) ly = -1;
    if (keys["KeyA"] || keys["ArrowLeft"]) lx = -1;
    if (keys["KeyD"] || keys["ArrowRight"]) lx = 1;
    const n = normalize2([lx, ly]);
    const [kvx, kvy] = applyDeadzone(n[0], n[1]);
    updateStickCircleHud("left-stick-dot", "left-stick-pointer", kvx, kvy);
    updateControlsHelp("keyboard");
}

function updateInputVisualizerGamepad(mapped) {
    const deviceLabel = document.getElementById("input-device-label");
    if (deviceLabel) deviceLabel.textContent = "Input: Gamepad";
    const kbPanel = document.getElementById("keyboard-front-buttons");
    const gpPanel = document.getElementById("gamepad-front-buttons");
    if (kbPanel) kbPanel.classList.add("hidden");
    if (gpPanel) gpPanel.classList.remove("hidden");

    setInputKeyActive("pad-a", mapped.canter);
    const [lvx, lvy] = applyDeadzone(mapped.lxRaw, -mapped.lyRaw);
    updateStickCircleHud("left-stick-dot", "left-stick-pointer", lvx, lvy);
    updateControlsHelp("gamepad");
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

function updateTimerDisplay(remaining) {
    sessionRemainingSeconds = remaining;
    const timerVal = document.getElementById("timer-value");
    if (timerVal) timerVal.textContent = formatTime(remaining);
    const timer = document.getElementById("session-timer");
    if (!timer) return;
    timer.classList.remove("warning", "critical");
    if (remaining <= 30) timer.classList.add("critical");
    else if (remaining <= 60) timer.classList.add("warning");
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
                    entityNames = msg.entityNames || [];
                    entityCount = msg.entityCount || entityNames.length;
                    rebuildSkeletonPairs();
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

function magnitude2(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1]);
}

function normalize2(v) {
    const mag = magnitude2(v);
    if (mag <= 0.000001) return [0, 0];
    return [v[0] / mag, v[1] / mag];
}

function applyDeadzone2(v, deadzone = INPUT_DEADZONE) {
    return magnitude2(v) < deadzone ? [0, 0] : v;
}

function applyDeadzone(x, y, deadzone = 0.15) {
    const mag = Math.sqrt(x * x + y * y);
    if (mag < deadzone) return [0, 0];
    const scale = (mag - deadzone) / (1.0 - deadzone) / mag;
    return [x * scale, y * scale];
}

function getCameraRelativeAxes(localX, localY) {
    const c = Math.cos(cameraTheta);
    const s = Math.sin(cameraTheta);
    return [localX * c - localY * s, -localX * s - localY * c];
}

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
    if (gp) {
        const mapped = getMappedGamepadState(gp);
        updateInputVisualizerGamepad(mapped);
    } else {
        updateInputVisualizerKeyboard();
    }
}

function getActiveGamepad() {
    ensureGamepadIndex();
    if (gamepadIndex < 0) return null;
    return navigator.getGamepads()[gamepadIndex] || null;
}

function getGamepadInput(gp) {
    const mapped = getMappedGamepadState(gp);
    updateInputVisualizerGamepad(mapped);
    const [lx, ly] = applyDeadzone(mapped.lxRaw, -mapped.lyRaw);
    const cameraRelative = getCameraRelativeAxes(lx, ly);
    return {
        left_stick: cameraRelative,
        canter_boost: mapped.canter,
        walk_modifier: false,
        trot_modifier: false,
        canter_modifier: false,
        action_sit: mapped.rb,
        action_stand: mapped.lb,
        action_lie: mapped.l2,
    };
}

let touchLeftId = null;
let touchLeftStick = [0, 0];
let sprintTouchActive = false;

function getKeyboardInput() {
    updateInputVisualizerKeyboard();
    let lx = 0;
    let ly = 0;
    if (keys["KeyW"] || keys["ArrowUp"]) ly = 1;
    if (keys["KeyS"] || keys["ArrowDown"]) ly = -1;
    if (keys["KeyA"] || keys["ArrowLeft"]) lx = -1;
    if (keys["KeyD"] || keys["ArrowRight"]) lx = 1;

    const local = normalize2([lx, ly]);
    const cameraRelative = getCameraRelativeAxes(local[0], local[1]);
    const clampedAxes = applyDeadzone2(cameraRelative);

    return {
        left_stick: clampedAxes,
        canter_boost: false,
        walk_modifier: !!(keys["AltLeft"] || keys["AltRight"]),
        trot_modifier: !!(keys["ControlLeft"] || keys["ControlRight"]),
        canter_modifier: !!(keys["ShiftLeft"] || keys["ShiftRight"]),
        action_sit: !!keys["KeyR"],
        action_stand: !!keys["KeyT"],
        action_lie: !!keys["KeyV"],
    };
}

function getTouchInput() {
    updateInputVisualizerKeyboard();
    const [tx, ty] = applyDeadzone(touchLeftStick[0], -touchLeftStick[1]);
    updateStickCircleHud("left-stick-dot", "left-stick-pointer", tx, ty);
    const cameraRelative = getCameraRelativeAxes(touchLeftStick[0], touchLeftStick[1]);
    const clampedAxes = applyDeadzone2(cameraRelative);
    return {
        left_stick: clampedAxes,
        canter_boost: sprintTouchActive,
        walk_modifier: false,
        trot_modifier: false,
        canter_modifier: false,
        action_sit: false,
        action_stand: false,
        action_lie: false,
    };
}

function getInput() {
    const gp = getActiveGamepad();
    if (gp) return getGamepadInput(gp);
    const hasTouch = (
        touchLeftStick[0] !== 0 ||
        touchLeftStick[1] !== 0 ||
        sprintTouchActive
    );
    return hasTouch ? getTouchInput() : getKeyboardInput();
}

function sendInput(timestamp) {
    if (!ws || ws.readyState !== WebSocket.OPEN || sessionState !== "active") return;
    if (timestamp - lastInputSendAt < INPUT_SEND_INTERVAL_MS) return;
    ws.send(JSON.stringify(getInput()));
    lastInputSendAt = timestamp;
}

function createJointSpheres() {
    for (const sphere of jointSpheres) debugGroup.remove(sphere);
    jointSpheres = [];
    for (let i = 0; i < entityCount; i++) {
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.015, 6, 6), new THREE.MeshBasicMaterial({ color: 0xffaa00, depthTest: false }));
        sphere.renderOrder = 999;
        jointSpheres.push(sphere);
        debugGroup.add(sphere);
    }
}

function renderFrame(alpha) {
    if (!frameCurr || entityCount === 0) return;
    applyFrame(alpha);
    currentSpeed = framePrev
        ? THREE.MathUtils.lerp(framePrev.speed || 0, frameCurr.speed || 0, alpha)
        : (frameCurr.speed || 0);
    updateSpeedBar(currentSpeed);
    if (!debugEnabled) return;

    const axisOrigin = _pos.clone();
    axisOrigin.y += 0.01;
    axisX.position.copy(axisOrigin);
    axisY.position.copy(axisOrigin);
    axisZ.position.copy(axisOrigin);
    axisX.setDirection(new THREE.Vector3(1, 0, 0).applyQuaternion(_quat));
    axisY.setDirection(new THREE.Vector3(0, 1, 0).applyQuaternion(_quat));
    axisZ.setDirection(new THREE.Vector3(0, 0, 1).applyQuaternion(_quat));

    const skelAttr = skelLine.geometry.getAttribute("position");
    const pairCount = Math.min(skeletonPairs.length, 512);
    for (let p = 0; p < pairCount; p++) {
        const [childName, parentName] = skeletonPairs[p];
        const child = boneMap[childName];
        const parent = boneMap[parentName];
        if (!child || !parent) continue;
        skelAttr.setXYZ(p * 2, child.matrixWorld.elements[12], child.matrixWorld.elements[13], child.matrixWorld.elements[14]);
        skelAttr.setXYZ(p * 2 + 1, parent.matrixWorld.elements[12], parent.matrixWorld.elements[13], parent.matrixWorld.elements[14]);
    }
    skelAttr.needsUpdate = true;
    skelLine.geometry.setDrawRange(0, pairCount * 2);

    for (let i = 0; i < entityCount && i < jointSpheres.length; i++) {
        const bone = boneMap[entityNames[i]];
        if (bone) jointSpheres[i].position.set(bone.matrixWorld.elements[12], bone.matrixWorld.elements[13], bone.matrixWorld.elements[14]);
    }

    const ctrlLineAttr = ctrlTrajLine.geometry.getAttribute("position");
    for (let i = 0; i < TRAJ_SAMPLES; i++) {
        const p = interpolateVector(framePrev ? framePrev.ctrlTrajectory[i] : frameCurr.ctrlTrajectory[i], frameCurr.ctrlTrajectory[i], alpha, _interpVec);
        const d = interpolateVector(framePrev ? framePrev.ctrlTrajectoryDir[i] : frameCurr.ctrlTrajectoryDir[i], frameCurr.ctrlTrajectoryDir[i], alpha, _tmpDir);
        ctrlTrajSpheres[i].position.set(p.x, p.y + 0.01, p.z);
        ctrlLineAttr.setXYZ(i, p.x, p.y + 0.01, p.z);
        ctrlTrajArrows[i].position.set(p.x, p.y + 0.01, p.z);
        if (d.lengthSq() > 0.001) {
            ctrlTrajArrows[i].setDirection(d.clone().normalize());
        }
    }
    ctrlLineAttr.needsUpdate = true;

    const simLineAttr = simTrajLine.geometry.getAttribute("position");
    for (let i = 0; i < TRAJ_SAMPLES; i++) {
        const p = interpolateVector(framePrev ? framePrev.simTrajectory[i] : frameCurr.simTrajectory[i], frameCurr.simTrajectory[i], alpha, _interpVec);
        const d = interpolateVector(framePrev ? framePrev.simTrajectoryDir[i] : frameCurr.simTrajectoryDir[i], frameCurr.simTrajectoryDir[i], alpha, _tmpDir);
        simTrajSpheres[i].position.set(p.x, p.y + 0.012, p.z);
        simLineAttr.setXYZ(i, p.x, p.y + 0.012, p.z);
        simTrajArrows[i].position.set(p.x, p.y + 0.012, p.z);
        if (d.lengthSq() > 0.001) {
            simTrajArrows[i].setDirection(d.clone().normalize());
        }
    }
    simLineAttr.needsUpdate = true;

    for (let i = 0; i < contactEntityIndices.length; i++) {
        const idx = contactEntityIndices[i];
        if (idx >= 0) {
            const bone = boneMap[entityNames[idx]];
            if (!bone) continue;
            contactSpheres[i].position.set(bone.matrixWorld.elements[12], bone.matrixWorld.elements[13], bone.matrixWorld.elements[14]);
            const contact = framePrev ? THREE.MathUtils.lerp(framePrev.contacts[i], frameCurr.contacts[i], alpha) : frameCurr.contacts[i];
            contactSpheres[i].material.color.setRGB(1 - contact, contact, 0);
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

function animate(timestamp) {
    requestAnimationFrame(animate);
    const delta = timestamp - lastRenderAt;
    if (delta < MIN_RENDER_DT_MS) return;
    lastRenderAt = timestamp - (delta % MIN_RENDER_DT_MS);

    renderFrameCount++;
    renderFrame(getInterpolationAlpha(timestamp));
    updateCamera();
    updateSpeedOverlayPosition();
    updateFpsDisplay();
    renderer.render(scene, camera);
    sendInput(timestamp);
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

document.addEventListener("keydown", (e) => {
    keys[e.code] = true;
    if (e.code === "KeyG") setDebugEnabled(!debugEnabled);
    if (e.code === "KeyM") {
        meshVisible = !meshVisible;
        if (skinnedMesh) skinnedMesh.visible = meshVisible;
    }
});
document.addEventListener("keyup", (e) => {
    keys[e.code] = false;
});

function clearKeyboardState() {
    for (const code of Object.keys(keys)) {
        keys[code] = false;
    }
    refreshInputDeviceHud();
}

window.addEventListener("blur", clearKeyboardState);
window.addEventListener("focus", clearKeyboardState);
document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearKeyboardState();
});

const canvas = renderer.domElement;
canvas.addEventListener("wheel", (e) => {
    cameraDistance = Math.max(1.5, Math.min(15, cameraDistance + e.deltaY * 0.005));
    e.preventDefault();
}, { passive: false });

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
        touchLeftId = touch.identifier;
        handleMove(touch.clientX, touch.clientY, getCenter());
    }, { passive: false });

    joystickEl.addEventListener("touchmove", (e) => {
        e.preventDefault();
        for (const touch of e.changedTouches) {
            if (touch.identifier === touchLeftId) {
                handleMove(touch.clientX, touch.clientY, getCenter());
                break;
            }
        }
    }, { passive: false });

    const endHandler = (e) => {
        for (const touch of e.changedTouches) {
            if (touch.identifier === touchLeftId) {
                touchLeftId = null;
                handleEnd();
                break;
            }
        }
    };

    joystickEl.addEventListener("touchend", endHandler);
    joystickEl.addEventListener("touchcancel", endHandler);
}

if ("ontouchstart" in window || navigator.maxTouchPoints > 0) {
    const joystickLeft = document.getElementById("joystick-left");
    const knobLeft = document.getElementById("knob-left");
    const joystickRight = document.getElementById("joystick-right");
    const knobRight = document.getElementById("knob-right");
    const debugBtnMobile = document.getElementById("debug-btn-mobile");
    const sprintBtnMobile = document.getElementById("sprint-btn-mobile");

    if (joystickRight) joystickRight.style.display = "none";
    if (knobRight) knobRight.style.display = "none";

    requestAnimationFrame(() => {
        setupTouchJoystick(
            joystickLeft,
            knobLeft,
            (x, y) => {
                touchLeftStick = normalize2([x, y]);
            },
            () => {
                touchLeftStick = [0, 0];
            }
        );
    });

    if (debugBtnMobile) {
        debugBtnMobile.addEventListener("click", () => setDebugEnabled(!debugEnabled));
    }

    if (sprintBtnMobile) {
        const setSprintButtonState = (active) => {
            sprintTouchActive = active;
            sprintBtnMobile.classList.toggle("active", active);
        };

        sprintBtnMobile.addEventListener("touchstart", (e) => {
            e.preventDefault();
            setSprintButtonState(true);
        }, { passive: false });

        sprintBtnMobile.addEventListener("touchend", () => setSprintButtonState(false));
        sprintBtnMobile.addEventListener("touchcancel", () => setSprintButtonState(false));
        sprintBtnMobile.addEventListener("pointerdown", () => setSprintButtonState(true));
        sprintBtnMobile.addEventListener("pointerup", () => setSprintButtonState(false));
        sprintBtnMobile.addEventListener("pointercancel", () => setSprintButtonState(false));
    }

}

setDebugEnabled(false);
updateSpeedBar(0.0);
refreshInputDeviceHud();
requestAnimationFrame(() => refreshInputDeviceHud());
animate(0);

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
