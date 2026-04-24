import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const DEMO = {
    title: "Neural Human Locomotion",
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

const loader = new GLTFLoader();
const textureLoader = new THREE.TextureLoader();
const APP_BASE_URL = new URL("../", import.meta.url);

function resolveAppUrl(path) {
    if (!path) return "";
    if (/^[a-z][a-z\d+.-]*:/i.test(path) || path.startsWith("//")) return path;
    const normalizedPath = path.startsWith("/") ? `.${path}` : path;
    return new URL(normalizedPath, APP_BASE_URL).toString();
}

function createWebSocketUrl(path, params) {
    const url = new URL(resolveAppUrl(path));
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.search = params.toString();
    return url.toString();
}

const BONE_NAMES = DEMO.boneNames;
const TRAJ_SAMPLES = 16;
const CONTACT_BONE_INDICES = DEMO.contactBoneIndices;
const CONTACT_BONE_NAMES = CONTACT_BONE_INDICES.map((idx) => BONE_NAMES[idx]);
const BONE_PAIRS = DEMO.bonePairs;
const BONE_PARENT_BY_NAME = {};
for (const [parentIndex, childIndex] of BONE_PAIRS) {
    const parentName = BONE_NAMES[parentIndex];
    const childName = BONE_NAMES[childIndex];
    BONE_PARENT_BY_NAME[childName] = parentName;
}
const HEARTBEAT_INTERVAL = 10000;
const SESSION_REPLACED_CLOSE_CODE = 4001;
const SESSION_REPLACED_MESSAGE = "This demo is now active in another window.";
const CLIENT_ID_KEY = "ai4animation-human-client-id";
const CLIENT_ID = (() => {
    const existing = sessionStorage.getItem(CLIENT_ID_KEY);
    if (existing) return existing;
    const value = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
    sessionStorage.setItem(CLIENT_ID_KEY, value);
    return value;
})();
const CLIENT_STARTED_AT = Date.now();
const SERVER_FRAME_MS = 1000 / 30;
const INTERPOLATION_DELAY_MS = SERVER_FRAME_MS;
const INPUT_SEND_INTERVAL_MS = SERVER_FRAME_MS;
const EXAMPLE_KIND = "human";

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
let modelRoot = null;
let canonicalMeshes = [];
let canonicalSkeleton = null;
let canonicalSegments = [];
let canonicalBounds = null;
let canonicalAnchorPoints = [];
const canonicalRestByRole = {};
const canonicalRestInverseByRole = {};
const canonicalLocalRestByRole = {};
let uploadedGroup = null;
let uploadedSourceMeshes = null;
let uploadedAutoFitMatrix = null;
let uploadedRigMode = null;
let uploadedRigRoot = null;
let uploadedRigSkeleton = null;
let uploadedRigBones = [];
let uploadedRigRestWorldByBone = new Map();
let uploadedRigRestLocalByBone = new Map();
let uploadedRigOriginalRestWorldByBone = new Map();
let uploadedRigOriginalRestLocalByBone = new Map();
let uploadedRigOriginalBoneInverses = [];
let uploadedRigRestParentWorldByBone = new Map();
let uploadedRigRetargetPairs = [];
let uploadedRigDriverBoneByName = new Map();
let uploadedRigContactLocalByName = new Map();
let uploadedRigHandleLocalByName = new Map();
let uploadedRigEditedHandleRoles = new Set();
let activeLandmarkTarget = null;
const uploadedLandmarkPoints = { head: null, hips: null, leftHand: null, rightHand: null };
let activeObjectUrl = null;
let activeTextureUrl = null;
let activeTextureMap = null;
let activeTargetRestByRole = {};
let canonicalTextureTargets = [];
let canonicalMaterialTemplates = [];
let exampleAssets = [];
let exampleLoadToken = 0;
let selectedExampleUrl = "";
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
let facingDragButton = null;
let activeFacingDrag = false;
let directionMouseStart = null;
let directionRingVisible = true;
let rightStick = [0, 0];
let controlMode = "path";
let pathActive = false;
let pathLoop = false;
let pathDisplayVisible = true;
let pathQueueWaypoints = false;
let pathDragActive = false;
let pathWaypointIndex = 0;
const pathWaypoints = [];
const PATH_MAX_WAYPOINTS = 48;
const PATH_MIN_POINT_DISTANCE = 0.45;
const PATH_ARRIVAL_RADIUS = 0.34;
const PATH_SLOWDOWN_RADIUS = 1.15;
const DIRECTION_MOMENTUM = 0.01;
let cameraDistance = 5.0;
let cameraPhi = Math.PI / 16;
let cameraTheta = 0;
let isOrbitDragging = false;
let orbitPointerId = null;
let orbitLastX = 0;
let orbitLastY = 0;
const CAM_SELF_HEIGHT = 0.9;
const CAM_TARGET_HEIGHT = 0.8;
const CAM_SMOOTHING = 10.0;
let cameraTarget = new THREE.Vector3(0, CAM_TARGET_HEIGHT, 0);
let cameraPos = new THREE.Vector3(0, CAM_SELF_HEIGHT, cameraDistance);
let currentRootPos = new THREE.Vector3();
let currentRootQuat = new THREE.Quaternion();
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
const _sourceParentInverse = new THREE.Matrix4();
const _sourceLocalAnimated = new THREE.Matrix4();
const _matA = new THREE.Matrix4();
const _matB = new THREE.Matrix4();
const _matC = new THREE.Matrix4();
const _matD = new THREE.Matrix4();
const _matE = new THREE.Matrix4();
const _matF = new THREE.Matrix4();
const _matG = new THREE.Matrix4();
const _fitPos = new THREE.Vector3();
const _debugPosA = new THREE.Vector3();
const _debugPosB = new THREE.Vector3();
const _skinVertex = new THREE.Vector3();
const _skinWorld = new THREE.Vector3();
const _sourceLocalPos = new THREE.Vector3();
const _sourceLocalQuat = new THREE.Quaternion();
const _sourceLocalScale = new THREE.Vector3();
const _sourceRestLocalPos = new THREE.Vector3();
const _sourceRestLocalQuat = new THREE.Quaternion();
const _sourceRestLocalScale = new THREE.Vector3();
const _sourceRestLocalInvQuat = new THREE.Quaternion();
const _targetRestLocalPos = new THREE.Vector3();
const _targetRestLocalQuat = new THREE.Quaternion();
const _targetRestLocalScale = new THREE.Vector3();
const _targetLocalQuat = new THREE.Quaternion();
const _targetDeltaQuat = new THREE.Quaternion();
const _zeroVec = new THREE.Vector3();
const _desiredHandleWorld = new THREE.Vector3();
const _handleParentLocal = new THREE.Vector3();
const _offsetParentLocal = new THREE.Vector3();
const landmarkRaycaster = new THREE.Raycaster();
const landmarkPointer = new THREE.Vector2();
const landmarkMarkerGroup = new THREE.Group();
const rigEditGroup = new THREE.Group();
const rigEditHandleMap = new Map();
const rigEditJointMap = new Map();
const rigEditVisibleRadius = { primary: 0.015, secondary: 0.015 };
const rigEditPickRadius = { primary: 0.095, secondary: 0.07 };
const rigEditScreenPickRadius = { joint: 72, segment: 48 };
const rigEditVisibleScreenPickRadius = { joint: 64, segment: 42 };
const rigEditLineGeometry = new THREE.BufferGeometry();
const rigEditLineMaterial = new THREE.LineBasicMaterial({
    color: 0xe6e8f2,
    transparent: true,
    opacity: 0.92,
    depthTest: false,
});
const rigEditLine = new THREE.LineSegments(rigEditLineGeometry, rigEditLineMaterial);
const rigEditPrimaryBones = new Set(["Hips", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]);
const rigEditRoles = [...DEMO.boneNames];
const rigEditRoleSet = new Set(rigEditRoles);
const rigEditRoleToBone = Object.fromEntries(rigEditRoles.map((boneName) => [boneName, boneName]));
const rigEditLinkedDragRoles = {
    LeftFoot: ["LeftFoot", "LeftToeBase"],
    RightFoot: ["RightFoot", "RightToeBase"],
};
let activeRigHandle = null;
let rigEditPointerId = null;
let rigEditEnabled = false;
let rigEditHasManualPoints = false;
const rigEditPoints = Object.fromEntries(rigEditRoles.map((boneName) => [boneName, null]));
const rigEditBasePoints = Object.fromEntries(rigEditRoles.map((boneName) => [boneName, null]));
const rigEditEditedRoles = new Set();
const rigEditInputDebug = {
    pointerDowns: 0,
    pointerMoves: 0,
    pointerUps: 0,
    mouseDowns: 0,
    mouseMoves: 0,
    mouseUps: 0,
    dragUpdates: 0,
    lastEvent: null,
    lastSource: null,
    lastPickedRole: null,
    lastPickMethod: null,
    lastPickDistance: null,
    lastActiveRole: null,
    lastBlockedReason: null,
    lastClientX: null,
    lastClientY: null,
};
const rigDragPlane = new THREE.Plane();
const rigDragOffset = new THREE.Vector3();
const rigDragHit = new THREE.Vector3();
const rigDragNormal = new THREE.Vector3();
const rigDragStartScenePoint = new THREE.Vector3();
let rigDragStartPointsByRole = new Map();
let rigDragLinkedRoles = [];
const facingGizmoGroup = new THREE.Group();
const facingLineGeometry = new THREE.BufferGeometry();
const facingLineMaterial = new THREE.LineBasicMaterial({
    color: 0x9fb2eb,
    transparent: true,
    opacity: 0.95,
    depthTest: false,
});
const facingLine = new THREE.Line(facingLineGeometry, facingLineMaterial);
const facingHandle = new THREE.Mesh(
    new THREE.SphereGeometry(0.08, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xbfd4ff, depthTest: false })
);
const facingRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.78, 0.022, 10, 64),
    new THREE.MeshBasicMaterial({ color: 0x8596c8, transparent: true, opacity: 0.88, depthTest: false })
);
const facingDisplayVector = new THREE.Vector2(0, 1);
const facingDragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const facingDragPoint = new THREE.Vector3();
const facingGizmoOrigin = new THREE.Vector3();
let fitState = createDefaultFitState();

const titleNode = document.getElementById("demo-title");
if (titleNode) titleNode.textContent = DEMO.title;
document.title = DEMO.title;
const viewportEl = document.getElementById("viewport");
const importInput = document.getElementById("import-glb-input");
const importButton = document.getElementById("import-glb-btn");
const restoreButton = document.getElementById("restore-model-btn");
const assetStatus = document.getElementById("asset-status");
const examplePicker = document.getElementById("example-picker");
const exampleButton = document.getElementById("example-picker-button");
const exampleButtonValue = document.getElementById("example-picker-value");
const exampleMenu = document.getElementById("example-picker-menu");
const fitPanel = document.getElementById("fit-panel");
const fitYawLeftButton = document.getElementById("fit-yaw-left-btn");
const fitYawRightButton = document.getElementById("fit-yaw-right-btn");
const fitFlipButton = document.getElementById("fit-flip-btn");
const fitMirrorButton = document.getElementById("fit-mirror-btn");
const fitResetButton = document.getElementById("fit-reset-btn");
const fitSkeletonButton = document.getElementById("fit-skeleton-btn");
const fitEditButton = document.getElementById("fit-edit-btn");
const fitApplyButton = document.getElementById("fit-apply-btn");
const fitAutoButton = document.getElementById("fit-auto-btn");
const fitYawInput = document.getElementById("fit-yaw");
const fitPitchInput = document.getElementById("fit-pitch");
const fitRollInput = document.getElementById("fit-roll");
const fitScaleInput = document.getElementById("fit-scale");
const fitWidthInput = document.getElementById("fit-width");
const fitLengthInput = document.getElementById("fit-length");
const fitLiftInput = document.getElementById("fit-lift");
const fitForwardInput = document.getElementById("fit-forward");
const fitYawValue = document.getElementById("fit-yaw-value");
const fitPitchValue = document.getElementById("fit-pitch-value");
const fitRollValue = document.getElementById("fit-roll-value");
const fitScaleValue = document.getElementById("fit-scale-value");
const fitWidthValue = document.getElementById("fit-width-value");
const fitLengthValue = document.getElementById("fit-length-value");
const fitLiftValue = document.getElementById("fit-lift-value");
const fitForwardValue = document.getElementById("fit-forward-value");
const directionRingToggle = document.getElementById("direction-ring-toggle");
const markHeadButton = document.getElementById("mark-head-btn");
const markHipsButton = document.getElementById("mark-hips-btn");
const markLeftHandButton = document.getElementById("mark-left-hand-btn");
const markRightHandButton = document.getElementById("mark-right-hand-btn");
const applyMarksButton = document.getElementById("apply-marks-btn");
const clearMarksButton = document.getElementById("clear-marks-btn");
const modeManualButton = document.getElementById("mode-manual-btn");
const modePathButton = document.getElementById("mode-path-btn");
const pathControls = document.getElementById("path-controls");
const pathPlayButton = document.getElementById("path-play-btn");
const pathClearButton = document.getElementById("path-clear-btn");
const pathLoopToggle = document.getElementById("path-loop-toggle");
const pathDisplayToggle = document.getElementById("path-display-toggle");
const pathQueueToggle = document.getElementById("path-queue-toggle");
const pathStatus = document.getElementById("path-status");

if (importButton) importButton.textContent = "Import GLB";
if (restoreButton) restoreButton.textContent = "Default Model";
if (importInput) importInput.setAttribute("accept", ".glb,.gltf,model/gltf-binary,model/gltf+json");

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
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
viewportEl.appendChild(renderer.domElement);

function resizeViewport() {
    const width = Math.max(1, viewportEl.clientWidth);
    const height = Math.max(1, viewportEl.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
}
resizeViewport();

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
scene.add(landmarkMarkerGroup);
rigEditGroup.renderOrder = 1002;
rigEditLine.renderOrder = 1002;
rigEditGroup.add(rigEditLine);
scene.add(rigEditGroup);
facingGizmoGroup.renderOrder = 1001;
facingRing.rotation.x = Math.PI / 2;
facingRing.renderOrder = 1001;
facingRing.userData.facingControl = true;
facingHandle.renderOrder = 1002;
facingHandle.userData.facingControl = true;
facingLine.renderOrder = 1001;
facingLine.frustumCulled = false;
facingGizmoGroup.add(facingRing, facingLine, facingHandle);
scene.add(facingGizmoGroup);

const pathGroup = new THREE.Group();
const pathLineGeometry = new THREE.BufferGeometry();
const pathLine = new THREE.Line(
    pathLineGeometry,
    new THREE.LineBasicMaterial({ color: 0x79d6bd, transparent: true, opacity: 0.9, depthTest: false })
);
const pathPointGeometry = new THREE.SphereGeometry(0.07, 14, 14);
const pathPointMaterial = new THREE.MeshBasicMaterial({ color: 0x79d6bd, depthTest: false });
const pathTargetMaterial = new THREE.MeshBasicMaterial({ color: 0xffd166, depthTest: false });
const pathPointGroup = new THREE.Group();
const pathTargetMarker = new THREE.Mesh(
    new THREE.RingGeometry(0.18, 0.24, 32),
    new THREE.MeshBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.92, side: THREE.DoubleSide, depthTest: false })
);
pathLine.renderOrder = 1000;
pathPointGroup.renderOrder = 1001;
pathTargetMarker.renderOrder = 1001;
pathTargetMarker.rotation.x = -Math.PI / 2;
pathTargetMarker.visible = false;
pathGroup.visible = false;
pathGroup.add(pathLine, pathPointGroup, pathTargetMarker);
scene.add(pathGroup);

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

function clearMatrixMap(map) {
    for (const key of Object.keys(map)) delete map[key];
}

function rebuildCanonicalLocalRest() {
    clearMatrixMap(canonicalLocalRestByRole);
    for (const name of BONE_NAMES) {
        const rest = canonicalRestByRole[name];
        if (!rest) continue;
        const parentName = BONE_PARENT_BY_NAME[name];
        const parentRest = parentName ? canonicalRestByRole[parentName] : null;
        const localRest = parentRest
            ? parentRest.clone().invert().multiply(rest)
            : rest.clone();
        canonicalLocalRestByRole[name] = localRest;
    }
}

function resetActiveTargetRest() {
    activeTargetRestByRole = {};
    for (const [name, matrix] of Object.entries(canonicalRestByRole)) {
        activeTargetRestByRole[name] = matrix.clone();
    }
}

function cloneMatrixMap(map) {
    const clone = new Map();
    for (const [key, matrix] of map) clone.set(key, matrix.clone());
    return clone;
}

function maxMatrixAbsDelta(a, b) {
    if (!a || !b) return Infinity;
    let maxDelta = 0;
    for (let index = 0; index < 16; index++) {
        maxDelta = Math.max(maxDelta, Math.abs(a.elements[index] - b.elements[index]));
    }
    return maxDelta;
}

function cloneMatrixWithPosition(matrix, position) {
    const currentPosition = new THREE.Vector3();
    const currentRotation = new THREE.Quaternion();
    const currentScale = new THREE.Vector3();
    matrix.decompose(currentPosition, currentRotation, currentScale);
    return new THREE.Matrix4().compose(position.clone(), currentRotation, currentScale);
}

function applyFrame(alpha) {
    if (!frameCurr) return;
    if (!framePrev) {
        frameCurr.rootMatrix.decompose(_pos, _quat, _scl);
        currentRootPos.copy(_pos);
        currentRootQuat.copy(_quat);
        for (let i = 0; i < entityCount; i++) {
            const bone = boneMap[entityNames[i]];
            if (bone) bone.matrixWorld.copy(frameCurr.entityMatrices[i]);
        }
        return;
    }
    interpolateTransform(framePrev.rootMatrix, frameCurr.rootMatrix, alpha, _pos, _quat, _scl);
    currentRootPos.copy(_pos);
    currentRootQuat.copy(_quat);
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
    debugGroup.visible = debugEnabled && !rigEditEnabled;
    updateDebugLabel();
    if (fitSkeletonButton) fitSkeletonButton.classList.toggle("active", !!debugEnabled);
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
    if (state === "replaced") stopHeartbeat();
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

function showBusyOverlay(message, title = "Server Busy") {
    const overlay = document.getElementById("busy-overlay");
    if (overlay) {
        const heading = overlay.querySelector("#busy-box h1");
        if (heading) heading.textContent = title;
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
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
    const params = new URLSearchParams(window.location.search);
    params.set("client_id", CLIENT_ID);
    params.set("started_at", `${CLIENT_STARTED_AT}`);
    ws = new WebSocket(createWebSocketUrl(DEMO.wsPath, params));
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
        const dot = document.getElementById("connection-dot");
        const text = document.getElementById("connection-text");
        if (dot) dot.classList.add("connected");
        if (text) text.textContent = "Connected";
        lastInputSendAt = 0;
        startHeartbeat();
    };
    ws.onclose = (event) => {
        const dot = document.getElementById("connection-dot");
        const text = document.getElementById("connection-text");
        if (dot) dot.classList.remove("connected");
        if (text) text.textContent = "Disconnected";
        stopHeartbeat();
        if (event.code === SESSION_REPLACED_CLOSE_CODE || sessionState === "replaced") {
            if (text) text.textContent = "Moved";
            setSessionState("replaced");
            showBusyOverlay(SESSION_REPLACED_MESSAGE, "Session Moved");
            return;
        }
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
                case "replaced":
                    setSessionState("replaced");
                    showBusyOverlay(msg.message, "Session Moved");
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
        s.userData.entityIndex = i;
        jointSpheres.push(s);
        debugGroup.add(s);
    }
}

function setAssetStatus(text, tone = "ok") {
    if (!assetStatus) return;
    assetStatus.textContent = text;
    assetStatus.classList.remove("tone-ok", "tone-warn", "tone-error");
    assetStatus.classList.add(
        tone === "error" ? "tone-error" : tone === "warn" ? "tone-warn" : "tone-ok"
    );
}

function createDefaultFitState() {
    return {
        yawDegrees: 0,
        pitchDegrees: 0,
        rollDegrees: 0,
        mirror: false,
        scale: 1,
        width: 1,
        length: 1,
        lift: 0,
        forward: 0,
    };
}

function updateLandmarkButtons() {
    const mapping = {
        head: markHeadButton,
        hips: markHipsButton,
        leftHand: markLeftHandButton,
        rightHand: markRightHandButton,
    };
    for (const [name, button] of Object.entries(mapping)) {
        if (!button) continue;
        button.classList.toggle("active", activeLandmarkTarget === name || !!uploadedLandmarkPoints[name]);
    }
}

function updateFitUi() {
    const active = !!uploadedSourceMeshes;
    if (fitPanel) fitPanel.classList.toggle("hidden", !active);
    updateLandmarkButtons();
    updateRigEditOverlay();
    if (fitSkeletonButton) fitSkeletonButton.classList.toggle("active", !!debugEnabled || rigEditEnabled);
    if (fitEditButton) fitEditButton.classList.toggle("active", rigEditEnabled);
    const fitHintText = document.getElementById("fit-hint-text");
    const fitGuidance = document.getElementById("fit-guidance");
    const preservedRig = uploadedRigMode === "preserved";
    if (fitHintText) {
        fitHintText.textContent = rigEditEnabled
            ? "Drag any skeleton joint or line directly in the viewport."
            : "Click a visible skeleton joint or line, or use Edit Skeleton.";
    }
    if (fitGuidance) {
        fitGuidance.innerHTML = rigEditEnabled
            ? (preservedRig
                ? "Line the skeleton up over the uploaded body, then click <strong>Apply</strong> to recalibrate the preserved skeleton landmarks."
                : "Line the skeleton up over the body, then click <strong>Apply</strong> to rebuild the canonical fallback bind.")
            : (preservedRig
                ? "Show the skeleton, then drag a joint directly or use <strong>Edit Skeleton</strong>."
                : "Show the skeleton, then drag a joint directly or use <strong>Edit Skeleton</strong>.");
    }
    if (!active) return;
    if (fitYawInput) fitYawInput.value = fitState.yawDegrees.toFixed(0);
    if (fitPitchInput) fitPitchInput.value = fitState.pitchDegrees.toFixed(0);
    if (fitRollInput) fitRollInput.value = fitState.rollDegrees.toFixed(0);
    if (fitScaleInput) fitScaleInput.value = fitState.scale.toFixed(2);
    if (fitWidthInput) fitWidthInput.value = fitState.width.toFixed(2);
    if (fitLengthInput) fitLengthInput.value = fitState.length.toFixed(2);
    if (fitLiftInput) fitLiftInput.value = fitState.lift.toFixed(2);
    if (fitForwardInput) fitForwardInput.value = fitState.forward.toFixed(2);
    if (fitYawValue) fitYawValue.textContent = `${Math.round(fitState.yawDegrees)}°`;
    if (fitPitchValue) fitPitchValue.textContent = `${Math.round(fitState.pitchDegrees)}°`;
    if (fitRollValue) fitRollValue.textContent = `${Math.round(fitState.rollDegrees)}°`;
    if (fitScaleValue) fitScaleValue.textContent = fitState.scale.toFixed(2);
    if (fitWidthValue) fitWidthValue.textContent = fitState.width.toFixed(2);
    if (fitLengthValue) fitLengthValue.textContent = fitState.length.toFixed(2);
    if (fitLiftValue) fitLiftValue.textContent = fitState.lift.toFixed(2);
    if (fitForwardValue) fitForwardValue.textContent = fitState.forward.toFixed(2);
    if (fitMirrorButton) fitMirrorButton.classList.toggle("active", !!fitState.mirror);
}

function updateViewportCursor() {
    if (!renderer?.domElement) return;
    renderer.domElement.style.cursor =
        activeLandmarkTarget ? "crosshair" :
        (activeRigHandle || activeFacingDrag || isOrbitDragging || pathDragActive ? "grabbing" :
            controlMode === "path" ? "crosshair" : "grab");
}

function clearLandmarkMarkers() {
    while (landmarkMarkerGroup.children.length) {
        landmarkMarkerGroup.remove(landmarkMarkerGroup.children[0]);
    }
}

function rebuildLandmarkMarkers() {
    clearLandmarkMarkers();
    const palette = {
        head: 0x8ec5ff,
        hips: 0xffc86b,
        leftHand: 0x98f0a4,
        rightHand: 0xf7a7ff,
    };
    for (const [name, point] of Object.entries(uploadedLandmarkPoints)) {
        if (!point) continue;
        const marker = new THREE.Mesh(
            new THREE.SphereGeometry(0.08, 12, 12),
            new THREE.MeshBasicMaterial({ color: palette[name] || 0xffffff, depthTest: false })
        );
        marker.position.copy(point);
        marker.renderOrder = 1001;
        landmarkMarkerGroup.add(marker);
    }
    updateLandmarkButtons();
}

function getCanonicalRigHandleMap() {
    const map = {};
    for (const [role, boneName] of Object.entries(rigEditRoleToBone)) {
        const matrix = activeTargetRestByRole[boneName];
        map[role] = matrix ? new THREE.Vector3().setFromMatrixPosition(matrix) : null;
    }
    return map;
}

function buildUploadedRestWorldByBoneFromLocal(localMap = uploadedRigRestLocalByBone) {
    const worldByBone = new Map();
    const sortedBones = [...uploadedRigBones].sort((a, b) => getBoneDepth(a) - getBoneDepth(b));
    for (const bone of sortedBones) {
        const local = localMap.get(bone);
        if (!local) continue;
        const parent = bone.parent;
        const parentWorld = parent?.isBone
            ? (worldByBone.get(parent) || uploadedRigRestWorldByBone.get(parent))
            : uploadedRigRestParentWorldByBone.get(bone);
        worldByBone.set(bone, parentWorld ? parentWorld.clone().multiply(local) : local.clone());
    }
    return worldByBone;
}

function clearRigDragTracking() {
    rigDragStartPointsByRole = new Map();
    rigDragLinkedRoles = [];
}

function getRigEditDragRoles(role) {
    return rigEditLinkedDragRoles[role] || [role];
}

function beginRigEditDrag(role, startScenePoint) {
    clearRigDragTracking();
    rigDragStartScenePoint.copy(startScenePoint);
    rigDragLinkedRoles = getRigEditDragRoles(role).filter((candidate) => rigEditRoles.includes(candidate));
    const seedMap = getRigEditSeedMap();
    const liveMap = getLiveRigHandleMap();
    for (const linkedRole of rigDragLinkedRoles) {
        if (!rigEditPoints[linkedRole] && liveMap[linkedRole]) {
            rigEditPoints[linkedRole] = liveMap[linkedRole].clone();
        }
        if (!rigEditPoints[linkedRole] && seedMap[linkedRole]) {
            rigEditPoints[linkedRole] = seedMap[linkedRole].clone();
        }
        if (rigEditPoints[linkedRole]) {
            rigDragStartPointsByRole.set(linkedRole, rigEditPoints[linkedRole].clone());
        }
    }
    return rigDragStartPointsByRole.size > 0;
}

function updateRigEditDrag(role, desiredWorldPosition) {
    if (!desiredWorldPosition) return false;
    if (!rigDragLinkedRoles.length || !rigDragStartPointsByRole.size) {
        beginRigEditDrag(role, desiredWorldPosition);
    }
    if (!rigDragLinkedRoles.length || !rigDragStartPointsByRole.size) return false;
    const delta = desiredWorldPosition.clone().sub(rigDragStartScenePoint);
    for (const linkedRole of rigDragLinkedRoles) {
        const startPoint = rigDragStartPointsByRole.get(linkedRole);
        if (startPoint) {
            rigEditPoints[linkedRole] = startPoint.clone().add(delta);
            rigEditEditedRoles.add(linkedRole);
        }
    }
    return true;
}

function getPreservedRigRestHandleMap() {
    const map = {};
    if (!uploadedAutoFitMatrix || !uploadedRigBones.length) return map;
    const fitMatrix = uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix());
    const restWorldByBone = buildUploadedRestWorldByBoneFromLocal();
    for (const role of rigEditRoles) {
        map[role] = getUploadedRigHandlePosition(role, new THREE.Vector3(), { restWorldByBone, fitMatrix });
    }
    return map;
}

function getLiveRigHandleMap() {
    const map = {};
    for (const role of rigEditRoles) {
        const point = getRigHandlePosition(role, new THREE.Vector3());
        map[role] = point ? point.clone() : null;
    }
    return map;
}

function getRigEditSeedMap() {
    return uploadedRigMode === "preserved" ? getLiveRigHandleMap() : getCanonicalRigHandleMap();
}

function seedRigEditPointsFromMap(pointMap) {
    for (const role of rigEditRoles) {
        const point = pointMap[role] ? pointMap[role].clone() : null;
        rigEditPoints[role] = point;
        rigEditBasePoints[role] = point ? point.clone() : null;
    }
    rigEditEditedRoles.clear();
}

function hasRigEditPointState() {
    return rigEditRoles.some((role) => rigEditPoints[role] || rigEditBasePoints[role]);
}

function buildRigEditDisplayHandleMap({ fillMissingFromLive = false } = {}) {
    const seedMap = getRigEditSeedMap();
    const livePositions = fillMissingFromLive ? getLiveRigHandleMap() : {};
    const editPositions = {};
    for (const role of rigEditRoles) {
        if (!rigEditPoints[role] && seedMap[role]) {
            rigEditPoints[role] = seedMap[role].clone();
        }
        if (rigEditPoints[role]) {
            editPositions[role] = rigEditPoints[role].clone();
            continue;
        }
        const livePoint = livePositions[role];
        if (livePoint) {
            editPositions[role] = livePoint.clone();
            rigEditPoints[role] = livePoint.clone();
        }
    }
    return editPositions;
}

function getVisibleSkeletonHandleMap() {
    if (uploadedRigMode !== "preserved" || !uploadedSourceMeshes) return null;
    return getLiveRigHandleMap();
}

function isUploadedSkeletonDebugOverlayActive() {
    return !!debugEnabled && uploadedRigMode === "preserved" && !!uploadedSourceMeshes;
}

function ensureRigEditHandles() {
    if (rigEditJointMap.size) return;
    const palette = {
        Hips: 0xffc86b,
        Head: 0xbfd9ff,
        LeftHand: 0x98f0a4,
        RightHand: 0xf3a6ff,
        LeftFoot: 0x79f0d0,
        RightFoot: 0xe7ff79,
    };
    rigEditLineGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BONE_PAIRS.length * 2 * 3), 3));
    for (const boneName of DEMO.boneNames) {
        const isPrimary = rigEditPrimaryBones.has(boneName);
        const joint = new THREE.Mesh(
            new THREE.SphereGeometry(isPrimary ? rigEditVisibleRadius.primary : rigEditVisibleRadius.secondary, 8, 8),
            new THREE.MeshBasicMaterial({
                color: palette[boneName] || 0xf0d765,
                depthTest: false,
                transparent: true,
                opacity: isPrimary ? 1 : 0.94,
            })
        );
        joint.renderOrder = isPrimary ? 1003 : 1002;
        joint.userData.boneName = boneName;
        joint.userData.rigRole = boneName;
        const pickJoint = new THREE.Mesh(
            new THREE.SphereGeometry(isPrimary ? rigEditPickRadius.primary : rigEditPickRadius.secondary, 10, 10),
            new THREE.MeshBasicMaterial({
                color: 0xffffff,
                depthTest: false,
                depthWrite: false,
                transparent: true,
                opacity: 0,
            })
        );
        pickJoint.renderOrder = 1004;
        pickJoint.userData.boneName = boneName;
        pickJoint.userData.rigRole = boneName;
        rigEditHandleMap.set(boneName, pickJoint);
        rigEditJointMap.set(boneName, joint);
        rigEditGroup.add(joint);
        rigEditGroup.add(pickJoint);
    }
}

function clearRigEditPoints() {
    for (const role of rigEditRoles) {
        rigEditPoints[role] = null;
        rigEditBasePoints[role] = null;
    }
    rigEditEditedRoles.clear();
    activeRigHandle = null;
    clearRigDragTracking();
    rigEditHasManualPoints = false;
    rigEditEnabled = false;
    updateRigEditOverlay();
    updateViewportCursor();
}

function resetRigEditHandles() {
    seedRigEditPointsFromMap(getRigEditSeedMap());
    activeRigHandle = null;
    clearRigDragTracking();
    rigEditHasManualPoints = false;
    updateRigEditOverlay();
    updateViewportCursor();
}

function beginSkeletonEdit({ reseed = false } = {}) {
    if (!uploadedSourceMeshes) return false;
    ensureRigEditHandles();
    if (reseed || !hasRigEditPointState()) {
        seedRigEditPointsFromMap(getRigEditSeedMap());
    }
    rigEditEnabled = true;
    activeRigHandle = null;
    clearRigDragTracking();
    updateRigEditOverlay();
    updateFitUi();
    updateViewportCursor();
    return true;
}

function updateRigEditOverlay() {
    const active = !!uploadedSourceMeshes && rigEditEnabled;
    rigEditGroup.visible = active;
    debugGroup.visible = debugEnabled && !active;
    if (!active) return;
    ensureRigEditHandles();

    const editPositions = buildRigEditDisplayHandleMap({ fillMissingFromLive: true });

    const attr = rigEditLineGeometry.getAttribute("position");
    for (let index = 0; index < BONE_PAIRS.length; index++) {
        const [fromIndex, toIndex] = BONE_PAIRS[index];
        const from = editPositions[BONE_NAMES[fromIndex]];
        const to = editPositions[BONE_NAMES[toIndex]];
        if (from && to) {
            attr.setXYZ(index * 2, from.x, from.y, from.z);
            attr.setXYZ(index * 2 + 1, to.x, to.y, to.z);
        } else {
            attr.setXYZ(index * 2, 0, 0, 0);
            attr.setXYZ(index * 2 + 1, 0, 0, 0);
        }
    }
    attr.needsUpdate = true;

    for (const boneName of DEMO.boneNames) {
        const joint = rigEditJointMap.get(boneName);
        const pickJoint = rigEditHandleMap.get(boneName);
        const point = editPositions[boneName];
        if (joint) {
            joint.visible = !!point;
            if (point) joint.position.copy(point);
        }
        if (pickJoint) {
            pickJoint.visible = !!point;
            if (point) pickJoint.position.copy(point);
        }
    }
    rigEditGroup.updateMatrixWorld(true);
}

function getDebugBonePosition(boneName, target) {
    if (uploadedRigMode === "preserved") {
        const uploadedBone = uploadedRigDriverBoneByName.get(boneName);
        if (uploadedBone) return target.setFromMatrixPosition(uploadedBone.matrixWorld);
    }
    const bone = boneMap[boneName];
    if (!bone) return null;
    return target.setFromMatrixPosition(bone.matrixWorld);
}

function getUploadedRigHandlePosition(role, target, { restWorldByBone = null, fitMatrix = null } = {}) {
    if (uploadedRigMode !== "preserved") return null;
    const uploadedBone = uploadedRigDriverBoneByName.get(role);
    if (!uploadedBone) return null;
    const localOffset = uploadedRigHandleLocalByName.get(role);
    if (localOffset) target.copy(localOffset);
    else target.set(0, 0, 0);
    const matrix = restWorldByBone ? restWorldByBone.get(uploadedBone) : uploadedBone.matrixWorld;
    if (!matrix) return null;
    target.applyMatrix4(matrix);
    if (fitMatrix) target.applyMatrix4(fitMatrix);
    return target;
}

function getRigHandlePosition(boneName, target) {
    const uploadedPoint = getUploadedRigHandlePosition(boneName, target);
    if (uploadedPoint) return uploadedPoint;
    return getDebugBonePosition(boneName, target);
}

function getVisibleSkeletonBonePosition(boneName, target, visibleHandleMap = null) {
    const visiblePoint = visibleHandleMap?.[boneName];
    if (visiblePoint) return target.copy(visiblePoint);
    return getRigHandlePosition(boneName, target);
}

function getContactMarkerPosition(boneName, target) {
    if (uploadedRigMode === "preserved") {
        const uploadedBone = uploadedRigDriverBoneByName.get(boneName);
        const localOffset = uploadedRigContactLocalByName.get(boneName);
        if (uploadedBone && localOffset) {
            return target.copy(localOffset).applyMatrix4(uploadedBone.matrixWorld);
        }
    }
    return getDebugBonePosition(boneName, target);
}

function getFacingDisplayDirection() {
    const magnitude = Math.hypot(rightStick[0], rightStick[1]);
    if (magnitude > 1e-4) {
        facingDisplayVector.set(rightStick[0] / magnitude, rightStick[1] / magnitude);
        return facingDisplayVector;
    }
    _tmpDir.set(0, 0, 1).applyQuaternion(_quat);
    const planar = new THREE.Vector2(_tmpDir.x, _tmpDir.z);
    if (planar.lengthSq() > 1e-5) {
        planar.normalize();
        facingDisplayVector.copy(planar);
        return facingDisplayVector;
    }
    facingDisplayVector.set(0, 1);
    return facingDisplayVector;
}

function updateFacingGizmo() {
    const visible = !!modelRoot && !rigEditEnabled && directionRingVisible && controlMode === "manual" && !isUploadedSkeletonDebugOverlayActive();
    facingGizmoGroup.visible = visible;
    if (!visible) return;

    const dir = getFacingDisplayDirection();
    const radius = 0.78;
    const centerY = Math.max(0.04, currentRootPos.y + 0.04);
    facingGizmoOrigin.set(currentRootPos.x, centerY, currentRootPos.z);
    facingGizmoGroup.position.copy(facingGizmoOrigin);

    facingHandle.position.set(dir.x * radius, 0, dir.y * radius);
    const linePoints = new Float32Array([0, 0, 0, dir.x * radius, 0, dir.y * radius]);
    facingLineGeometry.setAttribute("position", new THREE.BufferAttribute(linePoints, 3));
}

const pathPickPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const pathPickPoint = new THREE.Vector3();

function updateInputVisualizerPath(move = [0, 0], face = [0, 0]) {
    const deviceLabel = document.getElementById("input-device-label");
    if (deviceLabel) deviceLabel.textContent = "Input: Path";
    const kbPanel = document.getElementById("keyboard-front-buttons");
    const gpPanel = document.getElementById("gamepad-front-buttons");
    const leftStickCol = document.getElementById("left-stick-column");
    if (kbPanel) kbPanel.classList.add("hidden");
    if (gpPanel) gpPanel.classList.add("hidden");
    if (leftStickCol) leftStickCol.classList.remove("hidden");
    updateStickCircleHud("left-stick-dot", "left-stick-pointer", move[0], move[1]);
    updateFacingStickVisualizer(face[0], face[1]);
    updateControlsHelp("keyboard");
}

function updatePathUi() {
    if (modeManualButton) modeManualButton.classList.toggle("active", controlMode === "manual");
    if (modePathButton) modePathButton.classList.toggle("active", controlMode === "path");
    if (pathControls) pathControls.classList.toggle("hidden", controlMode !== "path");
    if (pathPlayButton) {
        pathPlayButton.disabled = pathWaypoints.length === 0;
        pathPlayButton.textContent = pathActive ? "Pause" : "Play";
    }
    if (pathClearButton) pathClearButton.disabled = pathWaypoints.length === 0;
    if (pathLoopToggle) pathLoopToggle.checked = pathLoop;
    if (pathDisplayToggle) pathDisplayToggle.checked = pathDisplayVisible;
    if (pathQueueToggle) pathQueueToggle.checked = pathQueueWaypoints;
    if (pathStatus) {
        if (pathWaypoints.length === 0) {
            pathStatus.textContent = "No path points";
        } else {
            const mode = pathActive ? "Following" : "Paused";
            pathStatus.textContent = `${mode} ${Math.min(pathWaypointIndex + 1, pathWaypoints.length)}/${pathWaypoints.length}`;
        }
    }
}

function rebuildPathVisuals() {
    while (pathPointGroup.children.length) {
        pathPointGroup.remove(pathPointGroup.children[0]);
    }
    const showPathVisuals = controlMode === "path" && pathDisplayVisible;
    pathGroup.visible = showPathVisuals;
    updatePathLineGeometry();
    for (let index = 0; index < pathWaypoints.length; index++) {
        const point = pathWaypoints[index];
        const marker = new THREE.Mesh(
            pathPointGeometry,
            index === pathWaypointIndex ? pathTargetMaterial : pathPointMaterial
        );
        marker.position.set(point.x, 0.07, point.z);
        marker.renderOrder = 1001;
        pathPointGroup.add(marker);
    }
    const target = pathWaypoints[pathWaypointIndex];
    pathTargetMarker.visible = !!target && showPathVisuals;
    if (target) pathTargetMarker.position.set(target.x, 0.035, target.z);
}

function getPathLinePoints() {
    const points = pathWaypoints.map((point) => new THREE.Vector3(point.x, 0.055, point.z));
    if (points.length === 1) {
        points.unshift(new THREE.Vector3(currentRootPos.x, 0.055, currentRootPos.z));
    }
    return points;
}

function updatePathLineGeometry() {
    const points = getPathLinePoints();
    pathLineGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(points.flatMap((point) => [point.x, point.y, point.z])), 3)
    );
    if (points.length > 0) pathLineGeometry.computeBoundingSphere();
}

function setPathDisplayVisible(visible) {
    pathDisplayVisible = !!visible;
    if (pathDisplayToggle) pathDisplayToggle.checked = pathDisplayVisible;
    rebuildPathVisuals();
}

function setControlMode(mode) {
    controlMode = mode === "path" ? "path" : "manual";
    document.body.dataset.controlMode = controlMode;
    activeFacingDrag = false;
    isOrbitDragging = false;
    pathDragActive = false;
    rightMouseDown = false;
    facingDragButton = null;
    directionMouseStart = null;
    if (controlMode === "manual") {
        pathActive = false;
        rightStick = [0, 0];
    }
    updatePathUi();
    rebuildPathVisuals();
    updateFacingGizmo();
    updateViewportCursor();
}

function getGroundPointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
    landmarkRaycaster.setFromCamera(landmarkPointer, camera);
    if (!landmarkRaycaster.ray.intersectPlane(pathPickPlane, pathPickPoint)) return null;
    return pathPickPoint.clone();
}

function appendPathWaypoint(point) {
    const waypoint = point.clone();
    waypoint.y = 0;
    const last = pathWaypoints[pathWaypoints.length - 1];
    if (last && last.distanceTo(waypoint) < PATH_MIN_POINT_DISTANCE) return false;
    if (pathWaypoints.length >= PATH_MAX_WAYPOINTS) {
        pathWaypoints.shift();
        pathWaypointIndex = Math.max(0, pathWaypointIndex - 1);
    }
    const wasInactive = !pathActive;
    pathWaypoints.push(waypoint);
    if (pathWaypoints.length === 1) {
        pathWaypointIndex = 0;
    } else if (wasInactive) {
        pathWaypointIndex = pathWaypoints.length - 1;
    }
    pathActive = true;
    updatePathUi();
    rebuildPathVisuals();
    return true;
}

function replacePathWithWaypoint(point) {
    pathWaypoints.length = 0;
    pathWaypointIndex = 0;
    pathActive = false;
    rightStick = [0, 0];
    return appendPathWaypoint(point);
}

function beginPathPlacement(event) {
    if (controlMode !== "path" || rigEditEnabled || activeLandmarkTarget) return false;
    const point = getGroundPointFromEvent(event);
    if (!point) return false;
    return pathQueueWaypoints ? appendPathWaypoint(point) : replacePathWithWaypoint(point);
}

function handlePathPointer(event) {
    if (controlMode !== "path" || rigEditEnabled || activeLandmarkTarget) return false;
    const point = getGroundPointFromEvent(event);
    if (!point) return false;
    appendPathWaypoint(point);
    return true;
}

function togglePathPlayback() {
    if (pathWaypoints.length === 0) {
        pathActive = false;
    } else {
        pathActive = !pathActive;
        if (pathActive && pathWaypointIndex >= pathWaypoints.length) pathWaypointIndex = 0;
    }
    updatePathUi();
    rebuildPathVisuals();
}

function clearPath() {
    pathWaypoints.length = 0;
    pathWaypointIndex = 0;
    pathActive = false;
    pathDragActive = false;
    rightStick = [0, 0];
    updatePathUi();
    rebuildPathVisuals();
    updateInputVisualizerPath();
    updateViewportCursor();
}

function advancePathWaypoint() {
    if (pathWaypointIndex < pathWaypoints.length - 1) {
        pathWaypointIndex++;
    } else if (pathLoop && pathWaypoints.length > 0) {
        pathWaypointIndex = 0;
    } else {
        pathActive = false;
    }
    updatePathUi();
    rebuildPathVisuals();
}

function getPathInput() {
    if (pathWaypoints.length === 0 || !pathActive) {
        rightStick = [0, 0];
        updateInputVisualizerPath();
        return { left_stick: [0, 0], right_stick: rightStick, speed_toggle: false };
    }

    let target = pathWaypoints[pathWaypointIndex];
    let dx = target.x - currentRootPos.x;
    let dz = target.z - currentRootPos.z;
    let distance = Math.hypot(dx, dz);
    if (distance < PATH_ARRIVAL_RADIUS) {
        advancePathWaypoint();
        if (!pathActive || pathWaypoints.length === 0) {
            rightStick = [0, 0];
            updateInputVisualizerPath();
            return { left_stick: [0, 0], right_stick: rightStick, speed_toggle: false };
        }
        target = pathWaypoints[pathWaypointIndex];
        dx = target.x - currentRootPos.x;
        dz = target.z - currentRootPos.z;
        distance = Math.hypot(dx, dz);
    }

    const invDistance = distance > 1e-5 ? 1 / distance : 0;
    const face = [dx * invDistance, dz * invDistance];
    const speedScale = THREE.MathUtils.clamp(distance / PATH_SLOWDOWN_RADIUS, 0.2, 1);
    const visualMove = [face[0] * speedScale, face[1] * speedScale];
    const move = [face[0] * speedScale, -face[1] * speedScale];
    const backendFace = [face[0], -face[1]];
    rightStick = backendFace;
    updateInputVisualizerPath(visualMove, face);
    return { left_stick: move, right_stick: backendFace, speed_toggle: false };
}

function setDirectionRingVisible(visible) {
    directionRingVisible = !!visible;
    if (directionRingToggle) directionRingToggle.checked = directionRingVisible;
    if (!directionRingVisible) activeFacingDrag = false;
    updateFacingGizmo();
}

function pickFacingControl(event) {
    if (rigEditEnabled || !facingGizmoGroup.visible) return false;
    const rect = canvas.getBoundingClientRect();
    landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
    landmarkRaycaster.setFromCamera(landmarkPointer, camera);
    const hit = landmarkRaycaster.intersectObjects([facingHandle, facingRing], false)[0];
    return !!hit;
}

function updateFacingFromPointer(event) {
    const rect = canvas.getBoundingClientRect();
    landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
    landmarkRaycaster.setFromCamera(landmarkPointer, camera);
    facingDragPlane.constant = -Math.max(0.04, currentRootPos.y + 0.04);
    if (!landmarkRaycaster.ray.intersectPlane(facingDragPlane, facingDragPoint)) return;
    const dir = new THREE.Vector2(
        facingDragPoint.x - currentRootPos.x,
        facingDragPoint.z - currentRootPos.z
    );
    if (dir.lengthSq() < 1e-4) return;
    dir.normalize();
    rightStick = [dir.x, dir.y];
    updateFacingStickVisualizer(dir.x, dir.y);
    updateFacingGizmo();
}

function normalizeYawDegrees(value) {
    let wrapped = value;
    while (wrapped > 180) wrapped -= 360;
    while (wrapped < -180) wrapped += 360;
    return wrapped;
}

function setModelVisibility(visible) {
    meshVisible = visible;
    for (const mesh of canonicalMeshes) {
        mesh.visible = visible && !uploadedGroup;
    }
    if (uploadedGroup) uploadedGroup.visible = visible;
}

function cloneMaterial(material) {
    if (!material) return material;
    if (Array.isArray(material)) return material.map((entry) => entry.clone());
    return material.clone();
}

function normalizeUploadedMaterial(material) {
    const normalized = cloneMaterial(material);
    const apply = (entry) => {
        if (!entry) return entry;
        if ("metalness" in entry && typeof entry.metalness === "number") entry.metalness = 0;
        if ("roughness" in entry && typeof entry.roughness === "number") {
            entry.roughness = Math.min(0.85, Math.max(0.45, entry.roughness));
        }
        if ("color" in entry && entry.color) entry.color.setRGB(1, 1, 1);
        entry.needsUpdate = true;
        return entry;
    };
    return Array.isArray(normalized) ? normalized.map(apply) : apply(normalized);
}

function cloneMaterialWithTexture(material, texture) {
    const cloned = cloneMaterial(material);
    const applyTexture = (entry) => {
        if (!entry) return entry;
        if ("map" in entry) entry.map = texture;
        if ("color" in entry && entry.color) entry.color.setRGB(1, 1, 1);
        if ("metalness" in entry && typeof entry.metalness === "number") entry.metalness = 0;
        entry.needsUpdate = true;
        return entry;
    };
    return Array.isArray(cloned) ? cloned.map(applyTexture) : applyTexture(cloned);
}

function captureCanonicalMaterials() {
    canonicalMaterialTemplates = canonicalTextureTargets.map((mesh) => ({
        mesh,
        material: cloneMaterial(mesh.material),
    }));
}

function disposeActiveTexture() {
    if (activeTextureMap) {
        activeTextureMap.dispose();
        activeTextureMap = null;
    }
    if (activeTextureUrl) {
        URL.revokeObjectURL(activeTextureUrl);
        activeTextureUrl = null;
    }
}

function restoreCanonicalSurface({ keepStatus = false } = {}) {
    clearUploadedBinding({ revokeObjectUrl: true, resetSource: true });
    disposeActiveTexture();
    resetActiveTargetRest();
    for (const { mesh, material } of canonicalMaterialTemplates) {
        mesh.material = cloneMaterial(material);
        if (Array.isArray(mesh.material)) {
            for (const entry of mesh.material) {
                if (entry) entry.needsUpdate = true;
            }
        } else if (mesh.material) {
            mesh.material.needsUpdate = true;
        }
    }
    setModelVisibility(meshVisible);
    if (!keepStatus) {
        setAssetStatus("Built-in Geno body · default model restored", "ok");
    }
}

async function loadCanonicalTexture(file) {
    setAssetStatus(`${file?.name || "file"} rejected · import a rigged GLB from Texturizer`, "error");
}

function normalizeBoneName(name) {
    return (name || "")
        .toLowerCase()
        .replace(/mixamorig/g, "")
        .replace(/[^a-z0-9]+/g, " ")
        .split(/\s+/)
        .filter((part) => part && !/^(armature|skeleton|rig|ctrl|def|org|jnt|joint)$/.test(part))
        .join("");
}

function inferCanonicalBoneByName(name) {
    const normalized = normalizeBoneName(name);
    if (/end$/.test(normalized)) return null;
    const patterns = [
        ["Hips", [/^hips?$/, /pelvis/, /^root$/, /spinebase/]],
        ["Spine", [/^spine$/]],
        ["Spine1", [/spine1/, /chest/, /upperbody/, /uppertorso/]],
        ["Spine2", [/spine2/, /upperchest/, /thorax/]],
        ["Spine3", [/spine3/]],
        ["Neck", [/^neck$/, /neck1/]],
        ["Head", [/^head$/, /headtop/, /headend/]],
        ["LeftShoulder", [/leftshoulder/, /leftclavicle/, /lclavicle/, /claviclel/]],
        ["LeftArm", [/leftarm$/, /leftupperarm/, /lupperarm/, /upperarml/, /arml$/]],
        ["LeftForeArm", [/leftforearm/, /leftlowerarm/, /llowerarm/, /forearml/, /lowerarml/, /elbowl/]],
        ["LeftHand", [/lefthand$/, /leftwrist/, /handl$/, /wristl/]],
        ["RightShoulder", [/rightshoulder/, /rightclavicle/, /rclavicle/, /clavicler/]],
        ["RightArm", [/rightarm$/, /rightupperarm/, /rupperarm/, /upperarmr/, /armr$/]],
        ["RightForeArm", [/rightforearm/, /rightlowerarm/, /rlowerarm/, /forearmr/, /lowerarmr/, /elbowr/]],
        ["RightHand", [/righthand$/, /rightwrist/, /handr$/, /wristr/]],
        ["LeftUpLeg", [/leftupleg/, /leftthigh/, /lthigh/, /uplegl/, /thighl/, /leftupperleg/]],
        ["LeftLeg", [/leftleg$/, /leftlowerleg/, /lcalf/, /leftcalf/, /lowerlegl/, /legl$/]],
        ["LeftFoot", [/leftfoot$/, /leftankle/, /footl$/, /anklel/]],
        ["LeftToeBase", [/lefttoe/, /lefttoebase/, /toel$/]],
        ["RightUpLeg", [/rightupleg/, /rightthigh/, /rthigh/, /uplegr/, /thighr/, /rightupperleg/]],
        ["RightLeg", [/rightleg$/, /rightlowerleg/, /rcalf/, /rightcalf/, /lowerlegr/, /legr$/]],
        ["RightFoot", [/rightfoot$/, /rightankle/, /footr$/, /ankler/]],
        ["RightToeBase", [/righttoe/, /righttoebase/, /toer$/]],
    ];
    for (const [driverName, matchers] of patterns) {
        if (matchers.some((pattern) => pattern.test(normalized))) {
            return driverName;
        }
    }
    return null;
}

function getBoneDepth(bone) {
    let depth = 0;
    let current = bone.parent;
    while (current) {
        if (current.isBone) depth += 1;
        current = current.parent;
    }
    return depth;
}

function buildUploadedRigRetargetPairs(fitMatrix) {
    if (!uploadedRigBones.length) return [];
    const canonicalRestPositions = {};
    for (const boneName of DEMO.boneNames) {
        const matrix = canonicalRestByRole[boneName];
        if (!matrix) continue;
        canonicalRestPositions[boneName] = new THREE.Vector3().setFromMatrixPosition(matrix);
    }
    return uploadedRigBones
        .map((bone) => {
            const inferredName = inferCanonicalBoneByName(bone.name);
            let driverName = inferredName;
            let proximityMapped = false;
            if (!driverName) {
                const restWorld = uploadedRigRestWorldByBone.get(bone);
                if (restWorld) {
                    _fitPos.setFromMatrixPosition(restWorld).applyMatrix4(fitMatrix);
                    let bestName = "Hips";
                    let bestScore = Infinity;
                    for (const [candidateName, candidatePos] of Object.entries(canonicalRestPositions)) {
                        let score = _fitPos.distanceToSquared(candidatePos);
                        const isLeftCandidate = candidatePos.x < -0.02;
                        const isRightCandidate = candidatePos.x > 0.02;
                        if ((_fitPos.x < -0.02 && isRightCandidate) || (_fitPos.x > 0.02 && isLeftCandidate)) score *= 3.0;
                        if (_fitPos.y > canonicalRestPositions.Hips.y && /(Leg|Foot|Toe)/.test(candidateName)) score *= 1.8;
                        if (_fitPos.y < canonicalRestPositions.Hips.y && /(Head|Neck|Spine|Shoulder|Arm|Hand)/.test(candidateName)) score *= 1.8;
                        if (score < bestScore) {
                            bestScore = score;
                            bestName = candidateName;
                        }
                    }
                    driverName = bestName;
                    proximityMapped = true;
                } else {
                    driverName = "Hips";
                    proximityMapped = true;
                }
            }
            return { bone, driverName, depth: getBoneDepth(bone), proximityMapped };
        })
        .sort((a, b) => a.depth - b.depth);
}

function rebuildUploadedRigDriverLookup() {
    uploadedRigDriverBoneByName = new Map();
    for (const { bone, driverName } of uploadedRigRetargetPairs) {
        const current = uploadedRigDriverBoneByName.get(driverName);
        const exact = bone.name === driverName;
        const currentExact = current?.name === driverName;
        if (!current || (exact && !currentExact) || (!currentExact && getBoneDepth(bone) < getBoneDepth(current))) {
            uploadedRigDriverBoneByName.set(driverName, bone);
        }
    }
}

function rebuildUploadedRigContactAnchors() {
    uploadedRigContactLocalByName = new Map();
    if (!uploadedGroup || !uploadedRigSkeleton) return;

    withUploadedRigRestPose(() => {
    const samplesByRole = new Map(CONTACT_BONE_NAMES.map((name) => [name, []]));
    uploadedGroup.updateMatrixWorld(true);
    uploadedRigSkeleton.update();
    uploadedGroup.traverse((mesh) => {
        if (!mesh.isSkinnedMesh || !mesh.geometry?.getAttribute("position") || !mesh.skeleton) return;
        const position = mesh.geometry.getAttribute("position");
        const skinIndex = mesh.geometry.getAttribute("skinIndex");
        const skinWeight = mesh.geometry.getAttribute("skinWeight");
        if (!skinIndex || !skinWeight) return;

        const roleByIndex = new Map();
        for (const role of CONTACT_BONE_NAMES) {
            const bone = uploadedRigDriverBoneByName.get(role);
            const index = bone ? mesh.skeleton.bones.indexOf(bone) : -1;
            if (index >= 0) roleByIndex.set(index, role);
        }
        if (!roleByIndex.size) return;

        mesh.updateMatrixWorld(true);
        mesh.skeleton.update();
        for (let i = 0; i < position.count; i++) {
            let matched = false;
            for (let j = 0; j < 4; j++) {
                const role = roleByIndex.get(skinIndex.getComponent(i, j));
                if (role && skinWeight.getComponent(i, j) > 0.05) {
                    matched = true;
                    break;
                }
            }
            if (!matched) continue;

            _skinVertex.fromBufferAttribute(position, i);
            if (typeof mesh.applyBoneTransform === "function") mesh.applyBoneTransform(i, _skinVertex);
            else if (typeof mesh.boneTransform === "function") mesh.boneTransform(i, _skinVertex);
            _skinWorld.copy(_skinVertex).applyMatrix4(mesh.matrixWorld);

            for (let j = 0; j < 4; j++) {
                const role = roleByIndex.get(skinIndex.getComponent(i, j));
                const weight = skinWeight.getComponent(i, j);
                if (!role || weight <= 0.05) continue;
                samplesByRole.get(role).push({ point: _skinWorld.clone(), weight });
            }
        }
    });

    for (const role of CONTACT_BONE_NAMES) {
        const bone = uploadedRigDriverBoneByName.get(role);
        const samples = samplesByRole.get(role);
        if (!bone || !samples?.length) continue;
        samples.sort((a, b) => a.point.y - b.point.y);
        const sampleCount = Math.min(48, Math.max(8, Math.ceil(samples.length * 0.08)));
        const centroid = new THREE.Vector3();
        let totalWeight = 0;
        for (const sample of samples.slice(0, sampleCount)) {
            centroid.addScaledVector(sample.point, sample.weight);
            totalWeight += sample.weight;
        }
        if (totalWeight <= 1e-5) continue;
        centroid.multiplyScalar(1 / totalWeight);
        _matA.copy(bone.matrixWorld).invert();
        uploadedRigContactLocalByName.set(role, centroid.applyMatrix4(_matA).clone());
    }
    });
}

function withUploadedRigRestPose(callback) {
    if (!uploadedRigRoot || !uploadedRigBones.length) return callback();
    const saved = uploadedRigBones.map((bone) => ({
        bone,
        position: bone.position.clone(),
        quaternion: bone.quaternion.clone(),
        scale: bone.scale.clone(),
    }));
    try {
        for (const bone of uploadedRigBones) {
            const restLocal = uploadedRigRestLocalByBone.get(bone);
            if (!restLocal) continue;
            restLocal.decompose(bone.position, bone.quaternion, bone.scale);
            bone.updateMatrix();
        }
        uploadedRigRoot.updateMatrixWorld(true);
        if (uploadedRigSkeleton) uploadedRigSkeleton.update();
        return callback();
    } finally {
        for (const entry of saved) {
            entry.bone.position.copy(entry.position);
            entry.bone.quaternion.copy(entry.quaternion);
            entry.bone.scale.copy(entry.scale);
            entry.bone.updateMatrix();
        }
        uploadedRigRoot.updateMatrixWorld(true);
        if (uploadedRigSkeleton) uploadedRigSkeleton.update();
    }
}

function captureUploadedRigReference() {
    uploadedRigBones = [];
    uploadedRigRestWorldByBone = new Map();
    uploadedRigRestLocalByBone = new Map();
    uploadedRigOriginalRestWorldByBone = new Map();
    uploadedRigOriginalRestLocalByBone = new Map();
    uploadedRigRestParentWorldByBone = new Map();
    uploadedRigDriverBoneByName = new Map();
    uploadedRigContactLocalByName = new Map();
    uploadedRigHandleLocalByName = new Map();
    uploadedRigEditedHandleRoles = new Set();
    uploadedRigSkeleton = null;
    if (!uploadedRigRoot) return;

    uploadedRigRoot.updateMatrixWorld(true);
    uploadedRigRoot.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            child.material = normalizeUploadedMaterial(child.material);
            if (child.isSkinnedMesh) child.frustumCulled = false;
        }
        if (child.isSkinnedMesh && child.skeleton && !uploadedRigSkeleton) {
            uploadedRigSkeleton = child.skeleton;
        }
    });
    if (!uploadedRigSkeleton) return;

    uploadedRigBones = [...uploadedRigSkeleton.bones];
    uploadedRigOriginalBoneInverses = uploadedRigSkeleton.boneInverses.map((matrix) => matrix.clone());
    for (const bone of uploadedRigBones) {
        bone.updateMatrix();
        const restWorld = bone.matrixWorld.clone();
        const restLocal = bone.matrix.clone();
        uploadedRigRestWorldByBone.set(bone, restWorld.clone());
        uploadedRigRestLocalByBone.set(bone, restLocal.clone());
        uploadedRigOriginalRestWorldByBone.set(bone, restWorld.clone());
        uploadedRigOriginalRestLocalByBone.set(bone, restLocal.clone());
        if (!bone.parent?.isBone && bone.parent) {
            uploadedRigRestParentWorldByBone.set(bone, bone.parent.matrixWorld.clone());
        }
    }
}

function resetUploadedRigRestToOriginal() {
    uploadedRigRestWorldByBone = cloneMatrixMap(uploadedRigOriginalRestWorldByBone);
    uploadedRigRestLocalByBone = cloneMatrixMap(uploadedRigOriginalRestLocalByBone);
    uploadedRigContactLocalByName = new Map();
    uploadedRigHandleLocalByName = new Map();
    uploadedRigEditedHandleRoles = new Set();
    restoreUploadedRigBoneInverses();
}

function restoreUploadedRigBoneInverses() {
    if (!uploadedRigSkeleton || !uploadedRigOriginalBoneInverses.length) return;
    for (let index = 0; index < uploadedRigSkeleton.boneInverses.length; index++) {
        const originalInverse = uploadedRigOriginalBoneInverses[index];
        if (!originalInverse) continue;
        if (!uploadedRigSkeleton.boneInverses[index]) uploadedRigSkeleton.boneInverses[index] = new THREE.Matrix4();
        uploadedRigSkeleton.boneInverses[index].copy(originalInverse);
    }
    uploadedRigSkeleton.update();
}

function getPreservedRigHandleRestWorld(role, target, fitMatrix) {
    const uploadedBone = uploadedRigDriverBoneByName.get(role);
    if (!uploadedBone) return null;
    const restWorld = uploadedRigRestWorldByBone.get(uploadedBone);
    if (!restWorld) return null;
    const localOffset = uploadedRigHandleLocalByName.get(role);
    target.copy(localOffset || _zeroVec).applyMatrix4(restWorld).applyMatrix4(fitMatrix);
    return target;
}

function getCanonicalDriverDelta(driverName, target) {
    const driverBone = boneMap[driverName];
    const driverRestInverse = canonicalRestInverseByRole[driverName];
    if (!driverBone || !driverRestInverse) return null;
    return target.copy(driverBone.matrixWorld).multiply(driverRestInverse);
}

function getDesiredPreservedRigHandleWorld(role, fitMatrix, target) {
    if (!getPreservedRigHandleRestWorld(role, target, fitMatrix)) return null;
    const driverDelta = getCanonicalDriverDelta(role, _matF);
    if (!driverDelta) return null;
    target.applyMatrix4(driverDelta);
    return target;
}

function solveBonePositionForControlHandle(bone, parentWorld, localOffset, desiredHandleWorld) {
    if (!bone || !parentWorld || !localOffset || !desiredHandleWorld) return false;
    _matF.copy(parentWorld).invert();
    _handleParentLocal.copy(desiredHandleWorld).applyMatrix4(_matF);
    _matG.compose(_zeroVec, bone.quaternion, bone.scale);
    _offsetParentLocal.copy(localOffset).applyMatrix4(_matG);
    bone.position.copy(_handleParentLocal.sub(_offsetParentLocal));
    return true;
}

function shouldDriveUploadedBone(pair, drivenBones) {
    if (!pair.proximityMapped) return true;
    let parent = pair.bone.parent;
    while (parent) {
        if (drivenBones.has(parent)) return false;
        parent = parent.parent;
    }
    return true;
}

function applyPreservedUploadedRigPose() {
    if (uploadedRigMode !== "preserved" || !uploadedGroup || !uploadedRigRoot || !uploadedRigRetargetPairs.length) return;
    const fitMatrix = uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix());
    uploadedGroup.matrixAutoUpdate = false;
    uploadedGroup.matrix.copy(fitMatrix);
    uploadedGroup.matrixWorld.copy(fitMatrix);
    uploadedGroup.updateMatrixWorld(true);
    uploadedRigRoot.updateMatrixWorld(true);

    const drivenBones = new Set();
    for (const pair of uploadedRigRetargetPairs) {
        const { bone, driverName } = pair;
        if (!shouldDriveUploadedBone(pair, drivenBones)) continue;
        const driverBone = boneMap[driverName];
        const uploadedRestLocal = uploadedRigRestLocalByBone.get(bone);
        if (!driverBone || !uploadedRestLocal) continue;
        const usesEditedControlHandle = uploadedRigEditedHandleRoles.has(driverName)
            && uploadedRigDriverBoneByName.get(driverName) === bone
            && uploadedRigHandleLocalByName.has(driverName);
        const controlHandleOffset = usesEditedControlHandle ? uploadedRigHandleLocalByName.get(driverName) : null;

        const parentName = BONE_PARENT_BY_NAME[driverName];
        const parentDriverBone = parentName ? boneMap[parentName] : null;
        const sourceLocalRest = canonicalLocalRestByRole[driverName];
        if (parentDriverBone && sourceLocalRest && bone.parent?.isBone) {
            _sourceParentInverse.copy(parentDriverBone.matrixWorld).invert();
            _sourceLocalAnimated.multiplyMatrices(_sourceParentInverse, driverBone.matrixWorld);
            _sourceLocalAnimated.decompose(_sourceLocalPos, _sourceLocalQuat, _sourceLocalScale);
            sourceLocalRest.decompose(_sourceRestLocalPos, _sourceRestLocalQuat, _sourceRestLocalScale);
            uploadedRestLocal.decompose(_targetRestLocalPos, _targetRestLocalQuat, _targetRestLocalScale);
            _sourceRestLocalInvQuat.copy(_sourceRestLocalQuat).invert();
            _targetDeltaQuat.copy(_sourceRestLocalInvQuat).multiply(_sourceLocalQuat).normalize();
            _targetLocalQuat.copy(_targetRestLocalQuat).multiply(_targetDeltaQuat).normalize();
            bone.position.copy(_targetRestLocalPos);
            bone.quaternion.copy(_targetLocalQuat);
            bone.scale.copy(_targetRestLocalScale);
            if (usesEditedControlHandle) {
                const desiredHandle = getDesiredPreservedRigHandleWorld(driverName, fitMatrix, _desiredHandleWorld);
                if (desiredHandle) {
                    solveBonePositionForControlHandle(bone, bone.parent.matrixWorld, controlHandleOffset, desiredHandle);
                }
            }
        } else {
            const driverRestInverse = canonicalRestInverseByRole[driverName];
            const uploadedRestWorld = uploadedRigRestWorldByBone.get(bone);
            if (!driverRestInverse || !uploadedRestWorld) continue;
            const desiredWorld = _matA
                .copy(driverBone.matrixWorld)
                .multiply(_matB.copy(driverRestInverse))
                .multiply(_matC.copy(fitMatrix))
                .multiply(_matD.copy(uploadedRestWorld));
            const parent = bone.parent;
            const parentWorld = parent ? parent.matrixWorld : uploadedGroup.matrixWorld;
            const localMatrix = _matE.copy(parentWorld).invert().multiply(desiredWorld);
            localMatrix.decompose(bone.position, bone.quaternion, bone.scale);
            if (usesEditedControlHandle) {
                const desiredHandle = getDesiredPreservedRigHandleWorld(driverName, fitMatrix, _desiredHandleWorld);
                if (desiredHandle) {
                    solveBonePositionForControlHandle(bone, parentWorld, controlHandleOffset, desiredHandle);
                }
            }
        }
        bone.updateMatrix();
        bone.updateMatrixWorld(true);
        drivenBones.add(bone);
    }

    uploadedRigRoot.updateMatrixWorld(true);
    if (uploadedRigSkeleton) uploadedRigSkeleton.update();
}

function getSkinnedVertex(mesh, index, target) {
    target.fromBufferAttribute(mesh.geometry.getAttribute("position"), index);
    if (mesh.isSkinnedMesh && mesh.skeleton) {
        mesh.skeleton.update();
        if (typeof mesh.applyBoneTransform === "function") mesh.applyBoneTransform(index, target);
        else if (typeof mesh.boneTransform === "function") mesh.boneTransform(index, target);
    }
    return target.applyMatrix4(mesh.matrixWorld);
}

function bakeMeshGeometry(mesh) {
    const geometry = mesh.geometry.clone();
    const position = mesh.geometry.getAttribute("position");
    const baked = new Float32Array(position.count * 3);
    const vertex = new THREE.Vector3();
    for (let i = 0; i < position.count; i++) {
        getSkinnedVertex(mesh, i, vertex);
        baked[i * 3 + 0] = vertex.x;
        baked[i * 3 + 1] = vertex.y;
        baked[i * 3 + 2] = vertex.z;
    }
    geometry.setAttribute("position", new THREE.BufferAttribute(baked, 3));
    geometry.deleteAttribute("skinIndex");
    geometry.deleteAttribute("skinWeight");
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    return { geometry, material: normalizeUploadedMaterial(mesh.material), vertexCount: position.count };
}

function extractUploadGeometry(root) {
    const meshes = [];
    root.updateMatrixWorld(true);
    root.traverse((child) => {
        if (!child.isMesh || !child.geometry?.getAttribute("position")) return;
        meshes.push(bakeMeshGeometry(child));
    });
    return meshes;
}

function samplePointsFromMeshes(meshes, maxPoints = 2200) {
    const points = [];
    const total = meshes.reduce((sum, mesh) => sum + mesh.vertexCount, 0);
    const stride = Math.max(1, Math.floor(total / maxPoints));
    let seen = 0;
    const point = new THREE.Vector3();
    for (const mesh of meshes) {
        const position = mesh.geometry.getAttribute("position");
        for (let i = 0; i < position.count; i++) {
            if (seen % stride === 0) {
                point.fromBufferAttribute(position, i);
                points.push(point.clone());
            }
            seen += 1;
        }
    }
    return points.length ? points : [new THREE.Vector3()];
}

function computeBoundsFromPoints(points) {
    const box = new THREE.Box3();
    for (const point of points) box.expandByPoint(point);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    return { box, center, size, minY: box.min.y };
}

function transformPoints(points, matrix) {
    return points.map((point) => point.clone().applyMatrix4(matrix));
}

function chooseBestFitMatrix(samplePoints) {
    const rotations = [0, Math.PI * 0.5, Math.PI, Math.PI * 1.5];
    let bestMatrix = new THREE.Matrix4();
    let bestScore = Infinity;
    for (const angle of rotations) {
        const rotation = new THREE.Matrix4().makeRotationY(angle);
        const rotatedPoints = transformPoints(samplePoints, rotation);
        const rotatedBounds = computeBoundsFromPoints(rotatedPoints);
        const horizontal = Math.max(rotatedBounds.size.x, rotatedBounds.size.z, 1e-3);
        const canonicalHorizontal = Math.max(canonicalBounds.size.x, canonicalBounds.size.z, 1e-3);
        const scale = Math.min(
            canonicalBounds.size.y / Math.max(rotatedBounds.size.y, 1e-3),
            canonicalHorizontal / horizontal
        );
        const scaledBounds = {
            center: rotatedBounds.center.clone().multiplyScalar(scale),
            minY: rotatedBounds.minY * scale,
        };
        const translation = new THREE.Vector3(
            canonicalBounds.center.x - scaledBounds.center.x,
            canonicalBounds.minY - scaledBounds.minY,
            canonicalBounds.center.z - scaledBounds.center.z
        );
        const fit = rotation.clone();
        fit.premultiply(new THREE.Matrix4().makeScale(scale, scale, scale));
        fit.premultiply(new THREE.Matrix4().makeTranslation(translation.x, translation.y, translation.z));

        let score = 0;
        for (const anchor of canonicalAnchorPoints) {
            let nearest = Infinity;
            for (const point of samplePoints) {
                const transformed = point.clone().applyMatrix4(fit);
                nearest = Math.min(nearest, transformed.distanceToSquared(anchor));
            }
            score += nearest;
        }
        if (score < bestScore) {
            bestScore = score;
            bestMatrix = fit;
        }
    }
    return bestMatrix;
}

function buildUserFitMatrix() {
    const pivot = new THREE.Vector3(canonicalBounds.center.x, canonicalBounds.minY, canonicalBounds.center.z);
    const translateToPivot = new THREE.Matrix4().makeTranslation(pivot.x, pivot.y, pivot.z);
    const translateFromPivot = new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z);
    const scale = new THREE.Matrix4().makeScale(
        fitState.scale * fitState.width * (fitState.mirror ? -1 : 1),
        fitState.scale,
        fitState.scale * fitState.length
    );
    const rotation = new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(
            THREE.MathUtils.degToRad(fitState.pitchDegrees),
            THREE.MathUtils.degToRad(fitState.yawDegrees),
            THREE.MathUtils.degToRad(fitState.rollDegrees),
            "YXZ"
        )
    );
    const offset = new THREE.Matrix4().makeTranslation(0, fitState.lift, fitState.forward);
    return offset.clone().multiply(translateToPivot).multiply(rotation).multiply(scale).multiply(translateFromPivot);
}

function getCanonicalLandmark(name) {
    if (name === "head") return new THREE.Vector3().setFromMatrixPosition(canonicalRestByRole.Head);
    if (name === "hips") return new THREE.Vector3().setFromMatrixPosition(canonicalRestByRole.Hips);
    if (name === "leftHand") return new THREE.Vector3().setFromMatrixPosition(canonicalRestByRole.LeftHand);
    if (name === "rightHand") return new THREE.Vector3().setFromMatrixPosition(canonicalRestByRole.RightHand);
    return null;
}

function buildHumanoidBasis(hips, head, leftHand, rightHand) {
    if (!hips || !head || !leftHand || !rightHand) return null;
    const up = head.clone().sub(hips);
    const right = rightHand.clone().sub(leftHand);
    if (up.lengthSq() < 1e-5 || right.lengthSq() < 1e-5) return null;
    up.normalize();
    right.addScaledVector(up, -right.dot(up)).normalize();
    const forward = new THREE.Vector3().crossVectors(right, up);
    if (forward.lengthSq() < 1e-5) return null;
    forward.normalize();
    right.copy(new THREE.Vector3().crossVectors(up, forward).normalize());
    return { right, up, forward };
}

function computeLandmarkCorrectionMatrix() {
    const srcHips = uploadedLandmarkPoints.hips;
    const srcHead = uploadedLandmarkPoints.head;
    const srcLeft = uploadedLandmarkPoints.leftHand;
    const srcRight = uploadedLandmarkPoints.rightHand;
    const tgtHips = getCanonicalLandmark("hips");
    const tgtHead = getCanonicalLandmark("head");
    const tgtLeft = getCanonicalLandmark("leftHand");
    const tgtRight = getCanonicalLandmark("rightHand");
    if (!srcHips || !srcHead || !srcLeft || !srcRight || !tgtHips || !tgtHead || !tgtLeft || !tgtRight) {
        return null;
    }

    const srcBasis = buildHumanoidBasis(srcHips, srcHead, srcLeft, srcRight);
    const tgtBasis = buildHumanoidBasis(tgtHips, tgtHead, tgtLeft, tgtRight);
    if (!srcBasis || !tgtBasis) return null;

    const srcBasisMatrix = new THREE.Matrix4().makeBasis(srcBasis.right, srcBasis.up, srcBasis.forward);
    const tgtBasisMatrix = new THREE.Matrix4().makeBasis(tgtBasis.right, tgtBasis.up, tgtBasis.forward);
    const rotation = tgtBasisMatrix.clone().multiply(srcBasisMatrix.clone().invert());
    const torsoScale = tgtHead.distanceTo(tgtHips) / Math.max(srcHead.distanceTo(srcHips), 1e-4);
    const shoulderScale = tgtRight.distanceTo(tgtLeft) / Math.max(srcRight.distanceTo(srcLeft), 1e-4);
    const scale = Math.sqrt(torsoScale * shoulderScale);

    return new THREE.Matrix4()
        .makeTranslation(tgtHips.x, tgtHips.y, tgtHips.z)
        .multiply(rotation)
        .multiply(new THREE.Matrix4().makeScale(scale, scale, scale))
        .multiply(new THREE.Matrix4().makeTranslation(-srcHips.x, -srcHips.y, -srcHips.z));
}

function captureCanonicalReference() {
    canonicalMeshes = [];
    modelRoot.traverse((child) => {
        if (!child.isSkinnedMesh) return;
        child.frustumCulled = false;
        canonicalMeshes.push(child);
    });
    canonicalSkeleton = skinnedMesh ? skinnedMesh.skeleton : null;

    const baked = extractUploadGeometry(modelRoot);
    const points = samplePointsFromMeshes(baked, 2600);
    canonicalBounds = computeBoundsFromPoints(points);
    canonicalAnchorPoints = ["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "Hips"]
        .map((name) => canonicalRestByRole[name])
        .filter(Boolean)
        .map((matrix) => new THREE.Vector3().setFromMatrixPosition(matrix));

    const boneIndexByObject = new Map();
    canonicalSegments = [];
    if (canonicalSkeleton) {
        canonicalSkeleton.bones.forEach((bone, index) => boneIndexByObject.set(bone, index));
        for (const bone of canonicalSkeleton.bones) {
            const childIndex = boneIndexByObject.get(bone);
            const parent = bone.parent?.isBone ? bone.parent : null;
            const parentIndex = parent ? boneIndexByObject.get(parent) : childIndex;
            const start = parent
                ? new THREE.Vector3().setFromMatrixPosition(parent.matrixWorld)
                : new THREE.Vector3().setFromMatrixPosition(bone.matrixWorld);
            const end = new THREE.Vector3().setFromMatrixPosition(bone.matrixWorld);
            const lengthSq = Math.max(start.distanceToSquared(end), 1e-6);
            canonicalSegments.push({ parentIndex, childIndex, start, end, lengthSq });
        }
    }
}

function buildSkinAttributes(geometry, segments = canonicalSegments) {
    const position = geometry.getAttribute("position");
    const skinIndex = new Uint16Array(position.count * 4);
    const skinWeight = new Float32Array(position.count * 4);
    const vertex = new THREE.Vector3();
    const delta = new THREE.Vector3();
    for (let i = 0; i < position.count; i++) {
        vertex.fromBufferAttribute(position, i);
        const weightsByBone = new Map();
        for (const segment of segments) {
            delta.subVectors(segment.end, segment.start);
            const lengthSq = segment.lengthSq;
            let t = 0;
            if (lengthSq > 1e-6) {
                t = THREE.MathUtils.clamp(vertex.clone().sub(segment.start).dot(delta) / lengthSq, 0, 1);
            }
            const closest = segment.start.clone().lerp(segment.end, t);
            const distSq = Math.max(vertex.distanceToSquared(closest), 1e-4);
            const baseWeight = 1 / distSq;
            const parentWeight = baseWeight * (segment.parentIndex === segment.childIndex ? 0.0 : 1 - t);
            const childWeight = baseWeight * (segment.parentIndex === segment.childIndex ? 1.0 : t + 0.15);
            weightsByBone.set(segment.parentIndex, (weightsByBone.get(segment.parentIndex) || 0) + parentWeight);
            weightsByBone.set(segment.childIndex, (weightsByBone.get(segment.childIndex) || 0) + childWeight);
        }
        const topBones = [...weightsByBone.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
        const total = topBones.reduce((sum, [, weight]) => sum + weight, 0) || 1;
        for (let j = 0; j < topBones.length; j++) {
            skinIndex[i * 4 + j] = topBones[j][0];
            skinWeight[i * 4 + j] = topBones[j][1] / total;
        }
    }
    geometry.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute("skinWeight", new THREE.Float32BufferAttribute(skinWeight, 4));
}

function clearUploadedBinding({ revokeObjectUrl = false, resetSource = false } = {}) {
    if (uploadedGroup) {
        scene.remove(uploadedGroup);
        uploadedGroup = null;
    }
    uploadedRigRoot = null;
    uploadedRigSkeleton = null;
    uploadedRigBones = [];
    uploadedRigRestWorldByBone = new Map();
    uploadedRigRestLocalByBone = new Map();
    uploadedRigOriginalRestWorldByBone = new Map();
    uploadedRigOriginalRestLocalByBone = new Map();
    uploadedRigOriginalBoneInverses = [];
    uploadedRigRestParentWorldByBone = new Map();
    uploadedRigRetargetPairs = [];
    uploadedRigDriverBoneByName = new Map();
    uploadedRigContactLocalByName = new Map();
    uploadedRigHandleLocalByName = new Map();
    uploadedRigEditedHandleRoles = new Set();
    uploadedRigMode = null;
    if (revokeObjectUrl && activeObjectUrl) {
        URL.revokeObjectURL(activeObjectUrl);
        activeObjectUrl = null;
    }
    if (resetSource) {
        uploadedSourceMeshes = null;
        uploadedAutoFitMatrix = null;
        fitState = createDefaultFitState();
        activeLandmarkTarget = null;
        resetActiveTargetRest();
        uploadedLandmarkPoints.head = null;
        uploadedLandmarkPoints.hips = null;
        uploadedLandmarkPoints.leftHand = null;
        uploadedLandmarkPoints.rightHand = null;
        clearRigEditPoints();
    }
    updateFitUi();
    rebuildLandmarkMarkers();
    updateViewportCursor();
    setModelVisibility(meshVisible);
}

function bindUploadToCanonicalSkeleton(meshes, fitMatrix, objectUrl) {
    clearUploadedBinding();
    uploadedGroup = new THREE.Group();
    activeObjectUrl = objectUrl;
    for (const mesh of meshes) {
        const geometry = mesh.geometry.clone();
        geometry.applyMatrix4(fitMatrix);
        if (!geometry.getAttribute("normal")) {
            geometry.computeVertexNormals();
        }
        buildSkinAttributes(geometry);
        const skinned = new THREE.SkinnedMesh(geometry, cloneMaterial(mesh.material));
        skinned.castShadow = true;
        skinned.receiveShadow = true;
        skinned.frustumCulled = false;
        skinned.bindMode = "detached";
        skinned.bind(canonicalSkeleton, new THREE.Matrix4());
        uploadedGroup.add(skinned);
    }
    scene.add(uploadedGroup);
    setModelVisibility(meshVisible);
}

function rebuildUploadedBinding() {
    if (!uploadedSourceMeshes || !uploadedAutoFitMatrix) return;
    const fitMatrix = uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix());
    if (uploadedRigMode === "preserved") {
        uploadedRigRetargetPairs = buildUploadedRigRetargetPairs(fitMatrix);
        rebuildUploadedRigDriverLookup();
        restoreUploadedRigBoneInverses();
        if (!uploadedRigContactLocalByName.size) rebuildUploadedRigContactAnchors();
        if (uploadedGroup) {
            uploadedGroup.matrixAutoUpdate = false;
            uploadedGroup.matrix.copy(fitMatrix);
            uploadedGroup.matrixWorld.copy(fitMatrix);
            uploadedGroup.updateMatrixWorld(true);
        }
        applyPreservedUploadedRigPose();
        setModelVisibility(meshVisible);
        return;
    }
    bindUploadToCanonicalSkeleton(uploadedSourceMeshes, fitMatrix, activeObjectUrl);
}

function commitRigEditPointsToActiveTargetRest() {
    let changed = false;
    for (const role of rigEditRoles) {
        const point = rigEditPoints[role];
        const boneName = rigEditRoleToBone[role];
        const matrix = boneName ? (activeTargetRestByRole[boneName] || canonicalRestByRole[boneName]) : null;
        if (!point || !boneName || !matrix) continue;
        activeTargetRestByRole[boneName] = cloneMatrixWithPosition(matrix, point);
        changed = true;
    }
    return changed;
}

function commitRigEditPointsToPreservedRigHandles() {
    if (uploadedRigMode !== "preserved" || !uploadedRigBones.length) return false;
    let changed = false;
    const fitMatrix = uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix());

    for (const role of rigEditRoles) {
        if (!rigEditEditedRoles.has(role)) continue;
        const point = rigEditPoints[role];
        const boneName = rigEditRoleToBone[role];
        const bone = boneName ? uploadedRigDriverBoneByName.get(boneName) : null;
        if (!point || !bone) continue;
        const restWorld = uploadedRigRestWorldByBone.get(bone);
        if (!restWorld) continue;
        const restPoint = point.clone();
        const driverDelta = getCanonicalDriverDelta(boneName, _matA);
        if (driverDelta) restPoint.applyMatrix4(_matB.copy(driverDelta).invert());
        const restHandleMatrix = _matC.copy(fitMatrix).multiply(restWorld);
        const localOffset = restPoint.applyMatrix4(_matD.copy(restHandleMatrix).invert());
        uploadedRigHandleLocalByName.set(role, localOffset.clone());
        uploadedRigEditedHandleRoles.add(role);
        changed = true;
    }

    return changed;
}

function applyLandmarkCorrection() {
    if (!uploadedSourceMeshes || !uploadedAutoFitMatrix) return;
    const correction = computeLandmarkCorrectionMatrix();
    if (!correction) {
        setAssetStatus("Mark head, hips, left hand, and right hand to align", "warn");
        return;
    }
    const currentFit = uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix());
    uploadedAutoFitMatrix = currentFit.premultiply(correction);
    fitState = createDefaultFitState();
    activeLandmarkTarget = null;
    rigEditHasManualPoints = false;
    if (uploadedRigMode === "preserved") resetUploadedRigRestToOriginal();
    uploadedLandmarkPoints.head = null;
    uploadedLandmarkPoints.hips = null;
    uploadedLandmarkPoints.leftHand = null;
    uploadedLandmarkPoints.rightHand = null;
    updateFitUi();
    rebuildLandmarkMarkers();
    updateViewportCursor();
    rebuildUploadedBinding();
    setAssetStatus("Applied landmark alignment", "ok");
}

function applyRigEditAlignment() {
    if (!uploadedSourceMeshes || !uploadedAutoFitMatrix) return;
    if (!rigEditRoles.some((role) => rigEditPoints[role]) && !rigEditEditedRoles.size) {
        setAssetStatus("No skeleton edits to apply", "warn");
        return;
    }
    if (uploadedRigMode === "preserved") {
        if (!rigEditEditedRoles.size) {
            setAssetStatus("No skeleton edits to apply", "warn");
            return;
        }
        const updatedPreservedHandles = commitRigEditPointsToPreservedRigHandles();
        if (!updatedPreservedHandles) {
            setAssetStatus("No preserved rig bones matched the edited skeleton", "warn");
            return;
        }
        rebuildUploadedBinding();
        seedRigEditPointsFromMap(getRigEditSeedMap());
        rigEditHasManualPoints = false;
        rigEditEnabled = false;
        rigEditEditedRoles.clear();
        updateRigEditOverlay();
        updateFitUi();
        updateViewportCursor();
        setAssetStatus("Applied skeleton landmark edits to preserved rig", "ok");
        return;
    }
    const updatedTargetRest = commitRigEditPointsToActiveTargetRest();
    if (!updatedTargetRest && uploadedRigMode !== "preserved") {
        setAssetStatus("No skeleton edits to apply", "warn");
        return;
    }
    rebuildUploadedBinding();
    seedRigEditPointsFromMap(getRigEditSeedMap());
    rigEditHasManualPoints = false;
    rigEditEnabled = false;
    rigEditEditedRoles.clear();
    updateRigEditOverlay();
    updateFitUi();
    updateViewportCursor();
    setAssetStatus(uploadedRigMode === "preserved" ? "Applied edited skeleton to preserved rig" : "Applied edited skeleton", "ok");
}

function restoreAutoRigFit() {
    if (!uploadedSourceMeshes) return;
    rigEditHasManualPoints = false;
    resetActiveTargetRest();
    if (uploadedRigMode === "preserved") resetUploadedRigRestToOriginal();
    uploadedAutoFitMatrix = chooseBestFitMatrix(samplePointsFromMeshes(uploadedSourceMeshes));
    rebuildUploadedBinding();
    resetRigEditHandles();
    rigEditEnabled = false;
    updateFitUi();
    updateViewportCursor();
    setAssetStatus(uploadedRigMode === "preserved" ? "Restored automatic preserved skeleton alignment" : "Restored automatic skeleton alignment", "ok");
}

async function loadDefaultModel() {
    const gltf = await loader.loadAsync(resolveAppUrl(DEMO.modelPath));
    const model = gltf.scene;
    scene.add(model);
    modelRoot = model;
    boneMap = {};
    activeTargetRestByRole = {};
    clearMatrixMap(canonicalRestByRole);
    clearMatrixMap(canonicalRestInverseByRole);
    clearMatrixMap(canonicalLocalRestByRole);
    skinnedMesh = null;
    canonicalTextureTargets = [];
    model.updateMatrixWorld(true);
    model.traverse((child) => {
        if (child.isMesh) {
            canonicalTextureTargets.push(child);
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.isSkinnedMesh && !skinnedMesh) {
                skinnedMesh = child;
                child.frustumCulled = false;
            }
        }
        if (child.isBone) {
            boneMap[child.name] = child;
            child.matrixAutoUpdate = false;
            child.matrixWorldAutoUpdate = false;
            const restMatrix = child.matrixWorld.clone();
            canonicalRestByRole[child.name] = restMatrix.clone();
            canonicalRestInverseByRole[child.name] = restMatrix.clone().invert();
            activeTargetRestByRole[child.name] = restMatrix.clone();
        }
    });
    rebuildCanonicalLocalRest();
    captureCanonicalMaterials();
    captureCanonicalReference();
    rigEditEnabled = false;
    resetRigEditHandles();
    setModelVisibility(meshVisible);
    setAssetStatus("Built-in Geno body · GLB upload ready", "ok");
}

async function loadUploadedModel(file) {
    if (!canonicalSkeleton || !canonicalBounds) {
        setAssetStatus("Canonical humanoid rig is still loading", "warn");
        return;
    }
    clearUploadedBinding({ revokeObjectUrl: true, resetSource: true });
    const objectUrl = URL.createObjectURL(file);
    setAssetStatus(`Loading ${file.name}...`, "warn");
    try {
        const gltf = await loader.loadAsync(objectUrl);
        const sceneRoot = gltf.scene;
        const bakedMeshes = extractUploadGeometry(sceneRoot);
        if (!bakedMeshes.length) {
            URL.revokeObjectURL(objectUrl);
            setAssetStatus(`${file.name} rejected · no mesh geometry found`, "error");
            return;
        }
        uploadedSourceMeshes = bakedMeshes;
        uploadedAutoFitMatrix = chooseBestFitMatrix(samplePointsFromMeshes(bakedMeshes));
        fitState = createDefaultFitState();
        resetActiveTargetRest();
        rigEditHasManualPoints = false;
        rigEditEnabled = false;
        updateFitUi();
        activeObjectUrl = objectUrl;
        let preservedRig = false;
        sceneRoot.updateMatrixWorld(true);
        sceneRoot.traverse((child) => {
            if (child.isSkinnedMesh && child.skeleton) preservedRig = true;
        });
        if (preservedRig) {
            uploadedGroup = new THREE.Group();
            uploadedGroup.matrixAutoUpdate = false;
            uploadedRigRoot = sceneRoot;
            uploadedGroup.add(uploadedRigRoot);
            scene.add(uploadedGroup);
            uploadedRigMode = "preserved";
            captureUploadedRigReference();
            rebuildUploadedBinding();
            setAssetStatus(`${file.name} loaded · driving preserved uploaded rig`, "ok");
        } else {
            rebuildUploadedBinding();
            setAssetStatus(`${file.name} loaded · no rig detected · fallback rebinding to canonical humanoid rig`, "warn");
        }
        resetRigEditHandles();
        updateFitUi();
    } catch (error) {
        URL.revokeObjectURL(objectUrl);
        setAssetStatus(`${file.name} failed to load`, "error");
        console.error("Failed to load uploaded GLB:", error);
    }
}

function handleImportedAsset(file) {
    if (!file) return;
    resetExamplePicker();
    const name = (file.name || "").toLowerCase();
    if (name.endsWith(".glb") || name.endsWith(".gltf") || file.type === "model/gltf-binary" || file.type === "model/gltf+json") {
        return loadUploadedModel(file);
    }
    setAssetStatus(`${file.name} rejected · import a GLB from Texturizer`, "error");
}

function setExamplePickerLabel(label) {
    if (exampleButtonValue) exampleButtonValue.textContent = label;
}

function setExamplePickerDisabled(disabled) {
    if (exampleButton) exampleButton.disabled = disabled;
    if (examplePicker) examplePicker.classList.toggle("is-disabled", disabled);
}

function closeExampleMenu() {
    if (!examplePicker || !exampleButton || !exampleMenu) return;
    examplePicker.classList.remove("is-open");
    exampleButton.setAttribute("aria-expanded", "false");
    exampleMenu.hidden = true;
}

function openExampleMenu() {
    if (!examplePicker || !exampleButton || !exampleMenu || exampleButton.disabled || !exampleAssets.length) return;
    examplePicker.classList.add("is-open");
    exampleButton.setAttribute("aria-expanded", "true");
    exampleMenu.hidden = false;
}

function focusExampleOption(offset = 0) {
    if (!exampleMenu) return;
    const options = Array.from(exampleMenu.querySelectorAll(".example-option"));
    if (!options.length) return;
    const activeIndex = options.indexOf(document.activeElement);
    const nextIndex = activeIndex < 0 ? 0 : (activeIndex + offset + options.length) % options.length;
    options[nextIndex].focus();
}

function resetExamplePicker() {
    selectedExampleUrl = "";
    setExamplePickerLabel(exampleAssets.length ? "Choose example" : "No examples");
    closeExampleMenu();
}

function replaceExampleOptions(label, items = []) {
    if (!exampleButton || !exampleMenu) return;
    selectedExampleUrl = "";
    setExamplePickerLabel(label);
    exampleMenu.innerHTML = "";

    items.forEach((item) => {
        const option = document.createElement("button");
        option.type = "button";
        option.className = "example-option";
        option.setAttribute("role", "option");
        option.setAttribute("aria-selected", "false");
        option.textContent = item.name || item.fileName;
        option.addEventListener("click", async () => {
            selectedExampleUrl = item.url;
            setExamplePickerLabel(item.name || item.fileName);
            for (const sibling of exampleMenu.querySelectorAll(".example-option")) {
                sibling.classList.toggle("is-selected", sibling === option);
                sibling.setAttribute("aria-selected", sibling === option ? "true" : "false");
            }
            closeExampleMenu();
            await loadExampleAsset(item);
        });
        exampleMenu.appendChild(option);
    });

    setExamplePickerDisabled(items.length === 0);
    closeExampleMenu();
}

async function loadExampleManifest() {
    if (!exampleButton || !exampleMenu) return;
    replaceExampleOptions("Loading examples...");
    try {
        const response = await fetch(resolveAppUrl(`/api/rigged-mesh/${EXAMPLE_KIND}`), { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        exampleAssets = Array.isArray(payload.items) ? payload.items : [];
        replaceExampleOptions(exampleAssets.length ? "Choose example" : "No examples", exampleAssets);
    } catch (error) {
        exampleAssets = [];
        replaceExampleOptions("Examples unavailable");
        console.error("Failed to load human mesh examples:", error);
    }
}

async function loadExampleAsset(example) {
    if (!example) return;
    const token = ++exampleLoadToken;
    setExamplePickerDisabled(true);
    setAssetStatus(`Fetching ${example.name || example.fileName}...`, "warn");
    try {
        const response = await fetch(resolveAppUrl(example.url), { cache: "force-cache" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        if (token !== exampleLoadToken) return;
        const file = new File(
            [blob],
            example.fileName || `${example.name || "example"}.glb`,
            { type: blob.type || "model/gltf-binary" },
        );
        await loadUploadedModel(file);
    } catch (error) {
        setAssetStatus(`${example.name || example.fileName} unavailable`, "error");
        console.error("Failed to load human mesh example:", error);
    } finally {
        if (token === exampleLoadToken) {
            setExamplePickerDisabled(exampleAssets.length === 0);
        }
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
    if (controlMode === "path") {
        updateInputVisualizerPath();
        return;
    }
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
        setModelVisibility(!meshVisible);
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
    if (controlMode === "path") updateInputVisualizerPath();
    else updateInputVisualizerKeyboard();
}

window.addEventListener("blur", clearKeyboardState);
window.addEventListener("focus", clearKeyboardState);
document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearKeyboardState();
});

const canvas = renderer.domElement;
canvas.addEventListener("pointerdown", (event) => {
    rigEditInputDebug.pointerDowns += 1;
    if (activeLandmarkTarget && uploadedGroup) {
        const rect = canvas.getBoundingClientRect();
        landmarkPointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        landmarkPointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        landmarkRaycaster.setFromCamera(landmarkPointer, camera);
        const hit = landmarkRaycaster.intersectObject(uploadedGroup, true).find((entry) => entry.object.isMesh);
        if (!hit) return;
        uploadedLandmarkPoints[activeLandmarkTarget] = hit.point.clone();
        activeLandmarkTarget = null;
        rebuildLandmarkMarkers();
        updateViewportCursor();
        setAssetStatus("Landmark captured", "ok");
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
    }
    if (beginRigEditPointerDrag(event, "pointer")) {
        rigEditPointerId = event.pointerId;
        if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId);
        event.preventDefault();
        return;
    }
});

canvas.addEventListener("pointermove", (event) => {
    if (rigEditPointerId === null || event.pointerId !== rigEditPointerId) return;
    rigEditInputDebug.pointerMoves += 1;
    if (updateRigEditPointerDrag(event, "pointer")) event.preventDefault();
});

function releaseRigEditPointer(event, source) {
    if (rigEditPointerId === null || event.pointerId !== rigEditPointerId) return;
    rigEditInputDebug.pointerUps += 1;
    endRigEditPointerDrag(source);
    if (rigEditPointerId !== null && canvas.releasePointerCapture) {
        try {
            canvas.releasePointerCapture(rigEditPointerId);
        } catch (_error) {
            // The browser can release capture before pointercancel reaches us.
        }
    }
    rigEditPointerId = null;
    event.preventDefault();
}

canvas.addEventListener("pointerup", (event) => releaseRigEditPointer(event, "pointer"));
canvas.addEventListener("pointercancel", (event) => releaseRigEditPointer(event, "pointercancel"));

function pickRigHandle(event) {
    if (!uploadedSourceMeshes || !rigEditEnabled || !rigEditHandleMap.size) return null;
    const rect = canvas.getBoundingClientRect();
    landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
    landmarkRaycaster.setFromCamera(landmarkPointer, camera);
    const handles = [...rigEditHandleMap.values()];
    const hit = landmarkRaycaster.intersectObjects(handles, false)[0];
    if (hit) {
        rigEditInputDebug.lastPickMethod = "raycast-handle";
        rigEditInputDebug.lastPickDistance = hit.distance;
        return hit.object.userData.rigRole;
    }

    return pickScreenSkeletonRole(event, {
        source: "edit",
        jointRadius: rigEditScreenPickRadius.joint,
        segmentRadius: rigEditScreenPickRadius.segment,
        methodPrefix: "edit",
    });
}

function pickVisibleSkeletonHandle(event) {
    if (!uploadedSourceMeshes || rigEditEnabled || !debugEnabled) return null;
    return pickScreenSkeletonRole(event, {
        source: "debug",
        jointRadius: rigEditVisibleScreenPickRadius.joint,
        segmentRadius: rigEditVisibleScreenPickRadius.segment,
        methodPrefix: "visible",
    });
}

function distanceToSegmentSq(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const lengthSq = dx * dx + dy * dy;
    if (lengthSq < 1e-6) return (px - ax) ** 2 + (py - ay) ** 2;
    const t = THREE.MathUtils.clamp(((px - ax) * dx + (py - ay) * dy) / lengthSq, 0, 1);
    const x = ax + dx * t;
    const y = ay + dy * t;
    return (px - x) ** 2 + (py - y) ** 2;
}

function getRoleScreenPoints(source, rect) {
    const points = new Map();
    const visibleHandleMap = source === "debug" ? getVisibleSkeletonHandleMap() : null;
    for (const role of rigEditRoles) {
        let world = null;
        if (source === "edit") {
            const joint = rigEditJointMap.get(role);
            if (joint?.visible) world = joint.position;
        } else {
            world = getVisibleSkeletonBonePosition(role, new THREE.Vector3(), visibleHandleMap);
        }
        if (!world) continue;
        const projected = world.clone().project(camera);
        if (projected.z < -1 || projected.z > 1) continue;
        points.set(role, {
            x: (projected.x * 0.5 + 0.5) * rect.width + rect.left,
            y: (-projected.y * 0.5 + 0.5) * rect.height + rect.top,
        });
    }
    return points;
}

function pickScreenSkeletonRole(event, { source, jointRadius, segmentRadius, methodPrefix }) {
    const rect = canvas.getBoundingClientRect();
    const screenPoints = getRoleScreenPoints(source, rect);
    let bestRole = null;
    let bestDistanceSq = jointRadius * jointRadius;
    let bestMethod = null;
    for (const [role, point] of screenPoints) {
        const distanceSq = (event.clientX - point.x) ** 2 + (event.clientY - point.y) ** 2;
        if (distanceSq < bestDistanceSq) {
            bestDistanceSq = distanceSq;
            bestRole = role;
            bestMethod = `${methodPrefix}-screen-joint`;
        }
    }
    const segmentLimitSq = segmentRadius * segmentRadius;
    for (const [fromIndex, toIndex] of BONE_PAIRS) {
        const fromRole = BONE_NAMES[fromIndex];
        const toRole = BONE_NAMES[toIndex];
        const fromPoint = screenPoints.get(fromRole);
        const toPoint = screenPoints.get(toRole);
        if (!fromPoint || !toPoint) continue;
        const distanceSq = distanceToSegmentSq(event.clientX, event.clientY, fromPoint.x, fromPoint.y, toPoint.x, toPoint.y);
        if (distanceSq >= bestDistanceSq || distanceSq >= segmentLimitSq) continue;
        const fromDistanceSq = (event.clientX - fromPoint.x) ** 2 + (event.clientY - fromPoint.y) ** 2;
        const toDistanceSq = (event.clientX - toPoint.x) ** 2 + (event.clientY - toPoint.y) ** 2;
        bestDistanceSq = distanceSq;
        bestRole = toDistanceSq <= fromDistanceSq ? toRole : fromRole;
        bestMethod = `${methodPrefix}-screen-segment`;
    }
    rigEditInputDebug.lastPickMethod = bestMethod;
    rigEditInputDebug.lastPickDistance = bestRole ? Math.sqrt(bestDistanceSq) : null;
    return bestRole;
}

function rememberRigInputEvent(event, source, name) {
    rigEditInputDebug.lastEvent = name;
    rigEditInputDebug.lastSource = source;
    rigEditInputDebug.lastClientX = event?.clientX ?? null;
    rigEditInputDebug.lastClientY = event?.clientY ?? null;
}

function beginRigEditPointerDrag(event, source) {
    if (event.button !== 0 || activeLandmarkTarget) {
        rigEditInputDebug.lastBlockedReason = event.button !== 0 ? "non-primary-button" : "landmark-target-active";
        return false;
    }
    rememberRigInputEvent(event, source, `${source}:down`);
    let pickedRole = pickRigHandle(event);
    let pickedFromVisibleSkeleton = false;
    if (!pickedRole) {
        const visibleRole = pickVisibleSkeletonHandle(event);
        if (visibleRole && beginSkeletonEdit()) {
            pickedRole = visibleRole;
            pickedFromVisibleSkeleton = true;
            setAssetStatus("Skeleton editing enabled", "ok");
        }
    }
    rigEditInputDebug.lastPickedRole = pickedRole || null;
    if (!pickedRole) {
        rigEditInputDebug.lastBlockedReason = "no-rig-handle-picked";
        return false;
    }

    activeRigHandle = pickedRole;
    const rect = canvas.getBoundingClientRect();
    landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
    landmarkRaycaster.setFromCamera(landmarkPointer, camera);
    camera.getWorldDirection(rigDragNormal).normalize();
    const handleMesh = rigEditHandleMap.get(pickedRole);
    let handleWorld = null;
    if (!pickedFromVisibleSkeleton && handleMesh?.visible) {
        handleWorld = handleMesh.position.clone();
    } else {
        handleWorld = getRigHandlePosition(pickedRole, new THREE.Vector3()) || currentRootPos.clone();
    }
    rigEditPoints[pickedRole] = handleWorld.clone();
    rigDragPlane.setFromNormalAndCoplanarPoint(rigDragNormal, handleWorld);
    if (landmarkRaycaster.ray.intersectPlane(rigDragPlane, rigDragHit)) {
        rigDragOffset.copy(handleWorld).sub(rigDragHit);
    } else {
        rigDragOffset.set(0, 0, 0);
    }
    beginRigEditDrag(pickedRole, handleWorld);
    rigEditInputDebug.lastActiveRole = activeRigHandle;
    rigEditInputDebug.lastBlockedReason = null;
    updateViewportCursor();
    return true;
}

function updateRigEditPointerDrag(event, source) {
    if (!activeRigHandle) return false;
    rememberRigInputEvent(event, source, `${source}:move`);
    const rect = canvas.getBoundingClientRect();
    landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
    landmarkRaycaster.setFromCamera(landmarkPointer, camera);
    if (landmarkRaycaster.ray.intersectPlane(rigDragPlane, rigDragHit)) {
        const desiredWorldPosition = rigDragHit.clone().add(rigDragOffset);
        if (updateRigEditDrag(activeRigHandle, desiredWorldPosition)) {
            rigEditHasManualPoints = true;
            rigEditInputDebug.dragUpdates += 1;
            rigEditInputDebug.lastActiveRole = activeRigHandle;
            updateRigEditOverlay();
        }
    }
    return true;
}

function endRigEditPointerDrag(source) {
    if (!activeRigHandle) return false;
    rigEditInputDebug.lastEvent = `${source}:up`;
    activeRigHandle = null;
    rigEditInputDebug.lastActiveRole = null;
    clearRigDragTracking();
    updateViewportCursor();
    return true;
}

canvas.addEventListener("mousedown", (e) => {
    rigEditInputDebug.mouseDowns += 1;
    if (e.button === 2 && controlMode === "manual") {
        rightMouseDown = true;
        facingDragButton = e.button;
        directionMouseStart = null;
        e.preventDefault();
        return;
    }
    if (e.button === 0 && beginRigEditPointerDrag(e, "mouse")) {
        e.preventDefault();
        return;
    }
    if (e.button === 0 && !activeLandmarkTarget) {
        if (controlMode === "path" && beginPathPlacement(e)) {
            pathDragActive = true;
            updateViewportCursor();
            e.preventDefault();
            return;
        }
        if (controlMode === "manual" && pickFacingControl(e)) {
            activeFacingDrag = true;
            updateFacingFromPointer(e);
            updateViewportCursor();
            e.preventDefault();
            return;
        }
        isOrbitDragging = true;
        orbitPointerId = null;
        orbitLastX = e.clientX;
        orbitLastY = e.clientY;
        updateViewportCursor();
        e.preventDefault();
        return;
    }
});
canvas.addEventListener("mouseup", (e) => {
    if (e.button === 0) {
        rigEditInputDebug.mouseUps += 1;
        endRigEditPointerDrag("mouse");
        activeFacingDrag = false;
        isOrbitDragging = false;
        pathDragActive = false;
        orbitPointerId = null;
        updateViewportCursor();
    }
    if (facingDragButton === e.button) {
        rightMouseDown = false;
        facingDragButton = null;
        directionMouseStart = null;
        rightStick = [0, 0];
    }
});
canvas.addEventListener("mousemove", (e) => {
    if (activeRigHandle) {
        rigEditInputDebug.mouseMoves += 1;
        updateRigEditPointerDrag(e, "mouse");
        return;
    }
    if (activeFacingDrag) {
        updateFacingFromPointer(e);
        return;
    }
    if (pathDragActive) {
        handlePathPointer(e);
        e.preventDefault();
        return;
    }
    if (isOrbitDragging) {
        const dx = e.clientX - orbitLastX;
        const dy = e.clientY - orbitLastY;
        orbitLastX = e.clientX;
        orbitLastY = e.clientY;
        cameraTheta -= dx * 0.008;
        cameraPhi = THREE.MathUtils.clamp(cameraPhi + dy * 0.006, -1.2, 1.1);
        return;
    }
    if (!rightMouseDown || controlMode !== "manual") return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / Math.max(rect.width, 1);
    const y = (e.clientY - rect.top) / Math.max(rect.height, 1);
    const pos = [x, y];
    if (!directionMouseStart) directionMouseStart = [...pos];
    else {
        directionMouseStart[0] += (pos[0] - directionMouseStart[0]) * DIRECTION_MOMENTUM;
        directionMouseStart[1] += (pos[1] - directionMouseStart[1]) * DIRECTION_MOMENTUM;
    }
    rightStick = [pos[0] - directionMouseStart[0], directionMouseStart[1] - pos[1]];
});
canvas.addEventListener("mouseleave", () => {
    if (rigEditPointerId !== null) return;
    activeRigHandle = null;
    clearRigDragTracking();
    activeFacingDrag = false;
    isOrbitDragging = false;
    pathDragActive = false;
    orbitPointerId = null;
    rightMouseDown = false;
    facingDragButton = null;
    directionMouseStart = null;
    rightStick = [0, 0];
    updateViewportCursor();
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
    if (controlMode === "path") return getPathInput();
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
    applyPreservedUploadedRigPose();
    updateRigEditOverlay();
    if (!debugEnabled) return;
    const visibleHandleMap = getVisibleSkeletonHandleMap();
    const skeletonOnlyOverlay = !!visibleHandleMap;
    axisX.visible = !skeletonOnlyOverlay;
    axisY.visible = !skeletonOnlyOverlay;
    axisZ.visible = !skeletonOnlyOverlay;
    ctrlTrajGroup.visible = !skeletonOnlyOverlay;

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
            const posA = getVisibleSkeletonBonePosition(entityNames[idxA], _debugPosA, visibleHandleMap);
            const posB = getVisibleSkeletonBonePosition(entityNames[idxB], _debugPosB, visibleHandleMap);
            if (posA && posB) {
                skelAttr.setXYZ(p * 2, posA.x, posA.y, posA.z);
                skelAttr.setXYZ(p * 2 + 1, posB.x, posB.y, posB.z);
            }
        }
    }
    skelAttr.needsUpdate = true;

    for (let i = 0; i < entityCount && i < jointSpheres.length; i++) {
        const entityName = entityNames[i];
        if (skeletonOnlyOverlay && !rigEditRoleSet.has(entityName)) {
            jointSpheres[i].visible = false;
            continue;
        }
        const pos = getVisibleSkeletonBonePosition(entityName, _debugPosA, visibleHandleMap);
        jointSpheres[i].visible = !!pos;
        if (pos) jointSpheres[i].position.copy(pos);
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
        if (skeletonOnlyOverlay) {
            contactSpheres[i].visible = false;
            continue;
        }
        const idx = contactEntityIndices[i];
        if (idx >= 0) {
            const pos = getContactMarkerPosition(entityNames[idx], _debugPosA);
            if (pos) {
                contactSpheres[i].visible = true;
                contactSpheres[i].position.copy(pos);
                const contact = framePrev ? THREE.MathUtils.lerp(framePrev.contacts[i], frameCurr.contacts[i], alpha) : frameCurr.contacts[i];
                contactSpheres[i].material.color.setRGB(1 - contact, contact, 0);
            } else {
                contactSpheres[i].visible = false;
            }
        } else {
            contactSpheres[i].visible = false;
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
    updateFacingGizmo();
    if (pathDisplayVisible && pathWaypoints.length > 0) updatePathLineGeometry();
    updateFpsDisplay();
    renderer.render(scene, camera);
    sendInput(timestamp);
}

setDebugEnabled(true);
refreshInputDeviceHud();
requestAnimationFrame(() => refreshInputDeviceHud());
updateViewportCursor();
animate(0);

if (importButton && importInput) {
    importButton.addEventListener("click", () => importInput.click());
    importInput.addEventListener("change", async (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        await handleImportedAsset(file);
        importInput.value = "";
    });
}

if (restoreButton) {
    restoreButton.addEventListener("click", () => {
        resetExamplePicker();
        restoreCanonicalSurface();
    });
}

if (exampleButton && exampleMenu) {
    exampleButton.addEventListener("click", () => {
        if (exampleMenu.hidden) openExampleMenu();
        else closeExampleMenu();
    });
    exampleButton.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowDown" && event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        openExampleMenu();
        focusExampleOption(0);
    });
    exampleMenu.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeExampleMenu();
            exampleButton.focus();
            return;
        }
        if (event.key === "ArrowDown") {
            event.preventDefault();
            focusExampleOption(1);
        }
        if (event.key === "ArrowUp") {
            event.preventDefault();
            focusExampleOption(-1);
        }
    });
    document.addEventListener("click", (event) => {
        if (examplePicker && !examplePicker.contains(event.target)) closeExampleMenu();
    });
    loadExampleManifest();
}

if (fitEditButton) {
    fitEditButton.addEventListener("click", () => {
        if (!uploadedSourceMeshes) return;
        if (rigEditEnabled) {
            rigEditEnabled = false;
            activeRigHandle = null;
            clearRigDragTracking();
            updateRigEditOverlay();
            updateFitUi();
            updateViewportCursor();
            setAssetStatus("Skeleton editing disabled", "ok");
            return;
        }
        beginSkeletonEdit();
        setAssetStatus("Skeleton editing enabled", "ok");
    });
}

if (fitSkeletonButton) {
    fitSkeletonButton.addEventListener("click", () => {
        setDebugEnabled(!debugEnabled);
        setAssetStatus(debugEnabled ? "Skeleton overlay enabled" : "Skeleton overlay hidden", "ok");
    });
}

if (fitApplyButton) {
    fitApplyButton.addEventListener("click", () => applyRigEditAlignment());
}

if (fitAutoButton) {
    fitAutoButton.addEventListener("click", () => restoreAutoRigFit());
}

function wireFitControl(input, key) {
    if (!input) return;
    input.addEventListener("input", () => {
        fitState[key] = parseFloat(input.value);
        updateFitUi();
        rebuildUploadedBinding();
    });
}

wireFitControl(fitYawInput, "yawDegrees");
wireFitControl(fitPitchInput, "pitchDegrees");
wireFitControl(fitRollInput, "rollDegrees");
wireFitControl(fitScaleInput, "scale");
wireFitControl(fitWidthInput, "width");
wireFitControl(fitLengthInput, "length");
wireFitControl(fitLiftInput, "lift");
wireFitControl(fitForwardInput, "forward");

function armLandmarkTarget(name, label) {
    activeLandmarkTarget = activeLandmarkTarget === name ? null : name;
    updateLandmarkButtons();
    updateViewportCursor();
    if (activeLandmarkTarget) {
        setAssetStatus(`Click the uploaded mesh to mark ${label}`, "warn");
    }
}

if (markHeadButton) markHeadButton.addEventListener("click", () => armLandmarkTarget("head", "head"));
if (markHipsButton) markHipsButton.addEventListener("click", () => armLandmarkTarget("hips", "hips"));
if (markLeftHandButton) markLeftHandButton.addEventListener("click", () => armLandmarkTarget("leftHand", "left hand"));
if (markRightHandButton) markRightHandButton.addEventListener("click", () => armLandmarkTarget("rightHand", "right hand"));
if (applyMarksButton) applyMarksButton.addEventListener("click", () => applyLandmarkCorrection());
if (clearMarksButton) {
    clearMarksButton.addEventListener("click", () => {
        activeLandmarkTarget = null;
        uploadedLandmarkPoints.head = null;
        uploadedLandmarkPoints.hips = null;
        uploadedLandmarkPoints.leftHand = null;
        uploadedLandmarkPoints.rightHand = null;
        rebuildLandmarkMarkers();
        updateViewportCursor();
        setAssetStatus("Cleared landmark marks", "ok");
    });
}

if (fitYawLeftButton) {
    fitYawLeftButton.addEventListener("click", () => {
        fitState.yawDegrees = normalizeYawDegrees(fitState.yawDegrees - 90);
        updateFitUi();
        rebuildUploadedBinding();
    });
}

if (fitYawRightButton) {
    fitYawRightButton.addEventListener("click", () => {
        fitState.yawDegrees = normalizeYawDegrees(fitState.yawDegrees + 90);
        updateFitUi();
        rebuildUploadedBinding();
    });
}

if (fitFlipButton) {
    fitFlipButton.addEventListener("click", () => {
        fitState.yawDegrees = normalizeYawDegrees(fitState.yawDegrees + 180);
        updateFitUi();
        rebuildUploadedBinding();
    });
}

if (fitMirrorButton) {
    fitMirrorButton.addEventListener("click", () => {
        fitState.mirror = !fitState.mirror;
        updateFitUi();
        rebuildUploadedBinding();
    });
}

if (fitResetButton) {
    fitResetButton.addEventListener("click", () => {
        resetActiveTargetRest();
        if (uploadedRigMode === "preserved") resetUploadedRigRestToOriginal();
        rebuildUploadedBinding();
        resetRigEditHandles();
        beginSkeletonEdit({ reseed: true });
        updateFitUi();
        updateViewportCursor();
        setAssetStatus("Reset skeleton edits", "ok");
    });
}

if (directionRingToggle) {
    directionRingToggle.checked = directionRingVisible;
    directionRingToggle.addEventListener("change", () => setDirectionRingVisible(directionRingToggle.checked));
}

if (modeManualButton) modeManualButton.addEventListener("click", () => setControlMode("manual"));
if (modePathButton) modePathButton.addEventListener("click", () => setControlMode("path"));
if (pathPlayButton) pathPlayButton.addEventListener("click", () => togglePathPlayback());
if (pathClearButton) pathClearButton.addEventListener("click", () => clearPath());
if (pathLoopToggle) {
    pathLoopToggle.addEventListener("change", () => {
        pathLoop = pathLoopToggle.checked;
        updatePathUi();
    });
}
if (pathDisplayToggle) {
    pathDisplayToggle.checked = pathDisplayVisible;
    pathDisplayToggle.addEventListener("change", () => setPathDisplayVisible(pathDisplayToggle.checked));
}
if (pathQueueToggle) {
    pathQueueToggle.checked = pathQueueWaypoints;
    pathQueueToggle.addEventListener("change", () => {
        pathQueueWaypoints = pathQueueToggle.checked;
        updatePathUi();
    });
}
setControlMode("path");

window.addEventListener("resize", () => {
    resizeViewport();
});

window.__ai4aBipedDebug = {
    getFacingHandleScreen() {
        const dir = getFacingDisplayDirection();
        const radius = 0.78;
        const point = new THREE.Vector3(
            currentRootPos.x + dir.x * radius,
            Math.max(0.04, currentRootPos.y + 0.04),
            currentRootPos.z + dir.y * radius
        ).project(camera);
        return {
            x: (point.x * 0.5 + 0.5) * renderer.domElement.clientWidth,
            y: (-point.y * 0.5 + 0.5) * renderer.domElement.clientHeight,
        };
    },
    getRightStick() {
        return [...rightStick];
    },
    getPathState() {
        return {
            controlMode,
            active: pathActive,
            loop: pathLoop,
            displayVisible: pathDisplayVisible,
            queueWaypoints: pathQueueWaypoints,
            visualsVisible: pathGroup.visible,
            waypointIndex: pathWaypointIndex,
            waypointCount: pathWaypoints.length,
        };
    },
    getRigEditState() {
        const pointSnapshot = {};
        const restSnapshot = getRigEditSeedMap();
        const liveSnapshot = getLiveRigHandleMap();
        for (const role of rigEditRoles) {
            const point = rigEditPoints[role];
            const rest = restSnapshot[role];
            const live = liveSnapshot[role];
            pointSnapshot[role] = point ? { x: point.x, y: point.y, z: point.z } : null;
            pointSnapshot[`${role}Rest`] = rest ? { x: rest.x, y: rest.y, z: rest.z } : null;
            pointSnapshot[`${role}Live`] = live ? { x: live.x, y: live.y, z: live.z } : null;
        }
        return {
            enabled: rigEditEnabled,
            hasManualPoints: rigEditHasManualPoints,
            editedRoles: [...rigEditEditedRoles],
            appliedControlRoles: [...uploadedRigEditedHandleRoles],
            uploadedRigMode,
            assetStatus: assetStatus ? assetStatus.textContent : "",
            points: pointSnapshot,
        };
    },
    getRigEditInputDebug() {
        return { ...rigEditInputDebug, pointerId: rigEditPointerId };
    },
    getSkeletonConsistency() {
        const visible = getVisibleSkeletonHandleMap() || {};
        const deltas = {};
        let maxDelta = 0;
        for (const role of rigEditRoles) {
            const shown = visible[role];
            const live = getRigHandlePosition(role, new THREE.Vector3());
            if (!shown || !live) {
                deltas[role] = null;
                continue;
            }
            const delta = shown.distanceTo(live);
            deltas[role] = delta;
            maxDelta = Math.max(maxDelta, delta);
        }
        return {
            uploadedRigMode,
            liveOverlay: uploadedRigMode === "preserved" && !!uploadedSourceMeshes,
            maxDelta,
            deltas,
        };
    },
    getPreservedRigInvariants() {
        let boneInverseMaxDelta = 0;
        if (uploadedRigSkeleton) {
            for (let index = 0; index < uploadedRigSkeleton.boneInverses.length; index++) {
                boneInverseMaxDelta = Math.max(
                    boneInverseMaxDelta,
                    maxMatrixAbsDelta(uploadedRigSkeleton.boneInverses[index], uploadedRigOriginalBoneInverses[index])
                );
            }
        }
        const fitMatrix = uploadedAutoFitMatrix ? uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix()) : null;
        const controlErrors = {};
        let controlMaxDelta = 0;
        if (fitMatrix) {
            for (const role of uploadedRigEditedHandleRoles) {
                const desired = getDesiredPreservedRigHandleWorld(role, fitMatrix, new THREE.Vector3());
                const actual = getRigHandlePosition(role, new THREE.Vector3());
                const delta = desired && actual ? desired.distanceTo(actual) : null;
                controlErrors[role] = delta;
                if (delta !== null) controlMaxDelta = Math.max(controlMaxDelta, delta);
            }
        }
        return {
            uploadedRigMode,
            editedControlRoles: [...uploadedRigEditedHandleRoles],
            boneInverseMaxDelta,
            controlMaxDelta,
            controlErrors,
        };
    },
    getUploadedDriverMap() {
        return Object.fromEntries(rigEditRoles.map((role) => {
            const bone = uploadedRigDriverBoneByName.get(role);
            return [role, bone ? bone.name : null];
        }));
    },
    getRigBoneWorld(role) {
        const point = getRigHandlePosition(role, new THREE.Vector3());
        return point ? { x: point.x, y: point.y, z: point.z } : null;
    },
    getRigBoneScreen(role) {
        const point = getRigHandlePosition(role, new THREE.Vector3());
        if (!point) return null;
        const projected = point.project(camera);
        return {
            x: (projected.x * 0.5 + 0.5) * renderer.domElement.clientWidth,
            y: (-projected.y * 0.5 + 0.5) * renderer.domElement.clientHeight,
        };
    },
    getRigEditHandleScreen(role) {
        const handle = rigEditJointMap.get(role);
        if (!handle || !handle.visible) return null;
        const point = handle.position.clone().project(camera);
        return {
            x: (point.x * 0.5 + 0.5) * renderer.domElement.clientWidth,
            y: (-point.y * 0.5 + 0.5) * renderer.domElement.clientHeight,
        };
    },
    getRigEditHandleVisuals(role) {
        const visibleHandle = rigEditJointMap.get(role);
        const pickHandle = rigEditHandleMap.get(role);
        return {
            visibleRadius: visibleHandle?.geometry?.parameters?.radius ?? null,
            visibleOpacity: visibleHandle?.material?.opacity ?? null,
            pickRadius: pickHandle?.geometry?.parameters?.radius ?? null,
            pickOpacity: pickHandle?.material?.opacity ?? null,
            pickVisible: !!pickHandle?.visible,
        };
    },
};

loadDefaultModel()
    .then(() => connectWebSocket())
    .catch((error) => {
        setAssetStatus("Failed to load built-in Model.glb", "error");
        console.error("Failed to load default model:", error);
        connectWebSocket();
    });
