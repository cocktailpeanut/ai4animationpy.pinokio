import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const DEMO = {
    title: "Neural Animal Locomotion",
    modelPath: "/assets/quadruped/Dog.glb",
    wsPath: "/ws-interactive/quadruped",
    trajectorySamples: 16,
    contactCount: 4,
};

const loader = new GLTFLoader();
const EXAMPLE_KIND = "animal";
const CRITICAL_ROLES = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftHip",
    "LeftKnee",
    "LeftFoot",
    "RightHip",
    "RightKnee",
    "RightFoot",
    "Tail",
    "Tail1",
];
const SITE_ROLES = [
    "HeadSite",
    "LeftHandSite",
    "RightHandSite",
    "LeftFootSite",
    "RightFootSite",
    "Tail1Site",
];
const SITE_PARENT_ROLE = {
    HeadSite: "Head",
    LeftHandSite: "LeftHand",
    RightHandSite: "RightHand",
    LeftFootSite: "LeftFoot",
    RightFootSite: "RightFoot",
    Tail1Site: "Tail1",
};
const ROLE_ALIASES = {
    Hips: ["hips", "pelvis", "root", "body", "center", "cog"],
    Spine: ["spine", "spine0", "spine01", "torso", "back", "chest"],
    Spine1: ["spine1", "spine2", "spine02", "upperback", "upperchest"],
    Neck: ["neck", "neck1"],
    Head: ["head", "skull", "snout", "muzzle", "face"],
    HeadSite: ["headsite", "headend", "headtip", "nose", "snouttip"],
    LeftShoulder: ["leftshoulder", "leftscapula", "leftclavicle", "lshoulder", "lscapula", "lclavicle", "frontleftshoulder", "leftfrontshoulder"],
    LeftArm: ["leftarm", "leftforeleg", "leftfrontleg", "leftfrontupperleg", "leftupperarm", "larm", "lforeleg", "lfrontleg"],
    LeftForeArm: ["leftforearm", "leftlowerforeleg", "leftfrontlowerleg", "leftfrontshin", "leftelbow", "leftwrist", "lforearm"],
    LeftHand: ["lefthand", "leftpaw", "leftfrontpaw", "leftfrontfoot", "leftfootfront", "lhand", "lpaw"],
    LeftHandSite: ["lefthandsite", "leftpawsite", "leftfrontpawend", "leftfronttoeend", "lpawend"],
    RightShoulder: ["rightshoulder", "rightscapula", "rightclavicle", "rshoulder", "rscapula", "rclavicle", "frontrightshoulder", "rightfrontshoulder"],
    RightArm: ["rightarm", "rightforeleg", "rightfrontleg", "rightfrontupperleg", "rightupperarm", "rarm", "rforeleg", "rfrontleg"],
    RightForeArm: ["rightforearm", "rightlowerforeleg", "rightfrontlowerleg", "rightfrontshin", "rightelbow", "rightwrist", "rforearm"],
    RightHand: ["righthand", "rightpaw", "rightfrontpaw", "rightfrontfoot", "rightfootfront", "rhand", "rpaw"],
    RightHandSite: ["righthandsite", "rightpawsite", "rightfrontpawend", "rightfronttoeend", "rpawend"],
    LeftHip: ["leftupleg", "lefthip", "leftthigh", "leftrearupperleg", "leftbackupperleg", "leftupperleg", "lhip", "lthigh"],
    LeftKnee: ["leftleg", "leftknee", "leftshin", "leftrearleg", "leftbackleg", "leftlowerleg", "lleg", "lknee"],
    LeftFoot: ["leftfoot", "lefthindfoot", "leftrearfoot", "leftbackfoot", "lefthindpaw", "lfoot"],
    LeftFootSite: ["leftfootsite", "leftrearfootend", "lefthindpawend", "lefttoesite"],
    RightHip: ["rightupleg", "righthip", "rightthigh", "rightrearupperleg", "rightbackupperleg", "rightupperleg", "rhip", "rthigh"],
    RightKnee: ["rightleg", "rightknee", "rightshin", "rightrearleg", "rightbackleg", "rightlowerleg", "rleg", "rknee"],
    RightFoot: ["rightfoot", "righthindfoot", "rightrearfoot", "rightbackfoot", "righthindpaw", "rfoot"],
    RightFootSite: ["rightfootsite", "rightrearfootend", "righthindpawend", "righttoesite"],
    Tail: ["tail", "tailbase"],
    Tail1: ["tail1", "tail2", "tailtip", "tailend"],
    Tail1Site: ["tail1site", "tailtip", "tailend", "tailtipend"],
};
const LEFT_HINTS = ["left", "frontleft", "rearleft", "hindleft", "lfront", "lrear", "lfore", "lhind"];
const RIGHT_HINTS = ["right", "frontright", "rearright", "hindright", "rfront", "rrear", "rfore", "rhind"];
const ROLE_DESCENDANT_CHAINS = [
    ["Hips", ["Spine", "Spine1", "Neck", "Head"]],
    ["Tail", ["Tail1"]],
    ["LeftShoulder", ["LeftArm", "LeftForeArm", "LeftHand"]],
    ["RightShoulder", ["RightArm", "RightForeArm", "RightHand"]],
    ["LeftHip", ["LeftKnee", "LeftFoot"]],
    ["RightHip", ["RightKnee", "RightFoot"]],
];
const ROLE_ANCESTOR_FALLBACKS = [
    ["LeftArm", "LeftShoulder"],
    ["RightArm", "RightShoulder"],
    ["LeftKnee", "LeftHip"],
    ["RightKnee", "RightHip"],
    ["Head", "Neck"],
    ["Neck", "Spine1"],
    ["Spine1", "Spine"],
];

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
let modelRoot = null;
let canonicalMeshes = [];
let canonicalSkeleton = null;
let canonicalSegments = [];
let canonicalBounds = null;
let canonicalAnchorPoints = [];
let uploadedGroup = null;
let uploadedSourceMeshes = null;
let uploadedAutoFitMatrix = null;
let uploadedRigRoot = null;
let uploadedRigSkeleton = null;
let uploadedRigBones = [];
let uploadedRigRestWorldByBone = new Map();
let uploadedRigRetargetPairs = [];
let uploadedRigMode = null;
let activeLandmarkTarget = null;
const uploadedLandmarkPoints = {
    hips: null,
    chest: null,
    head: null,
    frontLeftPaw: null,
    frontRightPaw: null,
    rearLeftPaw: null,
    rearRightPaw: null,
    tail: null,
};
let skeletonPairs = [];
let contactEntityIndices = [];
let activeObjectUrl = null;
let activeRetargetMode = false;
let activeTargetRestByRole = {};
let actualToExpectedName = new Map();
let virtualBones = [];
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
let lastRenderAt = 0;
const keys = {};
let gamepadIndex = -1;
let currentRootPos = new THREE.Vector3();
let cameraDistance = 4.0;
let cameraPhi = Math.PI / 9;
let cameraTheta = 0;
let isOrbitDragging = false;
let orbitPointerId = null;
let orbitLastX = 0;
let orbitLastY = 0;
let activeFacingDrag = false;
let facingDragButton = null;
let directionRingVisible = true;
let rightStick = [0, 0];
let controlMode = "path";
let pathActive = false;
let pathLoop = false;
let pathDragActive = false;
let pathWaypointIndex = 0;
const pathWaypoints = [];
const PATH_MAX_WAYPOINTS = 48;
const PATH_MIN_POINT_DISTANCE = 0.55;
const PATH_ARRIVAL_RADIUS = 0.42;
const PATH_SLOWDOWN_RADIUS = 1.35;
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
const _retargetPrev = new THREE.Matrix4();
const _retargetCurr = new THREE.Matrix4();
const _retargetWork = new THREE.Matrix4();
const _matA = new THREE.Matrix4();
const _matB = new THREE.Matrix4();
const _matC = new THREE.Matrix4();
const _matD = new THREE.Matrix4();
const _matE = new THREE.Matrix4();
const referenceRestInverseByRole = {};
let fitState = createDefaultFitState();
const landmarkRaycaster = new THREE.Raycaster();
const landmarkPointer = new THREE.Vector2();
const landmarkMarkerGroup = new THREE.Group();
const rigEditGroup = new THREE.Group();
const rigEditHandleMap = new Map();
const rigEditLineGeometry = new THREE.BufferGeometry();
const rigEditLineMaterial = new THREE.LineBasicMaterial({
    color: 0xe6e8f2,
    transparent: true,
    opacity: 0.92,
    depthTest: false,
});
const rigEditLine = new THREE.LineSegments(rigEditLineGeometry, rigEditLineMaterial);
const rigEditRoles = ["hips", "chest", "head", "frontLeftPaw", "frontRightPaw", "rearLeftPaw", "rearRightPaw", "tail"];
const rigEditRoleToBone = {
    hips: "Hips",
    chest: "Spine1",
    head: "Head",
    frontLeftPaw: "LeftHand",
    frontRightPaw: "RightHand",
    rearLeftPaw: "LeftFoot",
    rearRightPaw: "RightFoot",
    tail: "Tail1",
};
const rigEditLinks = [
    ["hips", "chest"],
    ["chest", "head"],
    ["chest", "frontLeftPaw"],
    ["chest", "frontRightPaw"],
    ["hips", "rearLeftPaw"],
    ["hips", "rearRightPaw"],
    ["hips", "tail"],
];
let rigEditEnabled = false;
let activeRigHandle = null;
const rigDragPlane = new THREE.Plane();
const rigDragOffset = new THREE.Vector3();
const rigDragHit = new THREE.Vector3();
const rigDragNormal = new THREE.Vector3();
const facingGizmoGroup = new THREE.Group();
const facingLineGeometry = new THREE.BufferGeometry();
const facingLineMaterial = new THREE.LineBasicMaterial({
    color: 0xd8e1ff,
    transparent: true,
    opacity: 0.75,
    depthTest: false,
});
const facingLine = new THREE.Line(facingLineGeometry, facingLineMaterial);
const facingHandle = new THREE.Mesh(
    new THREE.SphereGeometry(0.09, 20, 20),
    new THREE.MeshBasicMaterial({ color: 0xd8e1ff, transparent: true, opacity: 0.9, depthTest: false }),
);
const facingRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.78, 0.022, 10, 64),
    new THREE.MeshBasicMaterial({ color: 0xd8e1ff, transparent: true, opacity: 0.72, depthTest: false }),
);
const facingDisplayVector = new THREE.Vector2(0, 1);
const facingDragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const facingDragPoint = new THREE.Vector3();
const facingGizmoOrigin = new THREE.Vector3();

const titleNode = document.getElementById("demo-title");
if (titleNode) titleNode.textContent = DEMO.title;
document.title = DEMO.title;
const importInput = document.getElementById("import-glb-input");
const importButton = document.getElementById("import-glb-btn");
const restoreButton = document.getElementById("restore-dog-btn");
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
const markTailButton = document.getElementById("mark-tail-btn");
const applyMarksButton = document.getElementById("apply-marks-btn");
const clearMarksButton = document.getElementById("clear-marks-btn");
const modeManualButton = document.getElementById("mode-manual-btn");
const modePathButton = document.getElementById("mode-path-btn");
const pathControls = document.getElementById("path-controls");
const pathPlayButton = document.getElementById("path-play-btn");
const pathClearButton = document.getElementById("path-clear-btn");
const pathLoopToggle = document.getElementById("path-loop-toggle");
const pathStatus = document.getElementById("path-status");

const styleSwitcher = document.getElementById("style-switcher");
if (styleSwitcher) styleSwitcher.style.display = "none";
const styleDropdown = document.getElementById("style-dropdown");
if (styleDropdown) styleDropdown.style.display = "none";
const styleDropdownLabel = document.getElementById("style-dropdown-label");
if (styleDropdownLabel) styleDropdownLabel.style.display = "none";

const viewportEl = document.getElementById("viewport");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x2a2a3e);
scene.fog = new THREE.Fog(0x2a2a3e, 15, 40);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 2, 6);

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
rigEditGroup.visible = false;
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
const pathPointGeometry = new THREE.SphereGeometry(0.08, 14, 14);
const pathPointMaterial = new THREE.MeshBasicMaterial({ color: 0x79d6bd, depthTest: false });
const pathTargetMaterial = new THREE.MeshBasicMaterial({ color: 0xffd166, depthTest: false });
const pathPointGroup = new THREE.Group();
const pathTargetMarker = new THREE.Mesh(
    new THREE.RingGeometry(0.22, 0.29, 32),
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

function updateFitUi() {
    const active = !!uploadedSourceMeshes;
    if (fitPanel) fitPanel.classList.toggle("hidden", !active);
    updateLandmarkButtons();
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
    if (fitSkeletonButton) fitSkeletonButton.classList.toggle("active", debugEnabled);
    if (fitEditButton) fitEditButton.classList.toggle("active", rigEditEnabled);
    updateRigEditOverlay();
}

function clearUploadedLandmarks() {
    for (const key of Object.keys(uploadedLandmarkPoints)) {
        uploadedLandmarkPoints[key] = null;
    }
}

function updateLandmarkButtons() {
    const mapping = {
        head: markHeadButton,
        hips: markHipsButton,
        tail: markTailButton,
    };
    for (const [name, button] of Object.entries(mapping)) {
        if (!button) continue;
        button.classList.toggle("active", activeLandmarkTarget === name || !!uploadedLandmarkPoints[name]);
    }
}

function updateViewportCursor() {
    if (!renderer?.domElement) return;
    renderer.domElement.style.cursor = activeRigHandle
        || activeFacingDrag
        || pathDragActive
        ? "grabbing"
        : activeLandmarkTarget
        ? "crosshair"
        : isOrbitDragging
            ? "grabbing"
            : rigEditEnabled
                ? "grab"
                : controlMode === "path"
                    ? "crosshair"
                    : "grab";
}

function normalizeYawDegrees(value) {
    let wrapped = value;
    while (wrapped > 180) wrapped -= 360;
    while (wrapped < -180) wrapped += 360;
    return wrapped;
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
        chest: 0xf2d16b,
        frontLeftPaw: 0xd5f6cf,
        frontRightPaw: 0xf0d8f8,
        rearLeftPaw: 0xcdf5ef,
        rearRightPaw: 0xf4f7b8,
        tail: 0x98f0a4,
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

function getRigEditPoint(role) {
    if (uploadedLandmarkPoints[role]) return uploadedLandmarkPoints[role].clone();
    const point = getCanonicalLandmark(role);
    if (!point || !uploadedAutoFitMatrix) return point;
    return point.applyMatrix4(uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix()));
}

function ensureRigEditHandles() {
    if (rigEditHandleMap.size) return;
    const palette = {
        hips: 0xffc86b,
        chest: 0xf2d16b,
        head: 0x8ec5ff,
        frontLeftPaw: 0xd5f6cf,
        frontRightPaw: 0xf0d8f8,
        rearLeftPaw: 0xcdf5ef,
        rearRightPaw: 0xf4f7b8,
        tail: 0x98f0a4,
    };
    for (const role of rigEditRoles) {
        const handle = new THREE.Mesh(
            new THREE.SphereGeometry(0.075, 16, 16),
            new THREE.MeshBasicMaterial({
                color: palette[role] || 0xffffff,
                transparent: true,
                opacity: 0.96,
                depthTest: false,
            })
        );
        handle.renderOrder = 1002;
        handle.userData.rigRole = role;
        rigEditHandleMap.set(role, handle);
        rigEditGroup.add(handle);
    }
}

function updateRigEditOverlay() {
    ensureRigEditHandles();
    const visible = !!uploadedSourceMeshes && rigEditEnabled;
    rigEditGroup.visible = visible;
    if (!visible) return;

    const worldPoints = new Map();
    for (const role of rigEditRoles) {
        const point = getRigEditPoint(role);
        const handle = rigEditHandleMap.get(role);
        if (!handle) continue;
        handle.visible = !!point;
        if (!point) continue;
        handle.position.copy(point);
        worldPoints.set(role, point);
    }

    const linePoints = [];
    for (const [a, b] of rigEditLinks) {
        const pa = worldPoints.get(a);
        const pb = worldPoints.get(b);
        if (!pa || !pb) continue;
        linePoints.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
    }
    rigEditLineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePoints, 3));
    rigEditLineGeometry.computeBoundingSphere();
}

function pickRigHandle(event) {
    if (!rigEditEnabled || !rigEditGroup.visible) return null;
    const rect = canvas.getBoundingClientRect();
    landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
    landmarkRaycaster.setFromCamera(landmarkPointer, camera);
    const hit = landmarkRaycaster.intersectObjects([...rigEditHandleMap.values()], false)[0];
    return hit ? hit.object.userData.rigRole : null;
}

function getFacingDisplayVector() {
    const magnitude = Math.hypot(rightStick[0], rightStick[1]);
    if (magnitude > 0.05) {
        facingDisplayVector.set(rightStick[0] / magnitude, rightStick[1] / magnitude);
        return facingDisplayVector;
    }
    const planar = new THREE.Vector2(
        Math.sin(cameraTheta),
        Math.cos(cameraTheta),
    );
    if (planar.lengthSq() > 0.0001) {
        planar.normalize();
        facingDisplayVector.copy(planar);
        return facingDisplayVector;
    }
    facingDisplayVector.set(0, 1);
    return facingDisplayVector;
}

function updateFacingGizmo() {
    const visible = !!modelRoot && !rigEditEnabled && directionRingVisible && controlMode === "manual";
    facingGizmoGroup.visible = visible;
    if (!visible) return;
    const radius = 0.78;
    const centerY = Math.max(0.04, currentRootPos.y + 0.04);
    const dir = getFacingDisplayVector();
    facingGizmoOrigin.set(currentRootPos.x, centerY, currentRootPos.z);
    facingGizmoGroup.position.copy(facingGizmoOrigin);
    facingHandle.position.set(dir.x * radius, 0, dir.y * radius);
    const linePoints = new Float32Array([0, 0, 0, dir.x * radius, 0, dir.y * radius]);
    facingLineGeometry.setAttribute("position", new THREE.BufferAttribute(linePoints, 3));
    facingLineGeometry.computeBoundingSphere();
}

const pathPickPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const pathPickPoint = new THREE.Vector3();

function updateInputVisualizerPath(move = [0, 0], face = [0, 0]) {
    const deviceLabel = document.getElementById("input-device-label");
    if (deviceLabel) deviceLabel.textContent = "Input: Path";
    const kbPanel = document.getElementById("keyboard-front-buttons");
    const gpPanel = document.getElementById("gamepad-front-buttons");
    if (kbPanel) kbPanel.classList.add("hidden");
    if (gpPanel) gpPanel.classList.add("hidden");
    updateStickCircleHud("left-stick-dot", "left-stick-pointer", move[0], move[1]);
    updateStickCircleHud("right-stick-dot", "right-stick-pointer", face[0], face[1]);
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
    pathGroup.visible = controlMode === "path";
    const points = pathWaypoints.map((point) => new THREE.Vector3(point.x, 0.06, point.z));
    pathLineGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(points.flatMap((point) => [point.x, point.y, point.z])), 3)
    );
    if (points.length > 0) pathLineGeometry.computeBoundingSphere();
    for (let index = 0; index < pathWaypoints.length; index++) {
        const point = pathWaypoints[index];
        const marker = new THREE.Mesh(
            pathPointGeometry,
            index === pathWaypointIndex ? pathTargetMaterial : pathPointMaterial
        );
        marker.position.set(point.x, 0.08, point.z);
        marker.renderOrder = 1001;
        pathPointGroup.add(marker);
    }
    const target = pathWaypoints[pathWaypointIndex];
    pathTargetMarker.visible = !!target && controlMode === "path";
    if (target) pathTargetMarker.position.set(target.x, 0.035, target.z);
}

function setControlMode(mode) {
    controlMode = mode === "path" ? "path" : "manual";
    document.body.dataset.controlMode = controlMode;
    activeFacingDrag = false;
    isOrbitDragging = false;
    pathDragActive = false;
    facingDragButton = null;
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
        return {
            left_stick: [0, 0],
            right_stick: rightStick,
            canter_boost: false,
            walk_modifier: !!(keys["AltLeft"] || keys["AltRight"]),
            trot_modifier: !!(keys["ControlLeft"] || keys["ControlRight"]),
            canter_modifier: !!(keys["ShiftLeft"] || keys["ShiftRight"]),
            action_sit: !!keys["KeyR"],
            action_stand: !!keys["KeyT"],
            action_lie: !!keys["KeyV"],
        };
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
            return {
                left_stick: [0, 0],
                right_stick: rightStick,
                canter_boost: false,
                walk_modifier: !!(keys["AltLeft"] || keys["AltRight"]),
                trot_modifier: !!(keys["ControlLeft"] || keys["ControlRight"]),
                canter_modifier: !!(keys["ShiftLeft"] || keys["ShiftRight"]),
                action_sit: !!keys["KeyR"],
                action_stand: !!keys["KeyT"],
                action_lie: !!keys["KeyV"],
            };
        }
        target = pathWaypoints[pathWaypointIndex];
        dx = target.x - currentRootPos.x;
        dz = target.z - currentRootPos.z;
        distance = Math.hypot(dx, dz);
    }

    const invDistance = distance > 1e-5 ? 1 / distance : 0;
    const face = [dx * invDistance, dz * invDistance];
    const speedScale = THREE.MathUtils.clamp(distance / PATH_SLOWDOWN_RADIUS, 0.2, 1);
    const move = [face[0] * speedScale, face[1] * speedScale];
    rightStick = face;
    updateInputVisualizerPath(move, face);
    return {
        left_stick: move,
        right_stick: face,
        canter_boost: false,
        walk_modifier: !!(keys["AltLeft"] || keys["AltRight"]),
        trot_modifier: !!(keys["ControlLeft"] || keys["ControlRight"]),
        canter_modifier: !!(keys["ShiftLeft"] || keys["ShiftRight"]),
        action_sit: !!keys["KeyR"],
        action_stand: !!keys["KeyT"],
        action_lie: !!keys["KeyV"],
    };
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

function setFacingFromPointer(event) {
    const rect = canvas.getBoundingClientRect();
    landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
    landmarkRaycaster.setFromCamera(landmarkPointer, camera);
    facingDragPlane.constant = -Math.max(0.04, currentRootPos.y + 0.04);
    if (!landmarkRaycaster.ray.intersectPlane(facingDragPlane, facingDragPoint)) return;
    const dir = new THREE.Vector2(
        facingDragPoint.x - currentRootPos.x,
        facingDragPoint.z - currentRootPos.z,
    );
    if (dir.lengthSq() < 0.0001) return;
    dir.normalize();
    rightStick = [dir.x, dir.y];
    updateFacingGizmo();
}

function getBoneDepth(bone) {
    let depth = 0;
    let cursor = bone;
    while (cursor?.parent?.isBone) {
        depth += 1;
        cursor = cursor.parent;
    }
    return depth;
}

function inferCanonicalBoneByName(name = "") {
    const norm = normalizeBoneName(name);
    if (!norm) return null;
    for (const role of Object.keys(boneMap)) {
        if (normalizeBoneName(role) === norm) return role;
    }
    for (const [role, aliases] of Object.entries(ROLE_ALIASES)) {
        if (!boneMap[role]) continue;
        if (normalizeBoneName(role) === norm || aliases.includes(norm)) return role;
        if (aliases.some((alias) => norm.includes(alias) || alias.includes(norm))) return role;
    }
    return null;
}

function buildUploadedRigRetargetPairs() {
    return uploadedRigBones
        .map((bone) => {
            const driverName = inferCanonicalBoneByName(bone.name);
            return driverName ? { bone, driverName, depth: getBoneDepth(bone) } : null;
        })
        .filter(Boolean)
        .sort((a, b) => a.depth - b.depth);
}

function captureUploadedRigReference(root) {
    uploadedRigRoot = root;
    uploadedRigSkeleton = null;
    uploadedRigBones = [];
    uploadedRigRestWorldByBone = new Map();
    uploadedRigRetargetPairs = [];
    if (!uploadedRigRoot) return;

    uploadedRigRoot.updateMatrixWorld(true);
    uploadedRigRoot.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            child.material = normalizeUploadedMaterial(child.material);
        }
        if (child.isSkinnedMesh) {
            child.frustumCulled = false;
            if (child.skeleton && !uploadedRigSkeleton) uploadedRigSkeleton = child.skeleton;
        }
    });
    if (!uploadedRigSkeleton) return;

    uploadedRigBones = [...uploadedRigSkeleton.bones];
    for (const bone of uploadedRigBones) {
        uploadedRigRestWorldByBone.set(bone, bone.matrixWorld.clone());
    }
    uploadedRigRetargetPairs = buildUploadedRigRetargetPairs();
}

function matrixMaxDelta(a, b) {
    if (!a || !b) return Infinity;
    let maxDelta = 0;
    for (let i = 0; i < 16; i++) {
        maxDelta = Math.max(maxDelta, Math.abs(a.elements[i] - b.elements[i]));
    }
    return maxDelta;
}

function bindUploadedMeshesToCanonicalDogSkeleton(root) {
    if (!root || !uploadedRigSkeleton) return 0;
    let reboundMeshes = 0;
    root.updateMatrixWorld(true);
    root.traverse((child) => {
        if (!child.isSkinnedMesh || !child.skeleton) return;
        const sourceBones = child.skeleton.bones;
        if (!sourceBones.length) return;

        const mappedBones = [];
        let matched = 0;
        let restMatched = 0;
        for (const sourceBone of sourceBones) {
            const driverName = inferCanonicalBoneByName(sourceBone.name);
            const driverBone = driverName ? boneMap[driverName] : null;
            mappedBones.push(driverBone || sourceBone);
            if (!driverName || !driverBone) continue;
            matched++;
            const sourceRest = uploadedRigRestWorldByBone.get(sourceBone);
            const driverRest = activeTargetRestByRole[driverName];
            if (matrixMaxDelta(sourceRest, driverRest) < 0.03) restMatched++;
        }

        const minimumMatches = Math.min(12, Math.max(6, Math.floor(sourceBones.length * 0.65)));
        const minimumRestMatches = Math.min(12, Math.max(6, Math.floor(matched * 0.65)));
        if (matched < minimumMatches || restMatched < minimumRestMatches) return;

        child.skeleton = new THREE.Skeleton(
            mappedBones,
            child.skeleton.boneInverses.map((inverse) => inverse.clone()),
        );
        child.frustumCulled = false;
        reboundMeshes++;
    });
    return reboundMeshes;
}

function applyPreservedUploadedRigPose() {
    if (uploadedRigMode !== "preserved" || !uploadedGroup || !uploadedRigRoot || !uploadedRigRetargetPairs.length) return;
    const fitMatrix = uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix());
    uploadedGroup.matrixAutoUpdate = false;
    uploadedGroup.matrix.copy(fitMatrix);
    uploadedGroup.matrixWorld.copy(fitMatrix);
    uploadedGroup.updateMatrixWorld(true);

    const desiredWorldByBone = new Map();
    for (const { bone, driverName } of uploadedRigRetargetPairs) {
        const driverBone = boneMap[driverName];
        const driverRestInverse = referenceRestInverseByRole[driverName];
        const uploadedRestWorld = uploadedRigRestWorldByBone.get(bone);
        if (!driverBone || !driverRestInverse || !uploadedRestWorld) continue;
        const desiredWorld = _matA
            .copy(driverBone.matrixWorld)
            .multiply(_matB.copy(driverRestInverse))
            .multiply(_matC.copy(fitMatrix))
            .multiply(_matD.copy(uploadedRestWorld));
        desiredWorldByBone.set(bone, desiredWorld.clone());
    }

    for (const { bone } of uploadedRigRetargetPairs) {
        const desiredWorld = desiredWorldByBone.get(bone);
        if (!desiredWorld) continue;
        const parent = bone.parent;
        const parentWorld = parent?.isBone
            ? (desiredWorldByBone.get(parent) || parent.matrixWorld)
            : (parent ? parent.matrixWorld : uploadedGroup.matrixWorld);
        const localMatrix = _matE.copy(parentWorld).invert().multiply(desiredWorld);
        localMatrix.decompose(bone.position, bone.quaternion, bone.scale);
        bone.updateMatrix();
    }

    uploadedRigRoot.updateMatrixWorld(true);
    if (uploadedRigSkeleton) uploadedRigSkeleton.update();
}

function normalizeBoneName(name = "") {
    return String(name).toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function hasAnyHint(norm, hints) {
    return hints.some((hint) => norm.includes(hint));
}

function getBoneChildren(bone) {
    return bone.children.filter((child) => child.isBone);
}

function inspectRig(root) {
    const bones = [];
    const skinnedMeshes = [];
    root.updateMatrixWorld(true);
    root.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
        if (child.isSkinnedMesh) {
            child.frustumCulled = false;
            skinnedMeshes.push(child);
        }
        if (child.isBone) {
            child.matrixAutoUpdate = false;
            child.matrixWorldAutoUpdate = false;
            bones.push(child);
        }
    });

    const descendantCount = new Map();
    const countDescendants = (bone) => {
        if (descendantCount.has(bone)) return descendantCount.get(bone);
        let count = 0;
        for (const child of getBoneChildren(bone)) {
            count += 1 + countDescendants(child);
        }
        descendantCount.set(bone, count);
        return count;
    };

    const rootBone = bones
        .filter((bone) => !bone.parent || !bone.parent.isBone)
        .sort((a, b) => countDescendants(b) - countDescendants(a))[0] || bones[0] || null;
    const rootPos = new THREE.Vector3();
    if (rootBone) rootBone.getWorldPosition(rootPos);

    const boneInfos = bones.map((bone) => {
        const worldPos = new THREE.Vector3();
        bone.getWorldPosition(worldPos);
        return {
            bone,
            name: bone.name || "",
            norm: normalizeBoneName(bone.name),
            worldPos,
            descendants: countDescendants(bone),
            isLeaf: getBoneChildren(bone).length === 0,
            isRoot: !bone.parent || !bone.parent.isBone,
        };
    });
    return { root, bones, boneInfos, skinnedMeshes, rootBone, rootPos };
}

function scoreBoneForRole(role, boneInfo, rootPos) {
    const norm = boneInfo.norm;
    if (!norm) return -Infinity;

    let score = 0;
    const roleNorm = normalizeBoneName(role);
    if (norm === roleNorm) score += 20;

    for (const alias of ROLE_ALIASES[role] || []) {
        if (norm === alias) score = Math.max(score, 18);
        else if (norm.startsWith(alias) || norm.endsWith(alias)) score = Math.max(score, 14);
        else if (norm.includes(alias)) score = Math.max(score, 10);
    }

    if (role === "Hips" && boneInfo.isRoot) score += 6;
    if ((role === "Head" || role.endsWith("Hand") || role.endsWith("Foot")) && boneInfo.isLeaf) score += 2;
    if ((role === "Spine" || role === "Spine1" || role === "Neck" || role === "Head") && boneInfo.worldPos.y > rootPos.y) {
        score += 1.5;
    }
    if ((role === "Tail" || role === "Tail1") && boneInfo.descendants > 0) {
        score += 1;
    }

    const leftRole = role.startsWith("Left");
    const rightRole = role.startsWith("Right");
    const leftHint = hasAnyHint(norm, LEFT_HINTS);
    const rightHint = hasAnyHint(norm, RIGHT_HINTS);
    if (leftRole) {
        if (leftHint) score += 4;
        if (rightHint) score -= 6;
        if (boneInfo.worldPos.x > rootPos.x + 0.01) score += 1.5;
        if (boneInfo.worldPos.x < rootPos.x - 0.01) score -= 1.5;
    }
    if (rightRole) {
        if (rightHint) score += 4;
        if (leftHint) score -= 6;
        if (boneInfo.worldPos.x < rootPos.x - 0.01) score += 1.5;
        if (boneInfo.worldPos.x > rootPos.x + 0.01) score -= 1.5;
    }
    return score;
}

function findBestBoneForRole(role, rig, used, minScore = 6) {
    let best = null;
    let bestScore = -Infinity;
    for (const boneInfo of rig.boneInfos) {
        if (used.has(boneInfo.bone)) continue;
        const score = scoreBoneForRole(role, boneInfo, rig.rootPos);
        if (score > bestScore) {
            bestScore = score;
            best = boneInfo.bone;
        }
    }
    return bestScore >= minScore ? best : null;
}

function collectDescendants(bone, depth = 1, out = []) {
    for (const child of getBoneChildren(bone)) {
        out.push({ bone: child, depth });
        collectDescendants(child, depth + 1, out);
    }
    return out;
}

function pickDescendantForRole(parentBone, role, rig, used, minScore = 1) {
    if (!parentBone) return null;
    let best = null;
    let bestScore = -Infinity;
    const descendants = collectDescendants(parentBone);
    for (const candidate of descendants) {
        if (used.has(candidate.bone)) continue;
        const boneInfo = rig.boneInfos.find((entry) => entry.bone === candidate.bone);
        if (!boneInfo) continue;
        const score = scoreBoneForRole(role, boneInfo, rig.rootPos) + Math.max(0, 4 - candidate.depth);
        if (score > bestScore) {
            bestScore = score;
            best = candidate.bone;
        }
    }
    return bestScore >= minScore ? best : null;
}

function pickAncestorForRole(childBone, role, rig, used, minScore = 1) {
    let pivot = childBone ? childBone.parent : null;
    let best = null;
    let bestScore = -Infinity;
    let depth = 1;
    while (pivot && pivot.isBone) {
        if (!used.has(pivot)) {
            const boneInfo = rig.boneInfos.find((entry) => entry.bone === pivot);
            if (boneInfo) {
                const score = scoreBoneForRole(role, boneInfo, rig.rootPos) + Math.max(0, 4 - depth);
                if (score > bestScore) {
                    bestScore = score;
                    best = pivot;
                }
            }
        }
        pivot = pivot.parent;
        depth += 1;
    }
    return bestScore >= minScore ? best : null;
}

function clearVirtualBones() {
    for (const bone of virtualBones) {
        if (bone.parent) bone.parent.remove(bone);
    }
    virtualBones = [];
}

function createVirtualSiteBone(role, parentBone) {
    if (!parentBone) return null;
    const bone = new THREE.Bone();
    bone.name = `__${role}`;
    bone.position.set(0, 0, 0);
    bone.matrixAutoUpdate = false;
    bone.matrixWorldAutoUpdate = false;
    parentBone.add(bone);
    bone.updateMatrixWorld(true);
    virtualBones.push(bone);
    return bone;
}

function setModelVisibility(visible) {
    meshVisible = visible;
    for (const mesh of canonicalMeshes) {
        mesh.visible = visible && !uploadedGroup;
    }
    if (uploadedGroup) {
        uploadedGroup.visible = visible;
    }
}

function cloneMaterial(material) {
    if (!material) return material;
    if (Array.isArray(material)) {
        return material.map((entry) => entry.clone());
    }
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

function getSkinnedVertex(mesh, index, target) {
    target.fromBufferAttribute(mesh.geometry.getAttribute("position"), index);
    if (mesh.isSkinnedMesh && mesh.skeleton) {
        mesh.skeleton.update();
        if (typeof mesh.applyBoneTransform === "function") {
            mesh.applyBoneTransform(index, target);
        } else if (typeof mesh.boneTransform === "function") {
            mesh.boneTransform(index, target);
        }
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

    return {
        geometry,
        material: normalizeUploadedMaterial(mesh.material),
        vertexCount: position.count,
    };
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
    return {
        box,
        center,
        size,
        minY: box.min.y,
    };
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
    const pivot = new THREE.Vector3(
        canonicalBounds.center.x,
        canonicalBounds.minY,
        canonicalBounds.center.z
    );
    const translateToPivot = new THREE.Matrix4().makeTranslation(
        pivot.x,
        pivot.y,
        pivot.z
    );
    const translateFromPivot = new THREE.Matrix4().makeTranslation(
        -pivot.x,
        -pivot.y,
        -pivot.z
    );
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
    const offset = new THREE.Matrix4().makeTranslation(
        0,
        fitState.lift,
        fitState.forward
    );

    return offset
        .clone()
        .multiply(translateToPivot)
        .multiply(rotation)
        .multiply(scale)
        .multiply(translateFromPivot);
}

function getCanonicalLandmark(name) {
    if (name === "head") return new THREE.Vector3().setFromMatrixPosition(activeTargetRestByRole.Head);
    if (name === "hips") return new THREE.Vector3().setFromMatrixPosition(activeTargetRestByRole.Hips);
    if (name === "chest") {
        const matrix = activeTargetRestByRole.Spine1 || activeTargetRestByRole.Spine;
        return matrix ? new THREE.Vector3().setFromMatrixPosition(matrix) : null;
    }
    if (name === "frontLeftPaw") {
        const matrix = activeTargetRestByRole.LeftHandSite || activeTargetRestByRole.LeftHand;
        return matrix ? new THREE.Vector3().setFromMatrixPosition(matrix) : null;
    }
    if (name === "frontRightPaw") {
        const matrix = activeTargetRestByRole.RightHandSite || activeTargetRestByRole.RightHand;
        return matrix ? new THREE.Vector3().setFromMatrixPosition(matrix) : null;
    }
    if (name === "rearLeftPaw") {
        const matrix = activeTargetRestByRole.LeftFootSite || activeTargetRestByRole.LeftFoot;
        return matrix ? new THREE.Vector3().setFromMatrixPosition(matrix) : null;
    }
    if (name === "rearRightPaw") {
        const matrix = activeTargetRestByRole.RightFootSite || activeTargetRestByRole.RightFoot;
        return matrix ? new THREE.Vector3().setFromMatrixPosition(matrix) : null;
    }
    if (name === "tail") {
        const matrix = activeTargetRestByRole.Tail1Site || activeTargetRestByRole.Tail1 || activeTargetRestByRole.Tail;
        return matrix ? new THREE.Vector3().setFromMatrixPosition(matrix) : null;
    }
    return null;
}

function computeLandmarkCorrectionMatrix() {
    const srcHips = uploadedLandmarkPoints.hips;
    const tgtHips = getCanonicalLandmark("hips");
    if (!srcHips || !tgtHips) return null;

    const weightedDirections = [
        ["head", 1.5],
        ["chest", 0.8],
        ["frontLeftPaw", 0.35],
        ["frontRightPaw", 0.35],
    ];
    const reverseDirections = [
        ["tail", 1.0],
        ["rearLeftPaw", 0.25],
        ["rearRightPaw", 0.25],
    ];
    const srcForward = new THREE.Vector3();
    const tgtForward = new THREE.Vector3();
    let directionWeight = 0;

    for (const [role, weight] of weightedDirections) {
        const src = uploadedLandmarkPoints[role];
        const tgt = getCanonicalLandmark(role);
        if (!src || !tgt) continue;
        srcForward.add(src.clone().sub(srcHips).multiplyScalar(weight));
        tgtForward.add(tgt.clone().sub(tgtHips).multiplyScalar(weight));
        directionWeight += weight;
    }
    for (const [role, weight] of reverseDirections) {
        const src = uploadedLandmarkPoints[role];
        const tgt = getCanonicalLandmark(role);
        if (!src || !tgt) continue;
        srcForward.add(srcHips.clone().sub(src).multiplyScalar(weight));
        tgtForward.add(tgtHips.clone().sub(tgt).multiplyScalar(weight));
        directionWeight += weight;
    }
    if (directionWeight <= 0) return null;

    const srcPlanar = new THREE.Vector3(srcForward.x, 0, srcForward.z);
    const tgtPlanar = new THREE.Vector3(tgtForward.x, 0, tgtForward.z);
    if (srcPlanar.lengthSq() < 1e-5 || tgtPlanar.lengthSq() < 1e-5) return null;
    srcPlanar.normalize();
    tgtPlanar.normalize();

    const angle = Math.atan2(
        srcPlanar.z * tgtPlanar.x - srcPlanar.x * tgtPlanar.z,
        srcPlanar.x * tgtPlanar.x + srcPlanar.z * tgtPlanar.z
    );

    const distanceRatios = [];
    for (const role of rigEditRoles) {
        if (role === "hips") continue;
        const src = uploadedLandmarkPoints[role];
        const tgt = getCanonicalLandmark(role);
        if (!src || !tgt) continue;
        const srcDistance = src.distanceTo(srcHips);
        const tgtDistance = tgt.distanceTo(tgtHips);
        if (srcDistance > 1e-4 && tgtDistance > 1e-4) {
            distanceRatios.push(tgtDistance / srcDistance);
        }
    }
    const scale = distanceRatios.length
        ? distanceRatios.reduce((sum, value) => sum + value, 0) / distanceRatios.length
        : 1;

    return new THREE.Matrix4()
        .makeTranslation(tgtHips.x, tgtHips.y, tgtHips.z)
        .multiply(new THREE.Matrix4().makeRotationY(angle))
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
    canonicalAnchorPoints = [
        "Head",
        "Tail1",
        "LeftHand",
        "RightHand",
        "LeftFoot",
        "RightFoot",
        "Hips",
    ]
        .map((name) => activeTargetRestByRole[name])
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

function buildSkinAttributes(geometry) {
    const position = geometry.getAttribute("position");
    const skinIndex = new Uint16Array(position.count * 4);
    const skinWeight = new Float32Array(position.count * 4);
    const vertex = new THREE.Vector3();
    const delta = new THREE.Vector3();

    for (let i = 0; i < position.count; i++) {
        vertex.fromBufferAttribute(position, i);
        const weightsByBone = new Map();

        for (const segment of canonicalSegments) {
            delta.subVectors(segment.end, segment.start);
            const lengthSq = segment.lengthSq;
            let t = 0;
            if (lengthSq > 1e-6) {
                t = THREE.MathUtils.clamp(
                    vertex.clone().sub(segment.start).dot(delta) / lengthSq,
                    0,
                    1
                );
            }
            const closest = segment.start.clone().lerp(segment.end, t);
            const distSq = Math.max(vertex.distanceToSquared(closest), 1e-4);
            const baseWeight = 1 / distSq;
            const parentWeight = baseWeight * (segment.parentIndex === segment.childIndex ? 0.0 : 1 - t);
            const childWeight = baseWeight * (segment.parentIndex === segment.childIndex ? 1.0 : t + 0.15);

            weightsByBone.set(
                segment.parentIndex,
                (weightsByBone.get(segment.parentIndex) || 0) + parentWeight
            );
            weightsByBone.set(
                segment.childIndex,
                (weightsByBone.get(segment.childIndex) || 0) + childWeight
            );
        }

        const topBones = [...weightsByBone.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4);
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
    uploadedRigRetargetPairs = [];
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
        rigEditEnabled = false;
        activeRigHandle = null;
        clearUploadedLandmarks();
    }
    updateFitUi();
    rebuildLandmarkMarkers();
    updateRigEditOverlay();
    setModelVisibility(meshVisible);
}

function bindUploadToCanonicalSkeleton(meshes, fitMatrix, objectUrl) {
    clearUploadedBinding();
    uploadedRigMode = "rebound";
    uploadedGroup = new THREE.Group();
    activeObjectUrl = objectUrl;

    for (const mesh of meshes) {
        const geometry = mesh.geometry.clone();
        geometry.applyMatrix4(fitMatrix);
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
    updateRigEditOverlay();
}

function rebuildUploadedBinding() {
    if (!uploadedSourceMeshes || !uploadedAutoFitMatrix) return;
    if (uploadedRigMode === "canonical") {
        if (uploadedGroup) {
            uploadedGroup.matrix.identity();
            uploadedGroup.matrixWorld.identity();
            uploadedGroup.updateMatrixWorld(true);
        }
        setModelVisibility(meshVisible);
        updateRigEditOverlay();
        return;
    }
    if (uploadedRigMode === "preserved") {
        applyPreservedUploadedRigPose();
        setModelVisibility(meshVisible);
        updateRigEditOverlay();
        return;
    }
    const fitMatrix = uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix());
    bindUploadToCanonicalSkeleton(uploadedSourceMeshes, fitMatrix, activeObjectUrl);
}

function applyLandmarkCorrection() {
    if (!uploadedSourceMeshes || !uploadedAutoFitMatrix) return;
    const correction = computeLandmarkCorrectionMatrix();
    if (!correction) {
        setAssetStatus("Mark hips plus head, chest, paws, or tail to align", "warn");
        return;
    }
    const currentFit = uploadedAutoFitMatrix.clone().premultiply(buildUserFitMatrix());
    uploadedAutoFitMatrix = currentFit.premultiply(correction);
    fitState = createDefaultFitState();
    activeLandmarkTarget = null;
    activeRigHandle = null;
    clearUploadedLandmarks();
    updateFitUi();
    rebuildLandmarkMarkers();
    rebuildUploadedBinding();
    setAssetStatus("Applied landmark alignment", "ok");
}

async function loadDefaultModel() {
    const gltf = await loader.loadAsync(DEMO.modelPath);
    clearVirtualBones();
    scene.add(gltf.scene);
    modelRoot = gltf.scene;
    const rig = inspectRig(gltf.scene);
    skinnedMesh = rig.skinnedMeshes[0] || null;
    boneMap = {};
    actualToExpectedName = new Map();
    activeTargetRestByRole = {};
    activeRetargetMode = false;

    for (const boneInfo of rig.boneInfos) {
        boneMap[boneInfo.bone.name] = boneInfo.bone;
        actualToExpectedName.set(boneInfo.bone, boneInfo.bone.name);
        activeTargetRestByRole[boneInfo.bone.name] = boneInfo.bone.matrixWorld.clone();
        referenceRestInverseByRole[boneInfo.bone.name] = boneInfo.bone.matrixWorld.clone().invert();
    }

    captureCanonicalReference();
    if (entityNames.length) {
        rebuildSkeletonPairs();
        createJointSpheres();
    }
    setModelVisibility(meshVisible);
    setAssetStatus("Built-in Dog.glb · canonical dog rig ready", "ok");
}

async function loadUploadedModel(file) {
    if (!canonicalSkeleton || !canonicalBounds) {
        setAssetStatus("Canonical dog rig is still loading", "warn");
        return;
    }
    clearUploadedBinding({ revokeObjectUrl: true, resetSource: true });
    const objectUrl = URL.createObjectURL(file);
    setAssetStatus(`Binding ${file.name} to canonical dog rig...`, "warn");
    try {
        const gltf = await loader.loadAsync(objectUrl);
        inspectRig(gltf.scene);
        captureUploadedRigReference(gltf.scene);
        const bakedMeshes = extractUploadGeometry(gltf.scene);
        if (!bakedMeshes.length) {
            URL.revokeObjectURL(objectUrl);
            setAssetStatus(`${file.name} rejected · no mesh geometry found`, "error");
            return;
        }

        const reboundMeshCount = bindUploadedMeshesToCanonicalDogSkeleton(gltf.scene);
        if (reboundMeshCount > 0) {
            uploadedRigMode = "canonical";
            uploadedGroup = new THREE.Group();
            uploadedGroup.matrixAutoUpdate = false;
            uploadedGroup.add(gltf.scene);
            scene.add(uploadedGroup);
            activeObjectUrl = objectUrl;
            uploadedSourceMeshes = bakedMeshes;
            uploadedAutoFitMatrix = new THREE.Matrix4();
            fitState = createDefaultFitState();
            updateFitUi();
            setModelVisibility(meshVisible);
            setAssetStatus(`${file.name} loaded · using canonical dog animation rig`, "ok");
            return;
        }

        if (uploadedRigSkeleton && uploadedRigRetargetPairs.length >= 12) {
            uploadedRigMode = "preserved";
            uploadedGroup = new THREE.Group();
            uploadedGroup.matrixAutoUpdate = false;
            uploadedGroup.add(gltf.scene);
            scene.add(uploadedGroup);
            activeObjectUrl = objectUrl;
            uploadedSourceMeshes = bakedMeshes;
            uploadedAutoFitMatrix = new THREE.Matrix4();
            fitState = createDefaultFitState();
            updateFitUi();
            setModelVisibility(meshVisible);
            applyPreservedUploadedRigPose();
            setAssetStatus(`${file.name} loaded · preserved uploaded dog rig`, "ok");
            return;
        }

        const samplePoints = samplePointsFromMeshes(bakedMeshes);
        uploadedSourceMeshes = bakedMeshes;
        uploadedAutoFitMatrix = chooseBestFitMatrix(samplePoints);
        fitState = createDefaultFitState();
        updateFitUi();
        activeObjectUrl = objectUrl;
        rebuildUploadedBinding();
        setAssetStatus(`${file.name} loaded · rebound to canonical dog rig`, "ok");
    } catch (error) {
        URL.revokeObjectURL(objectUrl);
        setAssetStatus(`${file.name} failed to load`, "error");
        console.error("Failed to load uploaded GLB:", error);
    }
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
        const response = await fetch(`/api/rigged-mesh/${EXAMPLE_KIND}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        exampleAssets = Array.isArray(payload.items) ? payload.items : [];
        replaceExampleOptions(exampleAssets.length ? "Choose example" : "No examples", exampleAssets);
    } catch (error) {
        exampleAssets = [];
        replaceExampleOptions("Examples unavailable");
        console.error("Failed to load animal mesh examples:", error);
    }
}

async function loadExampleAsset(example) {
    if (!example) return;
    const token = ++exampleLoadToken;
    setExamplePickerDisabled(true);
    setAssetStatus(`Fetching ${example.name || example.fileName}...`, "warn");
    try {
        const response = await fetch(example.url, { cache: "force-cache" });
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
        console.error("Failed to load animal mesh example:", error);
    } finally {
        if (token === exampleLoadToken) {
            setExamplePickerDisabled(exampleAssets.length === 0);
        }
    }
}

function rebuildSkeletonPairs() {
    skeletonPairs = [];
    for (const name of entityNames) {
        const bone = boneMap[name];
        if (!bone) continue;
        let parent = bone.parent;
        while (parent && parent.isBone && !actualToExpectedName.has(parent)) {
            parent = parent.parent;
        }
        if (!parent || !actualToExpectedName.has(parent)) continue;
        skeletonPairs.push([name, actualToExpectedName.get(parent)]);
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
            const role = entityNames[i];
            const bone = boneMap[role];
            if (!bone) continue;
            if (activeRetargetMode && referenceRestInverseByRole[role] && activeTargetRestByRole[role]) {
                _retargetWork.multiplyMatrices(frameCurr.entityMatrices[i], referenceRestInverseByRole[role]);
                _retargetWork.multiply(activeTargetRestByRole[role]);
                bone.matrixWorld.copy(_retargetWork);
            } else {
                bone.matrixWorld.copy(frameCurr.entityMatrices[i]);
            }
        }
        return;
    }

    interpolateTransform(framePrev.rootMatrix, frameCurr.rootMatrix, alpha, _pos, _quat, _scl);
    currentRootPos.copy(_pos);
    for (let i = 0; i < entityCount; i++) {
        const role = entityNames[i];
        const bone = boneMap[role];
        if (!bone) continue;
        if (activeRetargetMode && referenceRestInverseByRole[role] && activeTargetRestByRole[role]) {
            _retargetPrev.multiplyMatrices(framePrev.entityMatrices[i], referenceRestInverseByRole[role]);
            _retargetPrev.multiply(activeTargetRestByRole[role]);
            _retargetCurr.multiplyMatrices(frameCurr.entityMatrices[i], referenceRestInverseByRole[role]);
            _retargetCurr.multiply(activeTargetRestByRole[role]);
            interpolateTransform(_retargetPrev, _retargetCurr, alpha, _interpPos, _interpQuat, _interpScl);
        } else {
            interpolateTransform(framePrev.entityMatrices[i], frameCurr.entityMatrices[i], alpha, _interpPos, _interpQuat, _interpScl);
        }
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
    if (fitSkeletonButton) fitSkeletonButton.classList.toggle("active", debugEnabled);
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
    const rxRaw = gp.axes[2] || 0;
    const ryRaw = gp.axes[3] || 0;

    return {
        lb: physicalL2,
        rb: physicalR2,
        l2: physicalL1,
        r2: physicalR1,
        canter: faceBottom,
        lxRaw,
        lyRaw,
        rxRaw,
        ryRaw,
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
    updateStickCircleHud("right-stick-dot", "right-stick-pointer", rightStick[0], rightStick[1]);
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
    const [rvx, rvy] = applyDeadzone(mapped.rxRaw, -mapped.ryRaw);
    updateStickCircleHud("left-stick-dot", "left-stick-pointer", lvx, lvy);
    updateStickCircleHud("right-stick-dot", "right-stick-pointer", rvx, rvy);
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
    if (controlMode === "path") {
        updateInputVisualizerPath();
        return;
    }
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
    const [rx, ry] = applyDeadzone(mapped.rxRaw, -mapped.ryRaw);
    const cameraRelative = getCameraRelativeAxes(lx, ly);
    rightStick = getCameraRelativeAxes(rx, ry);
    return {
        left_stick: cameraRelative,
        right_stick: rightStick,
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
let touchRightStick = [0, 0];
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
        right_stick: rightStick,
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
    rightStick = touchRightStick;
    updateInputVisualizerKeyboard();
    const [tx, ty] = applyDeadzone(touchLeftStick[0], -touchLeftStick[1]);
    updateStickCircleHud("left-stick-dot", "left-stick-pointer", tx, ty);
    const cameraRelative = getCameraRelativeAxes(touchLeftStick[0], touchLeftStick[1]);
    const clampedAxes = applyDeadzone2(cameraRelative);
    return {
        left_stick: clampedAxes,
        right_stick: rightStick,
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
    if (controlMode === "path") return getPathInput();
    const gp = getActiveGamepad();
    if (gp) return getGamepadInput(gp);
    const hasTouch = (
        touchLeftStick[0] !== 0 ||
        touchLeftStick[1] !== 0 ||
        touchRightStick[0] !== 0 ||
        touchRightStick[1] !== 0 ||
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
    applyPreservedUploadedRigPose();
    currentSpeed = framePrev
        ? THREE.MathUtils.lerp(framePrev.speed || 0, frameCurr.speed || 0, alpha)
        : (frameCurr.speed || 0);
    updateSpeedBar(currentSpeed);
    updateRigEditOverlay();
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
    updateFacingGizmo();
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
        setModelVisibility(!meshVisible);
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

canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 0 && rigEditEnabled) {
        const pickedRole = pickRigHandle(event);
        if (pickedRole) {
            activeRigHandle = pickedRole;
            landmarkRaycaster.setFromCamera(landmarkPointer, camera);
            camera.getWorldDirection(rigDragNormal).normalize();
            const handleMesh = rigEditHandleMap.get(pickedRole);
            rigDragPlane.setFromNormalAndCoplanarPoint(rigDragNormal, handleMesh ? handleMesh.position : currentRootPos);
            if (landmarkRaycaster.ray.intersectPlane(rigDragPlane, rigDragHit)) {
                rigDragOffset.copy(handleMesh ? handleMesh.position : currentRootPos).sub(rigDragHit);
            } else {
                rigDragOffset.set(0, 0, 0);
            }
            orbitPointerId = event.pointerId;
            canvas.setPointerCapture(event.pointerId);
            updateViewportCursor();
            event.preventDefault();
            return;
        }
    }

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
        setAssetStatus("Landmark captured", "ok");
        updateViewportCursor();
        return;
    }

    if (event.button !== 0) return;
    if (controlMode === "path" && handlePathPointer(event)) {
        pathDragActive = true;
        orbitPointerId = event.pointerId;
        canvas.setPointerCapture(event.pointerId);
        updateViewportCursor();
        event.preventDefault();
        return;
    }
    if (controlMode === "manual" && pickFacingControl(event)) {
        activeFacingDrag = true;
        facingDragButton = event.button;
        orbitPointerId = event.pointerId;
        canvas.setPointerCapture(event.pointerId);
        setFacingFromPointer(event);
        updateViewportCursor();
        event.preventDefault();
        return;
    }

    isOrbitDragging = true;
    orbitPointerId = event.pointerId;
    orbitLastX = event.clientX;
    orbitLastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
    updateViewportCursor();
});

canvas.addEventListener("pointermove", (event) => {
    if (activeRigHandle) {
        const rect = canvas.getBoundingClientRect();
        landmarkPointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
        landmarkPointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
        landmarkRaycaster.setFromCamera(landmarkPointer, camera);
        if (landmarkRaycaster.ray.intersectPlane(rigDragPlane, rigDragHit)) {
            uploadedLandmarkPoints[activeRigHandle] = rigDragHit.clone().add(rigDragOffset);
            rebuildLandmarkMarkers();
            updateRigEditOverlay();
        }
        return;
    }

    if (activeFacingDrag) {
        setFacingFromPointer(event);
        event.preventDefault();
        return;
    }

    if (pathDragActive && event.pointerId === orbitPointerId) {
        handlePathPointer(event);
        event.preventDefault();
        return;
    }

    if (!isOrbitDragging || event.pointerId !== orbitPointerId) return;
    const dx = event.clientX - orbitLastX;
    const dy = event.clientY - orbitLastY;
    orbitLastX = event.clientX;
    orbitLastY = event.clientY;
    cameraTheta -= dx * 0.008;
    cameraPhi = THREE.MathUtils.clamp(cameraPhi - dy * 0.006, -1.2, 1.1);
});

function endOrbitDrag(event) {
    if (pathDragActive && event.pointerId === orbitPointerId) {
        pathDragActive = false;
        orbitPointerId = null;
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
        updateViewportCursor();
        return;
    }
    if (activeFacingDrag && event.pointerId === orbitPointerId) {
        activeFacingDrag = false;
        facingDragButton = null;
        orbitPointerId = null;
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
        updateViewportCursor();
        return;
    }
    if (activeRigHandle) {
        activeRigHandle = null;
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
        updateViewportCursor();
        return;
    }
    if (!isOrbitDragging || event.pointerId !== orbitPointerId) return;
    isOrbitDragging = false;
    orbitPointerId = null;
    if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
    }
    updateViewportCursor();
}

canvas.addEventListener("pointerup", endOrbitDrag);
canvas.addEventListener("pointercancel", endOrbitDrag);
canvas.addEventListener("pointerleave", (event) => {
    if (event.pointerType !== "mouse") return;
    endOrbitDrag(event);
});

function setupTouchJoystick(joystickEl, knobEl, onMove, onEnd) {
    if (!joystickEl || !knobEl) return;
    const radius = joystickEl.offsetWidth / 2;
    const knobRadius = knobEl.offsetWidth / 2;
    const maxDist = radius - knobRadius;
    let activeTouchId = null;

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
        activeTouchId = touch.identifier;
        handleMove(touch.clientX, touch.clientY, getCenter());
    }, { passive: false });

    joystickEl.addEventListener("touchmove", (e) => {
        e.preventDefault();
        for (const touch of e.changedTouches) {
            if (touch.identifier === activeTouchId) {
                handleMove(touch.clientX, touch.clientY, getCenter());
                break;
            }
        }
    }, { passive: false });

    const endHandler = (e) => {
        for (const touch of e.changedTouches) {
            if (touch.identifier === activeTouchId) {
                activeTouchId = null;
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
        setupTouchJoystick(
            joystickRight,
            knobRight,
            (x, y) => {
                touchRightStick = normalize2([x, y]);
                rightStick = touchRightStick;
            },
            () => {
                touchRightStick = [0, 0];
                rightStick = [0, 0];
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

if (importButton && importInput) {
    importButton.addEventListener("click", () => importInput.click());
    importInput.addEventListener("change", async (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        resetExamplePicker();
        await loadUploadedModel(file);
        importInput.value = "";
    });
}

if (restoreButton) {
    restoreButton.addEventListener("click", () => {
        resetExamplePicker();
        clearUploadedBinding({ revokeObjectUrl: true, resetSource: true });
        setAssetStatus("Built-in Dog.glb · canonical dog rig ready", "ok");
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

function armLandmarkTarget(name) {
    activeLandmarkTarget = activeLandmarkTarget === name ? null : name;
    updateLandmarkButtons();
    updateViewportCursor();
    if (activeLandmarkTarget) {
        setAssetStatus(`Click the uploaded mesh to mark ${name}`, "warn");
    }
}

function resetUploadedAutoFit() {
    if (!uploadedSourceMeshes) return;
    const samplePoints = samplePointsFromMeshes(uploadedSourceMeshes);
    uploadedAutoFitMatrix = chooseBestFitMatrix(samplePoints);
    fitState = createDefaultFitState();
    activeLandmarkTarget = null;
    activeRigHandle = null;
    clearUploadedLandmarks();
    rebuildLandmarkMarkers();
    updateFitUi();
    rebuildUploadedBinding();
    setAssetStatus("Recomputed automatic quadruped fit", "ok");
}

if (fitSkeletonButton) fitSkeletonButton.addEventListener("click", () => setDebugEnabled(!debugEnabled));
if (fitEditButton) {
    fitEditButton.addEventListener("click", () => {
        rigEditEnabled = !rigEditEnabled;
        activeLandmarkTarget = null;
        activeRigHandle = null;
        updateFitUi();
        updateViewportCursor();
        setAssetStatus(rigEditEnabled ? "Fit Rig on · drag animal landmarks, then Apply" : "Fit Rig off", "ok");
    });
}
if (fitApplyButton) fitApplyButton.addEventListener("click", () => applyLandmarkCorrection());
if (fitAutoButton) fitAutoButton.addEventListener("click", () => resetUploadedAutoFit());
if (markHeadButton) markHeadButton.addEventListener("click", () => armLandmarkTarget("head"));
if (markHipsButton) markHipsButton.addEventListener("click", () => armLandmarkTarget("hips"));
if (markTailButton) markTailButton.addEventListener("click", () => armLandmarkTarget("tail"));
if (applyMarksButton) applyMarksButton.addEventListener("click", () => applyLandmarkCorrection());
if (clearMarksButton) {
    clearMarksButton.addEventListener("click", () => {
        activeLandmarkTarget = null;
        activeRigHandle = null;
        clearUploadedLandmarks();
        rebuildLandmarkMarkers();
        updateRigEditOverlay();
        setAssetStatus("Cleared landmark marks", "ok");
        updateViewportCursor();
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
        fitState = createDefaultFitState();
        activeLandmarkTarget = null;
        activeRigHandle = null;
        clearUploadedLandmarks();
        rebuildLandmarkMarkers();
        updateFitUi();
        rebuildUploadedBinding();
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
setControlMode("path");

loadDefaultModel()
    .then(() => connectWebSocket())
    .catch((error) => {
        setAssetStatus("Failed to load built-in Dog.glb", "error");
        console.error("Failed to load default model:", error);
        connectWebSocket();
    });

window.addEventListener("resize", resizeViewport);
updateViewportCursor();
