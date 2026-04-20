from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from interactive_server import INTERACTIVE_DEMOS, mount_interactive_routes


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
CLIENT_DIR = ROOT_DIR / "webdemo" / "client"

CRANBERRY_ASSETS = APP_DIR / "Demos" / "_ASSETS_" / "Cranberry"
GENO_ASSETS = APP_DIR / "Demos" / "_ASSETS_" / "Geno"

FRAME_DT = 1.0 / 30.0
MIN_SPEED = 0.25
MAX_SPEED = 1.75

sys.path.insert(0, str(APP_DIR))

from ai4animation import Actor, AI4Animation, FABRIK, Motion, RootModule, Rotation, Time, Vector3


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CRANBERRY_DEFS = _load_module(
    "cranberry_defs", CRANBERRY_ASSETS / "Definitions.py"
)
GENO_DEFS = _load_module("geno_defs", GENO_ASSETS / "Definitions.py")


class BaseDemo:
    title = ""
    description = ""
    category = ""
    model_path = ""
    asset_label = ""
    surface_label = ""
    supports_mirror = False
    sliders = []
    focus_name = None

    def __init__(self):
        self.mirror = False
        self._entity_names = []
        self._entity_indices = []

    def set_inputs(self, payload: dict):
        if self.supports_mirror:
            self.mirror = bool(payload.get("mirror", False))

    def _cache_actor_entities(self, actor):
        self.Actor = actor
        self._entity_names = list(actor.NameToEntity.keys())
        self._entity_indices = [
            actor.NameToEntity[name].Index for name in self._entity_names
        ]

    def _focus_matrix(self):
        if self.focus_name and self.focus_name in self.Actor.NameToEntity:
            entity = self.Actor.NameToEntity[self.focus_name]
            return AI4Animation.Scene.Transforms[entity.Index]
        if self._entity_indices:
            return AI4Animation.Scene.Transforms[self._entity_indices[0]]
        return np.eye(4, dtype=np.float32)

    def get_entity_names(self):
        return self._entity_names

    def get_markers(self):
        return []

    def get_frame_data(self):
        entity_transforms = np.asarray(
            AI4Animation.Scene.Transforms[self._entity_indices], dtype=np.float32
        ).reshape(-1)
        root_transform = np.asarray(self._focus_matrix(), dtype=np.float32).reshape(-1)
        return {
            "root": root_transform.tolist(),
            "entityMatrices": entity_transforms.tolist(),
            "markers": self.get_markers(),
        }


class ActorDemo(BaseDemo):
    title = "Actor Viewer"
    description = "Inspect the Cranberry character rig in a clean orbital viewer."
    category = "Character"
    model_path = "/assets/cranberry/Model.glb"
    asset_label = "Cranberry rig"
    surface_label = "Orbit + inspect"
    focus_name = CRANBERRY_DEFS.HipName

    def Start(self):
        entity = AI4Animation.Scene.AddEntity("Actor")
        actor = entity.AddComponent(
            Actor,
            str(CRANBERRY_ASSETS / "Model.glb"),
            CRANBERRY_DEFS.FULL_BODY_NAMES,
            True,
        )
        actor.Entity.SetPosition(Vector3.Create(0, 0, 0))
        self._cache_actor_entities(actor)

    def Update(self):
        self.Actor.Entity.SetRotation(Rotation.Euler(0, 120 * Time.TotalTime, 0))
        self.Actor.SyncFromScene()


class GlbLoadingDemo(BaseDemo):
    title = "GLB Motion"
    description = "Load a GLB motion clip onto Cranberry and flip it live."
    category = "Motion"
    model_path = "/assets/cranberry/Model.glb"
    asset_label = "Cranberry + GLB"
    surface_label = "Loop + mirror"
    supports_mirror = True
    focus_name = CRANBERRY_DEFS.HipName

    def Start(self):
        self.Motion = Motion.LoadFromGLB(
            str(APP_DIR / "Demos" / "GLBLoading" / "cranberry.glb"),
            names=CRANBERRY_DEFS.FULL_BODY_NAMES,
            floor=None,
        )
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            str(CRANBERRY_ASSETS / "Model.glb"),
            CRANBERRY_DEFS.FULL_BODY_NAMES,
        )
        self._cache_actor_entities(self.Actor)

    def Update(self):
        timestamp = Time.TotalTime % self.Motion.TotalTime
        self.Actor.SetTransforms(
            self.Motion.GetBoneTransformations(
                timestamps=timestamp,
                bone_names_or_indices=self.Actor.GetBoneNames(),
                mirrored=self.mirror,
            )
        )
        self.Actor.SyncToScene()


class BvhLoadingDemo(BaseDemo):
    title = "BVH Motion"
    description = "Preview raw BVH playback on Geno with just the essentials."
    category = "Motion"
    model_path = "/assets/geno/Model.glb"
    asset_label = "Geno + BVH"
    surface_label = "Loop + mirror"
    supports_mirror = True
    focus_name = GENO_DEFS.HipName

    def Start(self):
        self.BVHMotion = Motion.LoadFromBVH(
            str(APP_DIR / "Demos" / "BVHLoading" / "WalkingStickLeft_BR.bvh"),
            scale=0.01,
        )
        actor = AI4Animation.Scene.AddEntity("BVH Actor").AddComponent(
            Actor,
            str(GENO_ASSETS / "Model.glb"),
            GENO_DEFS.FULL_BODY_NAMES,
        )
        self._cache_actor_entities(actor)

    def Update(self):
        timestamp = Time.TotalTime % self.BVHMotion.TotalTime
        self.Actor.SetTransforms(
            self.BVHMotion.GetBoneTransformations(
                timestamps=timestamp,
                bone_names_or_indices=self.Actor.GetBoneNames(),
                mirrored=self.mirror,
            )
        )
        self.Actor.SyncToScene()


class InverseKinematicsDemo(BaseDemo):
    title = "Inverse Kinematics"
    description = "Move a wrist target and watch the arm solve in real time."
    category = "Interaction"
    model_path = "/assets/cranberry/Model.glb"
    asset_label = "Cranberry + target"
    surface_label = "Target + solve"
    focus_name = CRANBERRY_DEFS.HipName
    sliders = [
        {
            "id": "target_x",
            "label": "Target X",
            "min": -0.8,
            "max": 0.8,
            "step": 0.01,
            "default": 0.25,
        },
        {
            "id": "target_y",
            "label": "Target Y",
            "min": 0.7,
            "max": 2.0,
            "step": 0.01,
            "default": 1.3,
        },
        {
            "id": "target_z",
            "label": "Target Z",
            "min": -0.8,
            "max": 0.8,
            "step": 0.01,
            "default": 0.0,
        },
    ]

    def Start(self):
        actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            str(CRANBERRY_ASSETS / "Model.glb"),
            CRANBERRY_DEFS.FULL_BODY_NAMES,
        )
        self._cache_actor_entities(actor)

        self.IK = FABRIK(
            self.Actor.GetBone(CRANBERRY_DEFS.LeftShoulderName),
            self.Actor.GetBone(CRANBERRY_DEFS.LeftWristName),
        )

        wrist_position = self.Actor.GetBone(CRANBERRY_DEFS.LeftWristName).GetPosition()
        self.target_x = float(wrist_position[0])
        self.target_y = float(wrist_position[1])
        self.target_z = float(wrist_position[2])

        self.Target = AI4Animation.Scene.AddEntity("Target")
        self.Target.SetPosition(
            Vector3.Create(self.target_x, self.target_y, self.target_z)
        )
        self.Pose = self.Actor.GetTransforms().copy()

    def set_inputs(self, payload: dict):
        self.target_x = float(payload.get("target_x", self.target_x))
        self.target_y = float(payload.get("target_y", self.target_y))
        self.target_z = float(payload.get("target_z", self.target_z))

    def Update(self):
        self.Actor.SetTransforms(self.Pose)
        self.Target.SetPosition(
            Vector3.Create(self.target_x, self.target_y, self.target_z)
        )
        self.IK.Solve(
            self.Target.GetPosition(),
            self.Target.GetRotation(),
            max_iterations=10,
            threshold=0.001,
        )
        self.Actor.SyncToScene(self.IK.Bones)

    def get_markers(self):
        return [
            {
                "id": "target",
                "position": [self.target_x, self.target_y, self.target_z],
                "tone": "accent",
            }
        ]


DEMO_REGISTRY = {
    "actor": {
        "factory": ActorDemo,
        "title": ActorDemo.title,
        "description": ActorDemo.description,
        "category": ActorDemo.category,
        "modelPath": ActorDemo.model_path,
        "assetLabel": ActorDemo.asset_label,
        "surfaceLabel": ActorDemo.surface_label,
        "supportsMirror": ActorDemo.supports_mirror,
        "sliders": ActorDemo.sliders,
    },
    "glb-motion": {
        "factory": GlbLoadingDemo,
        "title": GlbLoadingDemo.title,
        "description": GlbLoadingDemo.description,
        "category": GlbLoadingDemo.category,
        "modelPath": GlbLoadingDemo.model_path,
        "assetLabel": GlbLoadingDemo.asset_label,
        "surfaceLabel": GlbLoadingDemo.surface_label,
        "supportsMirror": GlbLoadingDemo.supports_mirror,
        "sliders": GlbLoadingDemo.sliders,
    },
    "bvh-motion": {
        "factory": BvhLoadingDemo,
        "title": BvhLoadingDemo.title,
        "description": BvhLoadingDemo.description,
        "category": BvhLoadingDemo.category,
        "modelPath": BvhLoadingDemo.model_path,
        "assetLabel": BvhLoadingDemo.asset_label,
        "surfaceLabel": BvhLoadingDemo.surface_label,
        "supportsMirror": BvhLoadingDemo.supports_mirror,
        "sliders": BvhLoadingDemo.sliders,
    },
    "inverse-kinematics": {
        "factory": InverseKinematicsDemo,
        "title": InverseKinematicsDemo.title,
        "description": InverseKinematicsDemo.description,
        "category": InverseKinematicsDemo.category,
        "modelPath": InverseKinematicsDemo.model_path,
        "assetLabel": InverseKinematicsDemo.asset_label,
        "surfaceLabel": InverseKinematicsDemo.surface_label,
        "supportsMirror": InverseKinematicsDemo.supports_mirror,
        "sliders": InverseKinematicsDemo.sliders,
    },
}


app = FastAPI()
app.mount("/client", StaticFiles(directory=CLIENT_DIR), name="client")
app.mount(
    "/assets/cranberry",
    StaticFiles(directory=CRANBERRY_ASSETS),
    name="assets-cranberry",
)
app.mount("/assets/geno", StaticFiles(directory=GENO_ASSETS), name="assets-geno")

_session_lock = asyncio.Lock()
mount_interactive_routes(app, _session_lock)


def _reset_runtime():
    Time.TotalTime = 0.0
    Time.DeltaTime = 0.0
    Time.Timescale = 1.0


def _clamp_speed(value: float) -> float:
    return max(MIN_SPEED, min(MAX_SPEED, value))


def _serialize_demo(demo_id: str):
    config = DEMO_REGISTRY[demo_id]
    return {
        "id": demo_id,
        **{key: value for key, value in config.items() if key != "factory"},
    }


@app.get("/")
async def index():
    return FileResponse(CLIENT_DIR / "index.html")


@app.get("/api/manifest")
async def manifest():
    menu_demos = INTERACTIVE_DEMOS + [_serialize_demo(demo_id) for demo_id in DEMO_REGISTRY]
    return {
        "title": "Motion Playground",
        "subtitle": "A calm, local browser surface for trying rigs, motion clips, and quick interaction studies.",
        "useCase": "Best for casually flipping through assets when you want to see how they feel before opening a heavier workflow.",
        "sectionCaption": "Pick a demo, orbit around it, and make small adjustments without leaving the browser.",
        "highlights": ["Local only", f"{len(menu_demos)} demos", "Light + dark"],
        "demos": menu_demos,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "activeSession": _session_lock.locked()}


async def _receive_inputs(websocket: WebSocket, state: dict):
    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") != "input":
                continue
            state["paused"] = bool(payload.get("paused", state["paused"]))
            state["speed"] = _clamp_speed(float(payload.get("speed", state["speed"])))
            state["mirror"] = bool(payload.get("mirror", state["mirror"]))
            for key, value in payload.get("sliders", {}).items():
                state[key] = float(value)
    except WebSocketDisconnect:
        state["_disconnected"] = True
    except Exception as error:
        state["_receiver_error"] = f"{type(error).__name__}: {error}"


@app.websocket("/ws/{demo_id}")
async def websocket_endpoint(websocket: WebSocket, demo_id: str):
    await websocket.accept()

    if demo_id not in DEMO_REGISTRY:
        await websocket.send_json(
            {"type": "error", "message": f"Unknown demo '{demo_id}'."}
        )
        await websocket.close()
        return

    if _session_lock.locked():
        await websocket.send_json(
            {
                "type": "busy",
                "message": "Another viewer session is already active. Close the other tab and try again.",
            }
        )
        await websocket.close()
        return

    await _session_lock.acquire()
    state = {
        "paused": False,
        "speed": 1.0,
        "mirror": False,
        "_disconnected": False,
        "_receiver_error": None,
    }

    try:
        _reset_runtime()
        demo = DEMO_REGISTRY[demo_id]["factory"]()
        AI4Animation(demo, mode=AI4Animation.Mode.MANUAL)

        await websocket.send_json(
            {
                "type": "init",
                "entityNames": demo.get_entity_names(),
                "entityCount": len(demo.get_entity_names()),
                "frameDt": FRAME_DT,
                "demo": _serialize_demo(demo_id),
            }
        )

        receiver = asyncio.create_task(_receive_inputs(websocket, state))

        try:
            while True:
                tick_started = asyncio.get_running_loop().time()
                if state["_receiver_error"]:
                    print(f"Receiver error for demo '{demo_id}': {state['_receiver_error']}")
                    break
                if state["_disconnected"] or receiver.done():
                    break
                demo.set_inputs(state)
                if not state["paused"]:
                    AI4Animation.Update(FRAME_DT * state["speed"])
                try:
                    await websocket.send_json(
                        {
                            "type": "frame",
                            **demo.get_frame_data(),
                        }
                    )
                except (WebSocketDisconnect, RuntimeError):
                    break
                elapsed = asyncio.get_running_loop().time() - tick_started
                await asyncio.sleep(max(0.0, FRAME_DT - elapsed))
        finally:
            receiver.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        if _session_lock.locked():
            _session_lock.release()
