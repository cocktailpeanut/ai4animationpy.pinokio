from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
CLIENT_DIR = ROOT_DIR / "webdemo" / "client"
LOCAL_PROGRAM_DIR = ROOT_DIR / "webdemo" / "locomotion"

SOURCE_DEMO_DIRS = {
    "biped": APP_DIR / "Demos" / "Locomotion" / "Biped",
    "quadruped": APP_DIR / "Demos" / "Locomotion" / "Quadruped",
}
WEBPROGRAM_FILES = {
    "biped": LOCAL_PROGRAM_DIR / "Biped" / "WebProgram.py",
    "quadruped": LOCAL_PROGRAM_DIR / "Quadruped" / "WebProgram.py",
}
QUADRUPED_ASSETS = APP_DIR / "Demos" / "_ASSETS_" / "Quadruped"

INTERACTIVE_DEMOS = [
    {
        "id": "biped-live",
        "title": "Human",
        "description": "Drive Geno manually or draw a route for the neural locomotion controller to follow.",
        "category": "Interactive",
        "assetLabel": "Geno locomotion",
        "surfaceLabel": "Manual + Path",
        "launchPath": "/play/human",
    },
    {
        "id": "quadruped-live",
        "title": "Animal",
        "description": "Control the dog directly or sketch a path across the ground plane.",
        "category": "Interactive",
        "assetLabel": "Animal locomotion",
        "surfaceLabel": "Manual + Path",
        "launchPath": "/play/animal",
    },
]

FRAME_DT = 1.0 / 30.0
SESSION_LIMIT_SECONDS = int(os.environ.get("INTERACTIVE_SESSION_LIMIT_SECONDS", "3600"))
MAX_CPU_THREADS = max(2, int(os.environ.get("MAX_CPU_THREADS", "4")))

sys.path.insert(0, str(APP_DIR))
torch.set_num_interop_threads(1)

_cached_models: dict[str, object] = {}
_model_ready = {demo: False for demo in SOURCE_DEMO_DIRS}
_guidance_templates = {demo: {} for demo in SOURCE_DEMO_DIRS}
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _preload_model(demo_name: str):
    demo_dir = SOURCE_DEMO_DIRS[demo_name]
    local_path = demo_dir / "Network.pt"
    torch.set_num_threads(MAX_CPU_THREADS)
    model = torch.load(str(local_path), weights_only=False, map_location="cpu")
    model.eval()

    guidances = {}
    guidances_dir = demo_dir / "Guidances"
    for path in sorted(guidances_dir.iterdir()):
        if path.suffix != ".npz":
            continue
        with np.load(path, allow_pickle=True) as data:
            guidance_id = path.stem
            guidances[guidance_id] = {
                "Names": data["Names"].copy(),
                "Positions": data["Positions"].copy(),
            }
    _guidance_templates[demo_name] = guidances
    _cached_models[demo_name] = model
    _model_ready[demo_name] = True


def _ensure_demo_model_loaded(demo_name: str):
    with _model_lock:
        if _model_ready.get(demo_name, False):
            return
        _preload_model(demo_name)


def _load_web_program_class(demo_name: str):
    module_path = WEBPROGRAM_FILES[demo_name]
    module_name = f"local_webprogram_{demo_name}"
    for name in ("Definitions", "LegIK", "Sequence", module_name, "WebProgram"):
        if name in sys.modules:
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WebProgram


def _create_program(demo_name: str):
    from ai4animation import AI4Animation, Time

    with _inference_lock:
        Time.TotalTime = 0.0
        Time.DeltaTime = 0.0
        Time.Timescale = 1.0
        _ensure_demo_model_loaded(demo_name)
        WebProgram = _load_web_program_class(demo_name)
        program = WebProgram()
        program._preloaded_model = _cached_models[demo_name]
        program._preloaded_guidances = _guidance_templates[demo_name]
        AI4Animation(program, mode=AI4Animation.Mode.MANUAL)
        return program, AI4Animation


def _pack_frame(program) -> bytes:
    frame_data = program.get_frame_data()
    (
        root_matrix,
        entity_matrices,
        contacts,
        sim_traj_pos,
        sim_traj_dir,
        ctrl_traj_pos,
        ctrl_traj_dir,
    ) = frame_data[:7]
    current_speed = (
        float(program.GetCurrentSpeed())
        if hasattr(program, "GetCurrentSpeed")
        else 0.0
    )

    root_matrix_f32 = np.asarray(root_matrix, dtype=np.float32)
    entity_matrices_f32 = np.asarray(entity_matrices, dtype=np.float32)
    contacts_f32 = np.asarray(contacts, dtype=np.float32).reshape(-1)
    sim_traj_pos_f32 = np.asarray(sim_traj_pos, dtype=np.float32)
    sim_traj_dir_f32 = np.asarray(sim_traj_dir, dtype=np.float32)
    ctrl_traj_pos_f32 = np.asarray(ctrl_traj_pos, dtype=np.float32)
    ctrl_traj_dir_f32 = np.asarray(ctrl_traj_dir, dtype=np.float32)
    speed_f32 = np.asarray([current_speed], dtype=np.float32)

    return (
        root_matrix_f32.tobytes()
        + entity_matrices_f32.tobytes()
        + contacts_f32.tobytes()
        + sim_traj_pos_f32.tobytes()
        + sim_traj_dir_f32.tobytes()
        + ctrl_traj_pos_f32.tobytes()
        + ctrl_traj_dir_f32.tobytes()
        + speed_f32.tobytes()
    )


def _precise_sleep(target_time: float):
    remaining = target_time - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _neutralize_input(inp: dict):
    neutral = dict(inp)
    neutral["left_stick"] = [0.0, 0.0]
    neutral["right_stick"] = [0.0, 0.0]
    neutral["speed_toggle"] = False
    neutral["canter_boost"] = False
    neutral["walk_modifier"] = False
    neutral["trot_modifier"] = False
    neutral["canter_modifier"] = False
    neutral["action_sit"] = False
    neutral["action_stand"] = False
    neutral["action_lie"] = False
    return neutral


def _tick_paced(ai4animation, program, input_lock, pending_input, pending_input_timestamp, next_frame_time):
    _precise_sleep(next_frame_time)
    with input_lock:
        inp = pending_input.copy()
        last_input_at = pending_input_timestamp[0]
    if (time.perf_counter() - last_input_at) > 0.25:
        inp = _neutralize_input(inp)
    with _inference_lock:
        program.set_inputs(**inp)
        ai4animation.Update(FRAME_DT)
        return _pack_frame(program)


async def _run_active_session(websocket: WebSocket, demo_name: str):
    loop = asyncio.get_running_loop()
    program, AI4Animation = await asyncio.to_thread(_create_program, demo_name)
    if not getattr(program, "_ready", False):
        await websocket.send_json({"type": "error", "message": "Failed to initialize demo."})
        return

    remaining_seconds = SESSION_LIMIT_SECONDS
    await websocket.send_json(
        {
            "type": "init",
            "styles": getattr(program, "GuidanceNames", []),
            "entityNames": program.get_entity_names(),
            "entityCount": len(program.get_entity_names()),
            "sessionLimitSeconds": SESSION_LIMIT_SECONDS,
            "remainingSeconds": remaining_seconds,
            "avgInferenceMs": getattr(program, "AvgInferenceMs", None),
            "demo": demo_name,
        }
    )

    disconnected = asyncio.Event()
    input_lock = threading.Lock()
    pending_input = {
        "left_stick": [0.0, 0.0],
        "right_stick": [0.0, 0.0],
        "speed_toggle": False,
        "guidance_index": 0,
        "canter_boost": False,
        "walk_modifier": False,
        "trot_modifier": False,
        "canter_modifier": False,
        "action_sit": False,
        "action_stand": False,
        "action_lie": False,
    }
    pending_input_timestamp = [time.perf_counter()]

    async def receive_inputs():
        nonlocal pending_input
        try:
            while not disconnected.is_set():
                raw = await websocket.receive_text()
                payload = json.loads(raw)
                if payload.get("type") == "heartbeat":
                    continue
                with input_lock:
                    pending_input = payload
                    pending_input_timestamp[0] = time.perf_counter()
        except (WebSocketDisconnect, RuntimeError):
            disconnected.set()

    async def send_frames():
        next_frame_time = time.perf_counter()
        frame_count = 0
        started_at = time.perf_counter()
        try:
            while not disconnected.is_set():
                elapsed = time.perf_counter() - started_at
                remaining = max(0, SESSION_LIMIT_SECONDS - int(elapsed))
                if remaining <= 0:
                    await websocket.send_json({"type": "timeout", "message": "Your session has expired."})
                    disconnected.set()
                    break
                next_frame_time += FRAME_DT
                frame_bytes = await asyncio.to_thread(
                    _tick_paced,
                    AI4Animation,
                    program,
                    input_lock,
                    pending_input,
                    pending_input_timestamp,
                    next_frame_time,
                )
                await websocket.send_bytes(frame_bytes)
                frame_count += 1
                if frame_count % 60 == 0:
                    await websocket.send_json({"type": "time_update", "remainingSeconds": remaining})
                if frame_count % 15 == 0:
                    await websocket.send_json(
                        {
                            "type": "perf_update",
                            "avgInferenceMs": round(program.AvgInferenceMs, 2)
                            if getattr(program, "AvgInferenceMs", None) is not None
                            else None,
                        }
                    )
                now = time.perf_counter()
                if now > next_frame_time + FRAME_DT:
                    next_frame_time = now
        except (WebSocketDisconnect, RuntimeError):
            disconnected.set()

    await asyncio.gather(receive_inputs(), send_frames())


def mount_interactive_routes(app, session_manager):
    app.mount("/assets/quadruped", StaticFiles(directory=QUADRUPED_ASSETS), name="assets-quadruped")

    @app.get("/play/human")
    @app.get("/play/biped")
    async def play_biped():
        return FileResponse(
            CLIENT_DIR / "biped.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/play/animal")
    @app.get("/play/quadruped")
    async def play_quadruped():
        return FileResponse(
            CLIENT_DIR / "quadruped.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.websocket("/ws-interactive/{demo_name}")
    async def interactive_socket(websocket: WebSocket, demo_name: str):
        await websocket.accept()
        if demo_name not in SOURCE_DEMO_DIRS:
            await websocket.send_json({"type": "error", "message": f"Unknown demo '{demo_name}'."})
            await websocket.close()
            return
        session_token = await session_manager.acquire(websocket)
        if session_token is None:
            return
        try:
            await _run_active_session(websocket, demo_name)
        except Exception as error:
            traceback.print_exc()
            try:
                await websocket.send_json({"type": "error", "message": str(error)})
            except Exception:
                pass
        finally:
            await session_manager.release(websocket, session_token)
