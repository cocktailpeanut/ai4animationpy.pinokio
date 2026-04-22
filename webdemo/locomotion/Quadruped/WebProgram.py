import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
APP_DIR = ROOT_DIR / "app"
DEMO_DIR = APP_DIR / "Demos" / "Locomotion" / "Quadruped"
ASSETS_PATH = str(APP_DIR / "Demos" / "_ASSETS_" / "Quadruped")

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, ASSETS_PATH)
sys.path.insert(0, str(DEMO_DIR))
import Definitions

from ai4animation import (
    AI4Animation,
    Actor,
    FABRIK,
    FeedTensor,
    GuidanceModule,
    MotionModule,
    PID,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
    Time,
    TimeSeries,
    Transform,
    Vector3,
    Utility,
)
from LegIK import LegIK
from Sequence import Sequence

MIN_TIMESCALE = 1.0
MAX_TIMESCALE = 1.5
SYNCHRONIZATION_SENSITIVITY = 5
TIMESCALE_SENSITIVITY = 5

SEQUENCE_WINDOW = 0.5
SEQUENCE_LENGTH = 16
SEQUENCE_FPS = 30
PREDICTION_FPS = int(os.environ.get("PREDICTION_FPS", "10"))
CONTACT_POWER = 3.0
CONTACT_THRESHOLD = 2.0 / 3.0
INPUT_DEADZONE = 0.25
ACTION_TRIGGER_SPEED_MAX = 0.5

LOCOMOTION_MODES = {
    "walk": 0.7,
    "pace": 1.2,
    "trot": 2.0,
    "canter": 4.0,
}

class WebProgram:
    def __init__(self):
        self.PIDHistoryLength = 48
        self._preloaded_model = None
        self._preloaded_guidances = None
        self._ready = False

        self.LastInferenceMs = None
        self.AvgInferenceMs = None

        self.left_stick = [0.0, 0.0]
        self.right_stick = [0.0, 0.0]
        self.canter_boost = False
        self.walk_modifier = False
        self.trot_modifier = False
        self.canter_modifier = False
        self.action_sit = False
        self.action_stand = False
        self.action_lie = False

    def Start(self):
        self.Actor = AI4Animation.Scene.AddEntity("Actor_Dog").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Dog.glb"),
            Definitions.FULL_BODY_NAMES,
            True,
        )

        if self._preloaded_model is not None:
            self.Model = self._preloaded_model
        else:
            local_path = os.path.join(DEMO_DIR, "Network.pt")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.Model = torch.load(local_path, weights_only=False, map_location=device)
            self.Model.eval()

        self.SolverIterations = 1
        self.SolverAccuracy = 1e-3
        self.NetworkIterations = int(os.environ.get("NETWORK_ITERATIONS", "3"))

        self.Timescale = 1.0
        self.Synchronization = 1.0
        self.TrajectoryCorrection = 0.33
        self.GuidanceCorrection = 0.33

        self.PID = PID(kp=2.0, ki=0.03, kd=0.0)
        self.PIDSpeedHistory = np.zeros((3, self.PIDHistoryLength), dtype=np.float32)

        self.ControlSeries = TimeSeries(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.SimulationObject = RootModule.Series(self.ControlSeries)
        self.RootControl = RootModule.Series(self.ControlSeries)

        self.GuidanceControl = GuidanceModule.Guidance(
            "Guidance", self.Actor.GetBoneNames(), self.Actor.GetPositions().copy()
        )
        self.GuidanceTemplates = {}
        if self._preloaded_guidances is not None:
            for guidance_id, template in self._preloaded_guidances.items():
                self.GuidanceTemplates[guidance_id] = GuidanceModule.Guidance(
                    guidance_id, template["Names"], template["Positions"].copy()
                )
        else:
            guidances_dir = os.path.join(DEMO_DIR, "Guidances")
            for path in sorted(os.listdir(guidances_dir)):
                with np.load(os.path.join(guidances_dir, path), allow_pickle=True) as data:
                    guidance_id = Path(path).stem
                    self.GuidanceTemplates[guidance_id] = GuidanceModule.Guidance(
                        guidance_id, data["Names"], data["Positions"]
                    )

        self.GuidanceNames = sorted(self.GuidanceTemplates.keys())
        self.CurrentGuidanceState = "Sit"
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            self.CurrentGuidanceState
        ].Positions.copy()

        self.Previous = None
        self.Sequence = None

        self.ContactBones = [
            Definitions.LeftHandSiteName,
            Definitions.RightHandSiteName,
            Definitions.LeftFootSiteName,
            Definitions.RightFootSiteName,
        ]

        self.LeftHandIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.LeftForeArmName),
                self.Actor.GetBone(Definitions.LeftHandSiteName),
            )
        )
        self.RightHandIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.RightForeArmName),
                self.Actor.GetBone(Definitions.RightHandSiteName),
            )
        )
        self.LeftFootIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.LeftKneeName),
                self.Actor.GetBone(Definitions.LeftFootSiteName),
            )
        )
        self.RightFootIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.RightKneeName),
                self.Actor.GetBone(Definitions.RightFootSiteName),
            )
        )

        self._noise_buf = Tensor.ToDevice(torch.empty(1, self.Model.LatentDim))
        self._seed_buf = Tensor.ToDevice(torch.zeros(1, self.Model.LatentDim))

        self.Timestamp = Time.TotalTime
        self._ready = True

    def set_inputs(self, **inp):
        self.left_stick = inp.get("left_stick", [0.0, 0.0])
        self.right_stick = inp.get("right_stick", [0.0, 0.0])
        self.canter_boost = bool(inp.get("canter_boost", False))
        self.walk_modifier = bool(inp.get("walk_modifier", False))
        self.trot_modifier = bool(inp.get("trot_modifier", False))
        self.canter_modifier = bool(inp.get("canter_modifier", False))
        self.action_sit = bool(inp.get("action_sit", False))
        self.action_stand = bool(inp.get("action_stand", False))
        self.action_lie = bool(inp.get("action_lie", False))

    def _select_locomotion_mode_keyboard(self):
        if self.walk_modifier:
            return "walk"
        if self.trot_modifier:
            return "trot"
        if self.canter_modifier:
            return "canter"
        return "pace"

    def _apply_deadzone(self, value, threshold):
        return Vector3.Zero() if Vector3.Length(value) < threshold else value

    def GetCurrentSpeed(self):
        if self.Sequence is None:
            return 0.0
        return float(self.Sequence.GetLength() / SEQUENCE_WINDOW)

    def _UpdatePIDSpeedHistory(self, current_speed, target_speed, pid_speed):
        self.PIDSpeedHistory = np.roll(self.PIDSpeedHistory, -1, axis=1)
        self.PIDSpeedHistory[0, -1] = float(current_speed)
        self.PIDSpeedHistory[1, -1] = float(target_speed)
        self.PIDSpeedHistory[2, -1] = float(pid_speed)

    def Update(self):
        self.Control()
        if (
            self.Timestamp == 0.0
            or Time.TotalTime - self.Timestamp > 1.0 / PREDICTION_FPS
        ):
            self.Timestamp = Time.TotalTime
            self.Predict()
        self.Animate()

    def Control(self):
        current_speed = self.GetCurrentSpeed()
        move_axes = np.asarray(self.left_stick, dtype=np.float32)
        move_axes_magnitude = np.clip(np.linalg.norm(move_axes), 0.0, 1.0)

        if self.canter_boost:
            if move_axes_magnitude > INPUT_DEADZONE:
                desired_speed = move_axes_magnitude * LOCOMOTION_MODES["canter"]
            else:
                desired_speed = 0.0
        else:
            if move_axes_magnitude > INPUT_DEADZONE:
                desired_speed = LOCOMOTION_MODES[self._select_locomotion_mode_keyboard()]
            else:
                desired_speed = 0.0

        can_trigger_action_pose = current_speed < ACTION_TRIGGER_SPEED_MAX
        sit_active = can_trigger_action_pose and self.action_sit
        stand_active = can_trigger_action_pose and self.action_stand
        lie_active = can_trigger_action_pose and self.action_lie

        action_pose_active = sit_active or lie_active or stand_active
        target_speed = 0.0 if action_pose_active else desired_speed

        speed = current_speed + self.PID(current_speed, Time.DeltaTime, setpoint=target_speed)
        speed = max(0.0, speed)

        move_vector = Vector3.ClampMagnitude(
            self._apply_deadzone(Vector3.Create(move_axes[0], 0.0, move_axes[1]), INPUT_DEADZONE),
            1.0,
        )
        move_vector_length = Vector3.Length(move_vector)
        move_direction = (
            Vector3.Zero() if move_vector_length == 0.0 else move_vector / move_vector_length
        )
        facing_axes = np.asarray(self.right_stick, dtype=np.float32)
        facing_vector = Vector3.ClampMagnitude(
            self._apply_deadzone(Vector3.Create(facing_axes[0], 0.0, facing_axes[1]), INPUT_DEADZONE),
            1.0,
        )
        facing_vector_length = Vector3.Length(facing_vector)
        facing_direction = (
            Vector3.Zero() if facing_vector_length == 0.0 else facing_vector / facing_vector_length
        )

        if action_pose_active:
            speed = 0.0
            velocity = Vector3.Zero()
            direction = self.Actor.GetRootDirection()
        else:
            velocity = speed * move_direction
            direction = facing_direction if facing_vector_length > 0.0 else velocity

        self._UpdatePIDSpeedHistory(current_speed, target_speed, speed)

        position = Vector3.Lerp(
            self.SimulationObject.GetPosition(0),
            self.Actor.GetRootPosition(),
            self.Synchronization,
        )
        self.SimulationObject.Control(position, direction, velocity, Time.DeltaTime)

        speed = Vector3.Length(velocity)
        if sit_active:
            guidance_state = "Sit"
        elif lie_active:
            guidance_state = "Lie"
        elif stand_active:
            guidance_state = "Stand"
        elif speed < 0.1:
            guidance_state = "Sit" if self.Sequence is None else "Idle"
        elif speed < LOCOMOTION_MODES["pace"]:
            guidance_state = "Walk"
        elif speed < LOCOMOTION_MODES["trot"]:
            guidance_state = "Pace"
        elif speed < LOCOMOTION_MODES["canter"]:
            guidance_state = "Trot"
        else:
            guidance_state = "Canter"

        self.CurrentGuidanceState = guidance_state
        self.GuidanceControl.Positions = self.GuidanceTemplates[guidance_state].Positions.copy()
        self.RootControl.Transforms = self.SimulationObject.Transforms.copy()
        self.RootControl.Velocities = self.SimulationObject.Velocities.copy()

        if self.Sequence is not None:
            self.RootControl.Transforms = Transform.Interpolate(
                self.SimulationObject.Transforms,
                self.Sequence.Trajectory.Transforms,
                self.TrajectoryCorrection,
            )
            for i in range(self.RootControl.SampleCount):
                target = Transform.GetPosition(self.RootControl.Transforms)[i:]
                current = self.Actor.GetRootPosition().reshape(-1, 3)
                time_slice = self.RootControl.Timestamps[i:].reshape(-1, 1)
                self.RootControl.Velocities[i] = Tensor.Sum(
                    target - current, axis=0, keepDim=False
                ) / Tensor.Sum(time_slice, axis=0, keepDim=False)
            self.RootControl.Velocities = Vector3.Lerp(
                self.RootControl.Velocities,
                self.Sequence.Trajectory.Velocities,
                self.TrajectoryCorrection,
            )
            self.GuidanceControl.Positions = Vector3.Lerp(
                self.GuidanceControl.Positions,
                self.Sequence.SampleGuidance(0.0),
                self.GuidanceCorrection,
            )

    def Predict(self):
        inputs = FeedTensor("X", self.Model.InputDim)
        root = self.Actor.Root

        transforms = Transform.TransformationTo(self.Actor.GetTransforms(), root)
        velocities = Vector3.DirectionTo(self.Actor.GetVelocities(), root)
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(velocities)

        futureRootTransforms = Transform.TransformationTo(self.RootControl.Transforms, root)
        futureRootVelocities = Vector3.DirectionTo(self.RootControl.Velocities, root)
        inputs.FeedVector3(
            Transform.GetPosition(futureRootTransforms), x=True, y=False, z=True
        )
        inputs.FeedVector3(
            Transform.GetAxisZ(futureRootTransforms), x=True, y=False, z=True
        )
        inputs.FeedVector3(futureRootVelocities, x=True, y=False, z=True)
        inputs.Feed(self.GuidanceControl.Positions)

        noise = 0.0
        self._seed_buf.zero_()
        inference_started_at = time.perf_counter()
        with torch.inference_mode():
            outputs, _, _, _ = self.Model(
                inputs.GetTensor().reshape(1, -1),
                noise=(
                    0.5
                    - noise / 2.0
                    + noise * self._noise_buf.uniform_()
                ),
                iterations=self.NetworkIterations,
                seed=self._seed_buf,
            )
        inference_ms = (time.perf_counter() - inference_started_at) * 1000.0
        self.LastInferenceMs = inference_ms
        if self.AvgInferenceMs is None:
            self.AvgInferenceMs = inference_ms
        else:
            self.AvgInferenceMs = 0.85 * self.AvgInferenceMs + 0.15 * inference_ms

        outputs = outputs.reshape(SEQUENCE_LENGTH, -1)
        outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

        futureRootVectors = outputs.ReadVector3()
        futureRootDelta = Tensor.ZerosLike(futureRootVectors)
        for i in range(1, SEQUENCE_LENGTH):
            futureRootDelta[i] = futureRootDelta[i - 1] + futureRootVectors[i]
        futureRootTransforms = Transform.TransformationFrom(
            Transform.DeltaXZ(futureRootDelta), root
        )
        futureRootVelocities = Tensor.ZerosLike(futureRootVectors)
        futureRootVelocities[..., [0, 2]] = futureRootVectors[..., [0, 2]] * SEQUENCE_FPS
        futureRootVelocities = Vector3.DirectionFrom(
            futureRootVelocities, futureRootTransforms
        )

        futureMotionTransforms = Transform.TransformationFrom(
            Transform.TR(
                outputs.ReadVector3(self.Actor.GetBoneCount()),
                outputs.ReadRotation3D(self.Actor.GetBoneCount()),
            ),
            futureRootTransforms.reshape(SEQUENCE_LENGTH, 1, 4, 4),
        )
        futureMotionVelocities = Vector3.DirectionFrom(
            outputs.ReadVector3(self.Actor.GetBoneCount()),
            futureRootTransforms.reshape(SEQUENCE_LENGTH, 1, 4, 4),
        )

        raw_contacts = outputs.Read(4)
        futureContacts = Utility.SmoothStep(raw_contacts, CONTACT_THRESHOLD, CONTACT_POWER)
        futureGuidances = outputs.ReadVector3(self.Actor.GetBoneCount())

        self.Previous = self.Sequence
        self.Sequence = Sequence()
        self.Previous = self.Sequence if self.Previous is None else self.Previous
        self.Sequence.Timestamps = Tensor.LinSpace(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.Sequence.Trajectory = RootModule.Series(
            self.ControlSeries, futureRootTransforms, futureRootVelocities
        )
        self.Sequence.Motion = MotionModule.Series(
            self.ControlSeries,
            self.Actor.GetBoneNames(),
            futureMotionTransforms,
            futureMotionVelocities,
        )
        self.Sequence.Contacts = futureContacts
        self.Sequence.Guidances = futureGuidances

    def Animate(self):
        dt = Time.DeltaTime

        requiredSpeed = (
            Vector3.Distance(
                self.Actor.GetRootPosition(), self.SimulationObject.GetPosition(0)
            )
            + self.SimulationObject.GetLength()
        ) / SEQUENCE_WINDOW
        predictedSpeed = self.Sequence.GetLength() / SEQUENCE_WINDOW
        if requiredSpeed > 0.1 and predictedSpeed > 0.1:
            ts = requiredSpeed / predictedSpeed
            sync = 1.0
        else:
            ts = 1.0
            sync = 0.0
        self.Timescale = Tensor.InterpolateDt(
            self.Timescale, ts, dt, TIMESCALE_SENSITIVITY
        )
        self.Timescale = Tensor.Clamp(self.Timescale, MIN_TIMESCALE, MAX_TIMESCALE)
        self.Synchronization = Tensor.InterpolateDt(
            self.Synchronization, sync, dt, SYNCHRONIZATION_SENSITIVITY
        )

        sdt = dt * self.Timescale
        blend = (Time.TotalTime - self.Timestamp) * PREDICTION_FPS
        root = Transform.Interpolate(
            self.Previous.SampleRoot(sdt), self.Sequence.SampleRoot(sdt), blend
        )
        positions = Vector3.Lerp(
            self.Previous.SamplePositions(sdt), self.Sequence.SamplePositions(sdt), blend
        )
        rotations = Rotation.Interpolate(
            self.Previous.SampleRotations(sdt), self.Sequence.SampleRotations(sdt), blend
        )
        velocities = Vector3.Lerp(
            self.Previous.SampleVelocities(sdt), self.Sequence.SampleVelocities(sdt), blend
        )
        contacts = Tensor.Interpolate(
            self.Previous.SampleContacts(sdt), self.Sequence.SampleContacts(sdt), blend
        )

        self.Actor.Root = Transform.Interpolate(root, self.Actor.Root, self.Sequence.GetRootLock())
        self.Actor.SetTransforms(
            Transform.TR(
                Vector3.Lerp(self.Actor.GetPositions() + velocities * sdt, positions, 0.5),
                rotations,
            )
        )
        self.Actor.SetVelocities(velocities)

        self.Actor.RestoreBoneLengths()
        self.Actor.RestoreBoneAlignments()

        self.LeftHandIK.Solve(
            contact=contacts[0],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
        )
        self.RightHandIK.Solve(
            contact=contacts[1],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
        )
        self.LeftFootIK.Solve(
            contact=contacts[2],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
        )
        self.RightFootIK.Solve(
            contact=contacts[3],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
        )

        self.Actor.SyncToScene()
        self.Previous.Timestamps -= sdt
        self.Sequence.Timestamps -= sdt

    def get_frame_data(self):
        root = self.Actor.Root
        entity_names = list(self.Actor.NameToEntity.keys())
        entity_indices = [self.Actor.NameToEntity[name].Index for name in entity_names]
        entity_transforms = AI4Animation.Scene.Transforms[entity_indices]
        contacts = (
            self.Sequence.SampleContacts(0.0) if self.Sequence is not None else np.zeros(4)
        )

        sim_traj_pos = Transform.GetPosition(self.SimulationObject.Transforms).flatten()
        sim_traj_dir = Transform.GetAxisZ(self.SimulationObject.Transforms).flatten()
        ctrl_traj_pos = Transform.GetPosition(self.RootControl.Transforms).flatten()
        ctrl_traj_dir = Transform.GetAxisZ(self.RootControl.Transforms).flatten()

        return (
            root.flatten(),
            entity_transforms.flatten(),
            contacts,
            sim_traj_pos,
            sim_traj_dir,
            ctrl_traj_pos,
            ctrl_traj_dir,
        )

    def get_entity_names(self):
        return list(self.Actor.NameToEntity.keys())
