# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import os
import sys
import time
import numpy as np

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
APP_DIR = ROOT_DIR / "app"
DEMO_DIR = APP_DIR / "Demos" / "Locomotion" / "Biped"
ASSETS_PATH = str(APP_DIR / "Demos" / "_ASSETS_" / "Geno")

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, ASSETS_PATH)
sys.path.insert(0, str(DEMO_DIR))
import Definitions

from ai4animation import (
    Actor,
    AI4Animation,
    FABRIK,
    FeedTensor,
    GuidanceModule,
    MotionModule,
    ReadTensor,
    RootModule,
    Rotation,
    Tensor,
    Time,
    TimeSeries,
    Transform,
    Vector3,
    Utility
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

class WebProgram:
    """Locomotion Program adapted for web — no raylib, accepts external inputs.
    Mirrors the original Program.py as closely as possible.
    """

    def __init__(self):
        self.left_stick = [0.0, 0.0]
        self.right_stick = [0.0, 0.0]
        self.speed_toggle = False
        self.guidance_index = 0
        self._prev_guidance_index = 0
        self._ready = False

        self._preloaded_model = None
        self._preloaded_guidances = None
        self.LastInferenceMs = None
        self.AvgInferenceMs = None

    def Start(self):
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Model.glb"),
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

        self.Synchronization = 0.0
        self.Timescale = 1.0

        self.TrajectoryCorrection = 0.25
        self.GuidanceCorrection = 0.0

        self.ControlSeries = TimeSeries(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.SimulationObject = RootModule.Series(self.ControlSeries)

        self.RootControl = RootModule.Series(self.ControlSeries)
        self.GuidanceControl = GuidanceModule.Guidance(
            "Guidance", self.Actor.GetBoneNames(), self.Actor.GetPositions().copy()
        )

        self.GuidanceTemplates = {}
        if self._preloaded_guidances is not None:
            for id, template in self._preloaded_guidances.items():
                self.GuidanceTemplates[id] = GuidanceModule.Guidance(
                    id, template["Names"], template["Positions"].copy()
                )
        else:
            guidances_dir = os.path.join(DEMO_DIR, "Guidances")
            for path in sorted(os.listdir(guidances_dir)):
                with np.load(os.path.join(guidances_dir, path), allow_pickle=True) as data:
                    id = Path(path).stem
                    self.GuidanceTemplates[id] = GuidanceModule.Guidance(
                        id, data["Names"], data["Positions"]
                    )

        self.GuidanceNames = sorted(self.GuidanceTemplates.keys())
        self.GuidanceStyleIndex = 0
        self.SelectedGuidance = self.GuidanceNames[self.GuidanceStyleIndex]
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            self.SelectedGuidance
        ].Positions.copy()

        self.Previous = None
        self.Sequence = None

        self.ContactBones = [
            Definitions.LeftAnkleName,
            Definitions.LeftBallName,
            Definitions.RightAnkleName,
            Definitions.RightBallName,
        ]
        self.ContactIndices = self.Actor.GetBoneIndices(self.ContactBones)

        self.LeftLegIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.LeftHipName),
                self.Actor.GetBone(Definitions.LeftAnkleName),
            ),
            FABRIK(
                self.Actor.GetBone(Definitions.LeftAnkleName),
                self.Actor.GetBone(Definitions.LeftBallName),
            ),
        )
        self.RightLegIK = LegIK(
            FABRIK(
                self.Actor.GetBone(Definitions.RightHipName),
                self.Actor.GetBone(Definitions.RightAnkleName),
            ),
            FABRIK(
                self.Actor.GetBone(Definitions.RightAnkleName),
                self.Actor.GetBone(Definitions.RightBallName),
            ),
        )

        self._noise_buf = Tensor.ToDevice(torch.empty(1, self.Model.LatentDim))
        self._seed_buf = Tensor.ToDevice(torch.zeros(1, self.Model.LatentDim))

        self.Timestamp = Time.TotalTime
        self._prev_guidance_index = self.GuidanceStyleIndex
        self._ready = True

    def set_inputs(self, **inp):
        self.left_stick = inp.get("left_stick", [0.0, 0.0])
        self.right_stick = inp.get("right_stick", [0.0, 0.0])
        self.speed_toggle = inp.get("speed_toggle", False)
        self.guidance_index = inp.get("guidance_index", 0)

    def _set_guidance(self, index):
        self.GuidanceStyleIndex = index % len(self.GuidanceNames)
        self.SelectedGuidance = self.GuidanceNames[self.GuidanceStyleIndex]
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            self.SelectedGuidance
        ].Positions.copy()

    def Update(self):
        self.Control()
        if (
            self.Timestamp == 0.0
            or Time.TotalTime - self.Timestamp > 1.0 / PREDICTION_FPS
        ):
            self.Timestamp = Time.TotalTime
            self.Predict()
        if self.Sequence is not None:
            self.Animate()

    def Control(self):
        speed_sprint = 2.0
        speed_normal = 1.0
        speed = speed_sprint if self.speed_toggle else speed_normal

        if self.guidance_index != self._prev_guidance_index:
            self._set_guidance(self.guidance_index)
            self._prev_guidance_index = self.guidance_index

        velocity = speed * Vector3.ClampMagnitude(
            Vector3.Create(self.left_stick[0], 0, -self.left_stick[1]), 1.0
        )
        direction = Vector3.Create(self.right_stick[0], 0, -self.right_stick[1])
        position = Vector3.Lerp(
            self.SimulationObject.GetPosition(0),
            self.Actor.GetRootPosition(),
            self.Synchronization,
        )

        self.SimulationObject.Control(position, direction, velocity, Time.DeltaTime)

        speed = Vector3.Length(velocity)
        self.GuidanceControl.Positions = self.GuidanceTemplates[
            "Idle" if speed < 0.1 else self.SelectedGuidance
        ].Positions.copy()

        if self.Sequence is not None:
            self.RootControl.Transforms = Transform.Interpolate(
                self.SimulationObject.Transforms,
                self.Sequence.Trajectory.Transforms,
                self.TrajectoryCorrection,
            )
            for i in range(self.RootControl.SampleCount):
                target = Transform.GetPosition(self.RootControl.Transforms)[i:]
                current = self.Actor.GetRootPosition().reshape(-1, 3)
                time = self.RootControl.Timestamps[i:].reshape(-1, 1)
                self.RootControl.Velocities[i] = Tensor.Sum(
                    target - current, axis=0, keepDim=False
                ) / Tensor.Sum(time, axis=0, keepDim=False)
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
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        futureRootTransforms = Transform.TransformationTo(
            self.RootControl.Transforms, root
        )
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
        futureRootVelocities[..., [0, 2]] = (
            futureRootVectors[..., [0, 2]] * SEQUENCE_FPS
        )
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
        self.Sequence.Timestamps = Tensor.LinSpace(
            0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH
        )
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
            self.Previous.SamplePositions(sdt),
            self.Sequence.SamplePositions(sdt),
            blend,
        )
        rotations = Rotation.Interpolate(
            self.Previous.SampleRotations(sdt),
            self.Sequence.SampleRotations(sdt),
            blend,
        )
        velocities = Vector3.Lerp(
            self.Previous.SampleVelocities(sdt),
            self.Sequence.SampleVelocities(sdt),
            blend,
        )
        contacts = Tensor.Interpolate(
            self.Previous.SampleContacts(sdt), self.Sequence.SampleContacts(sdt), blend
        )

        self.Actor.Root = Transform.Interpolate(
            root, self.Actor.Root, self.Sequence.GetRootLock()
        )
        self.Actor.SetTransforms(
            Transform.TR(
                Vector3.Lerp(
                    self.Actor.GetPositions() + velocities * sdt, positions, 0.5
                ),
                rotations,
            )
        )
        self.Actor.SetVelocities(velocities)

        self.Actor.RestoreBoneLengths()
        self.Actor.RestoreBoneAlignments()

        self.LeftLegIK.Solve(
            ankleContact=contacts[0],
            ballContact=contacts[1],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
            poleTarget=Vector3.PositionFrom(
                Vector3.Create(0.0, 0.0, 1.0),
                self.Actor.GetBone(Definitions.LeftKneeName).GetTransform(),
            ),
            poleWeight=1.0,
        )
        self.RightLegIK.Solve(
            ankleContact=contacts[2],
            ballContact=contacts[3],
            maxIterations=self.SolverIterations,
            maxAccuracy=self.SolverAccuracy,
            poleTarget=Vector3.PositionFrom(
                Vector3.Create(0.0, 0.0, 1.0),
                self.Actor.GetBone(Definitions.RightKneeName).GetTransform(),
            ),
            poleWeight=1.0,
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
            self.Sequence.SampleContacts(0.0)
            if self.Sequence is not None
            else np.zeros(4)
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
