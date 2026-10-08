"""Controls do not represent a pretrained WAM or an RL-trained policy."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
import uuid

import h5py
import numpy as np

from robots_project.types import ActionSpec, ObservationBatch, PolicyOutput


class RandomPolicy:
    version = "random-native-v1"
    synthetic = False

    def __init__(self, action_spec: ActionSpec, chunk_length: int = 4, seed: int = 0):
        self.action_spec = action_spec
        self.chunk_length = chunk_length
        self.seed = seed
        self.rngs = {}

    def specs(self) -> dict:
        return {"policy_version": self.version, "action_spec_id": self.action_spec.id,
                "chunk_length": self.chunk_length, "synthetic": self.synthetic,
                "privileged_policy": False, "real_wam": False,
                "checkpoint_revision": None, "training_extras": None}

    def reset(self, env_ids, episode_ids) -> None:
        for slot, episode in zip(env_ids, episode_ids, strict=True):
            digest = hashlib.sha256(f"{self.seed}:{episode}".encode()).digest()
            self.rngs[slot] = np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def act(self, observation_batch, request_context) -> PolicyOutput:
        actions = np.stack([self.rngs[slot].uniform(
            self.action_spec.low, self.action_spec.high,
            (self.chunk_length, self.action_spec.dimension)).astype(np.float32)
            for slot in request_context["env_ids"]])
        return PolicyOutput(actions, self.action_spec.id, self.version,
                            request_context["request_id"], synthetic=self.synthetic)

    def close(self) -> None:
        self.rngs.clear()


class HoldPolicy(RandomPolicy):
    version = "panda-measured-joint-hold-v1"

    def __init__(self, action_spec: ActionSpec, chunk_length: int = 4):
        super().__init__(action_spec, chunk_length)
        controller = action_spec.semantics.get("control_mode")
        if action_spec.semantics.get("robot_id") != "panda" or controller not in {
                "pd_joint_delta_pos", "pd_joint_pos"} or action_spec.dimension != 8:
            raise ValueError("Hold is defined only for verified Panda joint controllers")
        self.controller = controller

    def act(self, observation_batch: ObservationBatch, request_context) -> PolicyOutput:
        qpos = observation_batch.policy["proprio"]["qpos"]
        actions = np.zeros((len(qpos), self.chunk_length, self.action_spec.dimension), np.float32)
        if self.controller == "pd_joint_pos":
            actions[:, :, :7] = qpos[:, None, :7]
        # The gripper is an absolute, normalized mimic-joint position, not a delta.
        gripper = 2 * (qpos[:, 7] - (-0.01)) / (0.04 - (-0.01)) - 1
        actions[:, :, 7] = np.clip(gripper[:, None], -1, 1)
        return PolicyOutput(actions, self.action_spec.id, self.version, request_context["request_id"])


class ReplayPolicy(RandomPolicy):
    version = "episode-replay-v1"

    def __init__(self, action_spec: ActionSpec, path: Path, chunk_length: int = 4):
        super().__init__(action_spec, chunk_length)
        import json
        self.path = Path(path)
        with h5py.File(path, "r") as f:
            metadata = json.loads(f.attrs["metadata_json"])
            if metadata["env_specs"]["action_spec"]["id"] != action_spec.id:
                raise ValueError("Replay action specification differs from the environment")
            if not metadata["complete"]:
                raise ValueError("Cannot replay an incomplete episode")
            self.actions = f["actions/executed"][:]
            self.source_metadata = metadata
        if len(self.actions) == 0:
            raise ValueError("Cannot replay an empty episode")

    def specs(self) -> dict:
        specs = super().specs()
        specs.update({"source_episode": self.source_metadata["episode_id"],
                      "source_path": str(self.path), "requires_initial_state_restore": True,
                      "privileged_policy": self.source_metadata.get("privileged_policy", False)})
        return specs

    def act(self, observation_batch, request_context) -> PolicyOutput:
        actions = []
        for step in request_context["origin_steps"]:
            indices = np.arange(int(step), int(step) + self.chunk_length)
            if step >= len(self.actions):
                raise ValueError("Replay exhausted before the episode ended")
            actions.append(self.actions[np.minimum(indices, len(self.actions) - 1)])
        return PolicyOutput(np.stack(actions), self.action_spec.id, self.version,
                            request_context["request_id"])


class MockWAM(RandomPolicy):
    version = "mock-wam-synthetic-v1"
    synthetic = True

    def __init__(self, *args, latency_s=0.0, failure=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.latency_s = latency_s
        self.failure = failure

    def act(self, observation_batch, request_context) -> PolicyOutput:
        if self.latency_s:
            time.sleep(self.latency_s)
        output = super().act(observation_batch, request_context)
        if self.failure == "nan":
            output.actions[0, 0, 0] = np.nan
        elif self.failure == "shape":
            output.actions = output.actions[..., :-1]
        offsets = np.arange(1, self.chunk_length + 1, dtype=np.int64)
        # Repeat real inputs only as a synthetic alignment fixture, never as WAM video.
        camera_predictions = {camera: np.repeat(rgb[:, None], self.chunk_length, axis=1)
                              for camera, rgb in observation_batch.policy.get("rgb", {}).items()}
        output.predictions = {"offset_steps": offsets,
                              "offset_times": offsets * request_context["dt_sim"],
                              "rgb": camera_predictions}
        output.prediction_spec = {"synthetic": True, "representation": "rgb",
                                  "preprocessing_version": "sensor-native-rgb-u8-v1",
                                  "includes_current_frame": False}
        return output


def request_id() -> str:
    return str(uuid.uuid4())
