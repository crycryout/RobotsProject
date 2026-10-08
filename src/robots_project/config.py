"""Strict, versioned single-run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

import yaml


@dataclass
class EnvConfig:
    schema_version: str = "1.0"
    backend: str = "maniskill"
    backend_version: str = "3.0.1"
    env_id: str = "PushCube-v1"
    robot_id: str = "panda"
    obs_mode: str = "state_dict"
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "physx_cuda"
    render_backend: str = "gpu"
    num_envs: int = 1
    episode_horizon: int = 50
    control_freq: int = 20
    sim_freq: int = 100
    camera: dict = field(default_factory=lambda: {
        "width": 128, "height": 128, "shader_pack": "minimal"})
    camera_ids: list[str] = field(default_factory=lambda: ["base_camera"])
    seed_manifest: str | None = None
    split: str = "smoke"
    seed: int = 17000
    reconfiguration_freq: int = 0
    auto_reset: bool = False
    termination: str = "collection"
    chunk_length: int = 4
    execution_horizon: int = 2
    policy_timeout_s: float = 30.0
    max_job_seconds: float = 7200.0
    retry_limit: int = 0

    def validate(self) -> "EnvConfig":
        if self.schema_version != "1.0":
            raise ValueError("Unsupported configuration schema_version")
        if self.backend not in {"maniskill", "fake"}:
            raise ValueError("backend must be maniskill or fake")
        if self.obs_mode not in {"state", "state_dict", "rgb"}:
            raise ValueError("obs_mode must be state, state_dict, or rgb")
        if self.sim_backend not in {"physx_cpu", "physx_cuda"}:
            raise ValueError("sim_backend must be physx_cpu or physx_cuda")
        if self.sim_backend == "physx_cpu" and self.num_envs != 1:
            raise ValueError("CPU simulation currently supports one slot per process")
        for name in ("num_envs", "episode_horizon", "control_freq", "sim_freq", "chunk_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.execution_horizon, int) or isinstance(self.execution_horizon, bool) or not 1 <= self.execution_horizon <= self.chunk_length:
            raise ValueError("Require 1 <= execution_horizon H <= chunk_length K")
        if self.sim_freq % self.control_freq:
            raise ValueError("sim_freq must be divisible by control_freq")
        if not math.isfinite(self.policy_timeout_s) or self.policy_timeout_s <= 0 or not 0 < self.max_job_seconds <= 7200:
            raise ValueError("Timeout must be positive and job limit must be in (0, 7200] seconds")
        if self.retry_limit not in range(4):
            raise ValueError("retry_limit must be an integer in [0, 3]")
        if self.termination not in {"collection", "evaluation"}:
            raise ValueError("termination must be collection or evaluation")
        if self.termination == "evaluation" and (self.reconfiguration_freq != 1 or self.auto_reset):
            raise ValueError("Evaluation requires reconfiguration_freq=1 and explicit full resets")
        if not isinstance(self.camera, dict) or set(self.camera) - {"width", "height", "shader_pack"}:
            raise ValueError("camera only supports width, height, shader_pack")
        for key in ("width", "height"):
            if not isinstance(self.camera.get(key), int) or self.camera[key] < 1:
                raise ValueError(f"camera.{key} must be a positive integer")
        if self.camera.get("shader_pack") not in {"minimal", "default"}:
            raise ValueError("Only raster shader packs minimal and default are supported")
        return self

    def resolved(self) -> dict:
        return asdict(self)


def load_env_config(path: Path | str, **overrides) -> EnvConfig:
    with Path(path).open(encoding="utf-8") as f:
        values = yaml.safe_load(f)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    values.update({k: v for k, v in overrides.items() if v is not None})
    return EnvConfig(**values).validate()
