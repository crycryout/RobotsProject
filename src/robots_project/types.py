"""Backend-independent data contracts; dimensions always come from specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class ActionSpec:
    id: str
    dimension: int
    low: np.ndarray
    high: np.ndarray
    semantics: dict

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self, actions: np.ndarray, batch_size: int, chunk_length: int) -> None:
        if actions.shape != (batch_size, chunk_length, self.dimension):
            raise ValueError(f"Policy actions shape {actions.shape}; expected "
                             f"{(batch_size, chunk_length, self.dimension)}")
        if actions.dtype.kind != "f" or not np.isfinite(actions).all():
            raise ValueError("Policy actions must be finite floating-point values")


@dataclass
class ObservationBatch:
    backend: Any
    policy: dict[str, Any]


@dataclass
class StepBatch:
    obs_before: ObservationBatch
    action_executed: np.ndarray
    obs_after_before_reset: ObservationBatch
    reward_env: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    success: np.ndarray
    episode_id: list[str]
    env_slot: np.ndarray
    step_index: np.ndarray
    dt_sim: float
    reset_observation: ObservationBatch | None = None
    reset_mask: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PolicyOutput:
    actions: np.ndarray
    action_spec_id: str
    policy_version: str
    request_id: str
    predictions: dict | None = None
    prediction_spec: dict | None = None
    policy_state: Any = None
    training_extras: dict | None = None
    training_extra_spec: dict | None = None
    synthetic: bool = False
    raw_model_output: np.ndarray | None = None
    conversion_metadata: dict | None = None


class PolicyAdapter(Protocol):
    def specs(self) -> dict: ...
    def reset(self, env_ids: list[int], episode_ids: list[str]) -> None: ...
    def act(self, observation_batch: ObservationBatch, request_context: dict) -> PolicyOutput: ...
    def close(self) -> None: ...


class EnvAdapter(Protocol):
    num_envs: int
    action_spec: ActionSpec

    def specs(self) -> dict: ...
    def reset(self, env_ids: list[int], episode_ids: list[str], seeds: list[int]) -> ObservationBatch: ...
    def step(self, actions: np.ndarray) -> StepBatch: ...
    def initial_state(self, slot: int) -> dict: ...
    def close(self) -> None: ...
