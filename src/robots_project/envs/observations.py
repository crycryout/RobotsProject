"""Snapshot native observations without deleting backend fields."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots_project.types import ObservationBatch


def snapshot(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: snapshot(v) for k, v in tree.items()}
    if hasattr(tree, "detach"):
        return tree.detach().cpu().numpy().copy()
    return np.asarray(tree).copy()


def take(tree: Any, indices: list[int] | np.ndarray) -> Any:
    if isinstance(tree, dict):
        return {k: take(v, indices) for k, v in tree.items()}
    return tree[indices].copy()


def select_observation(obs: ObservationBatch, indices: list[int]) -> ObservationBatch:
    return ObservationBatch(take(obs.backend, indices), take(obs.policy, indices))


def flatten(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            if "/" in key:
                raise ValueError("Observation keys may not contain '/' in HDF5 storage")
            result.update(flatten(value, f"{prefix}/{key}" if prefix else key))
        return result
    return {prefix or "state": np.asarray(tree)}


def standardize(native: Any, obs_mode: str) -> ObservationBatch:
    native = snapshot(native)
    policy: dict = {}
    if obs_mode in {"state", "state_dict"}:
        # Explicitly privileged; visual policies must reject this input.
        policy["diagnostic_state"] = native
    if isinstance(native, dict):
        if "agent" in native:
            policy["proprio"] = native["agent"]
        if "sensor_data" in native:
            policy["rgb"] = {cam: data["rgb"] for cam, data in native["sensor_data"].items()
                             if "rgb" in data}
    return ObservationBatch(native, policy)


def normalize_autoreset(obs: Any, info: dict, done: np.ndarray) -> tuple[Any, Any | None]:
    """ManiSkill 3.0.1 final_observation is a full batched native tree, with a mask.

    Preserve terminal values for ended slots and keep reset observations separate.
    Unknown auto-reset layouts are rejected instead of guessed.
    """
    if "final_observation" not in info:
        return snapshot(obs), None
    mask = snapshot(info.get("_final_observation", done)).astype(bool)
    if mask.shape != done.shape or not np.array_equal(mask, done):
        raise ValueError("Automatic-reset terminal mask differs from done slots")
    after = snapshot(obs)
    terminal = snapshot(info["final_observation"])

    def replace(a, b):
        if isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError("Automatic-reset terminal observation topology mismatch")
            return {k: replace(a[k], b[k]) for k in a}
        if a.shape != b.shape or a.shape[0] != len(done):
            raise ValueError("Unsupported automatic-reset observation shape")
        a[mask] = b[mask]
        return a

    return replace(after, terminal), snapshot(obs)
