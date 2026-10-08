"""Validate actual episode structure, hashes, and temporal/prediction contracts."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from robots_project.utils import array_hash, sha256_file


def datasets(group):
    result = {}
    group.visititems(lambda name, obj: result.update({name: obj}) if
                     isinstance(obj, h5py.Dataset) else None)
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_episode(path: Path, expected_checksum: str | None = None) -> dict:
    path = Path(path)
    checksum = sha256_file(path)
    require(expected_checksum is None or checksum == expected_checksum, "Episode checksum mismatch")
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["metadata_json"])
        require(meta["schema_version"] == "1.0", "Unsupported trajectory schema")
        require(meta["complete"] is True, "Incomplete episode")
        require(meta["episode_id"] == path.stem, "Episode ID differs from filename")
        T = meta["transition_count"]
        require(T > 0, "Empty completed episode")
        require(len(f["actions/executed"]) == T, "Transition count differs")
        spec = meta["env_specs"]["action_spec"]
        D = spec["dimension"]
        for name in ("raw", "denormalized", "executed", "clip_mask"):
            require(f[f"actions/{name}"].shape == (T, D), f"Action {name} shape mismatch")
        executed = f["actions/executed"][:]
        require(executed.dtype.kind == "f", "Executed actions require floating dtype")
        require(np.all(executed >= np.asarray(spec["low"]) - 1e-6) and
                np.all(executed <= np.asarray(spec["high"]) + 1e-6), "Actions violate specifications")
        for name, ds in datasets(f).items():
            if ds.dtype.kind == "f":
                # Read in bounded slices so validation scales to larger episodes.
                if ds.ndim == 0:
                    require(np.isfinite(ds[()]), f"Non-finite value at {name}")
                else:
                    for begin in range(0, len(ds), 32):
                        require(np.isfinite(ds[begin:begin + 32]).all(), f"Non-finite values at {name}")
        for name, ds in datasets(f["obs"]).items():
            require(len(ds) == T + 1, f"Observation T+1 convention failed at {name}")
            if name.endswith("/rgb"):
                require(ds.dtype == np.uint8 and ds.ndim == 4 and ds.shape[-1] == 3,
                        f"Native RGB dtype/shape mismatch at {name}")
        for name, ds in datasets(f["transitions"]).items():
            require(len(ds) == T, f"Transition field length differs at {name}")
        for name in ("terminated", "truncated", "success"):
            require(f["transitions/" + name].dtype.kind == "b", f"{name} must be boolean")
        indices = f["transitions/step_index"][:]
        require(np.array_equal(indices, np.arange(T)), "Non-contiguous step indices")
        require(not f["transitions/truncated"][:-1].any(), "Truncation crossed an episode boundary")
        if meta["termination"] == "collection":
            require(not f["transitions/terminated"][:-1].any(), "Termination crossed an episode boundary")
        require(bool(f["transitions/truncated"][-1] or f["transitions/terminated"][-1]),
                "Completed episode has no ending flag")
        dt = meta["env_specs"]["dt_sim"]
        require(np.allclose(f["transitions/dt_sim"][:], dt), "dt_sim differs from specifications")
        require(np.array_equal(f["observation_metadata/control_step"][:], np.arange(T + 1)),
                "Observation control-step ordering differs")
        require(np.allclose(f["observation_metadata/sim_time"][:], np.arange(T + 1) * dt),
                "Observation timestamp ordering differs")
        for key, expected_hash in meta["initial_state_hashes"].items():
            require(array_hash(f["initial_state/" + key][:]) == expected_hash,
                    "Initial-state hash mismatch")
        request_ids = f["transitions/request_id"].asstr()[:]
        action_indices = f["transitions/action_index"][:]
        require(set(request_ids) == set(f["requests"]), "Request identity mismatch")
        for request_id, group in f["requests"].items():
            request = json.loads(group.attrs["metadata_json"])
            require(request["episode_id"] == meta["episode_id"] and request["request_id"] == request_id,
                    "Request belongs to another episode")
            K, H = request["proposed_length"], request["execution_horizon"]
            require(1 <= H <= K, "Invalid chunk horizon")
            for field in ("raw_actions", "denormalized_actions", "planned_actions", "clip_mask"):
                require(group[field].shape == (K, D), f"Request {field} shape differs")
            require(array_hash(group["planned_actions"][:]) == request["action_sequence_hash"],
                    "Action sequence hash mismatch")
            require(array_hash(group["raw_actions"][:]) == request["raw_action_sequence_hash"],
                    "Raw action sequence hash mismatch")
            where = np.flatnonzero(request_ids == request_id)
            prefix = len(where)
            require(prefix == request["executed_prefix"] and prefix <= H, "Executed prefix mismatch")
            require(np.array_equal(where, request["origin_step"] + np.arange(prefix)),
                    "Request execution is not a contiguous prefix")
            require(np.array_equal(action_indices[where], np.arange(prefix)), "Per-slot action cursor differs")
            require(np.array_equal(executed[where], group["planned_actions"][:prefix]),
                    "Actually executed actions differ from request prefix")
            if "prediction" in group:
                pred = group["prediction"]
                offsets = pred["offset_steps"][:]
                require(offsets.ndim == 1 and (offsets > 0).all() and (np.diff(offsets) > 0).all(),
                        "Prediction offsets invalid")
                require(np.allclose(pred["offset_times"][:], offsets * dt), "Prediction timing differs")
                require(len(pred["valid"]) == len(offsets), "Prediction validity length mismatch")
                clip_steps = group["clip_mask"][:].any(axis=-1)
                expected_valid = np.asarray([o <= prefix and not clip_steps[:o].any() for o in offsets])
                require(np.array_equal(pred["valid"][:], expected_valid), "Prediction validity prefix mismatch")
    return {"episode_id": meta["episode_id"], "transitions": T, "sha256": checksum,
            "synthetic": meta["synthetic"], "valid": True}


def validate_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("kind") == "evaluation_suite":
        reports, errors = [], []
        for task in manifest["tasks"]:
            child = run_dir / task["task_id"]
            if sha256_file(child / "manifest.json") != task["sha256"]:
                errors.append({"path": task["manifest"], "error": "Task manifest checksum mismatch"})
            report = validate_run(child)
            reports.extend(report["episodes"])
            errors.extend(report["errors"])
        return {"valid": not errors, "episodes": reports, "errors": errors,
                "complete_run": manifest["complete"]}
    reports, errors = [], []
    ids = [row["episode_id"] for row in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        errors.append({"path": "manifest.json", "error": "Duplicate episode IDs"})
    listed = {row["trajectory"] for row in manifest["episodes"]}
    actual = {str(p.relative_to(run_dir)) for p in (run_dir / "episodes").glob("*.h5")}
    if actual != listed:
        errors.append({"path": "manifest.json", "error": "File list differs from manifest"})
    if list((run_dir / "episodes").glob("*.tmp")):
        errors.append({"path": "episodes", "error": "Interrupted temporary files require recovery"})
    for row in manifest["episodes"]:
        try:
            reports.append(validate_episode(run_dir / row["trajectory"], row["sha256"]))
        except (OSError, ValueError, KeyError) as error:
            errors.append({"path": row["trajectory"], "error": str(error)})
    return {"schema_version": "1.0", "valid": not errors, "episodes": reports, "errors": errors,
            "complete_run": len(ids) == len(manifest["planned_episodes"])}
