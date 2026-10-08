"""One native observation per time point, one atomic HDF5 file per episode."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

import h5py
import numpy as np

from robots_project.envs.observations import flatten
from robots_project.utils import array_hash, atomic_json, jsonable, sha256_file

SCHEMA_VERSION = "1.0"


def append_dataset(file, name, value):
    value = np.asarray(value)
    if value.dtype.kind not in "biuf":
        raise ValueError(f"Unsupported stored dtype {value.dtype} at {name}")
    if name not in file:
        kwargs = {}
        if value.ndim >= 3 and value.dtype == np.uint8:
            kwargs = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
        file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                            chunks=(1, *value.shape), dtype=value.dtype, **kwargs)
    ds = file[name]
    if ds.shape[1:] != value.shape or ds.dtype != value.dtype:
        raise ValueError(f"Shape/dtype changed at {name}")
    ds.resize(ds.shape[0] + 1, axis=0)
    ds[-1] = value


class EpisodeWriter:
    def __init__(self, run_dir, metadata, initial_observation, slot, initial_state):
        self.run_dir = Path(run_dir)
        self.metadata = {**metadata, "schema_version": SCHEMA_VERSION, "complete": False}
        self.episode_id = metadata["episode_id"]
        self.slot = slot
        self.path = self.run_dir / "episodes" / f"{self.episode_id}.h5"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.path.with_suffix(".h5.tmp")
        if self.path.exists() or self.temp.exists():
            raise FileExistsError(f"Episode already exists: {self.episode_id}")
        self.file = h5py.File(self.temp, "w")
        self.file.attrs["metadata_json"] = json.dumps(jsonable(self.metadata), sort_keys=True)
        self.steps = 0
        self.requests = {}
        self.success_once = False
        self.success_at_end = False
        self.return_env = 0.0
        self.first_success = None
        self.created = time.perf_counter()
        self.write_seconds = 0.0
        self.obs_keys = set(flatten(initial_observation.backend))
        self.initial_hashes = {}
        for key, value in flatten(initial_state).items():
            self.file.create_dataset("initial_state/" + key, data=value)
            self.initial_hashes[key] = array_hash(value)
        self._append_observation(initial_observation, 0.0)

    def _append_observation(self, observation, sim_time):
        native = flatten(observation.backend)
        if set(native) != self.obs_keys:
            raise ValueError("Native observation topology changed inside an episode")
        for key, value in native.items():
            append_dataset(self.file, "obs/" + key, value[self.slot])
        append_dataset(self.file, "observation_metadata/control_step", np.int64(self.steps))
        append_dataset(self.file, "observation_metadata/sim_time", np.float64(sim_time))

    def add_request(self, output, batch_index, origin_step, planned, denormalized, clip_mask,
                    execution_horizon, latency_s, dt_sim):
        group = self.file.require_group("requests/" + output.request_id)
        group.create_dataset("raw_actions", data=output.actions[batch_index])
        group.create_dataset("denormalized_actions", data=denormalized[batch_index])
        group.create_dataset("planned_actions", data=planned[batch_index])
        group.create_dataset("clip_mask", data=clip_mask[batch_index])
        if output.raw_model_output is not None:
            group.create_dataset("raw_model_output", data=output.raw_model_output[batch_index])
        K = output.actions.shape[1]
        metadata = {
            "episode_id": self.episode_id, "request_id": output.request_id,
            "origin_step": int(origin_step), "origin_sim_time": origin_step * dt_sim,
            "action_sequence_hash": array_hash(planned[batch_index]),
            "raw_action_sequence_hash": array_hash(output.actions[batch_index]),
            "execution_horizon": execution_horizon, "proposed_length": K,
            "executed_prefix": 0, "latency_seconds": latency_s,
            "synthetic": output.synthetic,
            "prediction_spec": output.prediction_spec,
            "model_conversion": output.conversion_metadata,
            "transforms": [{"operation": "identity_native_actions", "frequency_conversion": None,
                            "coordinate_transform": None, "normalization": None},
                           {"operation": "clip_to_action_space",
                            "clipped_components": int(clip_mask[batch_index].sum())}],
        }
        if output.predictions is not None:
            offsets = np.asarray(output.predictions["offset_steps"], dtype=np.int64)
            times = np.asarray(output.predictions["offset_times"], dtype=np.float64)
            if offsets.ndim != 1 or times.shape != offsets.shape or (offsets <= 0).any():
                raise ValueError("Invalid prediction offsets")
            if np.any(np.diff(offsets) <= 0) or not np.isfinite(times).all():
                raise ValueError("Prediction offsets must be finite and strictly increasing")
            if not np.allclose(times, offsets * dt_sim):
                raise ValueError("Prediction frequency mismatch requires a confirmed converter")
            group.create_dataset("prediction/offset_steps", data=offsets)
            group.create_dataset("prediction/offset_times", data=times)
            for camera, frames in output.predictions.get("rgb", {}).items():
                if frames.shape[0] != len(output.actions) or frames.shape[1] != len(offsets):
                    raise ValueError("Prediction RGB batch/time dimensions do not match offsets")
                if frames.dtype != np.uint8:
                    raise ValueError("Native RGB predictions require uint8 preprocessing")
                group.create_dataset("prediction/rgb/" + camera, data=frames[batch_index],
                                     compression="gzip", compression_opts=1)
            for name, latents in output.predictions.get("latents", {}).items():
                if latents.shape[:2] != (len(output.actions), len(offsets)):
                    raise ValueError("Prediction latent shape differs from offsets")
                group.create_dataset("prediction/latents/" + name, data=latents[batch_index])
        if output.training_extras is not None:
            if not output.training_extra_spec:
                raise ValueError("Training extras require declared semantics")
            metadata["training_extra_spec"] = output.training_extra_spec
            for key, values in output.training_extras.items():
                group.create_dataset("training_extras/" + key, data=values[batch_index])
        self.requests[output.request_id] = metadata
        group.attrs["metadata_json"] = json.dumps(jsonable(metadata), sort_keys=True)

    def append(self, step, raw_action, denormalized, clip_mask, request_id, action_index):
        begin = time.perf_counter()
        slot = self.slot
        if step.episode_id[slot] != self.episode_id or step.step_index[slot] != self.steps:
            raise ValueError("Episode ID or contiguous transition index mismatch")
        # Compare the entire before observation against the previous stored observation.
        for key, value in flatten(step.obs_before.backend).items():
            if not np.array_equal(self.file["obs/" + key][-1], value[slot]):
                raise ValueError(f"Non-contiguous observation at {key}")
        for name, value in {
            "actions/raw": raw_action, "actions/denormalized": denormalized,
            "actions/executed": step.action_executed[slot], "actions/clip_mask": clip_mask,
            "transitions/reward_env": step.reward_env[slot],
            "transitions/terminated": np.bool_(step.terminated[slot]),
            "transitions/truncated": np.bool_(step.truncated[slot]),
            "transitions/success": np.bool_(step.success[slot]),
            "transitions/step_index": np.int64(self.steps),
            "transitions/dt_sim": np.float64(step.dt_sim),
            "transitions/action_index": np.int64(action_index),
        }.items():
            append_dataset(self.file, name, value)
        ds = self.file.get("transitions/request_id")
        if ds is None:
            ds = self.file.create_dataset("transitions/request_id", (0,), maxshape=(None,),
                                          dtype=h5py.string_dtype("utf-8"))
        ds.resize(len(ds) + 1, axis=0)
        ds[-1] = request_id
        self.requests[request_id]["executed_prefix"] += 1
        self.success_at_end = bool(step.success[slot])
        self.success_once |= self.success_at_end
        self.return_env += float(step.reward_env[slot])
        self.steps += 1
        if self.success_at_end and self.first_success is None:
            self.first_success = self.steps
        self._append_observation(step.obs_after_before_reset, self.steps * step.dt_sim)
        if step.reset_observation is not None and step.reset_mask[slot]:
            group = self.file.require_group(f"reset_events/{self.steps}")
            for key, value in flatten(step.reset_observation.backend).items():
                group.create_dataset(key, data=value[slot])
            group.attrs["reason"] = "backend_auto_reset; never used as terminal observation"
        self.write_seconds += time.perf_counter() - begin

    def _finalize_requests(self, ending_reason):
        for key, metadata in self.requests.items():
            group = self.file["requests/" + key]
            metadata["ending_reason"] = ending_reason
            group.attrs["metadata_json"] = json.dumps(jsonable(metadata), sort_keys=True)
            if "prediction" not in group:
                continue
            offsets = group["prediction/offset_steps"][:]
            clips = group["clip_mask"][:].any(axis=-1)
            valid, reasons = [], []
            for offset in offsets:
                if offset > metadata["executed_prefix"]:
                    reason = "unexecuted_or_replanned_prefix"
                elif clips[:offset].any():
                    reason = "clipped_action_prefix"
                else:
                    reason = "valid"
                valid.append(reason == "valid")
                reasons.append(reason)
            group.create_dataset("prediction/valid", data=np.asarray(valid, dtype=bool))
            group.create_dataset("prediction/invalid_reason", data=reasons,
                                 dtype=h5py.string_dtype("utf-8"))

    def finish(self, ending_reason):
        self._finalize_requests(ending_reason)
        self.metadata.update({"complete": True, "transition_count": self.steps,
                              "ending_reason": ending_reason,
                              "initial_state_hashes": self.initial_hashes,
                              "elapsed_wall_seconds": time.perf_counter() - self.created,
                              "write_seconds": self.write_seconds})
        self.file.attrs["metadata_json"] = json.dumps(jsonable(self.metadata), sort_keys=True)
        self.file.flush()
        self.file.close()
        with self.temp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(self.temp, self.path)
        dir_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        checksum = sha256_file(self.path)
        dt = self.metadata["env_specs"]["dt_sim"]
        row = {
            "episode_id": self.episode_id, "attempt": self.metadata.get("attempt", 0),
            "task_id": self.metadata["task_id"], "split": self.metadata["split"],
            "seed": self.metadata["seed"], "env_slot": self.slot,
            "status": "complete", "success_once": int(self.success_once),
            "success_at_end": int(self.success_at_end), "return_env": self.return_env,
            "steps": self.steps, "steps_to_first_success": self.first_success,
            "time_to_first_success": self.first_success * dt if self.first_success else None,
            "wall_seconds": self.metadata["elapsed_wall_seconds"],
            "write_seconds": self.write_seconds, "ending_reason": ending_reason,
            "trajectory": str(self.path.relative_to(self.run_dir)), "sha256": checksum,
            "bytes_written": self.path.stat().st_size,
            "synthetic": self.metadata["synthetic"],
            "privileged_policy": self.metadata["privileged_policy"],
            "policy_version": self.metadata["policy_version"],
        }
        # Sidecars allow recovery if the process dies between the HDF5 rename and manifest update.
        atomic_json(self.path.with_suffix(".json"), row)
        return row

    def abort(self):
        if self.file.id.valid:
            self.file.flush()
            self.file.close()


class RunStore:
    def __init__(self, run_dir, manifest):
        self.path = Path(run_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        self.quarantined = []
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest["identity_hash"] != manifest["identity_hash"]:
                raise ValueError("Resume configuration, policy, or episode manifest differs")
            self.recover()
        else:
            self.manifest = manifest
            self.manifest.setdefault("episodes", [])
            self.manifest.setdefault("attempts", [])
            self.manifest.setdefault("pending_attempts", [])
            self.manifest.setdefault("recovery", [])
            self.save()

    def save(self):
        atomic_json(self.manifest_path, self.manifest)

    def recover(self):
        from robots_project.data.validation import validate_episode
        episode_dir = self.path / "episodes"
        episode_dir.mkdir(exist_ok=True)
        checksums = {r["episode_id"]: r["sha256"] for r in self.manifest["episodes"]}
        good = []
        for file in episode_dir.glob("*.h5"):
            try:
                report = validate_episode(file, checksums.get(file.stem))
                sidecar = file.with_suffix(".json")
                if not sidecar.exists():
                    atomic_json(sidecar, reconstruct_row(file, self.path))
                row = json.loads(sidecar.read_text())
                if row["episode_id"] != file.stem or row["sha256"] != report["sha256"]:
                    raise ValueError("Episode sidecar ID/checksum mismatch")
                good.append(row)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                self._quarantine(file, str(error))
                if file.with_suffix(".json").exists():
                    self._quarantine(file.with_suffix(".json"), "invalid episode sidecar")
        for file in episode_dir.glob("*.tmp"):
            self._quarantine(file, "interrupted write")
        self.manifest["episodes"] = sorted(good, key=lambda r: r["episode_id"])
        available = {row["episode_id"] for row in good}
        for row in self.manifest["attempts"]:
            if row["status"] == "complete" and row["episode_id"] not in available:
                row.update({"status": "infrastructure_error", "error_type": "TrajectoryIntegrityError",
                            "error": "Completed trajectory unavailable or invalid after recovery"})
                with (self.path / "errors.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
        recorded = {(r["episode_id"], r.get("attempt", 0)) for r in
                    self.manifest["attempts"] if r["status"] == "complete"}
        self.manifest["attempts"].extend(r for r in good if
                                       (r["episode_id"], r.get("attempt", 0)) not in recorded)
        for row in list(self.manifest.get("pending_attempts", [])):
            if row["episode_id"] not in available:
                self.add_error({**row, "status": "infrastructure_error",
                                "error_type": "InterruptedCollection",
                                "error": "Process stopped before the episode was atomically committed",
                                "steps": None})
        self.manifest["pending_attempts"] = []
        self.manifest["recovery"].extend(self.quarantined)
        self.save()

    def _quarantine(self, file, reason):
        destination = self.path / "quarantine" / f"{file.name}.{uuid.uuid4().hex[:8]}"
        destination.parent.mkdir(exist_ok=True)
        os.replace(file, destination)
        self.quarantined.append({"path": str(destination.relative_to(self.path)), "reason": reason})

    @property
    def completed_ids(self):
        return {row["episode_id"] for row in self.manifest["episodes"]}

    def add_complete(self, row):
        if row["episode_id"] in self.completed_ids:
            raise ValueError("Duplicate completed episode ID")
        self.manifest["episodes"].append(row)
        self.manifest["attempts"].append(row)
        self._remove_pending(row)
        self.save()

    def start_attempts(self, rows):
        self.manifest.setdefault("pending_attempts", []).extend(rows)
        self.save()

    def _remove_pending(self, row):
        self.manifest["pending_attempts"] = [item for item in self.manifest.get("pending_attempts", [])
                                            if (item["episode_id"], item["attempt"]) !=
                                            (row["episode_id"], row["attempt"])]

    def add_error(self, record):
        self.manifest["attempts"].append(record)
        self._remove_pending(record)
        with (self.path / "errors.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsonable(record), sort_keys=True, allow_nan=False) + "\n")
        self.save()


def reconstruct_row(path, run_dir):
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["metadata_json"])
        success = f["transitions/success"][:]
        found = np.flatnonzero(success)
        first = int(found[0]) + 1 if len(found) else None
        dt = meta["env_specs"]["dt_sim"]
        return {"episode_id": meta["episode_id"], "attempt": meta.get("attempt", 0),
                "task_id": meta["task_id"], "split": meta["split"], "seed": meta["seed"],
                "env_slot": meta["env_slot"], "status": "complete",
                "success_once": int(success.any()), "success_at_end": int(success[-1]),
                "return_env": float(f["transitions/reward_env"][:].sum(dtype=np.float64)),
                "steps": meta["transition_count"], "steps_to_first_success": first,
                "time_to_first_success": first * dt if first else None,
                "wall_seconds": meta["elapsed_wall_seconds"], "write_seconds": meta["write_seconds"],
                "ending_reason": meta["ending_reason"], "trajectory": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path), "bytes_written": path.stat().st_size,
                "synthetic": meta["synthetic"], "privileged_policy": meta["privileged_policy"],
                "policy_version": meta["policy_version"]}
