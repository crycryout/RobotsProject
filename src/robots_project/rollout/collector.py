"""Chunked collection. Reset observations are never paired with the previous episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from robots_project.data.store import EpisodeWriter, RunStore
from robots_project.envs.observations import select_observation
from robots_project.policies.controls import request_id
from robots_project.utils import dump_yaml, jsonable, provenance


def episode_manifest(config, count):
    if config.seed_manifest:
        records = [json.loads(line) for line in Path(config.seed_manifest).read_text().splitlines()
                   if line.strip()]
        records = [r for r in records if r["task_id"] == config.env_id and r["split"] == config.split]
        if len(records) < count:
            raise ValueError("Seed manifest has fewer requested episodes than the evaluation budget")
        records = records[:count]
    else:
        records = [{"episode_id": f"{config.split}-{config.env_id}-{config.seed + i}",
                    "task_id": config.env_id, "seed": config.seed + i,
                    "split": config.split, "scene_reference": "seed-and-frozen-batch-order"}
                   for i in range(count)]
    ids = [r["episode_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode IDs in requested manifest")
    for episode_id in ids:
        if not episode_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in episode_id):
            raise ValueError("Episode IDs must be safe neutral filenames")
    return records


def collect(env, policy, config, run_dir: Path, count: int, *, pause_after_episodes=None,
            on_transition=None, replay_source=None):
    from robots_project.policies.invoke import PolicyError, invoke
    if count < 1:
        raise ValueError("episodes must be positive")
    planned = episode_manifest(config, count)
    if config.termination == "evaluation" and count % config.num_envs:
        raise ValueError("Evaluation budget must be divisible by frozen num_envs; padding is forbidden")
    identity = {"config": config.resolved(), "policy": policy.specs(), "planned": planned}
    identity_hash = hashlib.sha256(json.dumps(jsonable(identity), sort_keys=True).encode()).hexdigest()
    manifest = {"schema_version": "1.0", "run_id": run_dir.name,
                "identity_hash": identity_hash, "planned_episodes": planned,
                "env_specs": env.specs(), "policy_specs": policy.specs(),
                "provenance": provenance(Path.cwd()), "complete": False}
    store = RunStore(run_dir, manifest)
    dump_yaml(run_dir / "config.resolved.yaml", config.resolved())
    from robots_project.utils import atomic_json
    atomic_json(run_dir / "provenance.json", store.manifest["provenance"])
    (run_dir / "errors.jsonl").touch(exist_ok=True)
    if store.completed_ids == {r["episode_id"] for r in planned}:
        store.manifest["complete"] = True
        store.save()
        return store.manifest
    start = time.perf_counter()
    writers = {}
    current = {}
    chunks = {}
    latencies = []
    step_times = []
    transitions = 0
    reset_seconds = 0.0
    collected = 0
    failed = False
    policy_failed = False

    def begin_batch(batch_records, slots):
        nonlocal reset_seconds
        ids = [r["episode_id"] for r in batch_records]
        seeds = [r["seed"] for r in batch_records]
        reset_start = time.perf_counter()
        observation = env.reset(slots, ids, seeds)
        if replay_source is not None:
            if config.num_envs != 1 or len(slots) != 1:
                raise ValueError("State-restored replay supports a single environment")
            observation = env.restore_episode(replay_source)
        reset_seconds += time.perf_counter() - reset_start
        policy.reset(slots, ids)
        for slot, record in zip(slots, batch_records, strict=True):
            current[slot] = record
            attempt = sum(r["episode_id"] == record["episode_id"] for r in store.manifest["attempts"])
            metadata = {**record, "env_slot": slot, "run_id": run_dir.name,
                        "attempt": attempt, "policy_version": policy.specs()["policy_version"],
                        "env_specs": env.specs(), "policy_specs": policy.specs(),
                        "provenance": store.manifest["provenance"],
                        "resolved_config": config.resolved(), "termination": config.termination,
                        "synthetic": bool(env.specs()["synthetic"] or policy.specs()["synthetic"]),
                        "privileged_policy": policy.specs()["privileged_policy"]}
            writers[slot] = EpisodeWriter(run_dir, metadata, observation, slot, env.initial_state(slot))
            chunks.pop(slot, None)
        return observation

    def close_writer(slot, reason):
        nonlocal collected
        row = writers.pop(slot).finish(reason)
        store.add_complete(row)
        chunks.pop(slot, None)
        current.pop(slot, None)
        collected += 1

    error_counts = {r["episode_id"]: sum(a["episode_id"] == r["episode_id"] and
                    a["status"] == "infrastructure_error" for a in store.manifest["attempts"])
                    for r in planned}
    pending = [r for r in planned if r["episode_id"] not in store.completed_ids and
               error_counts[r["episode_id"]] <= config.retry_limit]
    if not pending:
        store.manifest["complete"] = len(store.completed_ids) == len(planned)
        store.save()
        policy.close()
        env.close()
        return store.manifest
    # Evaluation resumes whole frozen batches and restores completed initial states.
    # Restrict partial-batch recovery to a full restart rather than changing initial conditions.
    if config.termination == "evaluation" and len(pending) % config.num_envs:
        raise ValueError("Partial evaluation batch requires a new run; frozen batch order cannot change")
    try:
        first = pending[:config.num_envs]
        pending = pending[len(first):]
        observation = begin_batch(first, list(range(len(first))))
        while writers:
            if time.perf_counter() - start > config.max_job_seconds:
                raise TimeoutError("Collection reached configured wall-time limit")
            active = sorted(writers)
            needs = [slot for slot in active if slot not in chunks or
                     chunks[slot]["cursor"] >= config.execution_horizon]
            if needs:
                context = {"env_ids": needs, "episode_ids": [current[s]["episode_id"] for s in needs],
                           "origin_steps": [writers[s].steps for s in needs],
                           "request_id": request_id(), "chunk_length": config.chunk_length,
                           "execution_horizon": config.execution_horizon, "dt_sim": 1 / config.control_freq}
                output, latency = invoke(policy, select_observation(observation, needs), context,
                                         env.action_spec, config.policy_timeout_s)
                latencies.append(latency)
                # Controls already use native units. Real-model normalization belongs in a confirmed adapter.
                denormalized = output.actions.astype(np.float32)
                planned_actions = np.clip(denormalized, env.action_spec.low, env.action_spec.high)
                clip_mask = planned_actions != denormalized
                for i, slot in enumerate(needs):
                    writers[slot].add_request(output, i, writers[slot].steps, planned_actions,
                                              denormalized, clip_mask, config.execution_horizon,
                                              latency, 1 / config.control_freq)
                    chunks[slot] = {"raw": output.actions[i], "denormalized": denormalized[i],
                                    "executed": planned_actions[i], "clip_mask": clip_mask[i],
                                    "request_id": output.request_id, "cursor": 0}
            actions = np.clip(np.zeros((env.num_envs, env.action_spec.dimension), np.float32),
                              env.action_spec.low, env.action_spec.high)
            # Inactive slots are stepped by the backend but excluded from every transition count.
            for slot in active:
                chunk = chunks[slot]
                actions[slot] = chunk["executed"][chunk["cursor"]]
            step_start = time.perf_counter()
            step = env.step(actions)
            step_times.append(time.perf_counter() - step_start)
            observation = step.obs_after_before_reset
            ended = []
            for slot in active:
                chunk = chunks[slot]
                cursor = chunk["cursor"]
                writers[slot].append(step, chunk["raw"][cursor], chunk["denormalized"][cursor],
                                     chunk["clip_mask"][cursor], chunk["request_id"], cursor)
                chunk["cursor"] += 1
                transitions += 1
                if on_transition is not None:
                    on_transition(step, slot)
                should_end = bool(step.truncated[slot]) or (
                    config.termination == "collection" and bool(step.terminated[slot]))
                if should_end:
                    reason = "truncated" if step.truncated[slot] else "terminated"
                    close_writer(slot, reason)
                    ended.append(slot)
            if pause_after_episodes is not None and collected >= pause_after_episodes:
                break
            if config.termination == "evaluation" and ended:
                if writers:
                    raise RuntimeError("Evaluation encountered a partial time-limit boundary")
                batch = pending[:config.num_envs]
                pending = pending[len(batch):]
                if batch:
                    observation = begin_batch(batch, list(range(len(batch))))
            elif pending and ended:
                records = pending[:len(ended)]
                slots = ended[:len(records)]
                pending = pending[len(records):]
                observation = begin_batch(records, slots)
    except (Exception, KeyboardInterrupt) as error:
        failed = True
        policy_failed = isinstance(error, PolicyError)
        for slot, writer in list(writers.items()):
            writer.abort()
            record = current[slot]
            store.add_error({**record, "attempt": writer.metadata["attempt"], "env_slot": slot,
                             "status": "infrastructure_error", "error_type": type(error).__name__,
                             "error": str(error), "steps": writer.steps,
                             "traceback": traceback.format_exc(),
                             "synthetic": writer.metadata["synthetic"],
                             "policy_version": policy.specs()["policy_version"]})
        print(f"Collection failed: {type(error).__name__}: {error}")
    finally:
        for writer in writers.values():
            writer.abort()
        # A timed-out policy thread is abandoned; never call its close() concurrently.
        if not policy_failed:
            policy.close()
        env.close()
        store.manifest["complete"] = len(store.completed_ids) == len(planned)
        store.manifest["last_invocation"] = {
            "elapsed_seconds": time.perf_counter() - start, "transitions": transitions,
            "completed_episodes": collected, "reset_seconds": reset_seconds,
            "policy_latency_seconds": latencies, "env_step_seconds": step_times,
            "failed": failed, "paused_for_test": pause_after_episodes is not None}
        store.save()
    return store.manifest
