"""Record official ManiSkill motion-planning controls without training a new policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from robots_project.config import load_env_config
from robots_project.data.store import EpisodeWriter, RunStore
from robots_project.envs import make_env
from robots_project.evaluation.artifacts import summarize, write_episode_table
from robots_project.policies.controls import HoldPolicy, request_id
from robots_project.types import PolicyOutput
from robots_project.utils import atomic_json, dump_yaml, jsonable, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--horizon", type=int, default=200)
    args = parser.parse_args()
    config = load_env_config(args.config, num_envs=1, sim_backend="physx_cpu",
                             control_mode="pd_joint_pos", episode_horizon=args.horizon,
                             chunk_length=1, execution_horizon=1, seed=args.seed,
                             split="control", reconfiguration_freq=0, auto_reset=False)
    if (args.run_dir / "manifest.json").exists():
        raise ValueError("Positive-control run already exists")
    env = make_env(config)
    import gymnasium as gym
    from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve as pick
    from mani_skill.examples.motionplanning.panda.solutions.push_cube import solve as push
    solutions = {"PushCube-v1": push, "PickCube-v1": pick}
    if config.env_id not in solutions:
        raise ValueError("No official solution configured for this task")
    record = {"episode_id": f"control-{config.env_id}-{args.seed}",
              "task_id": config.env_id, "seed": args.seed, "split": "control"}
    policy_specs = {"policy_version": "official-maniskill-motionplanning-v3.0.1",
                    "privileged_policy": True, "synthetic": False, "real_wam": False,
                    "source": "mani_skill.examples.motionplanning.panda.solutions"}
    manifest = {"schema_version": "1.0", "run_id": args.run_dir.name,
                "identity_hash": hashlib.sha256(json.dumps(jsonable(config.resolved()), sort_keys=True).encode()).hexdigest(),
                "planned_episodes": [record], "env_specs": env.specs(), "policy_specs": policy_specs,
                "provenance": provenance(Path.cwd()), "complete": False}
    store = RunStore(args.run_dir, manifest)
    writer = None
    start = time.perf_counter()
    failure = None

    class Capture(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            nonlocal writer
            if writer is not None:
                raise RuntimeError("Official solution unexpectedly reset an in-progress episode")
            obs = env.reset([0], [record["episode_id"]], [args.seed if seed is None else seed])
            writer = EpisodeWriter(args.run_dir, {**record, "env_slot": 0, "attempt": 0,
                "policy_version": policy_specs["policy_version"], "env_specs": env.specs(),
                "policy_specs": policy_specs, "provenance": store.manifest["provenance"],
                "resolved_config": config.resolved(), "termination": "evaluation",
                "synthetic": False, "privileged_policy": True}, obs, 0, env.initial_state(0))
            return obs.backend, {}

        def step(self, action):
            if writer is None or writer.steps >= config.episode_horizon:
                raise RuntimeError("Official solution exceeded the control horizon")
            array = np.asarray(action, dtype=np.float32).reshape(1, 1, -1)
            rid = request_id()
            output = PolicyOutput(array, env.action_spec.id, policy_specs["policy_version"], rid)
            planned = np.clip(array, env.action_spec.low, env.action_spec.high)
            clips = planned != array
            writer.add_request(output, 0, writer.steps, planned, array, clips, 1, 0.0,
                               1 / config.control_freq)
            transition = env.step(planned[:, 0])
            writer.append(transition, array[0, 0], array[0, 0], clips[0, 0], rid, 0)
            return (transition.obs_after_before_reset.backend, transition.reward_env,
                    transition.terminated, transition.truncated,
                    {"success": transition.success, "elapsed_steps": env.steps.copy()})

    capture = Capture(env.env)
    try:
        result = solutions[config.env_id](capture, seed=args.seed, debug=False, vis=False)
        if isinstance(result, int) and result == -1:
            raise RuntimeError("Official motion planner reported planning failure")
        hold = HoldPolicy(env.action_spec, chunk_length=1)
        while writer.steps < config.episode_horizon:
            if time.perf_counter() - start >= config.max_job_seconds:
                raise TimeoutError("Positive control exceeded its wall-time budget")
            output = hold.act(env.observation, {"request_id": request_id()})
            capture.step(output.actions[0, 0])
        row = writer.finish("truncated")
        store.add_complete(row)
        store.manifest["complete"] = True
        print(f"Official control: success_once={row['success_once']}, "
              f"success_at_end={row['success_at_end']}, steps={row['steps']}")
    except Exception as error:
        failure = str(error)
        if writer is not None:
            writer.abort()
        store.add_error({**record, "status": "infrastructure_error", "attempt": 0,
                         "steps": writer.steps if writer else 0, "error": str(error),
                         "error_type": type(error).__name__, "traceback": traceback.format_exc(),
                         "synthetic": False, "policy_version": policy_specs["policy_version"]})
        print(traceback.format_exc())
    finally:
        env.close()
        store.manifest["elapsed_seconds"] = time.perf_counter() - start
        store.save()
        dump_yaml(args.run_dir / "config.resolved.yaml", config.resolved())
        atomic_json(args.run_dir / "provenance.json", store.manifest["provenance"])
        (args.run_dir / "errors.jsonl").touch(exist_ok=True)
        write_episode_table(args.run_dir, store.manifest)
        summary = summarize(args.run_dir)
    return 0 if failure is None and summary["macro_success_once"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
