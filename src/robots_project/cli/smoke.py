"""Measure state and real sensor RGB capability independently."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from robots_project.config import load_env_config
from robots_project.envs import make_env
from robots_project.envs.observations import flatten
from robots_project.policies import make_policy
from robots_project.policies.controls import request_id
from robots_project.policies.invoke import invoke
from robots_project.utils import atomic_json, dump_yaml, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--sim-backend", choices=["physx_cpu", "physx_cuda"])
    parser.add_argument("--policy", choices=["random", "hold"], default="random")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"exit_code": 1, "capability": "UNTESTED"}
    env = policy = None
    start = time.perf_counter()
    try:
        if args.steps < 1:
            raise ValueError("steps must be positive")
        config = load_env_config(args.config, num_envs=args.num_envs, sim_backend=args.sim_backend)
        dump_yaml(args.output / "config.resolved.yaml", config.resolved())
        env = make_env(config)
        obs = env.reset(list(range(config.num_envs)), [f"smoke-{i}" for i in range(config.num_envs)],
                        [config.seed + i for i in range(config.num_envs)])
        report["specs"] = env.specs()
        report["observations"] = {key: {"shape": list(v.shape), "dtype": str(v.dtype),
                                       "min": float(v.min()), "max": float(v.max())}
                                  for key, v in flatten(obs.backend).items()}
        rgb = obs.policy.get("rgb", {})
        if config.obs_mode == "rgb":
            if set(rgb) != set(config.camera_ids):
                raise ValueError(f"Expected cameras {config.camera_ids}, got {list(rgb)}")
            for camera, frames in rgb.items():
                if frames.dtype != np.uint8 or frames.shape != (
                        config.num_envs, config.camera["height"], config.camera["width"], 3):
                    raise ValueError(f"Invalid RGB sensor shape/dtype for {camera}")
                if np.ptp(frames[0]) == 0:
                    raise ValueError(f"Constant sensor frame for {camera}")
                Image.fromarray(frames[0]).save(args.output / f"{camera}.png")
        initial_qpos = obs.policy.get("proprio", {}).get("qpos")
        policy = make_policy(args.policy, env.action_spec, config)
        policy.reset(list(range(config.num_envs)), env.episode_ids)
        sent = []
        done_counts = 0
        for i in range(args.steps):
            context = {"env_ids": list(range(config.num_envs)), "origin_steps": env.steps.tolist(),
                       "request_id": request_id(), "chunk_length": config.chunk_length,
                       "dt_sim": 1 / config.control_freq}
            output, _ = invoke(policy, obs, context, env.action_spec, config.policy_timeout_s)
            actions = np.clip(output.actions[:, 0], env.action_spec.low, env.action_spec.high)
            step = env.step(actions)
            sent.append(actions)
            obs = step.obs_after_before_reset
            done = step.terminated | step.truncated
            if done.any():
                slots = np.flatnonzero(done).tolist()
                done_counts += len(slots)
                ids = [f"smoke-{s}-{i}" for s in slots]
                obs = env.reset(slots, ids, [config.seed + 1000 + s + i for s in slots])
                policy.reset(slots, ids)
        report.update({"exit_code": 0, "capability": "RGB_OK" if rgb else "STATE_OK",
                       "transitions": args.steps * config.num_envs, "episode_ends": done_counts,
                       "action_min": np.stack(sent).min(axis=(0, 1)),
                       "action_max": np.stack(sent).max(axis=(0, 1))})
        if initial_qpos is not None:
            delta = np.abs(obs.policy["proprio"]["qpos"] - initial_qpos)
            report["max_qpos_displacement_rad"] = float(delta[:, :7].max())
            report["gripper_displacement_m"] = float(delta[:, 7:].max())
            if args.policy == "hold" and float(delta[:, :7].max()) > 0.02:
                raise ValueError("Hold displacement exceeded the declared 0.02 rad tolerance")
    except Exception as error:
        report.update({"exit_code": 1, "capability": "RGB_BLOCKED" if "rgb" in str(args.config) else "STATE_BLOCKED",
                       "error": str(error), "traceback": traceback.format_exc()})
        print(report["traceback"])
    finally:
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
        report["elapsed_seconds"] = time.perf_counter() - start
        report["provenance"] = provenance(Path.cwd())
        atomic_json(args.output / "report.json", report)
    print(f"{report['capability']}: {args.output / 'report.json'}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
