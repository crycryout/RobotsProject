"""Collect native simulator episodes with atomic recovery and action chunks."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from robots_project.config import load_env_config
from robots_project.envs import make_env
from robots_project.evaluation.artifacts import summarize, write_episode_table
from robots_project.policies import make_policy
from robots_project.rollout.collector import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=["random", "hold", "mock", "replay"], default="random")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-path", type=Path)
    parser.add_argument("--stop-after-episodes", type=int,
                        help="Controlled interruption for recovery acceptance; returns exit 3")
    args = parser.parse_args()
    config = load_env_config(args.config, num_envs=args.num_envs)
    if args.policy == "replay":
        if args.episodes != 1 or args.replay_path is None:
            raise ValueError("Replay collection requires --episodes 1 and --replay-path; use replay_check for state restoration")
        import h5py
        import json
        with h5py.File(args.replay_path, "r") as f:
            source = json.loads(f.attrs["metadata_json"])
        config.seed = source["seed"]
        config.split = "replay"
        config.seed_manifest = None
    run_dir = args.run_dir or Path("outputs") / f"{time.strftime('%Y%m%dT%H%M%S')}-{config.env_id}-{args.policy}"
    if (run_dir / "manifest.json").exists() and not args.resume:
        raise ValueError("Run exists; --resume is required to prevent accidental overwrites")
    if args.resume and not (run_dir / "manifest.json").exists():
        raise ValueError("Cannot resume a missing run")
    env = make_env(config)
    try:
        policy = make_policy(args.policy, env.action_spec, config, replay_path=args.replay_path)
        manifest = collect(env, policy, config, run_dir, args.episodes,
                           pause_after_episodes=args.stop_after_episodes,
                           replay_source=args.replay_path if args.policy == "replay" else None)
    finally:
        env.close()
    write_episode_table(run_dir, manifest)
    summary = summarize(run_dir)
    print(f"{run_dir}: {summary['completed']}/{summary['intended']} complete; "
          f"success_once={summary['macro_success_once']}; synthetic={summary['synthetic']}")
    if args.stop_after_episodes is not None and not manifest["complete"]:
        return 3
    return 0 if manifest["complete"] and not manifest.get("last_invocation", {}).get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
