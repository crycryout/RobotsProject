"""Replay a single episode on the same version and report native observation discrepancies."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from robots_project.config import EnvConfig
from robots_project.data.validation import datasets, validate_episode
from robots_project.envs import make_env
from robots_project.evaluation.artifacts import write_episode_table
from robots_project.policies.controls import ReplayPolicy
from robots_project.rollout.collector import collect
from robots_project.utils import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.atol <= 0:
        raise ValueError("Replay tolerance must be positive")
    validate_episode(args.trajectory)
    with h5py.File(args.trajectory, "r") as f:
        metadata = json.loads(f.attrs["metadata_json"])
    config = EnvConfig(**{**metadata["resolved_config"], "num_envs": 1, "split": "replay",
                          "seed": metadata["seed"], "seed_manifest": None,
                          "reconfiguration_freq": 0, "termination": "collection"}).validate()
    env = make_env(config)
    try:
        policy = ReplayPolicy(env.action_spec, args.trajectory, config.chunk_length)
        manifest = collect(env, policy, config, args.output, 1, replay_source=args.trajectory)
    finally:
        env.close()
    write_episode_table(args.output, manifest)
    report = {"source": str(args.trajectory), "complete": manifest["complete"],
              "tolerance": args.atol, "fields": {}, "matches": False,
              "limitation": "Same version/configuration replay check, not cross-version determinism."}
    if manifest["complete"]:
        target = args.output / manifest["episodes"][0]["trajectory"]
        with h5py.File(args.trajectory, "r") as source, h5py.File(target, "r") as replay:
            all_match = True
            for name, ds in datasets(source["obs"]).items():
                actual = replay["obs/" + name]
                if ds.shape != actual.shape:
                    detail = {"matches": False, "reason": "shape mismatch"}
                else:
                    difference = np.abs(ds[:].astype(np.float64) - actual[:].astype(np.float64))
                    match = np.allclose(ds[:], actual[:], atol=args.atol, rtol=0)
                    detail = {"matches": bool(match), "max_absolute_error": float(difference.max())}
                all_match &= detail["matches"]
                report["fields"][name] = detail
            report["matches"] = all_match
    atomic_json(args.output / "replay_report.json", report)
    print(f"Replay matches={report['matches']}: {args.output / 'replay_report.json'}")
    return 0 if report["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
