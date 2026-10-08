"""Run frozen validation/test manifests with complete-episode ManiSkill evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import yaml

from robots_project.config import load_env_config
from robots_project.envs import make_env
from robots_project.evaluation.artifacts import summarize, write_episode_table
from robots_project.evaluation.metrics import aggregate
from robots_project.policies import make_policy
from robots_project.rollout.collector import collect
from robots_project.utils import atomic_json, dump_yaml, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=["random", "hold"], default="random")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    values = yaml.safe_load(args.config.read_text())
    if not isinstance(values, dict) or set(values) != {
            "schema_version", "split", "episodes_per_task", "num_envs", "manifest", "tasks"}:
        raise ValueError("Invalid evaluation config keys")
    if values["schema_version"] != "1.0" or values["split"] not in {"validation", "test"}:
        raise ValueError("Evaluation requires a versioned validation/test split")
    records = [json.loads(line) for line in Path(values["manifest"]).read_text().splitlines() if line]
    from robots_project.evaluation.splits import validate_splits
    validate_splits(Path("manifests"))
    destination = args.run_dir or Path("outputs") / f"eval-{time.strftime('%Y%m%dT%H%M%S')}-{args.policy}"
    destination.mkdir(parents=True, exist_ok=True)
    all_rows, all_planned, manifests = [], [], []
    complete = True
    for entry in values["tasks"]:
        config = load_env_config(entry, num_envs=values["num_envs"], split=values["split"],
                                 seed_manifest=values["manifest"], termination="evaluation",
                                 reconfiguration_freq=1, auto_reset=False)
        task_dir = destination / config.env_id
        if (task_dir / "manifest.json").exists() and not args.resume:
            raise ValueError("Evaluation exists; explicit --resume required")
        env = make_env(config)
        try:
            policy = make_policy(args.policy, env.action_spec, config)
            manifest = collect(env, policy, config, task_dir, values["episodes_per_task"])
        finally:
            env.close()
        write_episode_table(task_dir, manifest)
        summarize(task_dir)
        all_rows.extend({**row, "trajectory": str(Path(config.env_id) / row["trajectory"])}
                        if row.get("trajectory") else row.copy() for row in manifest["attempts"])
        all_planned.extend(manifest["planned_episodes"])
        manifests.append({"task_id": config.env_id, "manifest": str(task_dir.relative_to(destination) / "manifest.json"),
                          "sha256": sha256_file(task_dir / "manifest.json")})
        complete &= manifest["complete"] and not manifest.get("last_invocation", {}).get("failed", False)
    summary = aggregate(all_rows, all_planned)
    summary["policy"] = args.policy
    summary["run_id"] = destination.name
    summary["policy_specs"] = manifest["policy_specs"]
    summary["method_status"] = {"pretrained_wam": "NOT_RUN", "task_only_rl": "NOT_RUN",
                                "task_plus_alignment_rl": "NOT_RUN"}
    atomic_json(destination / "summary.json", summary)
    atomic_json(destination / "manifest.json", {
        "schema_version": "1.0", "kind": "evaluation_suite", "run_id": destination.name,
        "planned_episodes": all_planned, "source_manifest_sha256": sha256_file(values["manifest"]),
        "policy_specs": manifest["policy_specs"], "suite_policy": args.policy,
        "method_status": summary["method_status"],
        "tasks": manifests, "complete": complete, "source_episode_count": len(records)})
    write_episode_table(destination, {"attempts": all_rows})
    dump_yaml(destination / "config.resolved.yaml", values)
    # Task subdirectories retain their exact provenance and error logs; suite paths are explicit.
    atomic_json(destination / "provenance.json", {"task_provenance": [
        str(Path(task["task_id"]) / "provenance.json") for task in manifests]})
    with (destination / "errors.jsonl").open("w") as out:
        for task in manifests:
            out.write((destination / task["task_id"] / "errors.jsonl").read_text())
    from robots_project.evaluation.artifacts import success_plots
    success_plots(summary, destination)
    print(f"{destination}: {summary['completed']}/{summary['intended']} complete; "
          f"macro_success_once={summary['macro_success_once']}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
