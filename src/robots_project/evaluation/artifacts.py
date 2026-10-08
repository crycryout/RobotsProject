"""Episode tables are the canonical input to deterministic summary regeneration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from robots_project.evaluation.metrics import aggregate
from robots_project.utils import atomic_json

FIELDS = ["episode_id", "attempt", "task_id", "split", "seed", "env_slot", "status",
          "success_once", "success_at_end", "return_env", "steps", "steps_to_first_success",
          "time_to_first_success", "wall_seconds", "write_seconds", "ending_reason", "trajectory",
          "sha256", "bytes_written", "synthetic", "privileged_policy", "policy_version",
          "error_type", "error"]


def write_episode_table(run_dir, manifest):
    rows = sorted(manifest["attempts"], key=lambda r: (r["episode_id"], r.get("attempt", 0)))
    path = Path(run_dir) / "episodes.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_episode_table(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("success_once", "success_at_end", "steps", "steps_to_first_success", "seed", "attempt"):
            row[key] = int(row[key]) if row[key] != "" else None
        for key in ("return_env", "time_to_first_success", "wall_seconds", "write_seconds"):
            row[key] = float(row[key]) if row[key] != "" else None
        row["synthetic"] = row["synthetic"].lower() == "true"
    return rows


def summarize(run_dir: Path, plots=True):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = read_episode_table(run_dir / "episodes.csv")
    summary = aggregate(rows, manifest["planned_episodes"])
    summary["run_id"] = manifest["run_id"]
    summary["policy_specs"] = manifest["policy_specs"]
    if manifest.get("kind") == "evaluation_suite":
        summary["policy"] = manifest["suite_policy"]
        summary["method_status"] = manifest["method_status"]
    atomic_json(run_dir / "summary.json", summary)
    if plots:
        success_plots(summary, run_dir)
    return summary


def success_plots(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = summary["per_task"]
    label = "SYNTHETIC INTERFACE FIXTURE" if summary["synthetic"] else "Infrastructure control"
    for metric in ("success_once", "success_at_end"):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names = list(tasks)
        values = [tasks[t][metric] for t in names]
        positions = list(range(len(names)))
        for pos, task, value in zip(positions, names, values, strict=True):
            if value is None:
                ax.text(pos, 0.05, "NOT_RUN", ha="center")
                continue
            interval = tasks[task][metric + "_wilson95"]
            ax.bar(pos, value, color="#296d98")
            ax.errorbar(pos, value, yerr=[[max(0, value - interval[0])],
                                        [max(0, interval[1] - value)]], capsize=5, color="black")
            ax.text(pos, min(1.05, interval[1] + 0.04),
                    f"{tasks[task][metric + '_count']}/{tasks[task]['completed']}", ha="center")
        ax.set_xticks(positions, names)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric)
        ax.set_title(label + (" (INCOMPLETE)" if not summary["complete"] else ""))
        fig.tight_layout()
        fig.savefig(destination / f"{metric}.png", dpi=160)
        plt.close(fig)
