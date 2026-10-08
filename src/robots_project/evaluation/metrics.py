"""Success counts, Wilson intervals, and equal task weighting."""

from __future__ import annotations

import math
import statistics


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def aggregate(rows: list[dict], planned: list[dict]) -> dict:
    ids = [r["episode_id"] for r in planned]
    if len(ids) != len(set(ids)):
        raise ValueError("Planned episode IDs are not unique")
    completed = [r for r in rows if r["status"] == "complete"]
    if len({r["episode_id"] for r in completed}) != len(completed):
        raise ValueError("Duplicate completed episode rows")
    if any(r["episode_id"] not in set(ids) for r in rows):
        raise ValueError("Attempt outside planned evaluation manifest")
    tasks = {}
    for task in sorted({r["task_id"] for r in planned}):
        expected = [r for r in planned if r["task_id"] == task]
        attempts = [r for r in rows if r["task_id"] == task]
        valid = [r for r in attempts if r["status"] == "complete"]
        failures = [r for r in attempts if r["status"] != "complete"]
        n = len(valid)
        once = sum(int(r["success_once"]) for r in valid)
        end = sum(int(r["success_at_end"]) for r in valid)
        attempted_ids = {r["episode_id"] for r in attempts}
        first_steps = [int(r["steps_to_first_success"]) for r in valid
                       if r.get("steps_to_first_success") not in (None, "")]
        first_times = [float(r["time_to_first_success"]) for r in valid
                       if r.get("time_to_first_success") not in (None, "")]
        tasks[task] = {
            "intended": len(expected), "attempted_episodes": len(attempted_ids),
            "attempts": len(attempts), "completed": n,
            "failed_attempts": len(failures), "missing": len(expected) - n,
            "complete": n == len(expected),
            "success_once_count": once, "success_at_end_count": end,
            "success_once": once / n if n else None,
            "success_at_end": end / n if n else None,
            "success_once_wilson95": wilson(once, n),
            "success_at_end_wilson95": wilson(end, n),
            "conservative_success_once": once / len(expected),
            "conservative_success_at_end": end / len(expected),
            "error_rate": len(failures) / len(attempts) if attempts else None,
            "return_env_mean": statistics.mean(float(r["return_env"]) for r in valid) if n else None,
            "steps_mean": statistics.mean(int(r["steps"]) for r in valid) if n else None,
            "steps_to_first_success_mean": statistics.mean(first_steps) if first_steps else None,
            "time_to_first_success_mean": statistics.mean(first_times) if first_times else None,
        }
    all_have_samples = all(v["completed"] for v in tasks.values())
    return {
        "schema_version": "1.0", "primary_metric": "success_once", "per_task": tasks,
        "complete": all(v["complete"] for v in tasks.values()),
        "intended": len(planned), "completed": len(completed), "attempts": len(rows),
        "synthetic": any(str(r.get("synthetic", False)).lower() == "true" for r in rows),
        "macro_success_once": statistics.mean(v["success_once"] for v in tasks.values())
                              if all_have_samples else None,
        "macro_success_at_end": statistics.mean(v["success_at_end"] for v in tasks.values())
                                if all_have_samples else None,
        "macro_conservative_success_once": statistics.mean(
            v["conservative_success_once"] for v in tasks.values()),
        "uncertainty_scope": "Finite-episode uncertainty of a fixed checkpoint; no training-seed variability.",
    }
