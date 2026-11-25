"""Cross-check the frozen base's per-task, merged, and summary files against the evaluated keys."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from audit import N, SEED, require, sha, stable_seed, tasks_from_queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="A freshly rebuilt four-policy audit")
    args = parser.parse_args()
    root, data = args.archive.resolve(strict=True), args.data
    keys = ["task", "level", "seed", "env"]
    merged = pd.read_parquet(root / "episodes_default.parquet").sort_values(keys).reset_index(drop=True)
    csv = pd.read_csv(root / "episodes_default.csv", float_precision="round_trip")
    files = sorted((root / "per_task").glob("episodes_screen_*.parquet"))
    original_rows = []
    for path in files:
        task = path.stem.removeprefix("episodes_screen_")
        rows = pq.read_table(path).to_pylist()
        require(len(rows) == N and {r["env"] for r in rows} == set(range(N)), path)
        seed = stable_seed(SEED, "track", task, "default")
        require(all((r["task"], r["level"], r["seed"], r["policy_version"], r["collection"], r["epoch"])
                    == (task, "default", seed, "base", "screen", 0) for r in rows), path)
        original_rows.extend(rows)
    require(len(files) == merged.task.nunique() == 26 and len(merged) == 312, root)
    require(not merged.duplicated(keys).any(), "Duplicate screening keys")
    for alternative in (pd.DataFrame.from_records(original_rows), csv):
        common = alternative[merged.columns].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(merged, common, check_dtype=False, check_exact=True)

    summary = pd.read_csv(root / "summary_default.csv").set_index("task").sort_index()
    by_task = merged.groupby("task").agg(episodes=("env", "size"), successes=("success", "sum"), seed=("seed", "first"))
    pd.testing.assert_frame_equal(by_task, summary[by_task.columns], check_dtype=False, check_exact=True)
    targets = merged[merged.task.isin(tasks_from_queue())].set_index(keys).sort_index()
    audited = pd.read_csv(data / "eval_episodes.csv")
    for model, group in audited.groupby("model"):
        recorded = group.set_index(keys).sort_index()
        pd.testing.assert_index_equal(targets.index, recorded.index)
        if model == "Base":
            for key in ["success", "steps"]:
                pd.testing.assert_series_equal(targets[key].astype(int), recorded[key].astype(int))
    require(set(audited.model) == {"Base", "T′", "A-video", "A-action"}, "Incomplete cohort")
    require(len(targets) == 120 and len(audited) == 480, "Unexpected cohort size")

    supplemental = [root / n for n in ("README.md", "episodes_default.csv", "episodes_default.parquet",
                    "episodes_specific_round0.csv", "episodes_specific_round0.parquet", "selection.json", "summary_default.csv")]
    snapshot = data / "screening_base_20261123.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in supplemental + files:
            archive.write(source, source.relative_to(root))
    with zipfile.ZipFile(snapshot) as archive:
        for source in supplemental + files:
            require(hashlib.sha256(archive.read(str(source.relative_to(root)))).hexdigest() == sha(source), source)
    result = dict(archive_path=str(root), snapshot_sha256=sha(snapshot), default_tasks=len(files),
                  default_episodes=len(merged), exact_shared_columns=list(merged.columns),
                  selected_tasks=sorted(targets.index.get_level_values("task").unique()),
                  selected_episodes=len(targets), selected_successes=int(targets.success.sum()),
                  selected_failures=int((~targets.success).sum()), four_policy_keys=len(targets),
                  base_raw_initial_states_available=False, base_raw_initial_states_source="archive README, Origin")
    (data / "screening_crosscheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"OK: {len(merged)} default rows, {len(files)} per-task originals, {len(merged.columns)} common fields match exactly.")
    print(f"OK: 26 task summaries match. {len(targets)} matched four-policy keys, {result['selected_successes']} Base successes.")


if __name__ == "__main__":
    main()
