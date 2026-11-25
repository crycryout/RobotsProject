"""Reconstruct the common v2 training window from the portable raw evidence."""
import argparse
import csv
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parents[1]
ARMS = {"T′": "T", "A-video": "AV", "A-action": "AA"}
TZ = ZoneInfo("Asia/Shanghai")


def local_time(timestamp):
    return datetime.fromtimestamp(timestamp, TZ).isoformat(timespec="milliseconds")


def record_table(name, rows, check_only):
    path = OUT / "data" / name
    if check_only:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            assert list(csv.DictReader(handle)) == [{k:str(v) for k,v in r.items()} for r in rows], path
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main(check_only=False):
    collections, runtimes = [], []
    with zipfile.ZipFile(OUT / "data/source_records.zip") as archive:
        for model, arm in ARMS.items():
            root = f"repo/training/runs/v2_{arm}_01/"
            state = json.loads(archive.read(root + "state.json"))
            status = json.loads(archive.read(root + "status.json"))
            history = [json.loads(line) for line in archive.read(root + "history.jsonl").splitlines()]
            ids = [f"e{epoch:02}c{index}" for epoch in range(1, 4) for index in range(2)]
            assert [row["collection"] for row in history] == ids and status["phase"] == "done"
            start, end = state["train_started"], status["time"]
            last_time, count, successes, steps = start, 0, 0, 0
            for row in history:
                tag = row["collection"]
                filename = root + f"counters/episodes_{tag}.parquet"
                episodes = pq.ParquetFile(io.BytesIO(archive.read(filename))).read(use_threads=False).to_pylist()
                n, k = len(episodes), sum(ep["success"] for ep in episodes)
                used = sum(ep["steps"] for ep in episodes)
                assert len({(e["task"], e["level"], e["seed"], e["env"]) for e in episodes}) == n
                assert (n, k, used) == (row["episodes"], row["successes"], row["env_steps"])
                assert n == 48 and last_time <= row["time"] <= end
                last_time = row["time"]
                steps, count, successes = steps + used, count + n, successes + k
                collections.append(dict(model=model, training_seed=20261123, epoch=row["epoch"],
                    collection=tag, episodes=n, success=k, success_pct=100*k/n,
                    collection_env_steps=used, cumulative_env_steps=steps,
                    history_timestamp_unix=row["time"], history_time_beijing=local_time(last_time),
                    seconds_since_train_started=round(last_time-start, 3),
                    source_episodes=filename, source_history=root+"history.jsonl"))
            assert state["env_steps"] == steps and count == 288
            duration = end - start
            seconds = round(duration)
            clock = f"{seconds//3600}:{seconds%3600//60:02d}:{seconds%60:02d}"
            runtimes.append(dict(model=model, training_seed=20261123, epochs="1–3",
                episodes=count, success=successes, cumulative_env_steps=steps,
                train_started_unix=start, done_unix=end, start_beijing=local_time(start),
                finish_beijing=local_time(end), elapsed_seconds=round(duration, 3),
                elapsed_hms=clock, state_source=root+"state.json", status_source=root+"status.json"))
            print(f"{model}: {successes}/{count}, {steps:,} env steps, {clock} wall time")
    record_table("training_collections.csv", collections, check_only)
    record_table("training_runtime.csv", runtimes, check_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    main(parser.parse_args().check_only)
