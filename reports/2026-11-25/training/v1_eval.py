"""Paired evaluation of two policies at once, one per GPU lane (train_v1.md §6.2, §9).

Both lanes run the same batches with the same seeds, so layouts and policy-noise seeds match and
episodes pair up by (task, level, batch, seed, env). Paired screening reuses the original
screening seeds. The two-live-model mode uses a separate "eval" stream.

Usage:
  python -m training.v1_eval <out_dir> --run training/runs/v1_T_01 --model base --model <checkpoint_dir> \
      [--tasks ...] [--level default] [--batches-per-task 3]
  python -m training.v1_eval <out_dir> --summarize
"""

import argparse
import hashlib
import json
import os
import pickletools
import re
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from training.online import CACHE, WORKSPACE, ResidentWorkers, read_json, run_command, write_json
from training.v1_reward import episode_chunks, rollout_logs
from training.v1_run import stable_seed


def trace_version(path):
    """Read the saved policy tag without loading tensor storage or executing pickle."""
    with zipfile.ZipFile(path) as archive:
        metadata, = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        strings = [arg for op, arg, _ in pickletools.genops(archive.read(metadata))
                   if op.name in ("UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8")]
    return strings[strings.index("policy_version") + 1]


def episode_rows(out, batch, model, version=None):
    rows = []
    for _, log in rollout_logs(out):
        if (log["task"] != batch["task"] or log["instruction_fields_used"] != [batch["level"]]
                or log["run"] != 0 or type(log["success"]) is not bool):
            raise ValueError(f"Unexpected rollout metadata in {out}")
        chunks = episode_chunks(log)
        if version is not None:
            for chunk in chunks:
                capture = out / "capture" / f"{chunk.key}_nft.pt"
                if trace_version(capture) != version:
                    raise ValueError(f"Captured policy differs from requested checkpoint in {capture}")
        path = [chunks[0].p_start] + [c.p_end for c in chunks]
        holds = [c.end for c in chunks if c.p_end == 0.5]
        rows.append({
            "model": model, "task": log["task"], "level": batch["level"], "batch": batch["index"], "seed": batch["seed"],
            "env": log["env_id"], "success": bool(log["success"]), "steps": log["final_step"], "final_p": path[-1],
            "ever_held": any(p >= 0.5 for p in path), "time_to_first_hold": holds[0] if holds else None,
            "drops": sum(1 for a, b in zip(path, path[1:]) if a == 0.5 and b == 0.0),
            "wrong_grab_events": sum(1 for e in log.get("events", []) if e.get("name") == "WRONG_OBJECT_GRABBED_FAILURE"),
        })
    return rows


def check_batch(rows, batch, num_envs, model):
    expected = (batch["task"], batch["level"], batch["index"], batch["seed"], model)
    ids = []
    for row in rows:
        if tuple(row[k] for k in ("task", "level", "batch", "seed", "model")) != expected:
            raise ValueError(f"Unexpected task, instruction, batch, seed or model for {expected}")
        if type(row["success"]) is not bool or type(row["env"]) is not int:
            raise ValueError(f"Invalid success or environment ID for {expected}")
        ids.append(row["env"])
    if sorted(ids) != list(range(num_envs)):
        raise ValueError(f"Incomplete or duplicate environments for {expected}: {ids}")


def policy_version(model):
    if model == "base":
        return "base"
    return hashlib.sha256((Path(model).expanduser() / "heads.pt").read_bytes()).hexdigest()


def check_runtime(cfg, batch, config, rows):
    noise = (cfg["events"].get("randomize_init_pose") or {}).get("params", {}).get("noise_m", 0)
    if (cfg["seed"] != batch["seed"] or cfg["num_envs"] != config["num_envs"]
            or float(noise) != float(config["init_pose_noise"])):
        raise ValueError(f"Actual reset configuration differs from the plan for {batch}")
    dt = cfg["sim"]["dt"] * cfg["decimation"]
    horizon = int(np.ceil(cfg["episode_length_s"] / dt))
    for row in rows:
        if not 0 < row["steps"] <= horizon or (not row["success"] and row["steps"] != horizon):
            raise ValueError(f"Unofficial truncation in {batch}, env {row['env']}")


def lane(root, config, index, model, queue, slot, scratch, status, lock):
    gpu, port = config["gpus"][index], 18731 + index
    worker = ResidentWorkers(root / f"lane_{index}")
    worker.root.mkdir(exist_ok=True)
    previous = None if model == "base" else Path(model).expanduser()
    version = policy_version(model)
    try:
        while True:
            with lock:
                if not queue:
                    break
                batch = queue.pop(0)
            name = f"{batch['task']}_{batch['level']}_b{batch['index']}"
            result = root / slot / f"{name}.json"
            if result.exists():
                saved = read_json(result)
                check_batch(saved["rows"], batch, config["num_envs"], model)
                if saved["version"] != version:
                    raise ValueError(f"Checkpoint changed for {result}")
                continue
            out = scratch / slot / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                out = Path(tempfile.mkdtemp(prefix=f"{name}_retry_", dir=out.parent))
            env = os.environ.copy()
            for key in ("DREAM_BRANCH", "REPLAY", "COSMOS_HEAD_ADAPTER", "DREAM_MAX_STEPS", "DREAM_NATIVE_REWARD",
                        "DREAM_ROLLOUT_DEADLINE"):
                env.pop(key, None)
            env.update(
                HF_HOME=str(CACHE), HF_HUB_OFFLINE="1", PYTHONPATH=f"{WORKSPACE}:/home/robot_share/cosmos-framework",
                DREAM_TASK_WORKERS="1", DREAM_ALL_LEVELS="0", DREAM_VIDEO_MODE="none", DREAM_ENDPOINT_ONLY="1",
                DREAM_LATENTS_ONLY="1", COSMOS_CAPTURE_NFT="1", COSMOS_UND_CACHE="8", DREAM_V1_PROGRESS="1",
                DREAM_INIT_POSE_NOISE=str(config["init_pose_noise"]), DREAM_EXTERNAL_POLICY="1",
                NUM_ENVS=str(config["num_envs"]), OUT=str(out), SIM_GPU=gpu, POLICY_GPU=gpu, PORT=str(port),
                DREAM_INSTRUCTION=batch["level"], SIM_SEED=str(batch["seed"]), COSMOS_POLICY_VERSION=version,
            )
            out.mkdir(exist_ok=True)
            worker.begin_rollout(env, out / "capture", previous, version)
            started = time.monotonic()
            with lock:
                status[f"lane{index}"] = {"model": model, "batch": name, "started": time.time()}
                write_json(root / "status.json", status)
            run_command(["bash", "dream_eval/run_pilot.sh", batch["task"]], env=env, log=out.with_suffix(".log"),
                        timeout=7200, peer=worker.policy)
            rows = episode_rows(out, batch, model, version)
            cfg = read_json(out / "rollouts" / batch["task"] / "env_cfg.json")
            check_runtime(cfg, batch, config, rows)
            check_batch(rows, batch, config["num_envs"], model)
            result.parent.mkdir(exist_ok=True)
            write_json(result, {"rows": rows, "rollout_s": time.monotonic() - started, "version": version})
            with lock:
                status.setdefault(f"done{index}", 0)
                status[f"done{index}"] += 1
                write_json(root / "status.json", status)
    except Exception as error:  # noqa: BLE001 - surface in status.json, the other lane keeps going
        with lock:
            status[f"lane{index}_error"] = repr(error)
            write_json(root / "status.json", status)
        raise
    finally:
        worker.close()


def summarize(root, versions=None):
    import pandas as pd

    plan = read_json(root / "plan.json")
    frames = []
    batches = plan["batches"]
    keys = [(b["task"], b["level"], b["index"]) for b in batches]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Empty or duplicated evaluation batches")
    if plan.get("baseline") == "screening":
        run = Path(plan["run"])
        for batch in batches:
            base = pd.read_parquet(run / "counters" / f"episodes_screen_{batch['task']}.parquet")
            if not base.policy_version.eq("base").all():
                raise ValueError(f"Non-base policy in screening for {batch['task']}")
            base = base.assign(batch=0, model="screening", model_index=0)
            check_batch(base.to_dict("records"), batch, plan["num_envs"], "screening")
            frames.append(base)
    for index, model in enumerate(plan["models"]):
        if model == "screening":
            continue
        seen_versions = set()
        for batch in sorted(batches, key=lambda b: f"{b['task']}_{b['level']}_b{b['index']}"):
            name = f"{batch['task']}_{batch['level']}_b{batch['index']}.json"
            path = root / f"model{index}" / name
            if not path.is_file():
                raise ValueError(f"Missing evaluation batch: {path}")
            saved = read_json(path)
            check_batch(saved["rows"], batch, plan["num_envs"], model)
            seen_versions.add(saved["version"])
            frames.append(pd.DataFrame(saved["rows"]).assign(model_index=index))
        expected_version = versions[index] if versions is not None else None
        if expected_version is None and (model == "base" or (Path(model).expanduser() / "heads.pt").is_file()):
            expected_version = policy_version(model)
        if (len(seen_versions) != 1 or (expected_version is not None and seen_versions != {expected_version})
                or (model != "base" and not re.fullmatch("[0-9a-f]{64}", next(iter(seen_versions))))):
            raise ValueError(f"Inconsistent checkpoint versions for {model}: {seen_versions}")
    data = pd.concat(frames, ignore_index=True)
    names = [Path(m).name if m not in ("base", "screening") else m for m in plan["models"]]
    print(f"models: 0={names[0]}  1={names[1]}")
    per = data.groupby(["task", "model_index"]).agg(n=("success", "size"), success=("success", "mean"),
                                                    final_p=("final_p", "mean")).round(3).unstack("model_index")
    print(per.to_string())
    a, b = (data[data.model_index == i].set_index(["task", "level", "batch", "seed", "env"]) for i in (0, 1))
    paired = a.join(b, lsuffix="_0", rsuffix="_1", how="inner", validate="one_to_one")
    if len(paired) != len(batches) * plan["num_envs"]:
        raise ValueError("The two policies do not have the full set of matched episodes")
    rng = np.random.default_rng(0)
    out = {}
    for metric in ("success", "final_p"):
        d = paired[f"{metric}_1"].astype(float) - paired[f"{metric}_0"].astype(float)
        draws = rng.choice(d.values, (10000, len(d))).mean(axis=1)
        out[metric] = (d.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5))
        print(f"paired Δ{metric} (model1 − model0), n={len(d)}: {d.mean():+.3f} "
              f"[{np.percentile(draws, 2.5):+.3f}, {np.percentile(draws, 97.5):+.3f}]")
    discordant = paired[paired.success_0 != paired.success_1]
    print(f"discordant pairs: model1 only {int((discordant.success_1).sum())}, model0 only {int((discordant.success_0).sum())}")
    summary = {"models": plan["models"], "n_pairs": len(paired),
               "paired": {k: list(map(float, v)) for k, v in out.items()}, "per_task": json.loads(per.to_json())}
    target = root / "summary.json"
    if target.exists():
        if read_json(target) != summary:
            raise ValueError(f"Stored summary differs from complete episode data: {target}. Use a fresh output.")
    else:
        write_json(target, summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--run", type=Path, default=WORKSPACE / "training/runs/v1_T_01")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--level", default="default")
    parser.add_argument("--batches-per-task", type=int, default=3)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--pair-screening", action="store_true",
                        help="Run one --model on both lanes with the screening seeds; base = stored screening episodes")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.summarize:
        return summarize(root)
    run = args.run.resolve()
    config = read_json(run / "config.json")
    if args.pair_screening:
        if len(args.model) != 1 or not args.tasks or args.level != "default":
            parser.error("--pair-screening needs one --model, explicit --tasks, and the default level")
        screened = read_json(run / "state.json")["screened"]
        tasks = sorted(args.tasks, key=lambda t: -screened[t]["rollout_s"])  # longest first balances the lanes
        batches = [{"task": t, "level": "default", "index": 0, "seed": screened[t]["seed"]} for t in tasks]
        models = ["screening", args.model[0]]
    else:
        if len(args.model) != 2:
            parser.error("Give exactly two --model values (base or a checkpoint directory)")
        tasks = args.tasks or read_json(run / "selection.json")["selected"]
        batches = [{"task": t, "level": args.level, "index": i,
                    "seed": stable_seed(config["seed"], "eval", t, args.level, i)}
                   for i in range(args.batches_per_task) for t in tasks]
        models = args.model
    root.mkdir(parents=True, exist_ok=True)
    plan = {"models": models, "tasks": tasks, "level": args.level, "batches": batches, "run": str(run),
            "baseline": "screening" if args.pair_screening else "run",
            "init_pose_noise": config["init_pose_noise"], "num_envs": config["num_envs"]}
    if (root / "plan.json").exists() and read_json(root / "plan.json") != plan:
        raise ValueError("Output directory holds a different evaluation plan")
    if (root / "summary.json").exists() and not (root / "plan.json").exists():
        raise ValueError("Stored summary lacks its evaluation plan")
    versions = ["screening" if m == "screening" else policy_version(m) for m in models]
    if (root / "summary.json").exists():
        return summarize(root, versions)
    write_json(root / "plan.json", plan)
    scratch = Path("/var/tmp/vis_rl") / f"eval_{root.parent.name}_{root.name}"  # unique per run and eval
    status, lock = {"pid": os.getpid(), "total_batches_per_lane": len(batches)}, threading.Lock()
    queue = list(batches)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = []
        for i in range(2):
            slot = 1 if args.pair_screening else i
            jobs.append(pool.submit(lane, root, config, i, models[slot], queue if args.pair_screening else list(batches),
                                    f"model{slot}", scratch, status, lock))
        for job in jobs:
            job.result()
    summarize(root, versions)
    status["finished"] = time.time()
    write_json(root / "status.json", status)


if __name__ == "__main__":
    main()
