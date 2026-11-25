"""Run detached, bounded-storage, on-policy simulator training."""

import argparse
import fcntl
import hashlib
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from training.task_split import LEVELS

WORKSPACE = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("HF_HOME", "~/hf_cache")).expanduser()
MODEL = CACHE / "models--nvidia--Cosmos3-Edge-Policy-DROID/snapshots/68b17b3c959ccd0999de9a972c5f7c8b57112f86"
ROUND = re.compile(r"round_\d+")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


@contextmanager
def measure_gpu(gpu):
    stats = {"samples": 0, "peak_mib": 0, "util_samples": 0, "mean_util_pct": 0.0}
    try:
        sampler = subprocess.Popen(
            ["nvidia-smi", "--id", gpu, "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "--loop-ms=500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except OSError:
        yield stats
        return

    def read():
        total_util = 0
        for line in sampler.stdout:
            memory, _, utilization = line.partition(",")
            if memory.strip().isdecimal():
                stats["samples"] += 1
                stats["peak_mib"] = max(stats["peak_mib"], int(memory))
            if utilization.strip().isdecimal():
                stats["util_samples"] += 1
                total_util += int(utilization)
                stats["mean_util_pct"] = total_util / stats["util_samples"]

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        yield stats
    finally:
        sampler.terminate()
        sampler.wait(timeout=10)
        reader.join(timeout=5)


def run_command(command, *, env, log, timeout, peer=None):
    with log.open("w") as stream:
        with subprocess.Popen(command, cwd=WORKSPACE, env=env, stdout=stream, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        code = process.wait(timeout=max(0, min(1, deadline - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        if peer is not None and peer.poll() is not None:
                            raise RuntimeError("Resident policy exited during rollout; see resident_policy.log")
                        if time.monotonic() >= deadline:
                            raise
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    if code:
        raise RuntimeError(f"{command[0]} exited {code}; see {log}")


class ResidentWorkers:
    def __init__(self, root):
        self.root = root
        self.policy = None
        self.actor = None
        self.logger = None

    def begin_rollout(self, env, capture_dir, previous, version):
        capture_dir.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "policy_manifest.json"
        write_json(manifest, {
            "capture_dir": str(capture_dir), "version": version,
            "adapter": str(previous / "heads.pt") if previous else None,
        })
        if self.policy is None:
            self.policy = subprocess.Popen(
                ["bash", "dream_eval/serve_policy.sh"], cwd=WORKSPACE,
                env=env | {"COSMOS_POLICY_MANIFEST": str(manifest), "DREAM_CAPTURE_DIR": str(self.root)},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                start_new_session=True,
            )
            self.logger = threading.Thread(target=self.drain_log, daemon=True)
            self.logger.start()
        if self.policy.poll() is not None:
            raise RuntimeError("Resident policy exited; see resident_policy.log")

    def drain_log(self):
        handler = RotatingFileHandler(self.root / "resident_policy.log", maxBytes=16 * 2**20, backupCount=2)
        try:
            for line in self.policy.stdout:
                handler.emit(logging.LogRecord("resident_policy", logging.INFO, "", 0, line.rstrip("\n"), (), None))
        finally:
            handler.close()

    def update(self, command, env, stage, timeout):
        if self.actor is None:
            with (self.root / "resident_actor.log").open("w") as log:
                self.actor = subprocess.Popen(
                    [sys.executable, "-u", "-m", "training.native_update", "--worker-loop"],
                    cwd=WORKSPACE, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True,
                )
        done = stage / "update.done"
        self.actor.stdin.write(json.dumps({
            "argv": command[3:], "done": str(done), "log": str(stage / "actor.log"),
        }) + "\n")
        self.actor.stdin.flush()
        deadline = time.monotonic() + timeout
        while not done.exists():
            if self.actor.poll() is not None:
                raise RuntimeError("Resident actor exited; see actor.log and resident_actor.log")
            if time.monotonic() > deadline:
                raise TimeoutError("Resident actor exceeded its update deadline")
            time.sleep(0.5)

    def close(self):
        if self.actor is not None:
            try:
                self.actor.stdin.close()
            except BrokenPipeError:
                pass
        for process in (self.actor, self.policy):
            if process is None or process.poll() is not None:
                continue
            if process is self.policy:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if self.logger:
            self.logger.join(timeout=5)


def completed_rounds(root):
    return sorted(path for path in root.iterdir() if ROUND.fullmatch(path.name) and (path / "round.json").is_file())


def record(root, state, config):
    history = read_json(root / "history.json", [])
    if not history or history[-1]["round"] < state["round"]:
        write_json(root / "history.json", (history + [state])[-config["keep_history"]:])
    rounds = completed_rounds(root)
    checkpoints = [path for path in rounds if (path / "heads.pt").is_file()]
    retain = set(rounds[-config["keep_checkpoints"]:] + checkpoints[-config["keep_checkpoints"]:])
    for path in rounds:
        if path not in retain:
            shutil.rmtree(path)


def run_round(root, config, prior, status, resident=None):
    index = prior["round"] + 1
    rng = random.Random(config["seed"] + index)
    reward_mode = config.get("reward_mode", "legacy")
    warmup = reward_mode == "legacy" and index == 1
    task = "BananaInBowlTask" if warmup else rng.choice(config["tasks"])
    level = "default" if warmup else rng.choice(LEVELS)
    seed = 0 if warmup else rng.randrange(1, 2**31)
    previous = root / prior["checkpoint"] if prior["checkpoint"] else None
    version = hashlib.sha256((previous / "heads.pt").read_bytes()).hexdigest() if previous else "base"
    name = f"round_{index:08d}"
    stage = root / f".{name}.pending"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    env = os.environ.copy()
    for key in ("COSMOS_HEAD_ADAPTER", "DREAM_BRANCH", "DREAM_MAX_STEPS", "REPLAY"):
        env.pop(key, None)
    env.update(
        HF_HOME=str(CACHE), HF_HUB_OFFLINE="1", PYTHONPATH=f"{WORKSPACE}:/home/robot_share/cosmos-framework",
        SIM_GPU=config["sim_gpu"], POLICY_GPU=config["policy_gpu"], OUT=str(stage / "rollout"),
        NUM_ENVS=str(config["num_envs"]), SIM_SEED=str(seed), DREAM_INSTRUCTION=level,
        DREAM_TASK_WORKERS="1", DREAM_ALL_LEVELS="0", DREAM_VIDEO_MODE="none", DREAM_ENDPOINT_ONLY="1",
        DREAM_LATENTS_ONLY="1", COSMOS_CAPTURE_NFT="1", COSMOS_UND_CACHE="8", COSMOS_POLICY_VERSION=version,
        PORT="18141", DREAM_NATIVE_REWARD="1" if reward_mode == "native" else "0",
    )
    if previous:
        env["COSMOS_HEAD_ADAPTER"] = str(previous / "heads.pt")
    status.update(round=index, task=task, level=level, policy_version=version, phase="rollout",
                  reward_mode=reward_mode)
    write_json(root / "status.json", status)
    started = time.monotonic()
    if resident is not None:
        resident.begin_rollout(env, stage / "rollout/capture", previous, version)
        env["DREAM_EXTERNAL_POLICY"] = "1"
    with measure_gpu(config["policy_gpu"]) as rollout_memory:
        run_command(["bash", "dream_eval/run_pilot.sh", task], env=env, log=stage / "collect.log", timeout=3600,
                    peer=resident.policy if resident else None)
    rollout_s = time.monotonic() - started
    summary = stage / "rollout/rollouts/episode_results.jsonl"
    episodes = [json.loads(line) for line in summary.read_text().splitlines() if line]
    if len(episodes) != config["num_envs"] or any(row["episode_step"] < 1 or
                                                  row["instruction_fields_used"] != [level] for row in episodes):
        raise RuntimeError("Incomplete rollout or unexpected prompt level")
    shutil.copyfile(summary, stage / "episodes.jsonl")
    write_json(stage / "input_stage_values.json", prior["stage_values"])
    status.update(phase="update", collected_env_steps=sum(row["episode_step"] for row in episodes))
    write_json(root / "status.json", status)
    command = [
        sys.executable, "-m", "training.native_update", str(stage / "rollout"), str(MODEL),
        "--output", str(stage), "--all-gen", "--rank", str(config["rank"]), "--no-checkpoint-blocks",
        "--learning-rate", str(config["learning_rate"]), "--accumulate", str(config["accumulate"]),
        "--samples", "0", "--mismatch-weight", "0", "--tensorboard", "--online-policy-version", version,
        "--reward-mode", reward_mode,
        "--values", str(stage / "input_stage_values.json"), "--round", str(index), "--port", "18242",
    ]
    if previous:
        command.extend(("--resume", str(previous)))
    actor_env = env | {"CUDA_VISIBLE_DEVICES": config["policy_gpu"]}
    with measure_gpu(config["policy_gpu"]) as update_memory:
        if resident is None:
            run_command(command, env=actor_env, log=stage / "actor.log", timeout=1800)
        else:
            resident.update(command, actor_env, stage, timeout=1800)
    update = read_json(stage / "update_metrics.json")
    if update["policy_version"] != version or update.get("reward_mode", "legacy") != reward_mode:
        raise RuntimeError("Actor used an unexpected rollout policy or reward mode")
    updated = bool(update["steps"])
    if updated and not all((stage / filename).is_file() for filename in ("heads.pt", "optimizer.pt")):
        raise RuntimeError("Missing resumable checkpoint files")
    for filename in ("policy.log", "sim.log"):
        source = stage / "rollout" / filename
        if source.exists():
            source.replace(stage / filename)
    cache_stats = stage / "rollout/capture/und_cache.json"
    if cache_stats.exists():
        cache_stats.replace(stage / "und_cache.json")
    shutil.rmtree(stage / "rollout")
    state = {
        "round": index, "task": task, "level": level, "seed": seed, "rollout_policy_version": version,
        "reward_mode": reward_mode,
        "checkpoint": name if updated else prior["checkpoint"],
        "env_steps": prior["env_steps"] + sum(row["episode_step"] for row in episodes),
        "optimizer_steps": prior["optimizer_steps"] + update["steps"],
        "stage_values": update["stage_values"], "prompt_rewards": update["prompt_level_rewards"],
        "rollout_s": rollout_s, "update_s": update.get("update_s", 0),
        "round_updates": update["steps"], "changed_adapter_tensors": update.get("changed_adapter_tensors", 0),
        "skip_reason": update.get("skip_reason"),
        "actor_load_s": update.get("load_s", 0), "actor_pid": update.get("actor_pid"),
        "actor_reused": update.get("actor_reused", False), "rollout_worker_pids": update.get("rollout_worker_pids", []),
        "rollout_gpu": rollout_memory, "update_gpu": update_memory,
    }
    write_json(stage / "round.json", state)
    stage.replace(root / name)
    record(root, state, config)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--resume", action="store_true", help="Clear a previous graceful stop request.")
    parser.add_argument("--resident", action="store_true", help="Keep the rollout and actor workers loaded.")
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(root / "config.json")
    train = [row["task"] for row in read_json(WORKSPACE / "training/task_split.json")["splits"]["train"]]
    reward_mode = config.get("reward_mode", "legacy")
    if reward_mode not in ("legacy", "native"):
        raise ValueError("Expected reward_mode to be legacy or native")
    if config.get("tasks") == "train":
        if reward_mode != "native":
            raise ValueError("The full split requires native rewards")
        config["tasks"] = train
    if (not isinstance(config.get("tasks"), list) or not config["tasks"]
            or not set(config["tasks"]) <= set(train)):
        raise ValueError("Online tasks must belong to the training split")
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = root / "STOP"
        if args.resume:
            stop.unlink(missing_ok=True)
        rounds = completed_rounds(root)
        state = read_json(rounds[-1] / "round.json") if rounds else {
            "round": 0, "checkpoint": None, "env_steps": 0, "optimizer_steps": 0, "stage_values": {},
        }
        if rounds and state.get("reward_mode", "legacy") != reward_mode:
            raise ValueError("Use a fresh run directory when changing the reward mode")
        status = {"pid": os.getpid(), "phase": "starting", "resident": args.resident, "reward_mode": reward_mode}
        write_json(root / "status.json", status)
        resident = ResidentWorkers(root) if args.resident else None
        try:
            if rounds:
                record(root, state, config)
            while not stop.exists():
                if shutil.disk_usage(root).free < config["min_free_gib"] * 2**30:
                    raise RuntimeError("Free disk below run threshold")
                state = run_round(root, config, state, status, resident)
                status.update(last_completed=state)
                write_json(root / "status.json", status)
            write_json(root / "status.json", status | {"phase": "stopped", "last_completed": state})
        except Exception as error:
            write_json(root / "status.json", status | {
                "phase": "error", "error": str(error), "last_completed": state,
            })
            raise
        finally:
            if resident is not None:
                resident.close()


if __name__ == "__main__":
    main()
