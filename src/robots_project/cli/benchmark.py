"""Measure a small, synchronized simulator/collector matrix on one visible GPU."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import statistics
import threading
import time

import numpy as np
import psutil
import yaml

from robots_project.config import load_env_config
from robots_project.envs import make_env
from robots_project.policies import make_policy
from robots_project.policies.controls import request_id
from robots_project.rollout.collector import collect
from robots_project.utils import atomic_json, dump_yaml, provenance


class ResourceMonitor:
    """Sample NVML total process memory and unrelated activity during the timed job."""

    def __init__(self, gpu_uuid):
        self.gpu_uuid = gpu_uuid
        self.peak_rss = 0
        self.peak_process_gpu = 0
        self.unrelated_pids = set()
        self.available = False
        self.stop = threading.Event()

    def __enter__(self):
        import pynvml
        self.nvml = pynvml
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(self.gpu_uuid)
            self.available = True
        except pynvml.NVMLError:
            self.handle = None

        def sample():
            while not self.stop.is_set():
                self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
                if self.available:
                    try:
                        processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                        graphics = self.nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle)
                        memory = {}
                        for process in [*processes, *graphics]:
                            if process.pid != os.getpid():
                                self.unrelated_pids.add(process.pid)
                            elif process.usedGpuMemory is not None and process.usedGpuMemory < 2**60:
                                memory[process.pid] = max(memory.get(process.pid, 0), process.usedGpuMemory)
                        self.peak_process_gpu = max(self.peak_process_gpu, memory.get(os.getpid(), 0))
                    except self.nvml.NVMLError:
                        self.available = False
                self.stop.wait(0.1)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        if self.handle is not None:
            self.nvml.nvmlShutdown()


def synced_time(env, name, bucket):
    import torch
    original = getattr(env.base, name)

    def wrapper(*args, **kwargs):
        if env.base.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        if env.base.device.type == "cuda":
            torch.cuda.synchronize()
        bucket.append(time.perf_counter() - start)
        return result

    setattr(env.base, name, wrapper)


def warmup(env, policy, config, steps):
    ids = [f"benchmark-warmup-{i}" for i in range(config.num_envs)]
    obs = env.reset(list(range(config.num_envs)), ids, [config.seed + i for i in range(config.num_envs)])
    policy.reset(list(range(config.num_envs)), ids)
    for index in range(steps):
        output = policy.act(obs, {"env_ids": list(range(config.num_envs)),
                                 "origin_steps": env.steps.tolist(), "request_id": request_id(),
                                 "chunk_length": config.chunk_length, "dt_sim": 1 / config.control_freq})
        result = env.step(np.clip(output.actions[:, 0], env.action_spec.low, env.action_spec.high))
        obs = result.obs_after_before_reset
        if result.truncated.any():
            ids = [f"benchmark-warmup-{index}-{i}" for i in range(config.num_envs)]
            obs = env.reset(list(range(config.num_envs)), ids, [config.seed + index + i for i in range(config.num_envs)])
            policy.reset(list(range(config.num_envs)), ids)


def measure(entry, n, repeat, settings, root, gpu_uuid):
    import torch
    config = load_env_config(entry, num_envs=n, termination="evaluation", reconfiguration_freq=1,
                             auto_reset=False, split="benchmark", seed=70000 + repeat * 1000,
                             episode_horizon=settings["episode_horizon"])
    if settings["measure_steps"] % config.episode_horizon:
        raise ValueError("measure_steps must contain complete common-horizon episodes")
    start = time.perf_counter()
    env = make_env(config)
    load_seconds = time.perf_counter() - start
    policy = make_policy("random", env.action_spec, config)
    physics, rendering = [], []
    synced_time(env, "_step_action", physics)
    synced_time(env, "_get_obs_sensor_data", rendering)
    warmup_start = time.perf_counter()
    warmup(env, policy, config, settings["warmup_steps"])
    warmup_seconds = time.perf_counter() - warmup_start
    physics.clear()
    rendering.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    run_dir = root / f"{config.env_id}-{config.obs_mode}-{n}-r{repeat}"
    with ResourceMonitor(gpu_uuid) as monitor:
        manifest = collect(env, policy, config, run_dir,
                           n * settings["measure_steps"] // config.episode_horizon)
    if not manifest["complete"] or manifest["last_invocation"]["failed"]:
        raise RuntimeError(f"Benchmark collection failed: {run_dir}")
    invocation = manifest["last_invocation"]
    elapsed = invocation["elapsed_seconds"]
    latencies = invocation["policy_latency_seconds"]
    bytes_written = sum(r["bytes_written"] for r in manifest["episodes"])
    episodes = len(manifest["episodes"])
    return {
        "task_id": config.env_id, "obs_mode": config.obs_mode, "num_envs": n, "repeat": repeat,
        "backend": config.sim_backend, "video_recording": False,
        "warmup_steps": settings["warmup_steps"], "measure_vector_steps": len(invocation["env_step_seconds"]),
        "transitions": invocation["transitions"], "completed_episodes": episodes,
        "wall_seconds": elapsed, "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
        "transitions_per_second": invocation["transitions"] / elapsed,
        "episodes_per_hour": episodes * 3600 / elapsed,
        "policy_latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
        "policy_latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
        "environment_step_p50_ms": float(np.percentile(invocation["env_step_seconds"], 50) * 1000),
        "environment_step_p95_ms": float(np.percentile(invocation["env_step_seconds"], 95) * 1000),
        "physics_seconds": sum(physics), "render_capture_seconds": sum(rendering),
        "env_step_seconds": sum(invocation["env_step_seconds"]),
        "serialization_write_seconds": sum(r["write_seconds"] for r in manifest["episodes"]),
        "reset_seconds": invocation["reset_seconds"],
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        "peak_process_gpu_bytes": monitor.peak_process_gpu if monitor.available else None,
        "peak_host_rss_bytes": monitor.peak_rss, "bytes_written": bytes_written,
        "exclusive_gpu_observed": not monitor.unrelated_pids if monitor.available else None,
        "unrelated_gpu_pids": sorted(monitor.unrelated_pids),
        "estimated_seconds_per_100_episodes": 100 * elapsed / episodes + load_seconds + warmup_seconds,
        "estimated_bytes_per_100_episodes": 100 * bytes_written / episodes,
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()
    values = yaml.safe_load(args.config.read_text())
    if set(values) != {"schema_version", "env_configs", "num_envs", "warmup_steps", "measure_steps", "repeats", "episode_horizon"}:
        raise ValueError("Invalid benchmark configuration")
    if values["schema_version"] != "1.0" or values["repeats"] < 1 or values["warmup_steps"] < 1:
        raise ValueError("Invalid benchmark schema/repeats/warmup")
    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
        raise ValueError("Benchmark requires one physical GPU selected by UUID")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in values["env_configs"]:
        for n in values["num_envs"]:
            for repeat in range(values["repeats"]):
                row = measure(entry, n, repeat, values, args.output, gpu_uuid)
                rows.append(row)
                atomic_json(args.output / "measurements.json", rows)
                print(f"{row['obs_mode']} N={n} repeat={repeat}: "
                      f"{row['transitions_per_second']:.1f} transitions/s", flush=True)
    with (args.output / "throughput.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = []
    for obs in sorted({r["obs_mode"] for r in rows}):
        for n in values["num_envs"]:
            group = [r for r in rows if r["obs_mode"] == obs and r["num_envs"] == n]
            grouped.append({"obs_mode": obs, "num_envs": n, "repeats": len(group),
                            "transitions_per_second_mean": statistics.mean(r["transitions_per_second"] for r in group),
                            "transitions_per_second_std": statistics.stdev(r["transitions_per_second"] for r in group) if len(group) > 1 else 0,
                            "peak_process_gpu_bytes": max(r["peak_process_gpu_bytes"] or 0 for r in group),
                            "estimated_seconds_per_100_episodes_mean": statistics.mean(r["estimated_seconds_per_100_episodes"] for r in group),
                            "estimated_bytes_per_100_episodes_mean": statistics.mean(r["estimated_bytes_per_100_episodes"] for r in group)})
    atomic_json(args.output / "summary.json", {"measurements": grouped, "provenance": provenance(Path.cwd()),
                "timing_definition": "CUDA synchronize before/after physics and sensor capture; CPU snapshots synchronously copy sensor data. Component durations overlap with env.step and must not be added. End-to-end includes resets, policy invocation, serialization, atomic finalization, and manifest writes; excludes initial load and warmup, which are reported separately.",
                "limitations": "Random native-control policy, raster 128x128, no video; sampling NVML at 0.1 seconds. Evaluation-style common horizons guarantee exactly 500 measured vector steps; no padding/reset transitions are counted."})
    dump_yaml(args.output / "config.resolved.yaml", values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
