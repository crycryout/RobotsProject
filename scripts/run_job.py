#!/usr/bin/env python3
"""Bound a single project job and record wall time, exit code and GPU-hour charges.

This is a single-process wrapper, not an experiment scheduler. It never inspects
environment secrets or stops processes outside its own process group.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu-uuid", default="GPU-92d5af20-d2aa-3a17-8bf5-866a22db81f1")
    parser.add_argument("--cpu", action="store_true", help="No GPU allocation; hides all CUDA devices")
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--budget-hours", type=float, default=12.0)
    parser.add_argument("--ledger-dir", type=Path, default=Path("outputs/jobs"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.max_seconds <= 7200:
        parser.error("A command and a 1..7200 second job limit are required")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Job name must be a safe filename")
    args.ledger_dir.mkdir(parents=True, exist_ok=True)
    # Hold a project-local lock for the job: initial single-GPU execution is sequential.
    with (args.ledger_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        used = sum(json.loads(p.read_text())["gpu_hours"] for p in args.ledger_dir.glob("*.json"))
        if not args.cpu and used + args.max_seconds / 3600 > args.budget_hours:
            print(f"Budget admission failed: {used:.6f} GPU-hours used, "
                  f"{args.max_seconds / 3600:.6f} requested reservation")
            return 4
        receipt = args.ledger_dir / f"{args.name}.json"
        log = args.ledger_dir / f"{args.name}.log"
        if receipt.exists() or log.exists():
            parser.error("Job name already exists; choose a new name")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else args.gpu_uuid
        environment["OMP_NUM_THREADS"] = "4"
        environment["MS_ASSET_DIR"] = str(Path("data/assets").resolve())
        environment["MS_SKIP_ASSET_DOWNLOAD_PROMPT"] = "1"
        environment["PYTHONFAULTHANDLER"] = "1"
        baseline = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                   "--format=csv,noheader"], capture_output=True, text=True)
        exclusive_start = not any(args.gpu_uuid in line for line in baseline.stdout.splitlines())
        start_unix = time.time()
        start = time.perf_counter()
        error = None
        process = None
        try:
            with log.open("w", encoding="utf-8") as output:
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                process = subprocess.Popen(command, env=environment, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=args.max_seconds)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exception:
                    error = type(exception).__name__
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 124 if error == "TimeoutExpired" else 130
        except OSError as exception:
            code = 1
            error = str(exception)
        elapsed = time.perf_counter() - start
        result = {"name": args.name, "command": command, "start_unix": start_unix,
                  "elapsed_seconds": elapsed, "exit_code": code, "error": error,
                  "gpu_uuid": None if args.cpu else args.gpu_uuid, "gpu_count": 0 if args.cpu else 1,
                  "gpu_hours": 0.0 if args.cpu else elapsed / 3600,
                  "cpu_only_seconds": elapsed if args.cpu else 0.0,
                  "exclusive_gpu_at_start": exclusive_start if not args.cpu else None,
                  "exclusive_gpu_throughout": "See per-job benchmark monitoring; start check alone is insufficient",
                  "max_seconds": args.max_seconds, "log": str(log)}
        temporary = receipt.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(receipt)
        print(f"{args.name}: exit={code}, wall={elapsed:.3f}s, gpu_hours={result['gpu_hours']:.6f}, log={log}")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
