"""Read-only machine diagnostics; never serialize the complete environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import psutil

from robots_project.utils import atomic_json, command_output, provenance


def inspect(output: Path) -> dict:
    disk = shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd())
    report = {
        "schema_version": "1.0",
        "time_unix": time.time(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_count_logical": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in
                          Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), None),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "python": sys.version,
        "python_executable": sys.executable,
        "physical_gpus": command_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version,mig.mode.current", "--format=csv,noheader"]),
        "gpu_topology": command_output(["nvidia-smi", "-L"]),
        "compute_processes": command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"]),
        "cuda_toolkit": command_output(["/usr/local/cuda/bin/nvcc", "--version"]),
        "vulkan_icd_files": [str(p) for p in Path("/usr/share/vulkan/icd.d").glob("*.json")],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "route": {"maniskill_state": "UNTESTED", "maniskill_rgb": "UNTESTED",
                  "robolab": "BLOCKED_HARDWARE"},
        "provenance": provenance(Path.cwd()),
    }
    try:
        import torch
        report["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "visible_devices": [{"process_local_index": i, "name": torch.cuda.get_device_name(i),
                                 "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", "UNKNOWN")),
                                 "total_memory": torch.cuda.get_device_properties(i).total_memory}
                                for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        report["torch"] = {"installed": False}
    atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/preflight.json"))
    args = parser.parse_args()
    report = inspect(args.output)
    print(f"Preflight saved: {args.output}; rendering must be tested by the smoke command.")
    return 0 if report["physical_gpus"]["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
