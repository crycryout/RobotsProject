# Acceptance status

Execution started on 2026-10-08. Acceptance is based on files and exit codes, not development estimates.

| Phase | Status | Evidence | Next dependency |
|---|---|---|---|
| P0 | PASSED | `docs/evidence/p0.json`, `uv.lock`, isolated Python 3.11.16 | None |
| P1 | IN_PROGRESS | Real state/RGB smoke and measured legal hold | Ten complete episodes on each task |
| P2 | IN_PROGRESS | Collector, atomic HDF5, validation and boundary tests implemented | Real collection and recovery acceptance |
| P3 | IN_PROGRESS | Episode statistics and deterministic summary generation implemented | Frozen manifests and real evaluation |
| P4 | NOT_RUN | None | Throughput measurements and official positive control |
| P5 | IN_PROGRESS | Policy/transition interfaces implemented | Versioned handoff and fixtures |
| P6 | NOT_RUN | No real model, reward or trained checkpoint delivered | Model, reward and training owners |

## P0 capability matrix

| Path | One environment | Four environments | Source |
|---|---|---|---|
| ManiSkill state | STATE_OK | STATE_OK | Native reset/step observations |
| ManiSkill sensor RGB | RGB_OK | RGB_OK | `sensor_data/base_camera/rgb`, uint8, 128 x 128 x 3 |
| RoboLab | BLOCKED_HARDWARE | BLOCKED_HARDWARE | Allocated hardware has H800 GPUs and no RTX GPU |

GPU 0 was selected by UUID and exposed as process-local CUDA device 0. GPU 1 is MIG-enabled and was left untouched. No existing Python environment, driver or unrelated process was changed.

SAPIEN reports that system libvulkan is unavailable and uses its bundled libvulkan. Actual sensor rendering passed without a driver change. PyPI downloads stalled; fixed simulation versions were installed from the existing uv cache. Development tools use a declared package mirror. The lock contains wheel hashes and the Python version is pinned separately.

Commands, exit codes, elapsed times, action ranges, image ranges and upstream source verification are recorded in `docs/evidence/p0.json`. Full local reports and sensor samples are under `outputs/p0/`; outputs and raw data are gitignored.

Next action: complete the CPU boundary tests, then collect twenty real episodes per task.
