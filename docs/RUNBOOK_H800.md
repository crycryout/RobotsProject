# H800 runbook

Run from the repository root. Python 3.11.16, PyTorch 2.7.1/CUDA 12.6, ManiSkill 3.0.1, SAPIEN 3.0.3 and Gymnasium 1.3.0 are pinned. The inspected driver is 580.173.02 on Linux x86_64. CUDA toolkit 13.0 is installed system-wide; the PyTorch wheel uses its own 12.6 runtime. No system environment or driver replacement is needed.

## Rebuild and select the allocated device

1. Rebuild the project environment:

   ```bash
   uv sync --frozen --extra sim --group dev
   ```

   The execution session used `uv sync --offline --extra sim --group dev` successfully with existing cached simulation wheels. A clean machine needs registry access. `uv.lock` records exact transitive versions, wheel URLs and hashes; `.python-version` pins the interpreter. The development-tool mirror is declared in `pyproject.toml`.

2. Activate the environment and select the inspected GPU by UUID:

   ```bash
   source .venv/bin/activate
   export CUDA_VISIBLE_DEVICES=GPU-92d5af20-d2aa-3a17-8bf5-866a22db81f1
   export OMP_NUM_THREADS=4
   export MS_ASSET_DIR="$PWD/data/assets"
   export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
   ```

   This UUID becomes process-local `cuda:0`. Recheck `nvidia-smi` before a job. GPU 1 has MIG instances and is not part of these runs. If GPU 0 is unavailable or its topology changes, record the blocker instead of resetting devices or stopping other processes. An inherited fleet schedule may change MIG topology; use preflight to check the current state.

3. Inspect hardware and native observations:

   ```bash
   python -m robots_project.cli.preflight --output outputs/preflight.json
   python -m robots_project.cli.smoke --config configs/envs/pushcube_state.yaml
   python -m robots_project.cli.smoke --config configs/envs/pushcube_rgb.yaml --output outputs/smoke-rgb
   ```

   State and sensor RGB are independent acceptance paths. The tested H800 raster route works with SAPIEN's bundled Vulkan loader. The warning about missing system libvulkan is retained in logs. Use `sensor_data/base_camera/rgb` from reset/step; viewer rendering is not a policy input. RoboLab remains blocked by the allocated hardware's lack of RTX.

## Collect and validate

```bash
python -m robots_project.cli.collect --config configs/envs/pushcube_rgb.yaml --policy random --num-envs 4 --episodes 20 --run-dir outputs/example-pushcube
python -m robots_project.cli.validate_trajectories --run-dir outputs/example-pushcube
python -m robots_project.cli.summarize --run-dir outputs/example-pushcube
```

Repeat with `pickcube_rgb.yaml` for the second task. `--policy hold` uses the measured native Panda hold; `--policy mock` produces only synthetic interface fixtures. A random-policy success rate of zero is recorded as a valid control result. Reusing a run directory requires `--resume`, which checks configuration/policy/manifest, source-file and dependency-lock identity, preserves completed checksums and isolates incomplete/corrupt files. The default infrastructure retry limit is zero. Episode starts are journaled before reset; interrupted partial episodes enter the error table and consume their predefined retry allowance. For recovery testing, `configs/envs/recovery_rgb.yaml` declares one retry before the run. A deliberate `--stop-after-episodes` exits 3; resumption never duplicates finished episodes.

`obs[0:T+1]` contains native same-episode observations. `actions/raw`, `denormalized`, `executed` and clipping masks are distinct. `requests/` preserves full proposed chunks and predictions; offsets beyond the executed prefix are invalid. Per-transition termination and truncation flags remain separate. Initial states are stored for diagnostics and replay, not passed silently to a visual WAM.

## Evaluation and measurement

```bash
python -m robots_project.cli.evaluate --config configs/eval/dev.yaml --policy random --run-dir outputs/example-validation
python -m robots_project.cli.summarize --run-dir outputs/example-validation
python -m robots_project.cli.benchmark --config configs/benchmarks/h800.yaml --output outputs/example-benchmark
```

Validation uses four frozen slots and twenty episodes per task. `configs/eval/test.yaml` reserves fifty RGB test episodes per task with one frozen slot; it is not used during infrastructure development. See `EVALUATION_PROTOCOL.md` before team experiments. Benchmark defaults are state/RGB, N=1/4/16, fifty warmup steps, five hundred measured steps and three repeats. No video is recorded. Measurements include end-to-end writes and separate synchronized components; nested component times must not be added together.

Use the bounded job wrapper for GPU-hour accounting and logs:

```bash
python scripts/run_job.py --name example-validation --max-seconds 600 -- python -m robots_project.cli.evaluate --config configs/eval/dev.yaml --policy random --run-dir outputs/bounded-validation
```

The wrapper reserves one local GPU, enforces a maximum 7200 seconds per job, records command/exit/log/wall time and charges GPU count x wall hours. Its project-local ledger limits aggregate supervised usage to twelve GPU-hours. Unique job names prevent overwritten evidence. It terminates only its own child process group on timeout. For pre-ledger exploratory work, resource evidence also records a conservative initial allocation window. Use `--cpu` only for jobs that truly require no GPU. This wrapper is not a scheduler.

## Official positive control

The main NumPy 2.4.6 environment exposed a native MPLib 0.1.1 initialization crash. Use the separately locked NumPy 1.26.4/OpenCV 4.11 control environment for official planning. It leaves the primary simulator stack unchanged.

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv-control" uv sync --frozen --project environments/control
python scripts/run_job.py --name example-control --max-seconds 600 -- env PYTHONPATH=src .venv-control/bin/python -m robots_project.cli.positive_control --config configs/envs/pushcube_rgb.yaml --seed 51000 --run-dir outputs/example-control
```

This uses the official same-version privileged-state motion-planning solution, CPU physics, native absolute joint control and a separately labeled 200-step control horizon. RGB rendering still uses the selected H800 and is charged as a GPU job. It is an infrastructure positive control, not a WAM or RL baseline. Every attempt is preserved; the CLI does not search seeds until success.

## Replay, correctness and handoff

```bash
python -m robots_project.cli.replay_check --trajectory EPISODE.h5 --output outputs/replay-check
python -m pytest -q
ruff check src tests scripts
```

Replay restores the initial state on the same task/controller/camera/version/backend and reports discrepancies. A single-slot source is the strongest comparison; changes in vectorization may alter physics numerics. No cross-version determinism claim is made. Read `docs/evidence/handoff.json` for actual sample paths and checksums, and `INTERFACE_CONTRACT.md` for model/reward/training responsibilities.

All CLI modules implement `--help`. Exit 0 means accepted execution, 1 a failed/incomplete run or validation, 2 usage failure, 3 intentional interruption, 4 GPU-budget rejection and 124 supervised timeout. Native signal failures are recorded by the job wrapper and return a nonzero shell exit. Never upload weights, raw HDF5 trajectories, large videos or credentials; outputs/data/checkpoints are gitignored.
