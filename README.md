# RobotsProject

Working simulation collection, rollout storage and evaluation infrastructure for RL post-training of world-action models. P0-P5 independent infrastructure is implemented and tested on the allocated H800. Real-WAM, scientific reward and RL checkpoint integration remain P6 team dependencies.

## Run

1. Install the locked simulation environment:

   ```bash
   uv sync --frozen --extra sim --group dev
   source .venv/bin/activate
   ```

2. Select the inspected device and verify current capabilities:

   ```bash
   export CUDA_VISIBLE_DEVICES=GPU-92d5af20-d2aa-3a17-8bf5-866a22db81f1
   export OMP_NUM_THREADS=4
   export MS_ASSET_DIR="$PWD/data/assets"
   export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
   python -m robots_project.cli.preflight --output outputs/preflight.json
   python -m robots_project.cli.smoke --config configs/envs/pushcube_rgb.yaml
   ```

3. Collect native-control trajectories:

   ```bash
   python -m robots_project.cli.collect --config configs/envs/pushcube_rgb.yaml --policy random --num-envs 4 --episodes 20 --run-dir outputs/example
   python -m robots_project.cli.validate_trajectories --run-dir outputs/example
   ```

4. Run the frozen development protocol and regenerate summaries:

   ```bash
   python -m robots_project.cli.evaluate --config configs/eval/dev.yaml --policy random --run-dir outputs/example-validation
   python -m robots_project.cli.summarize --run-dir outputs/example-validation
   ```

5. Verify infrastructure correctness:

   ```bash
   python -m pytest -q
   ruff check src tests scripts
   ```

## Measured acceptance

Both `PushCube-v1` and `PickCube-v1` completed twenty state and twenty real sensor-RGB episodes each. All eighty basic acceptance episodes passed trajectory validation. Random-policy task success was zero; this is an infrastructure control result. Official privileged-state motion planning subsequently produced a genuine successful episode on each task.

Real RGB interruption/recovery preserved completed checksums, quarantined four partial episode files and completed the twenty-episode budget without duplicates. Frozen development evaluation completed forty episodes and reproduced byte-identical statistics from its episode table.

On one H800, native random-policy collection with audited HDF5 achieved approximately 28.5, 59.5 and 81.1 RGB transitions/s at 1, 4 and 16 environments, respectively. These include serialization and resets and do not estimate real-WAM inference or RL training performance. Camera resolution was 128 x 128; official positive controls used separately labeled CPU physics and native absolute joint control.

## Documentation and evidence

| File | Purpose |
|---|---|
| [STATUS.md](docs/STATUS.md) | Phase acceptance, measured capabilities, results and remaining dependencies |
| [RUNBOOK_H800.md](docs/RUNBOOK_H800.md) | Rebuild, GPU selection, bounded jobs, controls and executable commands |
| [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) | Frozen splits, common horizons, statistics and error accounting |
| [INTERFACE_CONTRACT.md](docs/INTERFACE_CONTRACT.md) | Model, action, prediction, reward and training responsibilities |
| [Execution plan](docs/EXECUTION_PLAN.md) | Original 2026-10-08 plan; implementation targets are tracked in STATUS |

Compact measured receipts live in [docs/evidence](docs/evidence). Raw trajectories and local handoff samples remain under gitignored outputs. No weights, raw HDF5 data, large videos or secrets are uploaded. RoboLab remains `BLOCKED_HARDWARE` on the allocated H800-only route. Mock outputs are explicitly synthetic and excluded from WAM or RL figures. The instructor's final rubric and submission requirements remain unspecified.

Next action: read [STATUS.md](docs/STATUS.md), then use the runbook command for the required phase.
