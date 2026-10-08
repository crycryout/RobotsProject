# Acceptance status

P0-P5 independent infrastructure is accepted from actual simulator runs and software checks. P6 real-model and RL results remain NOT_RUN. Execution took place on 2026-10-08 and 2026-10-09 (Asia/Macao). This status does not establish compliance with an unspecified course rubric.

| Phase | Status | Acceptance evidence | Remaining dependency |
|---|---|---|---|
| P0 | PASSED | Actual hardware report, locked environments, separate state/RGB smoke tests | None for the selected ManiSkill route |
| P1 | PASSED | Two registered tasks, native adapters, measured movement and legal hold | Real-WAM input/action agreement belongs to P6 |
| P2 | PASSED | Eighty basic episodes, validated boundaries, atomic storage and real interrupted-write recovery | None for native collection |
| P3 | PASSED | Frozen disjoint manifests, forty validation episodes, deterministic summaries and error accounting | Formal checkpoint comparisons NOT_RUN |
| P4 | PASSED | Two official successful controls and eighteen measured throughput repetitions | Measurements do not cover WAM inference or training |
| P5 | PASSED | Versioned contracts, real state/RGB samples, synthetic fixture and same-version replay check | Scientific reward and trainer ownership unresolved |
| P6 | NOT_RUN | No real WAM, scientific reward implementation or trained checkpoint supplied | Model, reward, training and experiment owners |

## P0: measured capability and reproducibility

| Path | One environment | Four environments | Observation source |
|---|---|---|---|
| ManiSkill state | STATE_OK | STATE_OK | Native reset/step observations |
| ManiSkill sensor RGB | RGB_OK | RGB_OK | `sensor_data/base_camera/rgb`, uint8, 128 x 128 x 3 |
| RoboLab | BLOCKED_HARDWARE | BLOCKED_HARDWARE | Allocated H800 hardware has no RTX GPU |

GPU 0 was selected by UUID `GPU-92d5af20-d2aa-3a17-8bf5-866a22db81f1` and exposed as process-local CUDA device 0. GPU 1 is MIG-enabled and was left untouched. The inspected device has 81559 MiB, driver 580.173.02 and toolkit 13.0. The isolated PyTorch wheel uses CUDA runtime 12.6. No existing environment, driver or unrelated process was changed.

Python 3.11.16, ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.3.0 and PyTorch 2.7.1 are pinned in `.python-version`, `pyproject.toml` and `uv.lock`. Selected installed backend source files match upstream v3.0.1 commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`. SAPIEN used its bundled Vulkan loader after reporting unavailable system libvulkan; actual sensor frames passed. Stalled registry downloads were bypassed using cached pinned wheels, with a declared mirror for development tools.

`uv sync --frozen --offline --extra sim --group dev` exited 0. A fresh machine needs access to the recorded wheel registries. Full preflight, exact smoke commands, exit codes, shapes, ranges and source verification are recorded in [p0.json](evidence/p0.json). Local reports and sensor samples are in `outputs/p0/`.

## P1-P2: collection, boundaries and recovery

| Task | State episodes / transitions | Real RGB episodes / transitions | Validation |
|---|---|---|---|
| PushCube-v1 | 20 / 1000 | 20 / 1000 | All passed |
| PickCube-v1 | 20 / 1000 | 20 / 1000 | All passed |

The native random controller produced observable robot movement; the verified native hold preserved arm position. Random-policy success was zero on these smoke budgets. Original backend observations, separate termination/truncation flags, action proposals, executed actions, clipping records, per-slot histories and T+1 same-episode observations are retained. Native automatic-reset terminal extraction is covered by a pinned-structure fixture; production runs use explicit resets.

Representative accepted command, exit 0:

```bash
python -m robots_project.cli.collect --config configs/envs/pushcube_rgb.yaml --policy random --num-envs 4 --episodes 20 --run-dir outputs/p2-pushcube-rgb
python -m robots_project.cli.validate_trajectories --run-dir outputs/p2-pushcube-rgb
```

Other accepted run directories are `outputs/p2-pickcube-rgb`, `outputs/p2-pushcube-state` and `outputs/p2-pickcube-state`. Full summaries are in [p1-p2.json](evidence/p1-p2.json).

Recovery was tested on real four-slot RGB collection using `configs/envs/recovery_rgb.yaml`, with one infrastructure retry declared before execution. A ten-second supervised timeout exited 124 after four completed episodes and four pending partial episodes. Resumption exited 0 with twenty unique completed episodes, unchanged earlier checksums and four quarantined partial files. All twenty trajectories passed validation. The journal records twenty-four attempts, including four `InterruptedCollection` failures; error rate is 4/24, not zero. [recovery.json](evidence/recovery.json) records the accepted `outputs/p2-recovery-audit` run. Exact timeout/resume commands and elapsed times are in [resources.json](evidence/resources.json).

An earlier exploratory recovery run lacked started-attempt accounting. The journal was corrected and the real recovery acceptance was rerun; the earlier run remains local historical evidence and is not the accepted recovery result. Recovery after a partial frozen evaluation batch requires a new run to preserve original slot assignment.

## P3: frozen evaluation

Train, validation and test manifests have disjoint episode IDs and seeds. Validation reserves twenty episodes per task with four fixed slots; formal test reserves fifty RGB episodes per task with one fixed slot. These are held-out initial conditions on known tasks, not task-generalization evidence.

```bash
python -m robots_project.cli.evaluate --config configs/eval/dev.yaml --policy random --run-dir outputs/p3-validation-random
python -m robots_project.cli.summarize --run-dir outputs/p3-validation-random
```

Both commands exited 0. All forty validation episodes completed and passed trajectory validation without infrastructure errors. Each task had 0/20 successes for both `success_once` and `success_at_end`, Wilson 95% interval [0, 0.161125]. Regeneration from the canonical episode table produced byte-identical summaries. A software fixture separately verifies intermediate success followed by final failure, macro weighting, missing episodes and retry denominators. [p3.json](evidence/p3.json) contains the receipt; [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) defines the frozen protocol. Pretrained WAM, task-only RL and alignment-RL rows remain NOT_RUN. Formal test episodes were not consumed.

## P4: real controls and measured resources

Official same-version privileged-state motion-planning controls produced one genuine successful 200-step episode on each task, with `success_once=1` and `success_at_end=1`. Both trajectories passed validation. These controls use CPU physics and native absolute joint control, with real RGB rendered on the H800; they are explicitly labeled privileged infrastructure controls.

The main environment's MPLib 0.1.1 initialization crashed with SIGSEGV at `mplib/planner.py:65`. Both failed attempts and exit -11 are retained. A separately locked NumPy 1.26.4/OpenCV 4.11 environment initialized the planner successfully and ran the two controls. This is evidence of a working compatibility configuration, not proof of the native crash's exact root cause. Rebuild and successful control commands are in [RUNBOOK_H800.md](RUNBOOK_H800.md).

```bash
python -m robots_project.cli.benchmark --config configs/benchmarks/h800.yaml --output outputs/p4-h800-benchmark
```

Exit 0; eighteen repetitions completed, each with fifty warmup and five hundred measured vector steps. They produced 63000 genuine transitions in 1260 complete episodes. Rates include ordinary HDF5 collection and resets, without video:

| Environments | State transitions/s, mean +/- sample SD | RGB transitions/s, mean +/- sample SD |
|---|---|---|
| 1 | 32.51 +/- 0.09 | 28.51 +/- 0.12 |
| 4 | 75.37 +/- 0.42 | 59.47 +/- 0.31 |
| 16 | 112.25 +/- 0.51 | 81.14 +/- 0.25 |

NVML sampled the selected device every 0.1 seconds. Four state repetitions observed additional GPU process IDs; fourteen repetitions, including all nine RGB repetitions, observed exclusive access. This is a sampled observation, not a guarantee between samples. Per-repeat process IDs and exclusivity are retained in [p4.json](evidence/p4.json) and [throughput.csv](evidence/throughput.csv); unrelated processes were not stopped.

Recommended measured native-collection configuration: sixteen environments, 128 x 128 RGB, K=4/H=2. Peak process GPU memory was 3180711936 bytes, leaving approximately 76.2 GiB device memory headroom on the inspected configuration. For one hundred 50-action RGB episodes, measured scaling estimates about 66 seconds including load/warmup and 176 MB of HDF5 storage. This estimate applies to the measured native random policy and horizon, not a WAM or a 200-step workload. RGB serialization/write time was the main measured bottleneck. Nested synchronized physics/render timings are reported separately and must not be added to end-to-end time.

Supervised jobs consumed 0.291934 GPU-hours. Conservatively charging the entire pre-supervision window adds at most 0.238443 GPU-hours, yielding a total accounting upper bound of 0.530378 GPU-hours against the twelve-hour budget. No validation job approached two hours. CPU-only final checks are timed separately; other CPU/waiting work was not individually timed. [resources.json](evidence/resources.json) includes exact commands, exits, limits, elapsed times and the accounting definition.

## P5: integration and handoff

[INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) defines inputs, raw model actions, owner-confirmed conversions, timing, prediction identity, reward hooks and unresolved training semantics. Unconfirmed real-WAM semantics are UNKNOWN. A real visual policy with unconfirmed conversion, missing native RGB or diagnostic state in its policy input is rejected before execution. No probabilities, gradients or scientific reward values are fabricated.

[handoff.json](evidence/handoff.json) supplies hashes and local paths for a real state episode, a real RGB episode and an explicitly synthetic prediction-interface episode. Same-version replay of the state episode matched all observed fields at tolerance 0.0001; no cross-version determinism is claimed. The synthetic fixture produced one hundred camera-specific prediction pairs, of which fifty were valid executed-prefix pairs under H<K. The zero reward hook returned zero with real validity masks; this establishes interface behavior only. A compact transport fixture is in [mock_policy_output.json](fixtures/mock_policy_output.json).

Final correctness checks exited 0: **29 tests passed in 1.35 seconds**, and Ruff reported all checks passed. Tests cover partial slot endings, truncation/final observations, action chunks, prediction invalidation, malformed/timed-out policies, recovery, split integrity and aggregation. All nine implemented CLI modules provide `--help` and explicit failure exits. [checks.json](evidence/checks.json) records final checks.

## Remaining team work

| Required input | Status | Next acceptance |
|---|---|---|
| Real WAM and action/timing/preprocessing agreement | UNKNOWN / NOT_RUN | Offline inference, then one compatible closed-loop episode |
| Scientific alignment reward | UNKNOWN / NOT_RUN | Owner-specified metric and paired prediction validation |
| RL algorithm, probability/gradient semantics and checkpoints | UNKNOWN / NOT_RUN | Actual updates, task-only and alignment-RL checkpoints |
| Final experiments, generalization and course deliverables | NOT_RUN | Frozen comparisons and the instructor's supplied rubric |

Implementation is on local branch `implement/simulation-rollout-evaluation`, with phase commits. No remote push has been performed. Raw trajectories, weights, asset caches and large videos remain gitignored. [deviations.md](deviations.md) records implementation choices and compatibility work.

Next action (under two minutes): open [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) and identify the model action-confirmation and training-owner fields needed to start P6.
