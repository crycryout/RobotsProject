# Evaluation protocol v1

Frozen before real-model or RL results. The primary metric is `success_once`; `success_at_end` is always reported. Random-policy acceptance episodes belong to the `smoke` split and are not formal test data.

## Initial conditions and budgets

`manifests/train.jsonl`, `validation.jsonl` and `test.jsonl` contain distinct episode IDs and seeds across both tasks. Validation uses twenty episodes per task; test uses fifty. These are held-out initial conditions on two known tasks, not task generalization. Entire held-out tasks require an additional team-approved protocol.

ManiSkill 3.0.1, Panda, native joint delta position control, 20 Hz control, 100 Hz physics, a fifty-step horizon, proposed chunk length K=4, and executed horizon H=2 are fixed. The initial visual configuration uses the native `base_camera` raster observation at 128 x 128. Real-model preprocessing or action conversion requires a new explicit protocol revision and separate run.

Validation freezes four slots and batch ordering. Test freezes one slot. Do not compare validation against test. Each evaluation batch resets all slots and uses `reconfiguration_freq=1` as prescribed by the [official evaluation setup](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/docs/source/user_guide/reinforcement_learning/setup.md). Intermediate success/failure does not trigger reset. Raw termination signals are retained in trajectories; only the common time limit ends evaluation episodes.

The adapter deliberately uses explicit resets, avoiding ambiguous automatic-reset conventions. Every saved terminal observation belongs to its original episode. Automatic reset in collection uses the pinned full batched `final_observation` layout and stores reset observations as separate events.

Seeds alone do not guarantee matched states when batch count, ordering or reconfiguration changes. Initial state trees and their hashes are stored. Direct state-restored replay is provided for a single slot. Formal comparisons must match recorded initial-state hashes or keep the entire frozen seed/batch/configuration sequence and report this limitation. A partially completed evaluation batch cannot be resumed with changed slot assignments.

## Counts, errors and uncertainty

`episodes.csv` records every attempted episode, including infrastructure failures. The default retry limit is zero; any increase must precede the experiment. Timeouts, malformed outputs and simulator/write failures stop the run and remain auditable. Resuming an interrupted write retains completed episodes and quarantines partial files. Attempts are never retried until success.

The summary reports intended, attempted, completed and missing counts. Success counts and Wilson 95% intervals use completed episodes; conservative rates use the full intended budget and count missing episodes as failures. An incomplete run is explicitly incomplete. Error rate is failed infrastructure attempts divided by all recorded attempts.

Per-task metrics include environment reward return, steps, and time/steps to first success. No-success values are null. Auxiliary rewards are kept separate from task rewards. Macro averages weight tasks equally. They are unavailable if any task has no completed episode; conservative macro averages remain defined. No aggregate binomial confidence interval is invented for an average across tasks.

The Wilson interval measures episode sampling uncertainty at a fixed checkpoint. Three evaluation seed lists are not three independent training runs. When supplied, three distinct training seeds must be reported separately, with mean and standard deviation across training runs.

## Team comparisons

Pretrained WAM, task-only RL, and task-plus-alignment RL remain `NOT_RUN` until real checkpoints are supplied. Synthetic fixtures are labeled at episode and summary level and are excluded from real model figures. No training-convergence plot is generated without actual training logs.

Validation is available for checkpoint/reward selection. Formal test episodes must not be used for repeated tuning. Compare equal episode budgets and matched simulator, robot/controller, camera processing, executed chunk length and control frequency. Report both environment interactions and wall time for trained methods. Changing H, camera resolution or tasks is a recorded ablation.

`python -m robots_project.cli.summarize --run-dir RUN_DIR` regenerates deterministic statistics from `episodes.csv` and `manifest.json`. Task directories preserve per-run resolved configurations, provenance, error logs, checksummed trajectories and initial states. Evaluation suites additionally contain a combined episode table and per-task success plots.
