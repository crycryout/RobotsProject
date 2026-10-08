# Model, reward and training interface contract v1.0

Status: infrastructure contract implemented; real-WAM acceptance NOT_RUN. Unconfirmed model values are `UNKNOWN`. This document defines transport and evidence requirements and does not assign scientific reward design or an RL optimizer to the infrastructure owner.

## Ownership and confirmation

| Item | Owner | Current agreement |
|---|---|---|
| Simulator, observation capture, collection, storage, evaluation | Infrastructure owner | Native ManiSkill route implemented |
| Model inputs, action semantics, predictions, DROID conversion | Model owner | UNKNOWN; confirmation required before real execution |
| Alignment metric and scientific auxiliary rewards | Reward owner | UNKNOWN; zero hook available only as a control |
| Sampling unit, probabilities/gradients, optimizer, checkpoint updates | Training owner | Unassigned team responsibility; UNKNOWN |
| Ablation organization, checkpoint selection and final result analysis | Experiment owner | Single-run CLI and machine-readable artifacts supplied |

## Inputs and native debug actions

| Field | Infrastructure value | Real-model requirement |
|---|---|---|
| Camera identity/order | One native `base_camera` | UNKNOWN |
| Color/layout/range | RGB, uint8, B x 128 x 128 x 3, [0,255] | UNKNOWN |
| Preprocessing version | `sensor-native-rgb-u8-v1` | UNKNOWN |
| History frames / reset history | Per-episode policy reset; input history is policy-owned | UNKNOWN |
| Proprioception | Original `agent/qpos`, `agent/qvel`, preserved native fields | UNKNOWN |
| Instructions / goals | None for native random/hold controls | UNKNOWN |
| Privileged simulator state | Diagnostic/state-control path only | Must be excluded from visual WAM input and scientific reward unless explicitly agreed |

The selected Panda `pd_joint_delta_pos` controller has eight native components: seven normalized current-joint deltas in [-1,1] mapped to [-0.1,0.1] radians, followed by one normalized absolute mimic-gripper position mapped to [-0.01,0.04] meters. +1 opens and -1 closes; the small negative target follows the official controller's force-generating convention. Zero arm deltas plus the observed gripper position implement the measured legal hold. Robot, controller, bounds, dimension, control/physics frequency and backend version are included in the action specification ID.

These are debug-control semantics. An eight-dimensional DROID output is not this eight-dimensional joint action. Model per-component semantics, units, absolute/delta convention, coordinate frame, rotations, gripper direction, normalization statistics and control frequency remain UNKNOWN. There is no inferred DROID-to-ManiSkill converter. Truncation, padding or shape matching cannot establish compatibility. A real policy response must use the exact environment action specification ID and declare a model-owner-confirmed conversion. Unconfirmed real-WAM execution is rejected.

## Policy and timing

`PolicyAdapter.specs/reset/act/close` and `PolicyOutput` are defined in `src/robots_project/types.py`. `act` receives a selected observation batch and a request context containing slot IDs, episode IDs, origin control steps, request UUID, K, H and dt. Original backend observations retain their topology; standardized policy observations are additional views. Visual adapters must deliberately select RGB/proprioception, excluding `diagnostic_state`.

Actions have B x K x D shape, finite floating values, action specification ID, immutable policy version and matching request ID. Native controls use K=4 and H=2 by default. Every slot owns its own cursor and episode history. A boundary invalidates its remaining actions. A new request replans after H actions. Inactive slots and resets are excluded from transition counts.

A real converter must additionally return `raw_model_output` with its original B x K_model x D_model dimensions and `conversion_metadata` containing the target action specification ID and an owner confirmation reference. Both are preserved per request. No generic code truncates the original tensor. Native proposal, denormalized/native float32 actions and actually clipped/executed actions remain separately stored. Internal simulator-controller normalization is recorded through its native configuration; conversion operations between model and controller must be supplied by the adapter.

Predicted video or latents are optional and remain absent when unavailable. Optional training extras are stored only with an explicit `training_extra_spec`; old probabilities and values are never fabricated. Real model batch support, history length, checkpoint revision, predicted-view availability and predicted-time convention remain UNKNOWN.

`MockWAM` repeats sensor inputs as an explicitly synthetic fixture. Latency, timeout, malformed shape, non-finite actions and history-reset tests use it. Every resulting trajectory, prediction and summary is labeled synthetic. It is not a model-performance or training result.

## Prediction-to-state binding

Every stored prediction is bound to `(episode_id, request_id, origin_step, action_sequence_hash, prediction_offset_time)`. Requests retain raw, native denormalized and clipped planned action chunks, transformation records and actual executed-prefix length. Actual observations retain camera identity through native keys, control-step index, simulation time and preprocessing version. Images are stored once in the T+1 native observation sequence.

Version 1 accepts positive, increasing integer control-step offsets and timestamps equal to offset x dt. It excludes the current frame. A different frequency/interpolation convention needs an explicit converter and protocol revision. An old prediction beyond the executed prefix is invalid even if a later replan reaches the same timestamp. Clipped prefixes are conservatively invalidated. Episode boundaries and unknown/mismatched camera/preprocessing fields invalidate pairs.

`integration.alignment.pairs_from_episode` produces camera-specific pairs with validity and reasons. Missing cameras/predictions remain unavailable. Latent arrays may be preserved with an encoder/version specification; no pixel-level alignment is claimed for them, and the pixel pairing function does not score latents.

## Reward and training handoff

`RewardHook.compute(batch)` returns `RewardResult(auxiliary_reward, validity, diagnostics, reward_version)`. `ZeroRewardHook` is a working pipeline control. Environment rewards and original predictions remain unchanged. The reward owner supplies the scientific metric, normalization and version; reward composition belongs in the training configuration.

An inference endpoint or exported trajectory is not a differentiable model graph. The training owner must supply algorithm, online/offline requirements, sampling unit, policy probability/gradient semantics, meaningful training extras, policy update behavior and checkpoint revision. PPO/GRPO is not implemented or implied by this collector.

## Storage and experiment handoff

Schema 1.0 uses an atomic HDF5 file per episode, JSON sidecar, run manifest and canonical episode CSV. `obs[0:T+1]`, executed/raw/denormalized actions, environment rewards, independent termination/truncation signals, success flags, native initial states, request chunks, provenance, checksums, completed flags and error attempts are preserved. Automatic-reset observations are separate events. Starts are journaled before reset. A valid completed file can be recovered even if its sidecar/manifest update was interrupted; partial files are quarantined, entered as interrupted attempts and restarted only within the predefined retry allowance.

Use the commands in `RUNBOOK_H800.md`. Exit 0 means success, 1 means infrastructure/validation failure, 2 means argparse usage failure, 3 is a deliberate collection interruption, 4 is GPU budget admission failure, and 124 is a supervised timeout. No complex scheduler is introduced.

Real state and RGB samples and a synthetic model-interface episode are listed with hashes in `docs/evidence/handoff.json`. They remain local under gitignored outputs. No weights or raw trajectories are uploaded. Formal test manifests are distinct from smoke/control/fixture episodes. Real-model, task-only RL and task-plus-alignment RL rows remain NOT_RUN.
