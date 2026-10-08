# Independent Execution Plan: Simulation, Rollouts, and Evaluation

Version: 2026-10-08. Target repository: `crycryout/RobotsProject`.

## 0. Outcome and Current Status

A working, reproducible simulation collection and evaluation framework can be completed independently before the model and RL training components are delivered. The remaining integration work includes validating the real world-action model (WAM), checking prediction-alignment rewards, measuring RL post-training outcomes, and establishing generalization results.

This plan follows the responsibilities in the supplied one-page `RL WAM.pdf` proposal. It is not the instructor's complete grading specification. The final rubric, submission template, and exact deadline have not been supplied, so completing this plan alone does not establish compliance with every course requirement.

The repository was empty when this plan was prepared, with `main` as its default branch. The current deliverables are planning and execution-entry documents. No simulator installation, implementation, H800 experiment, or measured result is claimed. The directories, commands, and tests below describe implementation targets for the server-side coding session.

### 0.1 Responsibilities and Boundaries

| Workstream | Proposal responsibility | Treatment in this plan |
|---|---|---|
| Simulation environments, rollout infrastructure, evaluation | Infrastructure owner | Main implementation scope |
| WAM integration, state-prediction alignment, reward design | Model and reward owner | Receive interfaces and data; retain ownership of scientific design |
| Experiment management and ablation analysis | Experiment owner | Receive standard commands and output formats; avoid duplicate scheduling systems |
| RL optimization, backpropagation, checkpoint updates | Not explicitly assigned as a separate responsibility | Record as an unresolved team responsibility rather than assigning it implicitly |

Rollout infrastructure means invoking policies, executing environment actions, saving trajectories, handling episode boundaries, and evaluating outcomes. It does not automatically include implementing PPO/GRPO for a diffusion-based WAM, defining tractable policy probabilities, or designing an RL loss.

### 0.2 Five Independent Deliverables

1. H800 environment and rendering diagnostics with reproducible installation instructions.
2. A small ManiSkill task set with common environment and policy interfaces.
3. A trajectory collector supporting action chunks, vector environments, and recovery.
4. Fair evaluation, trajectory validation, resource measurements, and auditable results.
5. Integration contracts, sample data, and evidence-backed progress records.

Most modules in P0-P5 can be completed before real-model acceptance in P6. Track completion through evidence for each module rather than an unmeasured percentage.

## 1. Technical Route and Prerequisites

### 1.1 Main Route: ManiSkill, State Observations First, Real RGB Second

Begin with `PushCube-v1` and `PickCube-v1` as candidate tasks. Add a third task only after the first two pass acceptance. Verify task IDs, robot support, control modes, and camera keys against the pinned version's actual registry and source. A supported `pd_ee_delta_pose` controller may be used during initial debugging, but it must not be assumed compatible with DROID actions.

Use state observations to establish simulator and data-pipeline correctness, then separately validate real camera RGB observations. A state-only success does not establish the visual path required by a WAM. If real camera frames are unavailable, mark the visual path `PARTIAL / RGB_BLOCKED`.

Early tests can use random actions, a verified legal hold controller, and available official demonstration replay. A zero success rate for a random policy does not imply infrastructure failure. Conversely, successful replay does not demonstrate that a WAM can control the robot.

### 1.2 RoboLab: A Conditional Branch, Not the Default on H800-Only Hardware

At the time of review, the official RoboLab README requires an NVIDIA RTX GPU and identifies Isaac Sim/Isaac Lab as its simulation stack. Two H800 GPUs do not establish RTX rendering compatibility. Headless execution removes the display window, not the need for visual rendering. [S4]

- Record the actual server GPUs and rendering capabilities in P0. If only H800 GPUs are available, mark the RoboLab branch `BLOCKED_HARDWARE`; do not spend days attempting to bypass its stated hardware requirements.
- If a suitable RTX machine is allocated to the project, consider running simulation there and model inference on H800. Measure network round trips and image-transfer costs.
- Other machines mentioned outside this task are not automatically allocated. Do not connect to or consume resources on an unspecified machine.
- Complete ManiSkill and backend-independent data contracts first. Implement a RoboLab backend only after suitable RTX resources and an official smoke example have been verified.

### 1.3 Official Model Integrations Are References, Not Training Guarantees

NVIDIA provides a Cosmos policy-server and RoboLab-client example covering Edge-Policy-DROID and `BananaInBowlTask`. [S5] This provides an integration reference; it does not establish simulator compatibility on the current server or availability of gradients, policy probabilities, or predicted-frame export through that service.

The Cosmos model card specifies DROID 8D actions. Do not truncate an 8D output to 7D and pass it into a ManiSkill controller. Obtain the action semantics, coordinate frame, units, normalization, and timing from the model owner. [S6]

V-JEPA 2-AC is a latent action-conditioned world model used with image-goal planning. It is not a guaranteed lightweight, interface-compatible replacement for Cosmos. Switching models is a team-level methodology decision. [S7]

### 1.4 Version Strategy

Do not overwrite the server's existing PyTorch/CUDA environment without first inspecting it. Install dependencies in an isolated virtual or conda environment. Simulation and model serving may use separate environments connected through an explicit interface.

After a successful installation, pin Python, PyTorch, ManiSkill, SAPIEN, Gymnasium, and other dependencies. Record upstream commits. For containers, record an image digest rather than only a mutable `latest` tag.

Upstream branches can change. The references in this plan were reviewed on 2026-10-08; documentation version labels are not instructions to install an assumed latest stable release.

## 2. Completion Criteria and Time Budget

### 2.1 Advance by Acceptance Evidence

| Phase | Objective | Team dependency | Estimated active development | Acceptance evidence |
|---|---|---|---|---|
| P0 | Validate hardware, dependencies, and rendering | None | 2-6 hours | Preflight report, environment lock, route decision |
| P1 | Establish environment and policy interfaces | None | 4-8 hours | Smoke runs on two tasks, observation samples |
| P2 | Collect valid trajectories and recover interrupted runs | None | 6-12 hours | Valid trajectories, boundary tests, recovery report |
| P3 | Produce reproducible evaluation outputs | None | 4-8 hours | Episode table, summaries, evaluation manifests |
| P4 | Measure resources and obtain a working control policy | Official resources must be available | 3-6 hours | Throughput table, non-random control evidence |
| P5 | Deliver model, reward, and training contracts | Contract preparation is independent; real acceptance is not | 3-6 hours | Contracts, fixtures, integration checks |
| P6 | Integrate the real WAM and trained checkpoints | Required | 4-12 hours or more | Real-model closed loop and prediction alignment evidence |

The P0-P5 estimate totals 22-46 active development hours, excluding downloads, waiting for environment issues to be resolved, and long experiments. Coding assistance can accelerate implementation but cannot eliminate hardware compatibility issues or scientific uncertainty.

Do not use the proposal's approximately 13-hour training estimate as a whole-project budget. Without the exact model, tasks, steps, precision, update strategy, and measurement logs, it remains an unverified proposal statement.

### 2.2 Initial Resource Limits

Start on one available H800 allocated to the project. Leave the other GPU available until the single-GPU path is correct; later it may run independent experiments. Do not introduce DDP, Ray, Slurm, or a multi-machine system initially. Do not stop unrelated processes.

The suggested initial independent-stage budget is 12 GPU-hours in total, with a maximum of 2 hours for any individual validation job. These are planning limits, not performance promises. On reaching a limit, save results, report remaining work, and continue tasks that can run on CPU. The budget can be adjusted when actual measurements justify it.

Count GPU-hours as the sum of GPU count multiplied by wall-clock hours for each job. Report waiting and CPU-only work separately. Record whether each performance measurement had exclusive GPU access.

## 3. P0: Environment and Compatibility Preflight

### P0.1 Record the Server Environment

Implement:

```bash
python -m robots_project.cli.preflight --output outputs/preflight.json
```

Record the OS, CPU/RAM, available disk space, Python version, driver, GPU model/UUID/memory, CUDA runtime, PyTorch CUDA availability, and dependency versions. Include only relevant device information. Do not log tokens or dump the full process environment.

Inspect utilization with `nvidia-smi`. Record both physical GPU UUIDs and process-local device indices. After setting `CUDA_VISIBLE_DEVICES`, do not use an old physical index as if it were a process-local CUDA device index.

### P0.2 Create an Isolated Environment

Read the existing repository and any applicable `AGENTS.md` before making changes. Preserve existing work. Install a minimal dependency set, starting with simulation rather than the model stack, and use a separate asset cache.

Produce a rebuildable environment lock and the installation commands that actually succeeded. A `pip freeze` file alone is insufficient without Python and driver/runtime constraints.

### P0.3 Validate State and RGB Separately

Run, in order: single-environment state observations, a small vectorized state setup, single-environment real camera RGB, and a small vectorized RGB setup. Save exit codes, error summaries, image shapes/dtypes/ranges, configurations, and elapsed times.

ManiSkill visual rendering requires Vulkan configuration. [S1] Determine H800 compatibility through an actual local test; `torch.cuda.is_available()` is not a rendering test. Extract policy images from sensor observations returned by reset/step. Viewer images are for demonstration only.

### P0.4 Select an Explicit Route

- `STATE_OK + RGB_OK`: continue the complete main route.
- `STATE_OK + RGB_BLOCKED`: continue state collection and evaluation; leave visual acceptance incomplete and document required resources or administrator actions.
- GPU simulation fails: use a supported CPU backend for interface and correctness work where possible. Label all artifacts with the backend; do not claim H800 throughput validation.
- H800-only hardware: do not make RoboLab the installation priority. Missing model weights: continue with mock contracts and non-WAM policies.

Acceptance: `docs/STATUS.md` contains the measured capability matrix and next action. A failure in one branch must not stop independent work in the others.

## 4. P1: Environment, Observation, and Policy Interfaces

### P1.1 Freeze Task Configurations

For each task, create a YAML configuration containing backend, environment ID, robot ID, observation mode, control mode, reward mode, number of environments, episode horizon, control/simulation frequencies, camera configuration, image resolution, seed manifest, split, and backend version.

Keep separate initial state and RGB configurations. Low-resolution RGB may be used for debugging. Real-WAM integration must follow its preprocessing requirements; changing resolution creates a separate run rather than silently changing a comparison group.

Increase `num_envs` from 1 to 4 to 16, and expand further only after measurement. Do not start with 256 RGB environments.

### P1.2 Define an EnvAdapter

Expose reset, step, close, and specifications. Preserve the original backend observation alongside a standardized policy observation. Do not delete backend fields just to simplify the common interface.

A `StepBatch` must include at least `obs_before`, `action_executed`, `obs_after_before_reset`, `reward_env`, `terminated`, `truncated`, `success`, `episode_id`, `env_slot`, `step_index`, and `dt_sim`. An initial observation returned after a reset must be a separate field or event.

Privileged simulator state is for diagnostics and explicitly labeled state-based controls. Do not silently feed it into the visual WAM or the scientific reward implementation.

### P1.3 Define a PolicyAdapter

Suggested protocol:

```python
class PolicyAdapter:
    def specs(self) -> dict: ...
    def reset(self, env_ids, episode_ids) -> None: ...
    def act(self, observation_batch, request_context) -> "PolicyOutput": ...
    def close(self) -> None: ...
```

`PolicyOutput` contains actions with shape `[B, K, D]`, an `action_spec_id`, `policy_version`, and `request_id`. Optional fields include predictions, prediction specifications, policy state, and training extras. B is the number of environments in the request; K is the proposed action-chunk length; D comes from the action specification and must not be hardcoded in generic infrastructure.

Implement a RandomPolicy, a verified legal hold controller, and a ReplayPolicy. A zero vector does not necessarily mean holding still; verify it for the selected controller.

Also implement a MockWAM explicitly labeled `synthetic=true` for prediction-field, latency, timeout, and reset tests. Its outputs must not enter model-performance figures.

### P1.4 Independent Acceptance

Complete at least 10 full episodes on each of two real simulator tasks. At least one action sequence must produce observable robot movement. Save small sensor-frame samples, action ranges, and episode logs. If RGB is unavailable, keep visual acceptance explicitly pending.

## 5. P2: Rollout Collection and Trajectory Storage

### P2.1 Execute Action Chunks Correctly

A policy request produces K actions. Configure an execution horizon H satisfying `1 <= H <= K`; execute H actions, obtain new observations, and replan. If an episode ends inside a chunk, discard the remaining actions and reset that environment's policy history. Environment slots must not share action cursors or history.

Save the raw model output, denormalized actions, and actions actually sent to the simulator. Record every clipping operation, coordinate transform, interpolation, and frequency conversion in metadata. Reject real-WAM execution when its required action conversion has not been established.

### P2.2 Separate Termination, Truncation, and Automatic Reset

Use separate termination configurations for training collection and evaluation. Save `terminated` and `truncated` independently; do not collapse them into a single done flag and discard their meanings. Bootstrapping rules belong to the trainer. The collector must retain sufficient information without inventing those rules.

Check whether an automatically reset environment returns an observation from a new episode. Extract and preserve the previous episode's terminal observation. Inspect the pinned backend version's actual structure rather than assuming Gymnasium and ManiSkill wrappers return identical fields.

### P2.3 Design the Storage Format

A minimal implementation can use one HDF5 file per episode, or bounded HDF5 shards, with a JSON manifest. A database is unnecessary initially. If an official recorder is reused, extend it with the required metadata rather than maintaining two conflicting versions of ground truth.

| Category | Required fields |
|---|---|
| Identity | Schema version, run ID, episode ID, environment slot, task ID, split, seed, policy version |
| Sequences | `obs[0:T+1]`, raw actions, executed actions, environment rewards, termination flags, truncation flags, success flags |
| Predictions | Request ID, origin step, predicted offsets/timestamps, predicted RGB or latent references, validity masks |
| Reproduction | Resolved configuration, code commit, dependency-lock hash, checkpoint hash/revision, robot/control/camera specifications, initial state or a reproducible reference |
| Audit | Transition count, completion flag, ending reason, retry records, checksums, synthetic/privileged-policy labels |

Observations may be split into RGB, proprioception, and diagnostic state groups. The T+1 observation convention applies to complete contiguous episodes. Each transition's next observation must belong to the same episode. Store each image once rather than duplicating current and next images in every transition.

Save training extras such as old log probabilities, value estimates, and action masks only when a real implementation supplies them with defined semantics. Missing values must remain absent or null, never fabricated zeros. A collector suitable for RL does not automatically provide everything needed for on-policy WAM training.

### P2.4 Implement Recovery and Validation

Write to temporary files and atomically rename them after completion. During recovery, use episode IDs, checksums, and completion flags to identify finished episodes. Isolate corrupted or incomplete episodes, or restart them from the beginning. Exact resumption at an arbitrary physics step is not a default requirement.

Implement `validate_trajectories` to check shapes, dtypes, finite values, action specifications, time ordering, episode boundaries, and unique IDs. Saving an initial state does not guarantee identical trajectories across simulator versions. Replay on the same version and configuration and report discrepancies.

### P2.5 Cover the Important Correctness Risks

| Scenario | Required behavior |
|---|---|
| One vector-environment slot ends early | Other slots retain their own history, actions, and episode IDs |
| A time limit truncates an episode | Preserve the terminal observation and separate truncation from termination |
| A reset occurs inside a chunk, or H is smaller than K | Invalidate remaining actions; do not pair unexecuted predictions with ground truth |
| A policy times out or returns NaNs/wrong dimensions | Record an infrastructure error and stop that episode; do not silently substitute random actions |
| A write is interrupted and the process restarts | Do not duplicate completed episodes or count partial trajectories |

Acceptance: first validate boundaries using a controllable fake environment, then collect 20 complete episodes per task on two real simulator tasks. Fake-environment tests establish software correctness, not robotics results.

## 6. P3: Evaluation Protocol and Result Generation

### P3.1 Fix Tasks and Initial Conditions

Create manifests before training or inspecting final results. A development configuration can use 20 validation episodes and 50 test episodes per task. The independent random-policy smoke episodes are separate from formal test data. Increase to 100 test episodes per task per checkpoint only when the team budget supports it.

Record an independent episode ID, seed, and scene/object configuration or initial state. Keep train, validation, and test episode seeds and scene records disjoint. Different seeds establish held-out initial conditions, not held-out tasks. Task generalization requires entire tasks to be excluded from training.

Changing vectorization can change random-number consumption and initial-state sequences. Identical seeds alone do not establish matched evaluation conditions. Save and validate restored states where possible. Otherwise freeze the environment count, batch ordering, and reconfiguration rules, and document limitations. Faster and slower methods must receive the same episode budget.

### P3.2 Evaluate Complete Episodes Consistently

For ManiSkill, follow the pinned version's official evaluation approach: avoid early resets on intermediate success/failure, run to a common horizon, and record both `success_once` and `success_at_end`. Follow the documented object-randomization configuration. [S3] Do not automatically apply these evaluation settings to training collection or another backend.

Use `success_once` as the provisional primary metric and report `success_at_end` as well. Freeze the choice before formal team experiments. Task-specific adjustments are allowed before evaluation, not after inspecting which metric looks better.

| Metric | Meaning and caveat |
|---|---|
| `success_once` | Whether success occurred at any point in a complete episode |
| `success_at_end` | Whether the task was successful on the final valid step |
| `return_env` | Sum of environment task rewards, suitable for comparison across reward variants |
| Steps and time to first success | Use null, not zero, when success never occurs |
| Error rate | Fraction of attempted episodes with infrastructure failures such as timeouts, crashes, or illegal actions |

Save auxiliary rewards and combined training rewards separately. Comparing total returns under different auxiliary-reward weights does not establish improved task capability.

### P3.3 Statistics and Controlled Comparisons

The eventual real-model comparison includes pretrained WAM without RL, task-only RL, and task-plus-alignment RL. Mark missing checkpoints `NOT_RUN`; do not populate their rows with mock data.

Report successful episode counts, sample counts, and Wilson 95% intervals for each task. Use a macro average across tasks by default and include per-task sample counts. The binomial interval describes finite-episode uncertainty for a fixed checkpoint, not variation across training seeds.

When feasible, use three independent training seeds and report each result plus the mean and standard deviation. Evaluating one checkpoint with three sets of environment seeds is not equivalent to three training runs. Do not use the formal test set for checkpoint selection, reward-weight selection, or repeated tuning.

Match tasks, simulator settings, control frequency, executed chunk length, camera preprocessing, and evaluation budgets. For trained methods, report both environment interactions and wall-clock time. Changing execution horizon is an ablation, not an unnoticed change in the main comparison.

Do not silently exclude failures or retry until success. Preserve all attempts and apply a predefined infrastructure-retry limit. Report intended, completed, and failed episode counts. Mark incomplete evaluations as incomplete; when useful, also report a conservative result that counts missing episodes as failures.

### P3.4 Output Artifacts

Each run contains `config.resolved.yaml`, `provenance.json`, `episodes.csv`, `summary.json`, `errors.jsonl`, and `manifest.json`. The aggregation tool produces per-task success plots, final-step success plots, and throughput tables. Do not create training-convergence plots without actual training logs.

Acceptance: regenerating summaries from the same episode table produces identical aggregates. A fixture with intermediate success followed by final failure must yield `success_once=1` and `success_at_end=0`. Verify denominators and cross-task weighting.

## 7. P4: Working Controls and Resource Measurements

### P4.1 Obtain a Positive Control

Prefer demonstrations, official motion-planning examples, or existing policies for the same task and pinned ManiSkill version. Obtain at least one genuinely successful episode and label any privileged-state use. Do not implement a large IK or motion-planning system simply to prove the infrastructure works.

If suitable demonstrations are unavailable, an official state-based PPO baseline is optional. Treat it as an infrastructure control, not a WAM RL baseline or a substitute for the proposal's core comparison. Limit the first attempt to 2 GPU-hours, then save the checkpoint and measured success rate. Do not promise convergence within that budget or train indefinitely to obtain a desired result.

If no real successful episode is available, synthetic fixtures can still validate metric logic, but real positive-control acceptance remains `PENDING`.

### P4.2 Measure a Small Throughput Matrix

Measure state and RGB configurations separately at 1, 4, and 16 environments. Try 32 or 64 only when memory permits and measured throughput improves. For each configuration, warm up for 50 steps, measure 500 steps, and repeat three times. A real slow model may require a fixed-duration measurement instead; document the change.

Record transitions per second, episodes per hour, policy latency p50/p95, environment-step time, rendering time, serialization/write time, peak allocated/reserved GPU memory, total process GPU memory, host RSS, bytes written, and wall time. Define synchronization boundaries for asynchronous GPU timing. Report component timings and end-to-end time separately; overlapping durations cannot simply be added.

Measure normal collection without video recording separately from demonstration runs with recording. Vector steps per second multiplied by the number of environments is valid only when counting genuine transitions. Do not count resets or padding as collected data.

### P4.3 Estimate Later Costs

Use measured episodes per hour to estimate evaluation time. Account separately for model loading, resets, writes, and warmup. Use both GPUs only when independent processes measurably improve aggregate throughput; do not assume linear scaling.

Approximate raw single-camera RGB storage as:

```text
number_of_episodes * (actions_per_episode + 1) * height * width * 3 bytes
```

For 100 episodes, 200 actions each, and 128 x 128 RGB, this is approximately 0.99 GB in decimal units, excluding other data. At 256 x 256 it is approximately 3.95 GB. Multiply by camera count. Measure compression on a small sample before scaling. MP4 is suitable for demonstrations but should not be the sole ground-truth source for a pixel-alignment reward because of lossy encoding.

Acceptance: provide one measured recommended configuration, memory headroom, estimated cost per 100 episodes, and known bottlenecks. CUDA-kernel optimization is not the current objective.

## 8. P5: Integration Contracts That Can Be Prepared Independently

### P5.1 Model Contract

Deliver a versioned `docs/INTERFACE_CONTRACT.md`. Mark unconfirmed values `UNKNOWN` rather than guessing.

| Category | Required agreement |
|---|---|
| Inputs | Camera count/order, RGB/BGR, shape/range, history frames, proprioception, instructions, required goals |
| Actions | Dimension, per-component semantics, units, absolute versus delta, rotation convention, coordinate frame, normalization statistics, gripper direction, control frequency |
| Timing | Proposed chunk length K, executed horizon H, replanning behavior, physical timestamp of every predicted frame, inclusion of the current frame |
| Outputs | Actions, optional real predicted video/latents, checkpoint revision, episode reset behavior, batching support |
| Training | Online/offline collection, policy version, availability of meaningful probabilities/gradients, semantics of training extras |

Ownership of a DROID-to-ManiSkill conversion must be agreed with the model owner. The infrastructure owner can implement the adapter skeleton and checks; the model owner must confirm semantics. Matching tensor shapes is not sufficient evidence of correct robot control.

### P5.2 Pair Predictions with Actual States

Bind every prediction to:

```text
(episode_id, request_id, origin_step,
 action_sequence_hash, prediction_offset_time)
```

Ground-truth images must include camera identity, simulation time, control-step index, and preprocessing version.

If a request predicts K future steps but only H are executed, later actual states may be generated by newly planned actions. Do not score the old prediction beyond the executed prefix unless the method explicitly handles those changed conditions. Record validity and reasons for invalidation after resets, dropped frames, interpolation, clipping, or frequency mismatches.

Confirm that predicted and actual views correspond. For latent-only predictions, record the encoder/version and representation semantics; do not claim pixel-level alignment. Missing predictions must be unavailable, not blank images masquerading as model predictions.

### P5.3 Reward Hook and Training Handoff

Expose `RewardHook.compute(batch)` for completed transitions or segments. Return auxiliary rewards, validity masks, diagnostics, and a reward version. Preserve original environment rewards and predictions; reward composition belongs in training configuration.

A zero auxiliary-reward hook and clearly labeled synthetic hooks are allowed for pipeline tests. MSE, feature distances, normalization, and potential-based reward design remain with the reward owner. An offline export or inference-only HTTP endpoint is not a differentiable computation graph when training requires gradients through the world model.

The model may not provide conventional action log probabilities. Do not apply PPO indiscriminately or fabricate old log probabilities. The training owner must define the algorithm, sampling unit, on-policy/off-policy requirements, and update procedure.

### P5.4 Experiment-Management Handoff

Provide a single-run command, configuration schema, exit codes, run manifests, and machine-readable summaries. The experiment owner can use these to organize reward-weight and trainable-component ablations. Do not build a competing complex scheduler.

Deliver a small real state trajectory, a real RGB trajectory if available, an episode-summary example, and a synthetic model-interface fixture. Label their provenance and intended use separately.

## 9. P6: Integration Requiring Team Components or Additional Hardware

Advance in this order, expanding only after acceptance:

1. Run real-WAM inference on one offline observation and validate action semantics and prediction timing.
2. Execute one closed-loop episode on a compatible task, then expand to ten. If success is zero, inspect interfaces and domain mismatch before attributing failure to RL.
3. Evaluate that checkpoint with the common protocol and save provenance and failure cases.
4. Integrate a task-only RL checkpoint and then an auxiliary-reward checkpoint; verify split integrity.
5. Run the frozen final comparison and hand results to the experiment owner.

If the WAM cannot be used with ManiSkill, using the existing RoboLab-plus-RTX route requires suitable allocated resources and a team decision. Retain reusable schemas, evaluation logic, and collection components rather than rebuilding everything. Record model or task changes that alter the research question in `deviations.md`; do not silently change the proposal's scope.

## 10. Suggested Repository Layout

This is a target layout. Initially, only this plan and the README are delivered. Create components as they are implemented rather than populating many empty directories.

```text
README.md
pyproject.toml
docs/
  EXECUTION_PLAN.md
  STATUS.md
  INTERFACE_CONTRACT.md
  EVALUATION_PROTOCOL.md
  RUNBOOK_H800.md
  deviations.md
configs/
  envs/
  policies/
  eval/
  benchmarks/
manifests/
  validation.jsonl
  test.jsonl
src/robots_project/
  envs/
  policies/
  rollout/
  data/
  evaluation/
  cli/
tests/
  test_episode_boundaries.py
  test_action_chunks.py
  test_trajectory_integrity.py
  test_evaluation_metrics.py
  test_prediction_alignment.py
scripts/
outputs/                 # gitignored
data/                    # gitignored
checkpoints/             # gitignored
```

Allow local paths to be configured through project-specific environment variables or CLI arguments. Do not repurpose system `HOME` or `CODEX_HOME` as temporary project variables. Record upstream source commits rather than copying entire upstream repositories into this project by default.

### 10.1 Target CLI Acceptance Interface

These commands are implementation targets, not existing functionality. If different entry points are chosen, update the runbook with commands that actually work.

```bash
python -m robots_project.cli.preflight --output outputs/preflight.json
python -m robots_project.cli.smoke --config configs/envs/pushcube_state.yaml
python -m robots_project.cli.collect --config configs/envs/pushcube_rgb.yaml --policy random --episodes 20
python -m robots_project.cli.validate_trajectories --run-dir outputs/RUN_ID
python -m robots_project.cli.evaluate --config configs/eval/dev.yaml --policy random
python -m robots_project.cli.summarize --run-dir outputs/RUN_ID
python -m robots_project.cli.benchmark --config configs/benchmarks/h800.yaml
```

Provide `--help`, configuration validation, and explicit nonzero failure exits. Do not swallow exceptions to make a command appear successful.

## 11. Acceptance Checklists

### 11.1 Independent Infrastructure Completion

- [ ] P0: Actual-machine preflight and reproducible dependency lock; state and RGB capabilities reported separately.
- [ ] P1-P2: Two real tasks execute, trajectory and episode-boundary checks pass, and recovery avoids duplication.
- [ ] P3: Frozen evaluation protocol, episode tables, statistics, and auditable errors/incomplete episodes.
- [ ] P4: Measured throughput/memory/storage budget and a real positive control, or an explicitly incomplete positive-control status.
- [ ] P5: Integration contracts, sample data, runbook, and unresolved dependencies are ready for handoff.

If only the state path is validated, report exactly that and keep visual acceptance pending. Do not mark complete visual infrastructure as accepted.

### 11.2 Evidence Still Required for the Team Project

- [ ] Real WAM and simulator observation/action/timing compatibility.
- [ ] Actual RL updates and usable checkpoints, beyond a rollout or inference server.
- [ ] Controlled task-only versus auxiliary-reward comparisons and necessary ablations.
- [ ] Real training-stability, sample-efficiency, and task-performance results, including negative results where observed.
- [ ] Final report, presentation, and other deliverables required by the instructor's rubric.

## 12. Common Blockers and Stopping Rules

| Blocker | Work that can continue | Invalid substitute |
|---|---|---|
| RoboLab requires RTX; only H800 is available | ManiSkill, schemas, evaluation, RTX integration notes | Treating CUDA availability as RoboLab compatibility |
| Vulkan/RGB fails | State path, correctness tests, logging/statistics, concrete diagnostics | Black or random images presented as visual trajectories |
| Real model is not delivered | Mock contracts, random/replay policies, boundary tests | Mock outputs presented as WAM results |
| No real successful episode exists | Action/replay diagnostics and synthetic metric tests | Changing the task's success function to improve reported results |
| Log probabilities or training gradients are unavailable | Preserve raw data and document the missing training interface | Fabricating zero log probabilities and claiming PPO support |

If the same environment issue consumes more than two hours without new evidence, record it under blockers and switch to independent work. Proceed with authorized low-risk implementation without repeatedly asking for confirmation. Before replacing system drivers, using an unspecified machine, or materially changing the research scope, describe the concrete issue and a feasible alternative.

## 13. Execution Prompt for Server-Side Codex

Paste the following into a coding session opened in the repository:

> Read README.md and docs/EXECUTION_PLAN.md, inspect the current repository and applicable AGENTS.md files, and implement the independent simulation, rollout-infrastructure, and evaluation workstream. Execute P0-P5 using acceptance evidence to advance. Deliver working implementation rather than additional advice or placeholder code.
>
> First inspect the actual H800 hardware, drivers, isolated Python environment, and camera rendering. On H800-only hardware, do not assume that RoboLab's RTX requirement is satisfied; use the ManiSkill route. Validate state and RGB separately. Begin with one GPU, small tasks, and few environments before measuring throughput. Use an initial aggregate limit of 12 GPU-hours and a two-hour limit for any validation training attempt. Do not interrupt unrelated GPU processes.
>
> Implement policy-independent interfaces, action-chunk execution, correct terminal-observation and automatic-reset handling, recoverable trajectory storage, and common evaluation. Test episode boundaries, partial resets, H<K prediction alignment, aggregation denominators, and interrupted-write recovery. Validate with real simulator evidence. Synthetic or mock results are only interface tests and must never be reported as WAM or RL experiment results.
>
> Do not silently take ownership of reward research or the RL trainer. Record uncertain action semantics, prediction fields, and log-probability support in INTERFACE_CONTRACT.md. Complete work that does not depend on those unresolved items. Continue independent implementation when downloads or real-model components are unavailable.
>
> Record commands, exit codes, result paths, versions, and acceptance status in docs/STATUS.md after each phase. Produce RUNBOOK_H800.md, EVALUATION_PROTOCOL.md, INTERFACE_CONTRACT.md, and deviations.md. Report passed, partially passed, unrun, and blocked items with next dependencies. Make small commits by phase, preserve existing changes, and do not upload weights, raw trajectories, large videos, or secrets. This prompt does not independently authorize remote writes beyond the current session's existing authorization.
>
> Keep all authored project documentation, comments, status reports, and handoff text in English. Use role-based labels and neutral filenames; do not include personal names.

## 14. Sources and Evidence Boundaries

The workstream split comes from the supplied one-page `RL WAM.pdf` proposal. The original PDF is not uploaded to this public repository. The references below support technical choices; they are not the course grading specification. Record the actual upstream versions used during execution.

- [S1] [ManiSkill installation and Vulkan](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [S2] [ManiSkill quickstart](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/getting_started/quickstart.md)
- [S3] [ManiSkill RL setup and evaluation](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/reinforcement_learning/setup.md)
- [S4] [RoboLab hardware requirements](https://github.com/NVlabs/RoboLab#requirements)
- [S5] [Cosmos policy server and RoboLab client](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md)
- [S6] [Cosmos3-Edge-Policy-DROID model card](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)
- [S7] [Official V-JEPA 2 and V-JEPA 2-AC repository](https://github.com/facebookresearch/vjepa2)

Task counts, development estimates, resource limits, directory layouts, interface fields, and acceptance thresholds are engineering recommendations in this plan, not published benchmark results or completed measurements. Training-duration estimates, full task coverage, and generalization claims in the proposal still require actual verification.
