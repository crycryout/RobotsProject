# RobotsProject

Robotics course final project: RL Post-Training for World-Action Models.

This repository currently provides the execution plan for the **simulation, rollout infrastructure, and evaluation** workstream. It does not yet contain a completed implementation or H800 experiment results.

## Start Here

Read the [detailed execution plan](docs/EXECUTION_PLAN.md). It contains phases P0-P6, interface contracts, trajectory schemas, the evaluation protocol, resource budgets, acceptance checks, and a complete server-side Codex prompt.

Environment setup, policy-independent collection, trajectory validation, common evaluation, and handoff specifications can be developed independently. Final acceptance of the real WAM, auxiliary rewards, and RL updates requires team integration.

## Execute on the H800 Server

1. Clone this repository, or preserve local changes and update an existing checkout to the current `main` branch.
2. Start Codex in the repository directory.
3. Give it the following instruction.

```text
Read README.md and docs/EXECUTION_PLAN.md. Follow the execution prompt in Section 13 to implement P0-P5. First validate the actual H800 hardware and rendering path, then implement the independent simulation, rollout collection, trajectory validation, evaluation, and integration contracts. Run acceptance checks for each phase and record commands and evidence in docs/STATUS.md. Deliver working code, not just another plan. Do not report mock results as WAM experiments or silently take ownership of reward design and the RL trainer. When hardware or model dependencies block a component, document the blocker and continue independent work. Keep all authored project content in English and use role-based labels without personal names.
```

## Constraints to Check First

- **RoboLab officially requires an RTX GPU.** Two H800 GPUs do not establish compatibility with its full simulation/rendering stack. Begin with the ManiSkill route and actual-machine validation.
- **Validate state and RGB separately.** A working state environment does not establish a complete visual-WAM pipeline.
- **Action dimensions do not establish semantic compatibility.** Confirm coordinate frames, units, normalization, and control frequency before connecting DROID outputs to another controller.
- **The complete course rubric has not been supplied.** This plan follows the proposal's responsibility split; it does not establish completion of the whole team project.

Technical references and limitations are in Sections 1 and 14 of the plan. The original proposal PDF, model weights, and experiment datasets are not uploaded to this public repository.

## Documentation Convention

Use English for authored documentation, comments, status reports, and handoff text. Use neutral filenames and role-based descriptions without personal names.
