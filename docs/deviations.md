# Execution deviations

| Decision | Evidence | Effect on scope |
|---|---|---|
| Use ManiSkill raster rendering on the allocated H800 | Actual state and sensor-RGB tests passed | Main route retained; RoboLab remains BLOCKED_HARDWARE |
| Use SAPIEN's bundled Vulkan loader | Runtime warning and real frame captures | No system driver/Vulkan change required |
| Install pinned dependencies from the uv cache after registry stalls | `uv sync --offline --extra sim --group dev` exited 0 | Existing environments preserved; declared mirror only for development tools |
| Start with native Panda joint delta control | Controller source verified against upstream v3.0.1 | Debugging only; DROID conversion remains unconfirmed |
| Use explicit resets for frozen evaluation | Pinned automatic-reset source reviewed and normalization tested | Terminal observations remain native and raw termination flags are retained |
| Use official CPU-physics motion planning for the positive control | Official same-version solver, separately recorded configuration | Privileged infrastructure control; not a WAM baseline and not H800 simulation throughput evidence |
| Isolate official controls in a second locked environment | MPLib crashed during initialization in the main NumPy 2 environment; NumPy 1.26.4/OpenCV 4.11 control environment passed initialization and both tasks | Failed attempts remain recorded; main environment is preserved; exact native crash cause is unconfirmed |
| Measure complete common-horizon episodes for throughput | Fifty warmup steps and five hundred measured vector steps per repeat | Avoids reset/padding transition counts; training and slow-model throughput may differ |
| Journal episode attempts before reset | Final audit found interrupted partial episodes missing from attempt counts; real RGB recovery was rerun after correction | Recovery error rate includes four interruptions in twenty-four attempts; retries are declared before execution |

No model replacement, RL-method change or additional machine allocation was made. Model, reward and training integration still requires the responsible team roles.
