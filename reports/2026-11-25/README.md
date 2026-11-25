# Simulation, rollout and evaluation · 25 November 2026

[PowerPoint with five speaker notes](outputs/汇报.pptx) · [Projection PDF](outputs/汇报.pdf) · [Chinese talk and Q&A](outputs/讲稿与问答.md) · [One-page results](outputs/结果表.pdf) · [Complete download](outputs/汇报资料包.zip)

The presentation covers the RoboLab environment, the rollout pipeline, and evaluation of Cosmos3-Edge-Policy-DROID. The team originally ran the archived training and evaluation. This work checked the original episodes, reconstructed the metrics, and fixed validation in the evaluation pipeline. The [work log](outputs/本次工作总结.md) documents responsibilities and evidence.

| Policy | Training samples across three epochs | Fixed-policy successes on 120 screened-task starts |
|---|---:|---:|
| Frozen pretrained Base | — | 51/120 (42.5%) |
| Task-reward RL (T′) | 144/288 (50.0%) | 51/120 (42.5%) |
| A-video | 147/288 (51.0%) | 48/120 (40.0%) |
| A-action | 125/288 (43.4%) | 51/120 (42.5%) |

Each trained policy used one training seed. Fixed-policy evaluation used only the default instruction for 10 screened train-pool tasks excluded from v2 updates. The official test split remains unevaluated. The three trained arms match on archived physical initial states. Base can be matched by protocol keys; its original physical-state trajectories were unavailable. Both paired intervals against T′ include zero. [Protocol](outputs/评测协议.pdf) · [Task results and paired intervals](outputs/结果表.md) · [CSV](outputs/data/report_table.csv)

The [training chart](outputs/figures/training_env_steps.png) uses actual cumulative environment control steps. The two T′ videos replay sampled sensor frames, and the two pilot videos contain original continuous footage. The 480 evaluation records are archival. The separate single-episode reproduction scripts are included; a newly completed model-driven episode was not present in the published snapshot. [Run times](outputs/data/training_runtime.csv) · [Case videos and selection](outputs/cases/案例说明.md)

## Verify the published counts

Run from this directory with Python, NumPy and PyArrow installed.

```bash
python -B outputs/reproduce/verify_bundle.py
(cd outputs && sha256sum --quiet -c evidence/SHA256SUMS)
```

The source-record extracts in the package are sufficient for this check. For the full H800 audit or the evaluation regression tests, use a checkout of the source revision below, copy this report's `outputs/` into that checkout, and apply `outputs/evidence/evaluation.patch` once to the original code. The three files under `training/` here show the patched versions. The reproduction guide lists the original H800 paths and the commands for each test. [Source revision](https://github.com/CYQFWang/CosmosEdgeRL/tree/b82302e83f8d999e8cc2031a619b5ea6255dc192) · [Patch](outputs/evidence/evaluation.patch) · [Reproduction guide](outputs/复现.md)
