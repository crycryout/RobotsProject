# results_v2 — dream–reality alignment arms

Date: 2026-11-25. Design in `train_v2.md`; v1 background in `train_v1.md`.

## Summary

- **All three arms** (task only T′, alignment reward to video tokens A-video, alignment added to action preferences A-action) improve their training pairs by a similar, modest amount on tracked layouts (≈ +0.1 to +0.2 success vs base, at the edge of significance).
- **None** changes performance on untrained tasks.
- **Head to head** on identical layouts and noise:
  - A-video ≈ T′: +0.010 [−0.049, +0.073].
  - A-action is slightly worse: −0.066 [−0.125, −0.007].
- **The alignment gain never moved,** in any arm or epoch (≈ +0.03 throughout), including three extra epochs of A-video.
- **Conclusion:** rewarding sharp-half dream–reality alignment, as implemented, didn't engage the mechanism and didn't improve the policy beyond task-only training.

## 1. Setup

| Item | Value |
| --- | --- |
| Policy | Cosmos3-Edge-Policy-DROID, rank-16 LoRA on both heads and all 28 GEN blocks, DiffusionNFT (β 0.1, reference weight 0.01), lr 1e-4, 1 pass, 16 chunks × all σ per step, DDP on 2 H800 |
| Reward | r = ΔP + S, G = r + 0.95·G′, p^a = ½ + ½·clip(G / (2 − P)) (`train_v1.md`) |
| Alignment | a = SSIM(dream f16, real f16) − SSIM(real f0, real f16): side-camera band, grayscale, 80×21; bound B = 0.2 |
| Arms | **T′:** p^v = ½. **A-video:** p^v = ½ + ½·clip(a/B). **A-action:** p^a ← clip(p^a + 0.1·clip(a/B), 0, 1). |
| Training pairs (8) | BowlInBin `default`; ButterAboveRaisin both levels; FruitsOnion both levels; MustardInLeftBin `default`; SmallerObjectButterInBin both levels |
| Schedule | From the base policy, 3 epochs × 8 batches × 12 environments per arm. Identical seeds across arms: `default` = screening seeds (tracked layouts), `specific` = fresh per epoch but the same for every arm. Randomization 0.1. |
| Base reference | `training/runs/screening_base_20261123` (base policy, same `default` layouts) |
| Wall time | A-video 6.3 h, A-action 6.4 h, T′ 5.2 h; A-video epochs 4–6: 6.5 h more |

## 2. Training pairs

### 2.1 Success over all 8 pairs (96 episodes per epoch)

| Arm | Epoch 1 | Epoch 2 | Epoch 3 | Mean | Mean final P |
| --- | ---: | ---: | ---: | ---: | ---: |
| T′ | 0.500 | 0.552 | 0.448 | 0.500 | 0.514 |
| A-video | 0.500 | 0.500 | 0.531 | 0.510 | 0.528 |
| A-action | 0.417 | 0.427 | 0.458 | 0.434 | 0.448 |

### 2.2 Head to head vs T′ (paired on the same layouts and noise seeds, Δsuccess)

| Arm | Epoch 1 | Epoch 2 | Epoch 3 | Pooled (n = 288) |
| --- | --- | --- | --- | --- |
| A-video − T′ | +0.000 [−0.115, +0.115] | −0.052 [−0.156, +0.052] | +0.083 [−0.021, +0.188] | +0.010 [−0.049, +0.073] |
| A-action − T′ | −0.083 [−0.198, +0.031] | −0.125 [−0.208, −0.042] | +0.010 [−0.083, +0.104] | −0.066 [−0.125, −0.007] |

Pooled intervals treat the epochs as independent, although the same layouts repeat, so they are somewhat too narrow.

### 2.3 Tracked `default` layouts vs base (paired, n = 60 per epoch, Δsuccess)

| Arm | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | --- | --- | --- |
| T′ | +0.100 [−0.067, +0.250] | +0.183 [+0.033, +0.333] | +0.100 [−0.050, +0.234] |
| A-video | +0.167 [+0.017, +0.317] | +0.083 [−0.067, +0.233] | +0.217 [+0.067, +0.367] |
| A-action | +0.083 [−0.067, +0.233] | +0.050 [−0.083, +0.183] | +0.100 [−0.050, +0.250] |

### 2.4 Per pair, epochs 1–3 pooled (successes out of 36)

| Pair | Base ×3 | T′ | A-video | A-action |
| --- | ---: | ---: | ---: | ---: |
| BowlInBin `default` | 21 | 29 | 27 | 31 |
| ButterAboveRaisin `default` | 12 | 18 | 20 | 13 |
| ButterAboveRaisin `specific` | 0* | 7 | 7 | 5 |
| FruitsOnion `default` | 18 | 18 | 21 | 20 |
| FruitsOnion `specific` | 9* | 16 | 18 | 14 |
| MustardInLeftBin `default` | 9 | 20 | 21 | 12 |
| SmallerObjectButter `default` | 18 | 16 | 17 | 16 |
| SmallerObjectButter `specific` | 15* | 20 | 16 | 14 |

\* The `specific` base comes from v1 round 0, on different layouts (unpaired).

A-action's deficit comes mainly from MustardInLeftBin `default` (20 → 12 vs T′) and ButterAboveRaisin `default` (18 → 13).

### 2.5 A-video with 3 more epochs (resumed from the epoch-3 checkpoint)

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Success, all 8 pairs | 0.500 | 0.500 | 0.531 | 0.479 | 0.490 | 0.469 |
| Tracked `default`, Δ vs base | +0.167 | +0.083 | +0.217 | +0.117 | +0.050 | +0.050 |

No further gain; the gain over base drifts back toward zero (within noise). There is no matched T′ for epochs 4–6.

## 3. Untrained tasks (epoch-3 checkpoints)

10 untrained tasks; 120 episodes per model, paired with the screening base (same layouts and noise).

| Model | Success | Δ vs base | Δ final P |
| --- | ---: | --- | --- |
| Base | 0.425 | — | — |
| T′ epoch 3 | 0.425 | +0.000 [−0.067, +0.067] | −0.004 |
| A-video epoch 3 | 0.400 | −0.025 [−0.083, +0.033] | −0.013 |
| A-action epoch 3 | 0.425 | +0.000 [−0.067, +0.067] | +0.013 |
| v1 `e02c2` (reference) | 0.417 | −0.008 [−0.067, +0.050] | −0.008 |

- Arm vs T′: A-video −0.025 [−0.083, +0.033]; A-action 0.000 [−0.058, +0.058].
- Per task, every model is within 0–2 episodes of base. Tasks the base already solves (BananaInBowl, BananaOnPlate, SauceBottlesCrate, RubiksCube) are retained.

## 4. Mechanism diagnostics

| Quantity | T′ | A-video | A-action |
| --- | --- | --- | --- |
| Alignment gain, epochs 1 / 2 / 3 | +0.029 / +0.031 / +0.031 | +0.029 / +0.033 / +0.030 (epochs 4–6: +0.029 / +0.029 / +0.032) | +0.030 / +0.028 / +0.032 |
| Own SSIM (dream f16 vs real f16) | ≈ 0.971 | ≈ 0.972 | ≈ 0.972 |
| Mean \|p^v − ½\| | 0 | 0.083 | 0 |
| Mean \|p^a − ½\| | 0.123 | 0.126 | 0.118 |
| Video distance from base, per collection | 0.044–0.086 | 0.058–0.094 | 0.048–0.066 |
| Chunks used in updates (3 epochs) | 1,637 | 4,043 | 4,191 |
| Alignment coverage / clip | 95–97% / 0 | 95–97% / 0 | 95–97% / 0 |

**Reading:**
- **A-video changed the video stream** somewhat more than T′, but **not toward reality**: alignment is flat and identical across arms.
- **The signal is small:** own SSIM is ≈ 0.97, the gain spans a few hundredths (std ≈ 0.04), and the video preference moved only ±0.08 on average.
- **A-action's term** shifts every chunk's action preference by an amount uncorrelated with task outcome. Most chunks are neutral under the task reward, so this mostly adds update noise, which fits its small deficit.

## 5. What this does and doesn't show

**Holds:**
- Sharp-half alignment is a valid, label-free *measurement*. On the 21 Nov pilot it identified a chunk's own dream with AUROC 0.98–0.99, and beat a same-state counterfactual dream in 83–92% of cases.
- Dreams carry their own action's consequences in frames 0–16.

**Does not hold at this scale:**
- Using that measurement as a DiffusionNFT preference (to video or to actions) neither raised alignment nor improved success over task-only training, on trained or untrained tasks.

**Limitations:**
- One seed per arm. 96 episodes per epoch on training pairs, 120 on untrained tasks, so effects below about ±0.07–0.10 can't be resolved.
- No matched T′ for epochs 4–6.
- The alignment bound (0.2) and A-action's weight (0.1) were set once from pilot statistics, not swept.
- The update itself is weak: v1 found raw G carries little per-state contrast, and the learning-rate probe found larger steps mostly add drift. Small reward differences between arms may therefore be undetectable regardless of the reward.

## 6. Possible next steps

1. **Offline signal-strength test** with the lr-probe approach (≈ 30 min): does training video tokens directly on alignment, with a larger bound or weight, raise dream–reality agreement on held-out chunks? If not, the alignment route is closed at this model scale.
2. **Address the action-learning bottleneck independently of dreams,** for example per-state contrast from same-layout replica groups.
3. If an arm is rerun: two seeds, and a matched T′ for every horizon.

## 7. Artifacts

| Path | Content |
| --- | --- |
| `training/runs/v2_T_01`, `v2_AV_01`, `v2_AA_01` | Runs: config, history, counters (chunks / episodes / steps / update samples), epoch checkpoints, `check.sh` |
| `training/runs/v2_*/eval_untrained_ep3/` | Untrained-task evaluations (`summary.json`; per-batch rows under `model1/`) |
| `training/runs/screening_base_20261123/` | Base reference (26 tasks, `default`; 6 tasks, `specific`) |
| `training/runs/v2_queue.log`, `v2_eval_queue.log`, `v2_av_continue.log` | Queue logs |
| `training/align.py`, `dream_eval/alignment/test_alignment.py` | Alignment metric and its offline validation |
