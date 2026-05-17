# TD-MPC-Glass HopperHop — Live Dashboard

Goal: 5 seeds > 500 MPPI on HopperHop. Beat Phase-1b baseline finals
`[438, 526, 294, 187, 562]` (mean 401, 3-of-5 > 500).

Refresh with: `bash scripts/iter5_dashboard.sh`

## Hardware fleet

| Box | GPU | VRAM | Driver | sps | Stability | Notes |
|---|---|---|---|---|---|---|
| Local | RTX 4070 Ti | 12GB | 12.4 | ~540 | high | Fastest dev box. Mem 0.85 |
| ssh3:11271 | RTX 3060 Ti | 8GB | 12.x | ~100 | high | Slow but reliable. Mem 0.55 |
| ssh6:11115 | RTX 4060 | 8GB | 580 / CUDA 13 | ~540 | high | Mem 0.55. Long-running Phase-p s6 from earlier sesssion |
| 78.83.187.54:17637 | 2× RTX 3060 Lap | 6GB ea. | 580 / CUDA 13 | ~250 ea. | **flaky** | OOM-killed Phase-v s2 + Phase-x s2 (twice). Use mem 0.35 |
| ssh9:16233 | RTX 3090 | **24GB** | 535 | — | **BLOCKED** | Driver 535 / CUDA 12.2 incompatible with our JAX+mujoco_warp stack (jax 0.6 needs cuSPARSE 12.6, jax 0.4 missing `jax.tree.map_with_path` used by mujoco_warp). Release or upgrade driver. |

## Phase legend (Iteration 5)

| Phase | Path | What it changes | Hypothesis |
|---|---|---|---|
| **Phase-v** | 7 | Concat soft cluster S[n*(z)] (K=8) to z before pi/q. Architectural. | Policy can condition on which gait phase it's in. |
| **Phase-x** | 9 | NS=512→2048 MPPI samples. **Planner-only.** | Stuck seeds are search-failure, not learning-failure. |
| **Phase-y** | 10 | Hierarchical Glass: K_sub=8 + K_super=4 joint 2D-SE losses. | K=3 basin cap → need a coarser layer. |
| **Phase-P/Pa** | dead | Cluster entropy as intrinsic reward (static + decayed). | Non-stationary reward signal killed policy at convergence. |

## Live state (manually updated)

| Phase | Seed | Box | Best MPPI | At step | Status |
|---|---|---|---|---|---|
| Phase-v | 1 | local 4070 Ti | **218.0** | 7.5M | done (10M cap) |
| Phase-v | 2 | local 4070 Ti | 19.9 | 6.5M | **KILLED** — stuck seed, 4h49m wasted |
| Phase-v | 3 | ssh6 4060 | 232.0 | 5.75M | running |
| Phase-x | 1 v1 | 2x3060 GPU1 | **453.2** | 4.25M | **OOM-killed** (archived: `seed_1_died_at_4.5M.csv`) |
| Phase-x | 1 v2 | 2x3060 GPU1 | 278.0 | 2M | running, climbing |
| Phase-x | 2 v1 | 2x3060 GPU0 | 1.7 | 750k | OOM-killed (status=139) |
| Phase-x | 2 v2 | 2x3060 GPU0 | 5.8 | 2.75M | OOM-killed again (status=137) |
| Phase-x | 3 | local 4070 Ti | — | — | just launched |
| Phase-x | 4 | ssh9 3090 | — | — | TBD |
| Phase-y | 1 | ssh3 3060Ti | **185.7** | 1.75M | done (early-stop @ 3.25M, patience=1.5M) |
| Phase-y | 2 | ssh3 3060Ti | TBD | — | running (auto-queued after s1) |
| Phase-p s6 (legacy) | 6 | ssh6 4060 | 250.9 | 6.25M | older baseline, still climbing |

## Reference trajectory: Phase-p winner s4 → 538

```
250k=0.2  500k=7.8  750k=0.0  1.0M=3.7  1.25M=52.7  ← surge starts
1.5M=114  1.75M=113  2M=2.4 (crash)  2.25M=159  2.75M=230
3M=163  3.5M=243  3.75M=281  4M=271  4.25M=278  4.5M=344
5M=312  5.25M=24.1 (crash)  5.5M=347  6M=374  6.25M=373
6.5M=407  7M=384  7.5M=412  8M=426  8.5M=497  9M=422
9.5M=500  9.75M=501  10M=538
```

**Pattern**: surge → crash → recover higher → repeat. Multiple deep crashes
along the way (2M=2.4 trough, 5.25M=24 trough). Don't kill a run just because
one eval crashed — check 3M-window early-stop.

## Iteration-5 lessons so far

- §5.1: Phase-v s1 hit 91 then crashed — looked dead, called it too early.
- §5.2: It oscillates 91 → 1.6 → 94 → 42 → 117 → 145 → 185 → 209 → 218 (final). Crash≠death.
- §5.3: Phase-v s2 (stuck seed, peak 19.9 over 9.25M): **stuck seeds are exploration-bound,
  not architecture-bound.** Path 7/9/10 don't help bad basins. Path 4 (BC from winner)
  needed for the consistent-mean problem.

## Path falsifications (iteration 5)

- **Path P** (cluster entropy as intrinsic reward, static coef=0.1): single eval peak
  MPPI=91 at 1.25M, then collapsed to 2.4 at 2M, never recovered. Non-stationary reward.
- **Path Pa** (Path P + linear decay 0.1→0 over [500k, 3M]): peak 24.9 — **3.6× WORSE
  than static**. Decay doesn't fix it; coef=0.1 magnitude is inherently incompatible
  with HopperHop reward scale (max intrinsic per episode ≈ 210 vs target ~600).

## Pending paths

- Path 4 (Phase-s): behaviour cloning from a winner trajectory — **likely needed for stuck-seed rescue**
- Path 8 (Phase-w): multi-task (HopperHop + HopperStand)
- Path A: distributional Q (quantile regression) + n-step returns
- Path B: SAC-style entropy regularization with auto-tuned α
- Render 7 checkpoints with peak > 450 (blocked by Warp mempool issue on remote)
