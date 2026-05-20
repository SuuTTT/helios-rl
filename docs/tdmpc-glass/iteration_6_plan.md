# Iteration 6 — Pivot: stuck-seed problem is the real bottleneck

## Goals (two distinct success criteria)

| ID | Goal | Current state | Strategy |
|---|---|---|---|
| **G1 — Consistency** | All 5 seeds > MPPI 500 (mean > 500, std small) | 2-of-5 (s3=523, s8=525) | Fix the stuck-seed lottery (§7.B soft-reward bundle, §7.C gait penalty) |
| **G2 — Ceiling** | At least one seed > MPPI 600 | Phase-t s2 = **612** (knee + Glass) | Reward shaping. §6.2 Phase-q (knee + no Glass) measures ceiling sans Glass. §7.E stack combines all winners |

Both goals share the same root cause (basin lottery) but have different
prescriptions: G1 needs *every* seed to escape bad basins; G2 just needs
*one* seed to push the upper limit. The §7 experiments target both.

## §0. Complete phase ledger (iter 1 → iter 5)

All HopperHop best-MPPI values we've recorded, sorted by best result per phase:

| Iter | Phase | Feature | Best | Seeds (#>500 / #stuck < 100) |
|---|---|---|---|---|
| 4 | **t** (Path 5) | **knee penalty** (reward shaping) | **612** s2 | **2 of 3 > 500** (612, 534, 374); 0 stuck |
| 3 | **o** (Path 3) | Glass OFF after 2M (hybrid) | 577 s3 | 1 of 3 > 500 (577, 254, 33); 1 stuck |
| 3 | **f** (Path 2) | latent_action_smooth_coef=0.001 | 571 s1 | 1 of 5 > 500 (572, 284, 262, 266, 255); 0 stuck |
| 1b | baseline (random) | TD-MPC-Glass default | 562 s5 | 2 of 5 > 500 (562, 526, 526, 294, 227); 0 stuck |
| 4 | **p** (Path 1) | EXPL_UNTIL=500k | 538 s4 | 1 of 3 > 500 (538, 197, 27); 1 stuck |
| 5 | **x** (Path 9) | NS=2048 MPPI | 523 s3 | 2 of 6+ > 500 (523, 501, 453 archived, 287, 234, 15, 6); 2 stuck |
| 3 | j (Path 2 curriculum) | curriculum smoothing | 518 s2 | 1 of 5 > 500 (518, 452, 354, 322, 266) |
| 3 | h (combined) | smooth + ccoef | 490 s1 | 0 of 2 > 500 (490, 328) |
| 3 | g | consistency_coef=1.0 | 482 s2 | 0 of 2 > 500 (482, 427) |
| 5 | y (Path 10) | hierarchical Glass K_sup=4 | 462 s3 | 0 of 3 > 500 (462, 211, 185) |
| 1c | act_noise anneal | 0.30→0.10 | 412 s3 | 0 of 4; FALSIFIED (hurts winners) |
| 3 | k | smooth + λ_temporal=0.05 | 292 s1 | 0 of 1; too aggressive |
| 3 | l_v1 | TD-MPC2 + smoothing (Glass OFF) | 289 s1 | 0 of 1; Glass-OFF hurts |
| 5 | v (Path 7) | cluster soft-dist as pi/q obs | 232 s3 | 0 of 3 > 500 (232, 218, 19.9); 1 stuck |
| 3 | i (Path 2 weak) | smooth=1e-4 | 312 s1 | 0 of 1; too weak |
| 3 | m | basin perturbation | 286 s5 | 0 of 2; doesn't help |
| 5 | P / Pa | cluster entropy intrinsic reward | 91 / 25 | FALSIFIED — non-stationary |
| 3 | d_v1 | act_noise=0.40 | 114 | Warp crash at 1M |
| 3 | d_v2 | H=5 only | 198 | Worse than baseline |
| 3 | n_v1 | basin perturbation alt | 56 | FALSIFIED |

## §0.1 What WORKS (interventions that produced ≥1 seed > 500)

In order of best per-phase result:

| Intervention | Best | Verdict |
|---|---|---|
| **Knee-penalty reward shaping (Phase-t)** | **612** | Most consistent — 2 of 3 > 500. Benchmark-unfair. |
| Glass OFF after 2M (Phase-o) | 577 | Glass helps early then becomes load |
| Latent action smoothing 1e-3 (Phase-f, j) | 572, 518 | Stable across iters |
| EXPL_UNTIL=500k (Phase-p) | 538 | Bigger random phase = wider state coverage |
| Random baseline 1b | 526, 562 | The seed lottery does work — sometimes |
| NS=2048 MPPI (Phase-x) | 523 (s3), 501 (s8) | Planner helps when policy is good |
| Consistency_coef=1.0 (Phase-g) | 482 | Helps but caps below 500 |
| Hierarchical Glass (Phase-y) | 462 | Close — Path 10 viable extension |

## §0.2 What's FALSIFIED (don't retry)

- **Path P / Pa** — cluster entropy as intrinsic reward: non-stationary, kills convergence
- **Path 7 (Phase-v)** — cluster soft-dist as observation: doesn't escape basin
- **Phase 1c** — act_noise anneal 0.30→0.10: hurts winners
- **Phase-d v1** — noise=0.40: Warp CUDA 901 at ~1M
- **Phase-e** — Q-reset: implementation bug, masked the test
- **Phase-i** — smooth=1e-4: too weak
- **Phase-k** — λ_temporal=0.05: too aggressive
- **Phase-l** — Glass-OFF entirely (not "after 2M"): hurts
- **Phase-m, n** — basin perturbation via param noise: doesn't help

## §0.3 Statistical pattern across 23 phases

**Out of all unique algorithm interventions tried:**
- ~5 produced winners (≥1 seed > 500): smoothing, EXPL_UNTIL=500k, NS=2048, Glass-OFF-late, knee-penalty
- ~9 produced no winners (peak < 500)
- ~5 were falsified outright

**No intervention produces 5-of-5 winners** — even knee penalty (best mean we've seen) only hit 2-of-3 > 500.

**Stuck-seed pattern is identical across all interventions**: in any multi-seed
sweep, 1-of-5 to 2-of-5 seeds plateau at MPPI 0–100. Confirmed across:
- Phase 1b s4 = 227 (close to stuck), s3 = 294
- Phase-f s2-s5 mid-range (262-284)
- Phase-p s5 = 27 (truly stuck), s3 = 197
- Phase-o s5 = 33, s4 = 254
- Phase-v s2 = 19.9 (stuck)
- Phase-x s2 = 5.8, s4 = 15 (stuck)
- Phase-y s2 = 211 (mid)

This is the **basin lottery problem** described in §2.1.


Goal unchanged: 5 HopperHop seeds > MPPI 500. Iteration 5 tried 4 paths (P, 7, 9, 10),
got 1 clear winner (Phase-x s3 = 523), but failed to consistently reach 500. This
document pivots based on what we learned.

## §1. Iteration 5 results (complete log)

| Path | Run | Seed | Best MPPI | Verdict |
|---|---|---|---|---|
| P | Phase-P static | 1 | 91 (collapsed) | FALSIFIED — non-stationary intrinsic |
| P-anneal | Phase-Pa | 1 | 24.9 (collapsed) | FALSIFIED — decay didn't help |
| 7 (cluster-obs) | Phase-v | 1 | 218 | mid, oscillation |
| 7 (cluster-obs) | Phase-v | 2 | 19.9 | **stuck** |
| 7 (cluster-obs) | Phase-v | 3 | 232 | mid |
| 9 (NS=2048) | Phase-x | 1 v1 | 453 (OOM) | partial |
| 9 (NS=2048) | Phase-x | 2 | 5.8 | **stuck** |
| 9 (NS=2048) | Phase-x | 3 | **523.5** ✅ | **WINNER** |
| 9 (NS=2048) | Phase-x | 4 | ~15 | **stuck** |
| 9 (NS=2048) | Phase-x | 6 | 287.3 | mid |
| 9 (NS=2048) | Phase-x | 8 | 488 (still running) | close |
| 9 (NS=2048) | Phase-x | 9 | 234.3 | mid |
| 10 (hier Glass) | Phase-y | 2 | 211.1 | mid |
| 10 (hier Glass) | Phase-y | 3 | 461.8 | close |
| Path 9 NS=1024 | Phase-x s5 | 5 | (in progress) | side test |

**Pattern across all paths**: 1-of-N seeds wins big, 1-of-N gets stuck near 0,
the rest land mid-range (200-300). HIGH VARIANCE is the consistent problem.

## §2. Why is 5-of-5 > 500 so hard? (root-cause analysis)

### §2.1 The gait-basin lottery

HopperHop has at least two stable gait basins (video-confirmed iter 4):
1. **Foot-hop** → can reach 500+
2. **Knee-walk** → caps around 200-300

Which basin a policy lands in is determined by **random initial conditions**
during the EXPL_UNTIL=500k random-action phase + early policy gradient updates.
Once in a basin, the policy converges and can't escape via standard exploration.

**CORRECTION (per user)**: an earlier draft of this doc claimed
"foot-hop=K=4-7, knee-walk=K=3". That mapping is NOT supported by the data.
Phase-p winner s4 (=538) was K=3 cluster pattern; several K=4 seeds got stuck.
The cluster-count is not predictive of basin. Basin identity needs to be
verified per-run via video inspection or geom-trajectory analysis, not by
counting active clusters in Glass.

### §2.2 What we tried (none rescued stuck seeds)

| Intervention | Why it should help | Result |
|---|---|---|
| Larger EXPL_UNTIL (25k→500k) | More state coverage in random phase | helped winners but not stuck seeds |
| Latent action smoothing | Force coherent motion | helped winners (Phase-f, Phase-j) but not stuck |
| Cluster intrinsic reward (P) | Reward gait diversity | non-stationary, destabilizes everyone |
| Cluster as observation (Path 7) | Policy knows which gait it's in | mid-range only, doesn't escape basin |
| Hierarchical Glass (Path 10) | Coarser abstraction layer | mid-range only |
| Bigger MPPI (Path 9 NS=2048) | Planner finds better actions | helps winners surge faster but stuck seeds stay stuck |

### §2.3 The key insight (iter 5 §5.3 restated)

> Stuck seeds are EXPLORATION-bound, not architecture-bound. You can't escape a
> basin by changing pi/q architecture or planner samples — you need either (a)
> exploration that generates trajectories FROM A DIFFERENT BASIN, or (b) a way
> to **transplant a known-good policy/critic seed** so the agent starts in the
> winning basin.

This points squarely at **Path 4 (behaviour cloning from a winner)** as the
necessary intervention. We've been delaying it; iteration 6 makes it the top
priority.

## §3. Iteration 6 plan — what's running and what's next

### §3.1 Phase-q sweep — DONE (12 seeds), G2 ceiling test

Knee penalty alone, vanilla TD-MPC2 (no Glass), NS=2048, EXPL_UNTIL=500k, smoothing curriculum.

| Seed | Best MPPI | Notes |
|---|---|---|
| s1 | 269 | stuck-ish |
| s2 | 303 | stuck-ish |
| **s3** | **557** ✓ | G1 |
| s5 | 343 | mid (slow box, ended 10M) |
| s6 | 328 | mid |
| **s7** | **510** ✓ | G1 |
| **s8** | **529** ✓ | G1 |
| s9 | 286 | still 7.75M, slow |
| s10 | 242 | mid |
| s11 | 347 | mid |
| **s12** | **553** ✓ | G1 |

**Result: 4/12 G1 winners (33%), 0/12 G2 (max 557). Knee penalty alone caps near 560 — does not replicate Phase-t s2=612 (which had Glass+knee).** Sweep over-allocated (planned 5, ran 12); marginal value of seeds 6-12 was 2 extra G1 winners but no G2 break. No more phaseq seeds will be queued.

### §3.1b Phase-r1 (soft-reward bundle v1) — running, 5 seeds

v1 = stand_bonus 0.1 (clip((h - 0.4)/0.2, 0, 1)) + linear weight anneal 1.0 → 0.0 over [0, 3M] env steps. Skipped from v2-spec: speed_curriculum, last-200-step early bonus.

| Seed | Best | Last | Step | Notes |
|---|---|---|---|---|
| s1 | 276 | 276 | 4.75M | climbing |
| **s2** | **553** ✓ | 548 | 5.5M | G1 winner, killed @ 5.5M (SIGKILL/OOM) |
| s3 | 398 | 398 | 7.75M | climbing, near G1 |
| s4 | 200 | 90 | 5.25M | dropped, may recover |
| s5 | 157 | 0 | 3.0M | early |

**Preliminary: 1/5 G1 hits, s3 close, others still climbing. Soft-reward bundle is promising but no G2 (max 553).**

### §3.1c Phase-r2 (gait penalty bundle) — running, 4 seeds

Full §7.C: fall_penalty -0.1 when height < 0.45 m + action_smooth -0.005·mean((Δa)²).

| Seed | Best | Last | Step | Notes |
|---|---|---|---|---|
| s1 | 7 | 7 | 1.5M | stuck-seed pattern (action_smooth too aggressive?) |
| s2 | 464 | 402 | 10M | close to G1, DONE |
| s3 | 390 | 314 | 6.5M | climbing |
| s4 | 279 | 259 | 10M | mid, DONE |

**Preliminary: 0/4 G1 hits, max 464. Gait penalty looks slightly worse than knee/soft so far — action_smooth may be too restrictive. Worth letting in-flight finish before concluding.**

### §3.1d Cross-bundle G1 hit-rate summary

| Bundle | G1 hits / total | Max MPPI | Notes |
|---|---|---|---|
| **Phase-q knee** | **4 / 12 = 33 %** | 557 | most data |
| **Phase-r1 soft** | 1 / 5 = 20 % (partial) | 553 | 3 still climbing |
| **Phase-r2 gait** | 0 / 4 = 0 % (partial) | 464 | 2 still climbing |
| (reference) Phase-x NS=2048 | 2 / 5 = 40 % | 525 | iter 5 baseline |

**G2 broke**: 0 seeds across all bundles (only Phase-t s2=612 in iter 5 with Glass+knee combined hit it). The single-knob recipes don't reach Phase-t's surge.

### §3.1e Phase-r-stack — COLLAPSED, ABLATIONS RUNNING

Launched 5 seeds (s1 local, s2 ssh1, s3+s4 ssh17637, s5 ssh6_3080) with Glass + NS=2048 + EXPL_UNTIL=500k + smoothing curriculum + soft_stand_bonus + gait_fall_penalty + gait_action_smooth. **All 5 collapsed to best MPPI 5–16** (vs single-bundle bests 510–557). s1 hit early-stop at 6.5M with best=5.7.

Stacking is destructively combining — the three shaping components together break learning where each alone was OK or even winning. Killed remaining 4 in-flight runs at 15:15Z.

Two ablations launched 15:16Z to identify the killer:

| Variant | Drops | Seeds | Hypothesis |
|---|---|---|---|
| **nosmooth** | `gait_action_smooth` | local s1, ssh6_3080 s2 | action_smooth penalising the bursty hopping actions is the killer |
| **nosoft** | `soft_stand_bonus` | ssh1 s1, ssh3_3070 s2 | soft+gait both pulling on height (one rewards, one penalises) creates a double-bind |

First eval expected ~15 min; clear recovery (>100 by 1.5M) vs continued flatline by 3M will indicate which knob to drop. If both flatline, Glass+shaping interaction itself is broken (Phase-v style) and r-stack with current Glass needs a different recipe.

### §3.2 Stop / drop

- **Path 4 (BC)** — DEFERRED per user (2026-05-18). Only env-interaction for now.
- **Path 9 more seeds** — we have enough variance data (s3=523, s8≈525, s4=15, s6=287, s9=234, s7 finishing).
- **Path 7 (Phase-v)** — falsified, no new launches.
- **Path 10 more seeds** — Phase-y s3=462 close-but-not-500 is enough info.
- **Path A (dist-Q), Path B (SAC entropy), Path 8 (multi-task)** — don't address basin lottery.

### §3.3 Stuck-seed soft-reset (deferred, secondary)

If reward-shaping (§7) doesn't fix the 1-in-5 stuck-seed pattern, fall back to:
detect plateau at 3M (best MPPI < 100), load checkpoint from 1M, perturb pi
params with Gaussian noise, restart training. Same dynamics/encoder kept.

## §4. Workflow redesign for 4 GPU fleet

Iteration 5 surfaced flakiness, OOM kills, watcher bugs, manual interventions.
Iteration 6 codifies:

### §4.1 Box specialization

| Box | Role | Suited for |
|---|---|---|
| Local 4070 Ti (12GB) | **Hot dev + reference** | Run main experiment, baseline reruns, smoke tests |
| ssh3 3060Ti (8GB) | **Long-burn reliable** | Single seed full 10M run, sequential queue |
| ssh6 4060 (8GB, driver 580) | **Stable parallel** | NS=2048 runs, second seed in CI sweep |
| ssh17637 2× 3060 Lap (6GB ea., flaky) | **Best-effort / disposable** | Side experiments only, accept CSV loss |
| ~~ssh9 3090~~ | **BLOCKED** | Skip until driver upgrade |

### §4.2 Launcher hygiene (mandatory for iter 6)

Fix the foot-guns we hit in iter 5:

1. **`tee -a` not `tee`** in launchers — so the log doesn't get truncated on restart.
2. **CSV backup BEFORE every relaunch** — already added to watcher v2.
3. **Per-seed launcher** — don't use queue scripts that loop SEEDS="1 2 3" because the
   watcher's relaunch will re-run from seed 1, overwriting prior seeds.
4. **Sleeper waits for SPECIFIC process** — not "any tdmpc-glass process" (avoids racing the watcher).
5. **Watcher slot lifecycle** — when a job completes naturally (early-stop, status=0),
   REMOVE the slot from the watcher (don't relaunch a finished run).

### §4.3 New training script behaviour requests

- **Resume from latest checkpoint** on restart, NOT fresh from seed 0. The current
  behaviour (overwriting CSV + starting fresh) made us lose ~5 trajectories in iter 5.
- **Append to CSV** on resume, not overwrite. Easy: open in 'a' mode after checking
  if file exists with matching seed.

### §4.4 Streaming + dashboard improvements

- Single stream script with all boxes, 10-min cadence (current setup good).
- Dashboard updated to show "(dead)", "(running)", "(early-stop)" tags per seed.
- Snapshots auto-archived to `exp/tdmpc_glass/archive/<phase>/seed_<N>_v<n>.csv` for any restart.

## §6. Reference experiments (running now)

Two missing baselines that tell us **what the algorithm is adding** and **what's
physically achievable** on this env.

### §6.1 Phase-z — vanilla TD-MPC2 baseline (NO Glass), 5 seeds

Same training config as Phase-x (NS=2048, EXPL=500k, smooth=1e-3), but `--algos tdmpc2`
(strips the Glass head). Tells us if Glass is helping, neutral, or hurting.

Local 4070 Ti, sequential s1→s5, ~25h. **Launched 2026-05-18 08:19Z.**

### §6.2 Phase-q — knee penalty + vanilla TD-MPC2 ceiling, 5 seeds

Same config as Phase-z + `--knee_penalty_coef 0.1`. Measures practical ceiling
(benchmark-unfair). Phase-t s2 was 612 with knee + Glass; Phase-q strips Glass.

2x3060 GPU0+GPU1 (s1+s2 parallel), then sequential. **Launched 2026-05-18 07:17Z.**

### §6.3 Interpretation matrix

| Phase-z mean | Phase-q mean | Reading |
|---|---|---|
| both <400 | both <400 | Algorithm is the bottleneck (need redesign) |
| z~265, q>500 | reward signal is the missing ingredient | basin lottery is real, shaping cracks it — pursue §7 |
| z>500, q>600 | Glass is HURTING us | drop Glass entirely |
| z~265, q~265 | even shaping isn't enough | exploration bottleneck — revisit Path 4 BC |

## §7. Env-only roadmap (condensed)

**Principle**: focus on robustness (5/5 winners), not peak. **Eval always on original
reward** — only training reward is shaped. We collapse the 7-priority brainstorm
into 3 concrete experiments + a stacking experiment.

### §7.A — DONE: diagnostic logging (was §7.1)

Per-eval `full_reward_rate / standing_rate / fall_count / time_to_first_full` logged
to `seed_N_diag.csv`. Reward-signal-only (env-agnostic, zero training cost).
Implemented + smoke-tested 2026-05-18. **All new runs auto-capture this.**

After Phase-q s1+s2 + Phase-z s1+s2 evals, we'll have diag data on stuck vs
winner seeds to inform which §7 lever to pull next.

### §7.B Soft-reward bundle (combines old §7.2 + §7.3 + §7.4 + §7.7)

One Phase-r experiment containing:
1. **Soft standing tolerance**: `stand_soft = tolerance(height, 0.6→2.0, margin=0.4, long_tail)` — smooth instead of binary cutoff at 0.6 m.
2. **Speed curriculum**: target_speed ramps 0.5 → 2.0 m/s over training.
3. **Shaping anneal** (linear): 0–20% steps `r_train = 0.5·orig + 0.25·stand_soft + 0.25·speed_soft`; 20–60% mix; 60–100% pure original. **Eval always original.**
4. **Early bonus** (last 200 steps of training only): `r_train += 0.05·speed_soft` when timestep < 200 per episode — helps break 700 ceiling.

All four are reward-signal modifications, no buffer changes — relatively cheap
to implement as a single `compute_shaped_reward(env_step, total_steps, height, speed, base_reward)` function in `run_benchmark.py`. **CLI flag**: `--soft_reward_curriculum` (single switch turns all 4 on).

### §7.C Gait stabilization penalties (combines old §7.6)

Add to training reward only:
```python
fall_penalty = -0.1 if height < 0.45 else 0
action_smooth = -0.005 * mean((a_t - a_{t-1}) ** 2)
```
We already have `latent_action_smooth_coef=0.001` on the latent action signal.
This is the *env-action-space* version which is different. Single CLI flag
`--gait_penalty` to enable both.

### §7.D Good-state replay (old §7.5)

DEFER. This requires modifying the replay buffer and reset distribution — bigger
implementation lift than the reward-shaping bundles. Try §7.B and §7.C first.
If both fail, this becomes the next priority.

### §7.E Stacked best-of-best (combines all winners)

After §7.B/§7.C results land, run **Phase-r-stack**:
`Glass + NS=2048 + EXPL_UNTIL=500k + curriculum smoothing + soft-reward bundle + gait penalty`.

Stacks everything that has won at least once in iter 1-5. Best shot at 5/5 > 500
if reward shaping is the missing piece.

### §7.F Experiment ordering

| # | Experiment | When |
|---|---|---|
| 1 | Phase-q + Phase-z (running) | Wait for results (~10-25h) |
| 2 | Phase-r1 = §7.B soft-reward bundle, 3 seeds | Once local frees |
| 3 | Phase-r2 = §7.C gait penalty, 3 seeds | Parallel on free remote |
| 4 | Phase-r-stack = §7.E, 5 seeds | After r1+r2 — the headline experiment |
| 5 | §7.D good-state replay | Only if r1+r2+stack don't hit 5/5 > 500 |

## §9. Hyperparam audit vs official TD-MPC2

Triggered by user question 2026-05-18 "why all runs seem low?". Most settings
match the official paper; one possibly under-tuned:

| Param | Ours | Official TD-MPC2 | Status |
|---|---|---|---|
| latent_dim | 512 | 512 | ✓ |
| hidden | (512, 512) | (512, 512) | ✓ |
| num_bins (distributional Q) | 101 | 101 | ✓ |
| BS (batch size) | 256 | 256 | ✓ |
| lr | 3e-4 | 3e-4 | ✓ |
| gamma | 0.99 | 0.99 | ✓ |
| tau (EMA) | 0.01 | 0.01 | ✓ |
| H (MPPI horizon) | 3 | 3 | ✓ |
| NS (MPPI samples) | 512 default | 512 | ✓ (Phase-x overrides 2048) |
| consistency_coef | 2.0 | 2.0 | ✓ |
| **K_UPDATE** | **64** | ~1 grad step / env step (≈256 with N_ENVS=256) | **possibly under-training 4×** |
| EXPL_UNTIL | 25k default | 25k typical | ✓ (Path 1+ overrides 500k) |

**K_UPDATE caveat**: 64 gradient updates per batch of N_ENVS=256 env steps =
0.25 updates per env step. Official is closer to 1.0. We may be under-trained.

Risk of changing: invalidates all iter 1-5 baseline comparisons. Defer to a
focused Phase-r-Kup sweep (K_UPDATE ∈ {64, 128, 256}) after current Phase-z/q
results land. Not a current-iteration action.

### §9.1 Why runs at MPPI 200-300 aren't "low"

Mid-training runs at 200-300 look the same as Phase-x s3 did at 3-6M env-steps
before it surged to 523 between 6M-10M. The actually-stuck pattern is MPPI 0-50
indefinitely (Phase-x s4, Phase-v s2, Phase-x s7 currently). Mid-band runs are
**pre-surge**, not stuck.

## §10. Box reliability cost analysis (2x3060)

The flaky 2x3060 box (6 GB × 2, driver 580) caused measurable training loss.

### §10.1 Sps benchmarks (sps = env-steps / second per single seed)

| Box | sps | Source |
|---|---|---|
| Local 4070 Ti (12 GB) | ~540 | `ssh4070ti:exp/.../logs/phasez_baseline/HopperHop_seed_1.log` |
| 4060 (8 GB, driver 580) | ~540 | `ssh6.vast.ai:exp/.../logs/phasex_4060/HopperHop_seed_8.log` |
| 3060 Ti (8 GB) | **~118** | `ssh3.vast.ai:exp/.../logs/phasex_3060ti/HopperHop_seed_7.log` (es=4,848,640 sps=118) |
| 2× 3060 Laptop (6 GB ea., shared) | ~250 per slot | `78.83.187.54:exp/.../logs/phaseq_knee/HopperHop_seed_1.log` |

### §10.2 2x3060 incidents (last ~24h)

| Time (UTC) | Event | Cost |
|---|---|---|
| 2026-05-17 07:30 | Phase-x s2 v1 SIGSEGV @ 1.5M env steps | 5h GPU wasted, no usable result |
| 2026-05-17 08:24 | Phase-x s1 v1 OOM-killed @ 4.5M (peak 453) | trajectory truncated, archived as `seed_1_died_at_4.5M.csv` |
| 2026-05-17 11:39 | Phase-x s2 v2 OOM-killed @ 3.5M | another wasted slot |
| 2026-05-17 15:46 | Phase-x s1 v2 OOM-killed @ 5.25M (peak 380) | partial result |
| 2026-05-17 22:35 | SSH outage ~30 min | watcher relaunched s5 → backup CSV preserved (272.7 peak) |
| 2026-05-18 00:20 | SSH outage ~16 min | watcher relaunched s4 → **Phase-y s4 382.9 peak LOST** (v1 bug, fixed in v2) |
| 2026-05-18 02:08 | SSH outage ~30 min | watcher relaunched s5 again (245.5 peak archived) |
| 2026-05-18 04:00 | SSH outage; watcher misfire | overwrote Phase-y queue's seed_1 (185.7 peak LOST) |

### §10.3 Cost in numbers

- **Wasted GPU time**: ~15-20h across failed/OOM-killed seeds on 2x3060.
- **Lost trajectories**: 2 (Phase-y s4 v1 = 382.9; Phase-y s1 = 185.7).
- **Truncated trajectories**: 2 (Phase-x s1 v1 = 453; Phase-x s1 v2 = 380).
- **Manual interventions**: ~8 times across the day (kills, relaunches, backups, watcher reconfigs).

### §10.4 Counterfactual

If 2x3060 were as stable as the 4060 (which has been zero-incident), we would have:
- 2 more clean Phase-y seeds (Path 10 CI would have 4-5 seeds instead of 2)
- 2 more clean Phase-x seeds (Path 9 CI would have 7-8 instead of 5)
- ~15h less debug/intervention time
- Faster auto-queue progression — fewer cycles spent on relaunches

### §10.5 Recommendation

Continue using 2x3060 ONLY for **disposable side experiments** (NS=1024 ablation
test). Move **anything we care about** to the stable boxes (local, 4060, 3060Ti).
The autoqueue should never queue a primary G1/G2 experiment to 2x3060 without a
backup CSV strategy.

## §11. Top-line decisions (current as of 2026-05-18)

1. **Don't launch more Phase-x / Phase-v / Phase-y seeds** — variance characterised.
2. **Path 4 BC is DEFERRED** per user — try env-shaping first (§7).
3. **Wait for Phase-z + Phase-q results** before launching new experiments — the
   §6.3 interpretation matrix decides which §7 lever to pull.
4. **§7.A diagnostic logging is DONE & deployed** — every new run carries it. Use
   the `seed_N_diag.csv` to characterize stuck-vs-winner seeds.
5. **Next coding work** when local 4070 Ti frees: implement §7.B (soft-reward
   bundle) as a single `--soft_reward_curriculum` flag. Then §7.C gait penalty.
6. **Stretch experiment**: §7.E Phase-r-stack combines every winning ingredient
   ever (smoothing + EXPL=500k + NS=2048 + soft-reward + gait penalty). If this
   doesn't reach 5/5 > 500, the basin lottery is genuinely unsolvable without
   demonstration data (and we'd revisit Path 4 BC).
7. **Outstanding hygiene gap**: checkpoint-resume for `run_benchmark.py` is still
   missing. Every box-recycle event loses ~1-2M env-steps of training. Worth
   ~half a day of engineering when we have a calm window.
