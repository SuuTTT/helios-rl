# Iteration 6 — Pivot: stuck-seed problem is the real bottleneck

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

## §3. Iteration 6 plan

### §3.1 Stop iterating on Path 9 / 7 / 10

We have enough Phase-x data for the 95% CI plot (s3=523, s4=15, s6=287, s7+s8 finishing,
s9=234). Two more (s7, s8) will give us 5-6 seeds. Don't launch s10+. 
Path 7 (Phase-v) and Path 10 (Phase-y) are also sufficiently characterised.

### §3.2 Top priority: Path 4 (BC from winner — Phase-s)

Implementation:
1. **Collect demonstrations** from Phase-x s3 winner (peak 523).
   Run inference for ~5 episodes, save (obs, action) pairs. ~5k transitions.
2. **Pre-train pi via BC** for N updates on the demonstrations. Cross-entropy or MSE loss.
3. **Continue normal training** from BC-warmed pi.
4. **Smoke-test on 1 seed first**, then scale to all 5 to see if stuck-seed pattern disappears.

**Hypothesis to test**: BC pre-training puts pi in the foot-hop basin from
the start, breaking the random-init lottery.

### §3.3 Secondary: stuck-seed detection + soft reset

Implementation:
1. Detect "stuck" at e.g. 3M env-steps if best MPPI < 100.
2. On detection: load checkpoint from a small earlier window (e.g. 1M env-steps),
   add Gaussian noise to policy params, restart training.
3. Tests whether stuck seeds can be "kicked" into a different basin without
   throwing away the dynamics/encoder learning.

### §3.4 Skip / deprioritize

- Path A (distributional Q): bigger lift, doesn't address basin lottery.
- Path B (SAC entropy): same — doesn't address basin lottery.
- Path 8 (multi-task): adds complexity without addressing root cause.
- Path 9 more seeds: diminishing returns, we have variance data.

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

## §6. New experiments (per user, before Path 4)

Before implementing Path 4 (BC from winner), we need two missing reference points
to know **what the algorithm is actually adding** and **what's physically achievable**.

### §6.1 Q1: Vanilla TD-MPC2 baseline (NO Glass) — 5 seeds

We've been iterating on TD-MPC-Glass for many phases without a clean 5-seed
TD-MPC2 baseline using the same training config (EXPL_UNTIL=500k, NS=2048,
curriculum smoothing). If vanilla TD-MPC2 already gets 3-of-5 > 500, then
"Glass is the problem"; if it can't break 300 either, then "Glass is roughly
neutral and the basin issue is algorithm-agnostic".

Launcher: `scripts/run_phasez_baseline_local.sh` (NEW)
- `--algos tdmpc2` (no `-glass`)
- `--mppi_n_samples 2048`
- `--expl_until 500000`
- `--latent_action_smooth_coef 0.001`
- `--early_stop_patience 3000000`
- 5 seeds, 10M cap

### §6.2 Q2: Knee-penalty ceiling — 5 seeds

Phase-t seed 2 hit 612 (iter 4 §10.x). That was 1 seed, so we don't know if
it's reliable. Run 5 seeds with knee penalty to measure the practical
ceiling for HopperHop. Even though this is benchmark-unfair (modifies reward),
it tells us what the policy class is *physically capable of*.

Launcher: `scripts/run_phaseq_knee_5seed.sh` (NEW, no Glass)
- `--algos tdmpc2` (no `-glass`)
- `--knee_penalty_coef 0.1`
- `--knee_penalty_threshold 0.15`
- `--mppi_n_samples 2048`
- Otherwise same as §6.1

### §6.3 Interpretation matrix

| Baseline mean | Knee-penalty mean | What it means |
|---|---|---|
| both <400 | both <400 | Algorithm is the bottleneck — fundamental redesign needed |
| baseline ~265, knee >500 | reward shaping is the missing ingredient | basin lottery is real, knee penalty cracks it |
| baseline >500, knee >600 | Glass is HURTING us | drop Glass entirely |
| baseline ~265, knee ~265 | reward signal isn't enough either | exploration is the bottleneck — Path 4 BC needed |

After §6.1 and §6.2 results, we'll have evidence to either:
- Continue with Path 4 BC (if exploration is confirmed bottleneck)
- Drop Glass (if vanilla baseline matches or beats it)
- Accept reward shaping for the headline number (if knee penalty cracks 600 reliably)

## §5. Top-line decisions

1. **Stop launching more Phase-x NS=2048 seeds**. Let s7, s8 finish then declare Path 9 done.
2. **Implement Path 4 (Phase-s BC from winner)** next — top priority for stuck-seed rescue.
3. **Codify launcher + watcher hygiene** above before any new experiments.
4. **Don't drop Phase-x as "failed"** — it's our best benchmark-fair result (523 winner), just inconsistent.
5. **Open question**: is it worth the engineering effort to retrofit run_benchmark.py with checkpoint-resume? Would save many lost trajectories during box-recycle events.
