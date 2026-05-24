# Iteration 5 — TD-MPC-Glass HopperHop sweep (live)

Goal: get all 5 HopperHop seeds > 500 (ideally > 700), beat Phase-1b baseline
finals [438, 526, 294, 187, 562]. Iteration 4 culminated in three organic
winners (Phase-f s1, Phase-j s2, Phase-o s3) at MPPI~500–577 and a knee-penalty
winner Phase-t s2 = 612 (benchmark-unfair, parked). Iteration 5 explores
*benchmark-fair* paths in parallel across 4 GPUs.

  bash scripts/iter5_dashboard.sh — run anytime to see best MPPI + last eval across all 5 GPUs.
  
## Hardware (live)

| Box                | GPU              | Mem  | Currently running |
|--------------------|------------------|------|-------------------|
| Local              | RTX 4070 Ti      | 12GB | Phase-v seed 1 (Path 7) |
| ssh3:11271         | RTX 3060 Ti      | 8GB  | Phase-y seed 1 (Path 10) |
| 78.83.187.54:17637 | RTX 3060 Lap GPU0 | 6GB  | Phase-v seed 2 (Path 7) |
| 78.83.187.54:17637 | RTX 3060 Lap GPU1 | 6GB  | Phase-x seed 1 (Path 9) |
| ssh6:11115         | RTX 4060         | 8GB  | Phase-v seed 3 (Path 7) — launching |

## Phase legend (what each one tests)

| Phase | Path | What it changes | Hypothesis |
|-------|------|-----------------|------------|
| **Phase-v** | 7 | Concat soft cluster distribution S[n*(z)] (8-dim) to z before pi/q lookups. Architectural change; pi/q first layer is now (512+8)→512. Fully benchmark-fair. | Letting the policy see *which gait phase* it's in helps it commit to a coherent hop pattern rather than oscillating at gait transitions. |
| **Phase-x** | 9 | Larger MPPI sample count: NS=2048 vs default 512. **Planner-only** — training is unchanged from Phase-p baseline. | Tests whether stuck seeds are *search-failure* (planner doesn't find good action sequences) vs *learning-failure* (Q/dynamics models are wrong). If Phase-x rescues a stuck seed, the planner was the bottleneck. |
| **Phase-y** | 10 | Hierarchical Glass: K_sub=8 fine clusters AND K_super=4 coarse super-clusters trained jointly via two 2D-SE losses on the same prototype graph A. | Iteration 4 §7.4: K=3 basin cap suggests we need a coarser layer; super-clusters might capture the gross gait families (hop/stand/fall) while sub-clusters capture phase. |

## Background falsified (iteration 4 §11)

- **Phase-P** (cluster entropy as intrinsic reward, static coef=0.1): seed 1 peaked
  MPPI=91 @ 1.25M then collapsed to 2.4 @ 2M. Non-stationary reward signal.
- **Phase-Pa** (same with linear decay 0.1→0 over [500k, 3M]): seed 1 peaked
  24.9 @ 1.25M — **3.6× WORSE** than static. Decay didn't help; coef=0.1
  magnitude is intrinsically incompatible with HopperHop reward scale
  (max bonus per episode ≈ 210 vs target ~600).
- **Path P (cluster intrinsic reward) is dead** in both static and annealed forms.

## Standing convention

All Phase-v/x/y runs use:
- `--expl_until 500000` (Phase-p winner setting — random actions for first 500k env-steps)
- `--latent_action_smooth_coef 0.001` after curriculum warmup at 250k
- `--early_stop_patience 3000000` (halt if no new best MPPI for 3M env-steps)
- `--total_steps 10000000` (10M cap)
- 3 seeds per phase per box

## Live trajectory comparison (iteration's leading metric: MPPI eval)

The Phase-p WINNER seed 4 (final 538) tracked like this:
```
step=    250k  MPPI=  0.2
step=    500k  MPPI=  7.8
step=    750k  MPPI=  0.0
step=  1.00M  MPPI=  3.7
step=  1.25M  MPPI= 52.7   ← surge starts here
```
Use this as the "this run might be a winner" benchmark.

## Live results (updated in-session — see `exp/tdmpc_glass/HopperHop_<phase>/seed_<N>.csv`)

| Phase | Seed | Best MPPI so far | At env-step |
|-------|------|------------------|-------------|
| Phase-v | 1 | **91.1** | peak @ 2M, then collapsed (91 → 1.6 → 3.1 → 1.3 over 0.75M steps) |
| Phase-v | 2 | 0.0 | 250k (random phase) |
| Phase-v | 3 | — | launching |
| Phase-x | 1 | 0.0 | 250k (random phase) |
| Phase-y | 1 | 0.0 | <500k (random phase, see remote log) |

### §5.1 Worrying pattern: Path 7 mimics Path P failure

Phase-v seed 1 hit **exactly the same surge-then-collapse pattern** as Phase-P
(static intrinsic) and Phase-Pa (decayed intrinsic): single-eval peak at
MPPI≈91, then collapse to single digits over the next 0.75M env-steps with
no recovery. This is concerning because Path 7 (cluster as *observation* with
stop_gradient) shouldn't have the same non-stationary-reward issue as Path P
(cluster as *reward* signal).

**Working hypothesis**: glass params keep evolving via glass_loss after policy
starts converging. As they evolve, the soft cluster distribution S[n_star(z)]
that pi/q sees as input drifts. pi/q was optimized for the OLD distribution.
Convergence + drift = miscalibrated Q → policy collapse. The stop_gradient on
the cluster computation prevents pi/q gradients from corrupting Glass, but does
**not** prevent the drift of Glass's own params from corrupting pi/q's inputs.

If seeds 2 and 3 (on 2x3060 and 4060) reproduce this collapse, Path 7 is also
dead and we need to either: (a) freeze Glass params at glass_warmup boundary
when Path 7 is on, or (b) use a discrete one-hot cluster id (hard argmax) so
the input changes only when n_star flips, not continuously.

### §5.2 Update: Phase-v s1 is *not* dead — surge-crash is the normal pattern

A few hours later, Phase-v s1 came back from the trough:
```
2.00M MPPI=91  ← peak 1
2.25M MPPI=1.6  (crash)
3.50M MPPI=94   ← peak 2
3.75M MPPI=42
4.00M MPPI=117  ← peak 3
4.25M MPPI=145
4.75M MPPI=185
6.25M MPPI=209
7.50M MPPI=218  ← peak (final best)
```
Walking back §5.1: **Phase-v s1 oscillates with a rising upper envelope.** Path 7
is learning, just slowly. Final at 10M cap: best MPPI=218.

Crucially, Phase-x s1 (Path 9, no Glass-obs change) showed the *same* surge-crash
pattern (193 @ 2M → 11.1 @ 2.25M → 238 @ 2.75M → 329 @ 3M → 453 @ 4.25M). So
the surge-crash is **TD-MPC2's native climb pattern on HopperHop**, not caused by
Glass features. The Phase-p winner s4 had crashes at 2M=2.4 and 5.25M=24 too —
and still reached 538 by 10M.

**Working principle now**: a single crash does not falsify a run. Only verify
death after 3M env-steps with no new best (the early-stop threshold).

### §5.3 Lesson from Phase-v seed 2: stuck-seed is exploration-bound, not Glass-bound

Phase-v seed 2 ran for 4h49m / 9.25M env-steps and never crossed MPPI=20
(best=19.9 @ 6.5M). All 37 evals bouncing in 0–20 range, no surge. **Killed
early to free the local 4070 Ti for a better experiment.**

Why it failed:
- Random init dropped the policy in a degenerate gait basin (K=3 / "knee-walk"
  pattern from iteration 4 §10).
- The EXPL_UNTIL=500k random-action phase didn't visit enough diverse states
  for the encoder/dynamics to model alternative gaits.
- Glass cluster-obs (Path 7) didn't help, because *the cluster information is
  about which gait the policy IS using, not which it COULD use*. You can't
  escape a basin by being told you're in it — you need exploration that
  generates trajectories from a different basin.

**Implication**: Path 7 / Path 9 / Path 10 all fail the stuck-seed problem
because none of them change the action distribution during early exploration.
Stuck seeds need **exploration-side interventions** (e.g. Path 4 BC from a
winner, or a much longer EXPL_UNTIL with random+pi mixture). Architectural and
planner-side tweaks help winning seeds get faster but don't rescue losers.

This narrows our 5-seed-mean problem: even if Path 9 wins on 3-4 seeds, the
1-in-5 "stuck" pattern likely persists. To beat the iteration-4 finals
[438, 526, 294, 187, 562] consistently we likely need BOTH a winning architecture
(Path 9 candidate) AND a stuck-seed rescue (Path 4 candidate).

(Best MPPI is updated as evals stream in — check the monitor notifications.)
