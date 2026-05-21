# Iteration 8 — Helios plan to meet G1 and G2 (benchmark-fair)

Companion to `iteration_7_codex.md` (Codex's draft) and
`mppi_vs_pi_analysis.md`. Read those first for the K_UPDATE audit and the
pi-vs-MPPI evaluator mismatch.

Phase-eval re-score source:
- `phase_eval_rescore_2026-05-21.md`

Goal:
- **G1**: 5/5 HopperHop seeds > 500 by verified `best_any = max(best_pi, best_mppi)`, benchmark-fair (no reward shaping, no BC, no env edits, eval on original reward).
- **G2**: at least 1 seed > 600 by verified `best_any`, benchmark-fair.

## §0. Re-diagnosis (what 25 phases + Codex Phase-aa actually tell us)

Cross-phase G1/G2 tally as of 2026-05-21:

| Phase | n | G1 | G2 | Max | Notes |
|---|---|---|---|---|---|
| **Phase-t** (knee + Glass) | 4 | 2 | **1 (612)** | 612 | only G2 ever — unfair |
| Phase-1b_remote (early Glass) | 5 | 3 | 0 | 562 | best historical hit-rate |
| Phase-aa K=128 (Codex) | 5 | 1 | 0 | 539 | K_UPDATE audit; `best_any` does not change G1 count |
| Phase-aa K=256 (Codex) | 3 | 1 | 0 | 561 | same G1 hit-rate as K=128, higher `best_any` ceiling |
| Phase-q knee | 11 | 4 | 0 | 557 | shaping, unfair |
| Phase-o (Glass-off-late) | 3 | 1 | 0 | 578 | fair |
| Phase-r1 soft / Phase-r2 gait | 4+4 | 1+1 | 0 | 553 / 510 | shaping |
| Phase-r-stack | 5 | 0 | 0 | 16 | destructive interaction |
| Phase-z vanilla (10M) | 4 | 1 | 0 | 535 | fair; seeds 2 and 5 are `pi`-selected under `best_any` |

**Hard observations**:

1. **K=128 and K=256 do not change G1 hit-rate** (both 1/3 in smoke, K=128 was 1/5 in Phase-aa). Codex's K_UPDATE hypothesis explains *one* stuck-seed rescue (s1 went 234 → 538) but does not solve the seed-level distribution. **The basin-lottery survives the training-ratio fix.**
2. **No fair recipe has produced 5/5 G1** in 25 phases. Best fair was Phase-1b 3/5 (iter 1, before our regressions).
3. **G2 only fires under reward shaping** (Phase-t). The fair ceiling is ~560 across all converged winners.
4. **Per the rollout-video analysis** (suuttt.github.io blog §3): winners maintain *stable* clusters per gait phase; stuck seeds have clusters *oscillating within one gait phase*. Glass's structural-entropy loss minimises aggregate cut edges but does **not** enforce temporal stability within phases. So Glass learns rich descriptions of whatever gait is found — including the wrong one. *A good representation of the wrong gait is useless.*
5. **New Codex insight: MPPI is often worse than pi.** Across de-duplicated
HopperHop CSVs, MPPI < pi at the same eval step in **896 / 3133 = 28.6%** of
paired evals. Even when pi >= 400, MPPI is lower in **51 / 403 = 12.7%** of
cases. At run level, best pi beats best MPPI in **23 / 140 = 16.4%** of runs.
This means MPPI-only checkpointing can discard a genuinely better actor.
   A focused re-score of recent fair phases (`phaseaa`, `phasez`, `phasex`) is
   now recorded in `phase_eval_rescore_2026-05-21.md`; it changes checkpoint
   selection for several seeds, but does not materially change the 5/5 G1
   conclusion.

**Mechanism of basin lock** (per the blog):
- Three causes converge in the first ~200k steps: weight init, early exploration trajectory, MPPI noise.
- After ~200k env steps, Q has fit value estimates to whatever gait the policy produces; gradients push toward that gait's local optimum; loop self-reinforces.

## §1. Why every iter 1-6 intervention failed to fix this

| Lever | What it changed | Why basin-lock survives |
|---|---|---|
| Latent smoothing (Phase-f/j) | mid-game representation smoothness | doesn't touch basin entry |
| NS=2048 MPPI (Phase-x) | post-lock planning quality | better plans within the wrong gait |
| EXPL_UNTIL=500k (Phase-p) | random coverage before policy locks | helps sometimes (s4=538) but doesn't fix init/MPPI-noise causes |
| Knee penalty (Phase-t) | reward gradient → no-torso-contact → hop | works (G2 hit), unfair |
| Hierarchical Glass (Phase-y) | coarser partition | still describes whatever gait was learned |
| Cluster intrinsic (Path P/Pa) | exploration bonus | hand-off to extrinsic still locked basin |
| Reward stacks (r-stack) | shaping cocktail | destructive — pi froze, best=16 |
| K_UPDATE 64→128 (Codex aa) | training ratio fix | individual seeds train better, but lottery persists |
| MPPI-only selection | checkpoint/eval metric | can under-rank good deterministic actors when planner/model mismatch is high |

The pattern is consistent: **interventions improve post-basin-lock performance but do not change the basin-entry distribution**. Iteration 8 must change both basin-entry and measurement.

## §2. Four levers that change basin entry and measurement (proposed)

Four orthogonal mechanisms, in EV order. Each is benchmark-fair — pure algorithm
or measurement changes, eval reward unchanged.

### §2.0 Phase-eval — **Best-of-pi-or-MPPI checkpointing** (new, immediate)

**Idea**: stop treating MPPI as the sole evaluator/checkpoint selector. Save and
track:

```
best_pi.pkl
best_mppi.pkl
best_any.pkl = argmax(max(pi_reward, mppi_reward))
```

Dashboard and iteration reports should show `best_pi`, `best_mppi`, and
`best_any`. A seed counts as "candidate solved" if either pi or MPPI reaches
500, then gets video/original-reward rollout verification.

**Why this matters**: MPPI is a planner through the learned latent dynamics and
reward model. On HopperHop, small contact-timing errors can make MPPI search into
actions that are model-favoured but real-env-bad. The actor can sometimes be the
more reliable controller because it has already settled into a coherent gait.

**Implementation**:
- In `scripts/run_benchmark.py`, save `best_pi.pkl` when pi improves and
  `best_any.pkl` when either evaluator improves the run's best score.
- Add CSV/dashboard derived fields:
  - `best_pi`
  - `best_mppi`
  - `best_any`
  - `pi_minus_mppi_last`
  - warning if `pi - mppi >= 100`
- Render both best-pi and best-MPPI checkpoints when their scores differ by
  >=50.

**Decision impact**: this may not create new behavior, but it changes which
policies we preserve and inspect. It also prevents false negatives where a seed
looks below 500 by MPPI while pi is already near/over threshold.

**Risk**: pi can be noisy too. Mitigation: "candidate solved" requires a render
or extra rollout verification, not just one lucky pi row.

### §2.1 Phase-ar — **Auto-Restart on plateau detection** (highest-EV pragmatic)

**Idea**: monitor per-seed best-MPPI. If at 1.0M env steps no eval row >100, the seed is basin-locked. Reset policy + Q to fresh random init, **keep** encoder + dynamics + replay buffer + RNG offset. Continue training; restart-count capped at 3 per seed.

**Why this works**: each restart is a fresh basin-entry attempt with the encoder/dynamics already pre-warmed by the replay buffer. The policy gets a clean shot at the gait basin without re-learning physics. With per-attempt G1 ≈ 0.30, 3 attempts → 1 − 0.7³ = **66% per seed**; 5-seed sweep → 0.66⁵ × 5 ≈ 1.6 expected winners minimum, distribution shifts heavily toward 5/5.

**Implementation**:
- `--restart_on_plateau` CLI flag (default off)
- `--restart_check_at <N>` env steps (default 1_000_000)
- `--restart_threshold <V>` best-MPPI floor below which restart fires (default 100)
- `--restart_max_attempts <N>` (default 3)
- On restart: re-init `params['pi']` + `params['q']` with new RNG; preserve `params['enc']`, `params['dyn']`, replay buffer; clear best-MPPI tracker; append a row `step,reward,eval_type=restart` to the CSV for visibility.
- Total wall-time impact: ~30% worst-case if all 3 attempts trigger and converge separately.

**Risk**: encoder may have learned a "crawling-friendly" representation in the first attempt. Mitigation: also re-init encoder if attempt 2 still plateaus by 1.5M (escalating reset).

### §2.2 Phase-mpc-lite — **MPPI-gated planner distillation** (algorithmic upgrade)

**Idea**: add planner distillation only when MPPI is not clearly worse than pi:

```
L_mpc = mask * lambda_mpc * ||pi(z) - mppi_planned_action(z)||^2
mask = 0 if recent_eval(pi - mppi) >= 100 else 1
```

MPPI can discover actions the policy gradient misses, but the new analysis shows
it is also worse than pi in a large minority of evals. So distillation must be
gated: imitate MPPI only when the planner and actor are reasonably aligned.
Anneal `lambda_mpc` from 1.0 -> 0.0 over 3M env steps so pi becomes independent
late.

**Why benchmark-fair**: zero demonstrations, no env modification, eval unchanged. The planner is the model talking to itself. (TD-MPC2 doesn't currently do this — pi loss is exp(advantage)·log_pi, not MPPI imitation.)

**Implementation**: small loss addition inside `make_update_fn`; new flags
`--mpc_distill_coef`, `--mpc_distill_anneal_steps`, and
`--mpc_distill_disable_gap`.

**Risk**: if pi tracks MPPI too tightly during the early random-exploration phase
or during model/planner mismatch, it may inherit bad actions. Mitigation: only
apply after `EXPL_UNTIL`, and disable when `pi - mppi` exceeds the gap.

### §2.3 Phase-g2 — **Glass V2: temporal stability loss** (the blog's smoking gun)

**Idea**: per the blog, winning seeds have stable cluster assignments within a gait phase; stuck seeds oscillate within one phase. Add Glass loss term:

```
L_temp = (1 - cos_sim(S[n*(z_t)], S[n*(z_{t-1})]))
```

Penalises cluster oscillation between consecutive steps. Tests whether enforcing temporal stability of clusters → stable gait → better basins.

**Why this might work**: the structural-entropy loss minimises cut edges in the aggregate transition graph but does not penalise within-phase flicker. The temporal-stability loss bridges that gap.

**Implementation**: 5 lines added to `tdmpc_glass.py`'s Glass loss function. New flag `--glass_temp_stability_coef` (default 0; try 0.05).

**Risk**: too high → over-coarse partition collapsing to 1 cluster. Mitigation: balance against the existing lambda_balance hinge.

**Run note, 2026-05-21 18:40 UTC**:
- Current local Phase-g2 seed 1 is a direct/manual local run on this machine via
  `scripts/run_phaseg2_temp_stability.sh`, not a newly launched central-queue
  task. The queue entry `tc5930bf` records the earlier failed local attempt, so
  the dashboard queue state can look stale while the local process is still
  live.
- Local command lineage: `bash scripts/run_phaseg2_temp_stability.sh` →
  `scripts/run_benchmark.py --algos tdmpc-glass --tasks HopperHop --seed 1
  --k_update 128 --mppi_n_samples 2048 --expl_until 500000
  --glass_lambda_temp_stability 0.05`.
- Current local seed 1 status: running past 8.2M env steps; best_any so far is
  144.5 at 8.0M, so this seed appears stuck.
- Remote seed 2 is the meaningful positive Phase-g2 signal so far: running on
  `ssh6_4060`, best_any 570.6 at 5.5M.
- Seeds 3-5 marked `done` locally are not valid completed runs; they failed
  during JIT after disk pressure (`ptxas fatal: Internal error: writing file`)
  and should be rerun through the queue after the current slots clear.

### §2.4 What we explicitly DON'T propose

- More K_UPDATE seeds — Codex Phase-aa data already shows no hit-rate change.
- MPPI-only G1/G2 accounting — contradicted by `mppi_vs_pi_analysis.md`.
- Wider exploration (EXPL_UNTIL > 500k) — Phase-p tested this; helped one seed (s4=538) but did not change the 5-seed distribution. Worth re-running only if §2.1-2.3 all fail.
- Action prior (sine wave during exploration) — too close to "knowledge injection", borderline-fair.
- More reward shaping or stacks — falsified in iter 6.
- BC from a winner — DEFERRED per user from iter 5/6.
- Larger Glass (K_super, more prototypes) — falsified.
- H=5 MPPI horizon — falsified iter 2.

## §3. Experiment ladder

Each phase = 5 seeds. Run on the 5 stable fast boxes: local 4070 Ti, ssh6 4060, ssh1 2080 Ti, ssh3 3070, ssh6 3080. Use ssh3 3060 Ti as parallel slot for 6th seed if needed. Skip 2x3060 for headline runs (the 5.5M SIGKILL pattern eats winners).

| # | Phase | What | Why |
|---|---|---|---|
| 0 | **Phase-eval** | Add best-pi/best-any checkpointing and dashboard/reporting. Re-score recent Phase-aa/ab/z/x runs by `best_any`. | Immediate measurement fix. Avoids discarding actors better than MPPI. |
| 1 | **Phase-ar** | TD-MPC2 + NS=2048 + EXPL_UNTIL=500k + K=128 + **auto-restart** (3 attempts, threshold 100 @ 1M). 5 seeds. | Highest-EV behavioral lever: directly attacks basin lottery without reward shaping. Estimated 4-5 / 5 G1 candidates by best-any. |
| 2 | **Phase-mpc-lite** | Do **not** imitate MPPI blindly. Add planner-consistency only when `mppi >= pi - margin`; skip when MPPI is clearly worse. | Uses the new insight: MPPI is useful when aligned, harmful when model/planner mismatch is large. |
| 3 | **Phase-g2** | TD-MPC-Glass + NS=2048 + K=128 + **temporal-stability** (coef=0.05). 5 seeds. | Tests the blog's hypothesis about Glass V2. Salvages the Glass research direction. |
| 4 | **Phase-ar-stack** | Phase-ar + MPPI-gated distillation (only if §1 doesn't hit 5/5 alone). 5 seeds. | Conservative additive stack of the two best fair levers. |

**Decision rules**:
- After Phase-ar 5-seed results: if 5/5 G1 → done with G1. If 3-4 / 5 G1 → run Phase-ar-stack. If ≤ 2 / 5 → auto-restart insufficient, prioritise Phase-mpc-lite.
- Phase-mpc-lite and Phase-g2 can run in parallel with Phase-ar (different boxes).
- G2 is downstream — once 5/5 G1 is achievable, push the winning recipe to 10 seeds and check tail (one of them should hit 600+).

## §4. Implementation plan (code work before any launches)

Four code changes in `src/helios/algorithms/tdmpc2.py` + `tdmpc_glass.py` +
`scripts/run_benchmark.py`:

0. **Best-pi/best-any checkpointing** (Phase-eval): ~30 lines in `train_tdmpc2`
   - Track `best_pi`, `best_mppi`, and `best_any` separately.
   - Save `best_pi.pkl`, `best_mppi.pkl`, and `best_any.pkl`.
   - Store both evaluator scores and the selector (`pi` or `mppi`) in checkpoint payloads.
   - Update dashboard/scripts to display `best_pi`, `best_mppi`, `best_any`, and `pi_minus_mppi`.

1. **Auto-restart** (Phase-ar): ~50 lines in `train_tdmpc2`
   - Track `_last_improvement_step` per seed.
   - At each eval, if `env_steps >= restart_check_at` and `best < restart_threshold`, set `_restart_pending=True`.
   - On next batch step, re-init `params['pi']`, `params['q']`, `params['target_q']` (= params['q']) with a fresh PRNGKey; keep `params['enc']`, `params['dyn']`, `params['rew']`; reset opt state for pi+q (but keep enc+dyn opt state); clear best-MPPI; increment `_restart_count`.
   - Log `restart` rows to CSV.

2. **MPPI-gated MPC-distill** (Phase-mpc-lite): ~25 lines in `make_update_fn`
   - Add `mpc_action_target` arg only for states where MPPI is not clearly worse than pi.
   - Add loss term `mpc_distill_coef * mask * mean((pi(z) - mpc_action_target)**2)`.
   - Gate the mask using eval-time evidence first: if current `pi - mppi >= 100`, disable distill until the next eval.
   - Anneal coef linearly from 1.0 → 0.0 between `expl_until` and `expl_until + 3M`.
   - This replaces the earlier unconditional MPC-distill proposal, because MPPI is empirically worse than pi in a large minority of cases.

3. **Glass V2 temporal stability** (Phase-g2): ~10 lines in `tdmpc_glass.py`
   - In the existing Glass loss, after computing `S[n_star(z)]`, also compute `S[n_star(z_{t-1}}]`.
   - Add `temp_coef * (1 - cos_sim(...)).mean()` to total Glass loss.

Each change ships with a smoke test (30k steps locally, verify no crash + reward not NaN) before any 10M run.

## §5. Boxes idle right now (2026-05-21)

- **local 4070 Ti**: idle (Phase-rstack-nosmooth s1 reported best=0 — likely never escaped JIT or hit early-stop trivially; check log before launching new).
- **ssh3 3070**: idle.
- **ssh17637 (both GPUs)**: vast.ai box unreachable — skip for now.

**Recommended first action when you approve this plan**:
1. Implement §4.0 (best-pi/best-any checkpointing + dashboard surfacing) — low-risk, immediate.
2. Re-score Phase-aa/ab/z/x by `best_any`; update G1/G2 accounting.
3. Implement §4.1 (auto-restart) — ~1 h dev + smoke test.
4. Launch Phase-ar on the 5 reachable fast boxes (5 seeds in parallel). Expected runtime ~6 h on the slowest (3060 Ti).
5. Implement §4.2 and §4.3 in parallel while Phase-ar runs.

**Do not launch Phase-ar yet** until §4.0 and §4.1 are implemented and smoke-tested.

## §6. What success looks like

- 5/5 G1 from one phase = G1 met (research goal), where G1 is counted by
  verified `best_any >= 500`, not MPPI-only.
- ≥1/5 G2 from a fair phase = G2 met (a benchmark-fair break of 600), counted by
  verified `best_any >= 600`.
- If Phase-ar hits 5/5 G1 alone → strongest possible result: basin-lottery defeats by reset alone, simpler than every algorithmic intervention tried.
- If only Phase-ar-stack hits 5/5 → MPC-distill matters; document the joint recipe.
- If nothing in §3 hits 5/5 → the problem is harder than basin-entry; pivot to investigating whether `Hopper-Hop` itself has a representation pathology (the encoder might genuinely cannot represent the foot-strike state crisply).
