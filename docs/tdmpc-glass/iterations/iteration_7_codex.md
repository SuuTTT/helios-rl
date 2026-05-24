# Iteration 7 Codex - benchmark-fair path to 5/5 HopperHop > 500

Date: 2026-05-19.

Goal: get standard HopperHop seeds 1-5 to best MPPI >= 500 in one phase while
remaining benchmark-fair: no reward shaping, no behaviour cloning, no
environment edits, and eval on original reward.

## Iteration summary

| Iteration | Main result | Verdict |
|---|---|---|
| 1 | First TD-MPC-Glass beat early v24 checkpoints; fixed prototype graph reached 336 MPPI at 500k and 412 at 1M. | Glass is viable and cheap, but early wins did not imply 5-seed robustness. |
| 2 | Tested smoothing, consistency, H=5, Q-reset, and noise changes. One smoothing seed reached 571; noise anneal and H=5 failed. | Latent smoothing helps lucky seeds but does not rescue stuck ones. |
| 3 | Curriculum smoothing, basin/JIT perturbation fixes, proto temperature, Glass-off ablation. Winners stayed rare. | K=3/K=4 cluster count is not the root cause; gait basin is. |
| 4 | EXPL_UNTIL=500k, Glass-off-late, knee penalty, hierarchy design. Phase-p seed 4 slow-burned to 538; Phase-t reached 612 with reward shaping. | Wider exploration helps, reward shaping proves physical ceiling, but shaping is not benchmark-fair. |
| 5 | Path P cluster intrinsic, Path 7 cluster observation, Path 9 NS=2048, Path 10 hierarchical Glass. NS=2048 produced winners; cluster intrinsic and cluster observation did not rescue stuck seeds. | Planner scale helps good seeds; architecture alone does not escape bad basins. |
| 6 | Vanilla TD-MPC2 baselines, knee/soft/gait reward bundles, stack ablations, dashboard hardening. Reward bundles raised hit rate but stack collapsed. Vanilla TD-MPC2 + NS=2048 is competitive with Glass. | The next benchmark-fair lever is training ratio, not more reward shaping. |

## Current evidence

- The stuck-seed pattern persists across Glass, vanilla TD-MPC2, NS=2048,
  hierarchy, cluster observation, and exploration-only runs.
- Reward shaping can improve hit rate, but it is excluded for this iteration.
- Behaviour cloning from a winner is also excluded for this iteration.
- The strongest unresolved fair audit is `K_UPDATE`: active runs use 64 gradient
  updates per 256-env collection batch, about 0.25 updates per env step. Official
  TD-MPC2 is closer to 1 update per env step, so `K_UPDATE=128/256` is the
  highest-EV fair change.
- Latest dashboard snapshot showed vanilla Phase-z already had one seed above
  500 and two seeds near the threshold, so compare Glass against a strengthened
  vanilla baseline before adding more Glass complexity.

## Results to date (2026-05-20)

### Phase-aa — K_UPDATE smoke sweep (vanilla TD-MPC2, NS=2048)

3 seeds per K value. One seed per K was run as a check before expanding.

| K_UPDATE | Seed | Box | Best MPPI | @ step | Status |
|---|---|---|---|---|---|
| 64 | 1 | ssh17637 | 233.7 | 3.25M | stopped (stuck) |
| 128 | 1 | ssh6_4060 | **538.6** | 8.25M | **WINNER** |
| 128 | 2 | ssh17637 | 284.9 | 4.25M | running |
| 128 | 3 | ssh6_3080 | 307.1 | 9.5M | done |
| 128 | 4 | ssh17637 | 209.2 | 4.25M | running |
| 128 | 5 | ssh17637 | 331.2 | 6.75M | running |
| 256 | 1 | ssh1_2080ti | **561.1** | 6.5M | **WINNER** |
| 256 | 1 | ssh3_3060ti | **531.0** | 7.75M | **WINNER** (parallel run) |
| 256 | 2 | local | 331.4 | 10M | done |
| 256 | 3 | ssh6_3080 | 233.8 | 8.75M | done |

**K=128 3-seed summary**: mean = (538.6 + 284.9 + 307.1) / 3 = **376.9**; G1 (>500) = **1/3**

**K=256 3-seed summary**: mean = (561.1 + 331.4 + 233.8) / 3 = **375.4**; G1 (>500) = **1/3**

**Verdict**: K=128 and K=256 have identical G1 winner rates in the 3-seed smoke. K=128 is cheaper
(halved gradient FLOPs per env step) so Phase-ab uses K=128.

K=64 (current default) looks clearly worse — one seed stuck at 233 confirms the training ratio
hypothesis.

### Phase-ab — K=128 vanilla 5-seed run (in progress)

| Seed | Box | Best MPPI | @ step | Status |
|---|---|---|---|---|
| 1 | local | 248.5 | 7M | **stuck/regressing** |
| 2 | ssh17637 | — | — | just started |
| 3 | — | — | — | pending |
| 4 | — | — | — | pending |
| 5 | — | — | — | pending |

Seed 1 behaviour diag at 7M: standing_rate=14.5%, falls=8.7/ep, ttf=674 steps — stuck in
crawling gait, confirming a bad basin regardless of K.

### Phase-1b — Glass baseline rerun (K=64, 10M cap)

Reference to confirm TD-MPC-Glass behaviour at the old training ratio.

| Seed | Box | Best MPPI | @ step | Status |
|---|---|---|---|---|
| 1 | local | 267.2 | 9.5M | done |
| 2 | ssh6_4060 | 276.2 | 5M | running |
| 3 | ssh3_3070 | **419.9** | 6M | running (best) |
| 4 | ssh6_4060 | 232.3 | 4.25M | done |

Phase-1b at K=64 shows the same stuck-seed pattern as vanilla TD-MPC2, with no winner yet across
4 seeds. Seed-3 is the closest at 419.9 with healthy behaviour diag (standing 54.9%, ttf≈45 steps).

### Key diag observations

- **Winner signatures** (K=256 s1 @ 6.5M): standing_rate=95.7%, falls=0/ep, ttf=47 steps —
  stable, continuous hopping.
- **Stuck-seed signatures** (phase-ab s1 @ 7M): standing_rate=14.5%, falls=8.7/ep, ttf=674 —
  crawling, never airborne.
- **Approaching-winner** (phase-1b s3 @ 6M): standing_rate=54.9%, falls=11/ep, ttf=45 —
  partial hopping, still inconsistent.

The diag values are now surfaced live in the Run Inspector panel (Training Progress → Behaviour Diag).

## Implementation changes landed

- `scripts/run_benchmark.py` accepts `--k_update` and leaves the default at 64.
- Vanilla `tdmpc2` now gets the same per-seed checkpoint directory layout as
  `tdmpc-glass` for HopperHop runs.
- `--resume_checkpoint` and `--save_full_state` work for both TD-MPC2 and
  TD-MPC-Glass in the active runner.
- Resume is append-safe for per-seed eval CSVs: existing CSVs keep their rows.
- `--expl_mix_decay_steps` adds a benchmark-fair fallback exploration schedule:
  per-env random-policy action mixture with random probability decaying 1 -> 0.

## Experiment ladder

Run only stable boxes for headline phases: local 4070 Ti, ssh6 4060, ssh1 2080
Ti, ssh3 3070, ssh6 3080. Use 2x3060 only for disposable smoke tests.

1. **Phase-aa-codex: K_UPDATE smoke sweep**
   - Launcher: `scripts/run_phaseaa_codex_kupdate_sweep.sh`
   - Algorithm: vanilla `tdmpc2`
   - Seeds: 1-3
   - Config: `NS=2048`, `EXPL_UNTIL=500k`, latent smoothing `1e-3` after 250k,
     10M cap, 3M patience, full-state checkpoints.
   - Sweep: `K_UPDATE in {64,128,256}`.
   - Pick the smallest K that materially improves mean MPPI or stuck rate.

2. **Phase-ab-codex: selected vanilla 5-seed run**
   - Launcher: `scripts/run_phaseab_codex_tdmpc2_5seed.sh`
   - Seeds: 1-5.
   - Use the selected `K_UPDATE`.
   - Success: 5/5 seeds best MPPI >= 500.
   - If a seed is 450-499 at 10M and still climbing, resume it to 15M.

3. **Phase-ac-codex: selected Glass 5-seed comparison**
   - Launcher: `scripts/run_phaseac_codex_glass_5seed.sh`
   - Same selected `K_UPDATE`, seeds, and fair knobs.
   - Compare against Phase-ab to decide whether Glass helps, hurts, or is neutral
     once the training ratio is fixed.

4. **Phase-ad-codex: fair exploration-mixture fallback**
   - Launcher: `scripts/run_phasead_codex_explmix.sh`
   - Use only if Phase-ab/ac still leave seeds below 300.
   - Replace hard random exploration with `--expl_mix_decay_steps 2000000`.
   - This changes only action collection, not rewards or demonstrations.

## Do not run for this objective

- Reward shaping: knee penalty, soft stand bonus, gait fall/action penalties,
  and r-stack variants.
- Behaviour cloning or winner trajectory transplant.
- Cluster entropy intrinsic reward Path P/Pa.
- Cluster soft distribution as policy/Q observation Path 7.
- More Phase-y hierarchy seeds before the K_UPDATE audit.

## Monitoring workflow

- Web dashboard: `http://localhost:5055`
- Terminal dashboard: `bash scripts/iter5_dashboard.sh`
- Queue files: `scripts/queues/*.queue`
- Local/remote mirrored CSVs:
  `exp/tdmpc_glass/remote_mirror/<box>/HopperHop_<phase>/seed_*.csv`
- Checkpoints:
  `exp/tdmpc_glass/HopperHop_<phase>/seed_N/checkpoints/{latest_eval.pkl,best_mppi.pkl,latest_full.pkl}`
- For primary runs, always use `--save_full_state` so interrupted boxes can
  resume from `latest_full.pkl` instead of restarting from seed 0.
