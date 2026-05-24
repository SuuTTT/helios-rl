# Iteration 8 — Final analysis

Date opened: 2026-05-21
Finalized: 2026-05-22
Companion docs:
- `../analysis/phase_eval_rescore_2026-05-21.md`
- `../analysis/mppi_vs_pi_analysis.md`
- `iteration_9.md`

## Goal

- **G1**: 5/5 HopperHop seeds above 500 by verified
  `best_any = max(best_pi, best_mppi)`.
- **G2**: at least one benchmark-fair seed above 600 by verified `best_any`.
- Constraints: benchmark-fair only; no reward shaping, no demonstrations, no
  environment edits, original eval reward.

## Executive conclusion

Iteration 8 did **not** find a 5/5 G1 method.

Final result by phase:

| Phase | n | Mean best_any | Median | G1 | G2 | Max | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Historical Phase1b | 5 | 428.8 | 526.0 | 3/5 | 0/5 | 562.1 | best fair hit-rate so far |
| Phase1b_10M rerun | 5 | 389.5 | 354.6 | 1/5 | 0/5 | 550.0 | longer cap does not rescue weak seeds |
| **Phase-ar** auto-restart | 5 | 395.2 | 349.2 | 1/5 | 0/5 | 584.6 | threshold too weak; no restart rows fired |
| **Phase-mpc-lite** MPPI-gated distill | 5 | 243.4 | 246.1 | 0/5 | 0/5 | 252.4 | falsified as configured |
| **Phase-g2** temp-stability 0.05 | 5 | 359.0 | 289.1 | 1/5 | 0/5 | 570.6 | one winner, not robust |

The useful Iteration 8 result is diagnostic:

1. **Measurement was improved permanently.** Best-pi, best-MPPI, and best-any
   checkpointing/reporting are required because MPPI can underperform the
   deterministic actor.
2. **The K=128 / EXPL_UNTIL=500k / latent-smooth stack did not improve
   robustness.** It often hurt hard seed 1 basin entry.
3. **The old Phase1b-style Glass recipe remains the best fair hit-rate family.**
   Iteration 9 should iterate around Phase1b knobs and hard-seed probes rather
   than continuing to vary the failed Iteration 8 stack.
4. **Phase-ar was not a clean test of the restart hypothesis.** All five CSVs
   contain zero `restart` rows. With `restart_threshold=100`, the mechanism did
   not actually fire in the completed sweep, so the result falsifies this
   configuration, not the broader idea of stronger/harder restart semantics.

## What Iteration 8 tested

Iteration 8 had four concrete components:

| Component | Purpose | Status |
|---|---|---|
| Phase-eval | Track/save `best_pi`, `best_mppi`, and `best_any` | adopted permanently |
| Phase-ar | TD-MPC2 + K=128 + NS=2048 + EXPL_UNTIL=500k + plateau restart | completed, negative |
| Phase-mpc-lite | TD-MPC2 + MPPI-gated actor distillation | completed, negative |
| Phase-g2 | Glass V2 temporal-stability loss, coef 0.05 | completed, mixed but not robust |

Phase1b_10M was also analyzed as a control: it reran the historical Phase1b
Glass recipe with a longer 10M cap and 3M patience.

## Final per-seed results

Scores are canonical de-duplicated CSV readouts. `selector` is the evaluator
that achieved `best_any`.

### Historical Phase1b

Glass baseline recipe from the earlier best fair family.

| Seed | best_any | Step | Selector | best_pi | best_mppi | Last step |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 526.0 | 3.00M | mppi | 481.6 @ 2.50M | 526.0 @ 3.00M | 4.00M |
| 2 | 526.3 | 4.00M | mppi | 524.1 @ 4.00M | 526.3 @ 4.00M | 4.00M |
| 3 | 294.4 | 4.00M | mppi | 278.7 @ 4.00M | 294.4 @ 4.00M | 4.00M |
| 4 | 235.4 | 3.00M | pi | 235.4 @ 3.00M | 227.1 @ 3.50M | 4.00M |
| 5 | 562.1 | 4.00M | mppi | 538.8 @ 4.00M | 562.1 @ 4.00M | 4.00M |

Summary: G1 3/5, G2 0/5, mean 428.8.

### Phase1b_10M rerun

Same old Phase1b Glass knobs, longer cap.

| Seed | best_any | Step | Selector | best_pi | best_mppi | Last step |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 550.0 | 9.00M | mppi | 453.4 @ 10.00M | 550.0 @ 9.00M | 10.00M |
| 2 | 292.1 | 9.25M | mppi | 274.7 @ 9.00M | 292.1 @ 9.25M | 10.00M |
| 3 | 419.9 | 5.50M | mppi | 393.2 @ 6.25M | 419.9 @ 5.50M | 6.50M |
| 4 | 331.0 | 6.75M | mppi | 305.0 @ 6.75M | 331.0 @ 6.75M | 6.75M |
| 5 | 354.6 | 5.25M | mppi | 335.5 @ 4.25M | 354.6 @ 5.25M | 5.25M |

Summary: G1 1/5, G2 0/5, mean 389.5.

Interpretation:
- Longer training rescued seed 1 but did not rescue the weak seeds.
- Seed 4 improved from 235.4 to 331.0 in the best duplicate rerun, still far
  below G1.
- "Just run Phase1b longer" is not a 5/5 path.

### Phase-ar — auto-restart

Config:
- `tdmpc2`
- K=128
- NS=2048
- EXPL_UNTIL=500k
- latent smooth 0.001 after 250k
- `restart_threshold=100`
- `restart_check_at=1M`
- `restart_max_attempts=3`

| Seed | best_any | Step | Selector | best_pi | best_mppi | Last step | Restart rows |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 349.2 | 2.50M | mppi | 293.3 @ 2.50M | 349.2 @ 2.50M | 2.50M | 0 |
| 2 | 474.5 | 7.75M | mppi | 294.3 @ 8.50M | 474.5 @ 7.75M | 10.00M | 0 |
| 3 | 271.9 | 5.00M | mppi | 257.6 @ 4.00M | 271.9 @ 5.00M | 8.25M | 0 |
| 4 | 295.8 | 2.25M | mppi | 286.2 @ 1.75M | 295.8 @ 2.25M | 5.50M | 0 |
| 5 | 584.6 | 7.75M | mppi | 521.9 @ 7.75M | 584.6 @ 7.75M | 7.75M | 0 |

Summary: G1 1/5, G2 0/5, mean 395.2.

Interpretation:
- The configured auto-restart did not solve basin entry.
- Since restart rows are zero, the threshold was too permissive: seeds were
  above 100 early enough to avoid restart while still converging to sub-G1
  gaits.
- Future restart work should use `best_any`, a much higher threshold such as
  300 at 1M, and should verify restart rows in a smoke run before spending a
  full sweep.

### Phase-mpc-lite — MPPI-gated distill

Config:
- `tdmpc2`
- K=128
- NS=2048
- EXPL_UNTIL=500k
- latent smooth 0.001 after 250k
- `mpc_distill_coef=1.0`
- `mpc_distill_anneal_steps=3M`
- `mpc_distill_disable_gap=100`
- `mpc_distill_batch_size=16`

| Seed | best_any | Step | Selector | best_pi | best_mppi | Last step |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 246.1 | 6.50M | mppi | 231.1 @ 6.50M | 246.1 @ 6.50M | 7.25M |
| 2 | 247.6 | 7.00M | mppi | 225.9 @ 9.50M | 247.6 @ 7.00M | 10.00M |
| 3 | 252.4 | 7.75M | mppi | 240.3 @ 6.75M | 252.4 @ 7.75M | 7.75M |
| 4 | 233.1 | 7.50M | mppi | 222.0 @ 8.75M | 233.1 @ 7.50M | 10.00M |
| 5 | 237.7 | 6.00M | mppi | 224.7 @ 6.25M | 237.7 @ 6.00M | 6.75M |

Summary: G1 0/5, G2 0/5, mean 243.4.

Interpretation:
- This is the cleanest negative result in Iteration 8.
- Distilling toward MPPI, even with the simple eval-gap gate, appears to keep
  all seeds in weak local behavior.
- Do not continue this branch without a materially different gate or a rollout
  diagnostic showing MPPI is producing useful actions on the target states.

### Phase-g2 — Glass V2 temporal stability, coef 0.05

Config:
- `tdmpc-glass`
- K=128
- NS=2048
- EXPL_UNTIL=500k
- latent smooth 0.001 after 250k
- `glass_lambda_temp_stability=0.05`

| Seed | best_any | Step | Selector | best_pi | best_mppi | Last step |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 246.2 | 8.50M | mppi | 111.1 @ 8.25M | 246.2 @ 8.50M | 8.50M |
| 2 | 570.6 | 5.50M | mppi | 558.4 @ 8.50M | 570.6 @ 5.50M | 8.50M |
| 3 | 284.5 | 8.25M | mppi | 261.8 @ 10.00M | 284.5 @ 8.25M | 10.00M |
| 4 | 404.8 | 8.25M | mppi | 342.5 @ 9.00M | 404.8 @ 8.25M | 9.25M |
| 5 | 289.1 | 4.25M | pi | 289.1 @ 4.25M | 241.5 @ 5.25M | 6.25M |

Summary: G1 1/5, G2 0/5, mean 359.0.

Interpretation:
- Temporal stability has a real positive signal: seed 2 reached 570.6, and seed
  4 improved to 404.8.
- It is not robust at coefficient 0.05. Seeds 1, 3, and 5 remained sub-300.
- The likely failure mode is over-stabilizing the wrong early gait phase rather
  than helping the model discover a better basin.
- Follow-up should test this idea only inside the better Phase1b recipe or with
  a delayed/decayed schedule; raw Phase-g2 0.05 is not a 5/5 candidate.

## Measurement result: Phase-eval stays

Iteration 8 permanently changed how runs are judged:

- Save and report `best_pi`.
- Save and report `best_mppi`.
- Save and report `best_any`.
- Treat G1/G2 candidates as `best_any`, then verify with rollout/video.

Reason:
- Across de-duplicated HopperHop CSVs, MPPI is worse than pi in a substantial
  minority of matched evals.
- In Iteration 8 itself, Phase-g2 seed 5 is selected by pi, not MPPI.
- MPPI-only checkpointing can discard the better controller.

This was not enough to change the Iteration 8 G1/G2 conclusion, but it prevents
future false negatives and makes render/debug selection more reliable.

## What failed and why

### Phase-ar

The idea was plausible, but the implemented run did not exercise it. The
restart floor of 100 was below the score range of several bad basins. Those
seeds avoided restart but still finished around 270-350.

Actionable lesson:
- Any future restart probe must assert that restart rows appear in the CSV when
  the condition should fire.
- Use hard seed 1 or seed 4, threshold around 300 at 1M, and count by
  `best_any`, not MPPI-only.

### Phase-mpc-lite

The MPPI-gated distill branch compressed all seeds into a narrow 233-252 band.
That is worse than baseline variability and worse than Phase-ar/Phase-g2.

Actionable lesson:
- Do not use this loss as configured.
- The MPPI-vs-pi insight should affect checkpointing and maybe action-selection
  schedules, not naive imitation.

### Phase-g2

Temporal stability is not worthless, but the 0.05 recipe is too brittle. It
produced one strong winner, one mid result, and three weak results.

Actionable lesson:
- If revisited, use Phase1b-compatible knobs, lower coefficient, delayed start,
  or off-late schedules.
- Compare cluster diagnostics and rollout videos before promoting.

### Phase1b_10M

The longer rerun reduces confidence in "late escape" as an explanation. Several
seeds spent millions more steps without reaching G1.

Actionable lesson:
- Basin entry remains the problem; more wall-clock alone is not enough.

## Handoff to Iteration 9

Iteration 9 should use one-seed probes instead of full 5-seed sweeps until a
mechanism shows a strong signal.

Recommended direction:

1. **Return to Phase1b-style Glass knobs.**
   - `proto_temperature=0.7`
   - `assign_logits_init_scale=0.5`
   - `stopgrad_graph=true`
   - short exploration first, not EXPL_UNTIL=500k by default
2. **Probe hard seeds, especially seed 4.**
   - Historical Phase1b seed 4: 235.4.
   - Phase1b_10M seed 4: 331.0 best duplicate.
   - A real 5/5 method must rescue this seed.
3. **Keep best-any accounting and render both pi/MPPI when they disagree.**
4. **Treat the Iteration 8 stack as deprioritized.**
   - K=128 + EXPL_UNTIL=500k + latent smooth + temp stability is not the
     default path forward.
5. **If restart is revisited, make it a new hard-threshold probe.**
   - The Iteration 8 sweep did not test an actually firing restart mechanism.

Active Iteration 9 probe families derived from this conclusion:
- Phase1b + K=128.
- Phase1b + low temporal stability.
- Phase1b + Glass off at 2M.
- Hard-seed 4 versions of the above.

## Final verdict

Iteration 8 is complete and negative for G1/G2:

- **G1 not achieved.**
- **G2 not achieved.**
- Best fair hit-rate remains historical Phase1b at 3/5.
- Best fair score in Iteration 8 is Phase-ar seed 5 at 584.6, still below G2.
- The most useful product of the iteration is the measurement fix plus the
  decision to pivot toward Phase1b-derived hard-seed probes in Iteration 9.
