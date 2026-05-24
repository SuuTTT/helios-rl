# TD-MPC-Glass HopperHop — Iteration 2 lessons (Phase-d through Phase-h)

Date: 2026-05-14.
Context: Phase 1c (act-noise anneal 0.30→0.10) was falsified — it hurt
winners (seed 1: 438→277, seed 2: 526→221) while only mildly helping seed 3.
This document records what we tried next, what worked, what didn't, and
what's left.

Goal throughout: **all 5 HopperHop seeds > 500 MPPI**, ideally > 700, beating
the Phase 1b finals `[438, 526, 294, 187, 562]` (mean 401±158, with seeds 3
and 4 stuck below 300).

---

## 1. Result summary across all seeds (Phase-d → Phase-h)

| Phase | Knob vs Phase 1b | Hardware | Seed | Basin | Peak MPPI | Final/Last | Status |
|-------|------------------|----------|-----:|:-----:|----------:|-----------:|--------|
| **d v1** | H=5 + noise=0.40 | 4070 Ti | 1 | K=4 | 114 | crash | mjx-warp-901 @ 1M |
| **d v2** | H=5 only | 4070 Ti | 1 | K=4 | 199 | killed @ 1.5M | plateau, killed for pivot |
| **e** | Q-reset @ 1M/2M/3M | 5070(broken)→4060 | 1 | K=4 | 228 (pre-reset) | corrupted | bad Q-reset impl |
| **f** | latent_smooth=1e-3 | 4070 Ti | 1 | K=4 | **571** | 489 @ 4M | ✅ winner |
| **f** | (same) | 4070 Ti | 2 | K=4 | 284 | crash | mjx-warp-901 @ 1.25M |
| **f** | (same) | 4070 Ti | 3 | K=4 | 262 | 246 @ 8.25M | early-stop, stuck downstream |
| **f** | (same) | 4070 Ti | 4 | **K=3** | 266 | 151 @ 6M | early-stop, basin-capped |
| **f** | (same) | 4070 Ti | 5 | **K=3** | 255 | 244 @ 2.75M | early-stop, basin-capped |
| **g** | consistency_coef=1.0 | 4060 | 1 | K=4 | 427 | 373 @ 4M | partial lift |
| **g** | (same) | 4060 | 2 | K=4 | 482 | 471 @ 4M (pi=488) | partial lift, just under 500 |
| **h** | smooth=1e-3 + ccoef=1.0 | 4060 | 1 | K=4 | 462 (so far) | running @ 2.58M | climbing |
| **h** | (same) | 4060 | 2 | — | — | queued | — |

**Headline**: of 8 fully-evaluated seeds, exactly **one cleared peak MPPI > 500** (Phase-f seed 1 at 571).

---

## 2. What we falsified (with hard evidence)

### 2.1 Act-noise anneal 0.30→0.10 over 1M (Phase 1c — already falsified)
- Hurts winners (seed 1: 438→277, seed 2: 526→221)
- Only seed 3 nudged up (294→401 at 3.5M, but seed 3 was the stuck plateau)
- **Don't propose noise schedules below 0.30 again.**

### 2.2 EXPL_NOISE > 0.30 (Phase-d v1, falsified hard)
- Setting `--act_noise_start 0.40 --act_noise_end 0.40` triggered `Warp CUDA error 901` from `mujoco_warp/_src/solver.py:3303` at ~1M env steps, during `eval_pi` host-sync.
- Initially blamed solely on noise=0.40; **later refuted by Phase-f seed 2 hitting the same crash with default noise=0.30**.
- **Refined lesson**: the crash is sporadic — `cudaErrorCapturedEvent` from `wp.capture_while` is a "hopper drifted into a non-converging solver configuration" failure mode that any setup can hit, but noise≥0.40 raises probability dramatically.
- See `feedback_mjx_warp_901.md` in the agent memory.

### 2.3 MPPI horizon H=3 → 5 alone (Phase-d v2, falsified)
- Throughput dropped 449→397 sps (~20% slower) — expected.
- Seed 1 trajectory through 1.5M: pi flat at 174, MPPI crawling 188→193→199 (+11/500k).
- Linearly extrapolates to final ~250 — same Phase 1b seed-3 stuck pattern.
- **H=5 alone does not rescue stuck seeds.** Killed for pivot.

### 2.4 Naive Q-reset (Phase-e, falsified by implementation bug)
- REDQ-style spec: re-init `params["q"]` at 1M/2M/3M, keep `tp["q"]` (target Q).
- My implementation also did `opt = tx.init(params)` — re-initialised optax state for **all params**, not just Q.
- Result: pi/enc/dyn lose Adam momentum, then immediately do 64 gradient updates with a random-Q signal → pi gets random gradient targets, gets corrupted.
- Seed 1 trajectory: 750k pre-reset = 228 MPPI ; 1M post-reset = 3.2 MPPI (collapsed).
- **Q-reset is unproven, not refuted**: the implementation was wrong. A correct version needs (a) Q-only slice of opt state replaced, or (b) pi update paused for K env steps post-reset.

### 2.5 consistency_coef 2.0 → 1.0 alone (Phase-g, partial result)
- Both seeds K=4 ✅ — basin-neutral.
- Seed 1 peak 427, seed 2 peak 482. Both close to but **below 500**.
- **Mild improvement over Phase 1b stuck seeds (294/187)** but not enough on its own.

---

## 3. What worked partially: latent action smoothing (Phase-f)

`--latent_action_smooth_coef 1e-3` added a `mean(||π(z_t) - π(z_{t-1})||²)` term to the policy loss, computed over the consistency-loss rollout (blog §9 item 10, "big return gains on underactuated DMC tasks").

### 3.1 Seed 1: the breakthrough
| step | pi | MPPI |
|------|----|------|
| 1M | 354 | 417 |
| 1.25M | 428 | **502** ← first > 500 ever |
| 1.5M | 451 | 518 |
| 2M | 426 | 527 |
| 2.25M | 477 | 540 |
| 3M | 393 | **571** ← peak |
| 3.5M | 521 | 477 |
| 3.75M | 518 | 566 |
| 4M | 525 | 489 |

Peak MPPI **571.7** > Phase 1b's best seed (562). Deterministic pi reached 525.6 at 4M.

### 3.2 Seeds 3, 4, 5: smoothing made things *worse*

This is the unexpected finding. The basin survey at the end of every run:

| seed | basin | peak | comment |
|------|:-----:|-----:|---------|
| 1 | K=4 | 571 | smoothing helped |
| 2 | K=4 | 284 | crashed @ 1.25M (Warp 901) |
| 3 | K=4 | 262 | stuck downstream (same shape as Phase 1b seed 3) |
| **4** | **K=3** | 266 | **basin-flipped** (Phase 1b had this seed at K=4) |
| **5** | **K=3** | 255 | **basin-flipped** |

Blog §5.5 reports K=4 seeds average 403.7, K=3 seeds average 310.6 — i.e. K=3 has a structural ceiling around 300 that no downstream tuning can break. **Phase-f shoved 2/5 seeds into the K=3 basin** that Phase 1b had kept all 5 seeds out of.

Mechanism: the smoothing term is computed in the same `loss_fn` and shares the optax chain. It changes gradient direction from step 1 onward. Different gradients → different latent geometry → different `(prototype, assign_logits)` evolution → some seeds flip into the K=3 attractor early (basin is decided within ~100-250k env steps per blog §5.4).

**Net verdict on Phase-f**: smoothing is a powerful policy-side knob but interacts badly with Glass's basin discovery. The 1/5 win rate is misleading — it's the *same 3/5* K=4 winners as Phase 1b, with one of them lifted from 438 to 571 and one stuck identically, plus 2 seeds *demoted* to K=3.

---

## 4. Cross-cutting infra wins (deserves to stay)

### 4.1 `--early_stop_patience N` (commit 1bbcfb6)
Halt training when no new best MPPI has been recorded for N env-steps. Combined with `--total_steps 10_000_000`, this auto-stops on convergence:
- Phase-f seed 3 stopped at 8.25M (peak 262 @ 6.5M).
- Phase-f seed 4 stopped at 6M (peak 266 @ 4.25M).
- Phase-f seed 5 stopped at 2.75M (peak 255 @ 1.25M, fast trigger because it plateaued early).

Saves hours per stuck seed. Reuse for all future iterations.

### 4.2 Output tagging via `TDMPC_GLASS_OUTPUT_TAG`
Already documented in CLAUDE.md but was load-bearing for this iteration — `phased_v1_noise040` / `phased_v2_H5_only` / `phasef` / `phaseg` / `phaseh` directories all coexist without overwriting each other.

### 4.3 Two-GPU coordination
Local 4070 Ti for one experiment, remote 4060 for another. Rsync mirror script polls the remote every 60s. Combined with SSH-tail Monitors, gives real-time visibility into both. (The original remote 5070 had driver 570 + Blackwell sm_120 = JAX cuda12 wheels couldn't load CUBINs; **don't try to bootstrap a 5070 instance with driver < 580**.)

### 4.4 Glass diagnostic survey scripts
Reading `exp/benchmark/glass_diag/HopperHop_<tag>/seed_*/step_*.npz` and computing `K_active`/`H_cm`/`max_mass` is the **single most informative post-hoc check** — it told us the basin shift instantly. Worth turning into a small `scripts/summarise_glass_basin.py` for future runs.

---

## 5. The five hard-won lessons

1. **The basin choice (K=3 vs K=4) is the dominant determinant of HopperHop performance**, not the downstream policy. The blog said this; Phase-f re-confirmed it the hard way. Any future intervention must be checked for basin-stability before being declared a win.

2. **Smoothing perturbs basin choice.** Adding a policy-side regulariser changes the gradient flow that the Glass prototypes see (because they share an Adam). The smoothing→basin perturbation pipeline is the surprising mechanism here.

3. **Warp 901 is sporadic and not noise-specific** — don't waste cycles trying to "fix the noise". Use queue-level retry-with-a-different-seed instead.

4. **Naive `opt = tx.init(params)` after Q-reset destroys pi.** REDQ-style resets need either an opt-state surgery (replace only Q's slice) or a pi-pause window. The simpler "just re-init opt" path is wrong.

5. **`--early_stop_patience` is essential at 10M caps.** Without it we'd have burned ~12h on Phase-f seed 4 alone. With it, the budget collapses to the actual peak window.

---

## 6. What to run next on the local 4070 Ti (now idle)

Goal: rescue the K=3 seeds 4 and 5 *without* losing the seed-1 win.

The two strongest candidates, ranked by my read of evidence:

### 6.1 Phase-i: weaker smoothing (Recommended first)

```
--latent_action_smooth_coef 1e-4
```

Hypothesis: the basin perturbation scales with the gradient magnitude that the smoothing term injects. At 1e-3, smoothing was too strong (perturbed 2/5 basins). At 1e-4 (10× weaker), it may still help seed 1 a little but leave the Glass discovery process undisturbed → all 5 seeds stay K=4 like Phase 1b had.

- Single-knob change. Direct ablation against Phase-f and Phase 1b.
- Risk: 1e-4 is too weak to help seed 1; we'd lose the breakthrough.
- Mitigation: queue seeds 1, 2 first; if seed 1's surge is gone, kill and try 5e-4 instead.

### 6.2 Phase-j: curriculum smoothing

```
--latent_action_smooth_coef 1e-3  (full strength)
--latent_smooth_warmup_env_steps 250000   (NEW flag — would need ~15 LOC)
```

Phase 1b's blog §5.4 says the basin is decided in the first ~100-250k env steps. If smoothing is OFF during that window, basin decision is undisturbed (= Phase 1b's 5/5 K=4). Then smoothing turns on at 250k for the policy-side benefit.

- Requires a code change to schedule the coef (or a kwarg to `make_update_fn` that switches via `jax.lax.cond` on env_steps). ~30 min implementation.
- More principled fix; addresses the mechanism directly.

### 6.3 Phase-k: combine Phase-h (already in flight on remote) with K=4 forcing

```
--latent_action_smooth_coef 1e-3
--consistency_coef 1.0
--glass_proto_temperature 0.5   (sharper, encourages K=4 collapse)
```

Wait for Phase-h to finish first (~3h more), then if it lands ~470-490, this is the next refinement.

### Recommendation order
1. **Phase-i first** (1e-4 smooth, 4-line script change, no algorithm code edits, runnable in 30 seconds). Seeds 1, 2 with 10M cap + 1.5M patience.
2. If Phase-i seed 1 surges → run all 5 Phase-i seeds.
3. If Phase-i seed 1 does NOT surge → fall back to Phase-j (curriculum, needs code change).

---

## 7. Open questions / further work

- **Phase-h** (remote 4060, in flight): if seed 1 finishes above 500, the additive-knobs hypothesis is alive. Watch peak.
- **Render seed 1 vs seed 3 rollouts**: the existing `scripts/render_glass_rollout.py` failed on remote 4060 with a Warp mempool error. Retry on local 4070 Ti when it's idle (it has the working driver 12.4 + warp 1.12.1 combo). Output: an MP4 showing what good vs stuck hopping looks like and what each K=4 cluster encodes temporally.
- **Render Phase-f seed 4** (K=3 basin): would directly show what 3-cluster hopping looks like — likely the "balanced shuffle" the blog §5.6.1 hypothesises for seed-3-class plateaus, but applied at the basin level.
- **Proper Q-reset (Phase-e v2)**: implement only-Q opt-state surgery and pi-update pause. Open question whether it adds anything on top of Phase-i/j.
- **Distributional Q (quantile regression)** — blog §9 item 1, the bigger algorithmic change that's still untried.
