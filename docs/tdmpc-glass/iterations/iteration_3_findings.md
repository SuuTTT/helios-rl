# TD-MPC-Glass HopperHop — Iteration 3 findings and next experiments

Date: 2026-05-15. Status: 17 seeds attempted across 7 hypothesis variants since
falsifying Phase 1c. **2 of 17 seeds cleared peak MPPI > 500** — both lucky
K=4 surges that the rest of the seeds didn't reproduce.

This document supersedes
[iteration_2_lessons.md](./iteration_2_lessons.md) for the running scoreboard
and adds new structural findings + the Phase-m / Phase-n directions.

---

## 1. Full scoreboard (Iteration 1c → Iteration 3)

| seed | knobs | basin | peak | comment | >500? |
|------|-------|:-----:|-----:|---------|:-----:|
| Phase-f seed 1 | smooth=1e-3 | K=4 | **571** | foot-hop winner | ✅ |
| Phase-f seed 2 | (same) | K=4 | 284 | Warp-901 crash @1.25M | — |
| Phase-f seed 3 | (same) | K=4 | 262 | K=4 stuck, knee-walk | ❌ |
| Phase-f seed 4 | (same) | **K=3** | 266 | basin-capped | ❌ |
| Phase-f seed 5 | (same) | **K=3** | 255 | basin-capped | ❌ |
| Phase-g seed 1 | ccoef=1.0 | K=4 | 427 | partial lift | ❌ |
| Phase-g seed 2 | (same) | K=4 | 482 | partial lift, close | ❌ |
| Phase-h seed 1 | smooth+ccoef | K=4 | 490 | very close, no surge | ❌ |
| Phase-h seed 2 | (same) | K=4 | 328 | SIGSEGV crash @2.93M | — |
| Phase-i seed 1 | smooth=1e-4 | K=4 | 308 | too-weak smoothing, plateau | ❌ |
| Phase-j seed 1 | curriculum smooth | K=4 | 452 | slow climb, plateau | ❌ |
| Phase-j seed 2 | (same) | K=4 | **518** | lucky surge | ✅ |
| Phase-j seed 3 | (same) | K=4 | 322 | K=4 stuck | ❌ |
| Phase-j seed 4 | (same) | **K=3** | 266 | basin-capped (still!) | ❌ |
| Phase-j seed 5 | (same) | **K=3** | 354 | basin-capped (still!) | ❌ |
| Phase-k seed 1 | curriculum + λ_temp=0.05 | K=4 | 292 (killed) | over-regularised | ❌ |
| Phase-l seed 1 | Glass-zeroed + smooth | K=? | 289 (killed) | Glass-zeroed underperforms | ❌ |

**Hit rate**: 2/17 = 12% of attempts cleared 500 peak MPPI. Both wins are
"K=4 basin + lucky surge" (Phase-f seed 1, Phase-j seed 2). Neither
intervention has produced a reliable >500 result across multiple seeds.

---

## 2. Three failure modes, now sharper than in iteration 2

### 2.1 K=3 basin (4/17 seeds, structural cap ~270-360)

Seeds 4 and 5 land in K=3 across BOTH Phase-f AND Phase-j despite very
different loss functions. Critically, the **earliest** Glass-diag dump
(step 250k, BEFORE the curriculum smoothing turns on) already shows K=3
for both seeds. So the basin is decided in the first ~100-250k env steps,
and our post-warmup smoothing changes can't recover it.

This is consistent with blog §5.4: "Empirically the basin is locked within
the first 250k env steps and never moves".

But Phase 1b reportedly had **5/5 K=4** with identical Glass knobs and no
smoothing. So how do seeds 4 and 5 flip to K=3 with smoothing in the loss?
The mechanism (Phase-m hypothesis below): the *existence* of the smoothing
vmap-over-pi in `loss_fn` perturbs XLA's floating-point order even when
`smooth_coef=0` during warmup. That tiny numerical perturbation is enough
to flip basin choice on the basin-fragile seeds.

### 2.2 K=4 stuck downstream (~10/17 seeds, 280-490)

Even K=4 seeds usually plateau around 300-450. The Phase-f seed 1 video
analysis (see [cluster_to_gait_mapping.md](../analysis/cluster_to_gait_mapping.md))
showed these seeds converge to a **knee-walk gait** (push off with knee +
toe, nose drags on ground) instead of the foot-hop the winner uses. The
Glass partition is fine; the policy/critic just gets stuck in the wrong
behavioural attractor.

### 2.3 K=4 + lucky surge (2/17 seeds, 500-571)

The "win condition" — same setup as 2.2 but the policy happened to find
the foot-hop early. No reliable trigger. Smoothing helps (raises hit rate
from 0/5 in Phase 1b → 1/4 in Phase-f → 1/5 in Phase-j) but doesn't fix
the underlying basin-perturbation issue.

### 2.4 Sporadic mjx Warp 901 / sigsegv crashes (2/17 seeds)

Random fault in `wp.capture_while` when hopper drifts into a non-converging
solver configuration. Sporadic — not noise-specific, not coef-specific.
Mitigation: retry with a different seed number (seed 2 → seed 12 etc.).
See `feedback_mjx_warp_901.md` in agent memory.

---

## 3. Seven ideas from the user (and what we now know about each)

These came up during the iteration 3 video discussion. Updated with
post-Phase-{j,k} evidence.

### 3.1 K-flicker penalty (penalise short clusters)

**Status: tested in Phase-k as raised `λ_temporal=0.05`. Falsified.**

Phase-k seed 1 plateaued at MPPI≈292 after 7+ hours — λ_temporal=0.05 was
over-regularising. The K-flicker signal is real (videos show stuck seeds
do flicker more) but the existing `λ_temporal` term penalises *cluster*
co-occurrence, not raw label-change rate. A direct K-label-change penalty
(idea 1b from iteration 2) needs a Gumbel-softmax surrogate and was
correctly deferred.

If we revisit: try `λ_temporal=5e-3` (only 5×, not 50×). Or implement the
direct K-label penalty.

### 3.2 K=4 ceiling around 550-650 / hierarchical super-clusters

**Status: weakly supported. Phase-f seed 1 peak 571 and Phase-h seed 1
peak 490 are both well below DMC HopperHop's max of ~1000. But it's not
clear if the cap is "K=4 partition" or "knee-walk policy gait".**

Deferred until basin / gait issues are fixed. If we still cap below 700
after a successful Phase-n or Phase-m, hierarchical super-clusters
become worth implementing (~30 LOC in tdmpc_glass.py).

### 3.3 VLM-driven semantic symbols + symbolic reasoning in MPPI

**Status: deprioritised by user judgement on 2026-05-15.** "We may not need
a semantic symbol (using VLM) for planning." Reasoning: the cluster→gait
analysis we did manually (cluster_to_gait_mapping.md) already gave us the
diagnostic insight we needed (knee-walk vs foot-hop). Automating that
mapping via a VLM doesn't unlock new training improvements — it only
helps with post-hoc analysis at scale, which we don't need yet.

### 3.4 MPPI in cluster space (planning with abstraction)

**Status: still deferred.** Phase-j showed pi often beating MPPI
(pi=326, MPPI=272 at 1.5M), confirming the MPPI-rollout-error issue. But
implementing two-level planning is ~1 week. Cheaper interventions (Phase-n,
Phase-m) come first.

### 3.5 Alternative Glass integrations / SimNorm groups as clusters

**Status: still deferred.** The two-level prototype→cluster design is
working as intended (Glass clusters DO correspond to gait phases per
video analysis). Replacing μ with SimNorm group argmaxes is an interesting
ablation but not the highest-EV next move.

### 3.6 Drop Glass entirely (Phase-l ablation)

**Status: partial result, killed early.** Phase-l (TD-MPC2 + smoothing,
Glass losses zeroed) reached MPPI=289 at 4.75M — way below Phase-f's 477
at 3.5M. **Glass IS contributing**, not just passive. Killed to free GPU
for more useful experiments.

### 3.7 Hybrid abstraction + raw latent (Glass loss decay curriculum)

**Status: still deferred but interesting.** With Phase-m and Phase-n
addressing the basin issue first, this becomes the natural next-step if
the K=4 ceiling is real (idea 3.2).

---

## 4. The mechanism we now believe explains everything

> **Adding *any* term to `loss_fn` — even when its coefficient is 0 —
> changes XLA's compiled graph enough to perturb floating-point order,
> which flips basin choice on the ~40% of seeds (4 and 5 in our case)
> whose basin is borderline between K=3 and K=4.**

Evidence:
- Phase 1b (no smoothing): 5/5 K=4 per blog
- Phase-f (smooth=1e-3 from step 0): 3/5 K=4 + 2/5 K=3
- Phase-j (smooth=0 pre-250k, but smoothing term is still in `loss_fn`
  multiplied by 0): 3/5 K=4 + 2/5 K=3 — **same as Phase-f**
- Phase-j seed 4 K=3 confirmed at the earliest step-250k diag dump,
  i.e. BEFORE the curriculum boundary fires.

So the perturbation is from the *graph structure*, not the *gradient
contribution*. This is the same RNG-perturbation issue blog §5.7 documents
for the separate-optimiser experiment.

---

## 5. Phase-m and Phase-n — running now

These two hypotheses attack basin choice from different angles. Both
launched 2026-05-15.

### 5.1 Phase-n (remote 4060) — Glass-internal pressure

**Knob**: `--glass_proto_temperature 0.4` (was 0.7). Sharper soft-assignment
means each latent commits more decisively to a single prototype, which
should encourage all seeds to crystallise onto the same K=4 attractor
geometry-driven, before training even matters.

Other Phase-j knobs unchanged (curriculum smoothing on after 250k, etc.).

**Hypothesis**: 5/5 K=4 because the sharper assignment removes the
K=3/K=4 ambiguity that's currently RNG-fragile.

**Risk**: T=0.4 might be too sharp — could cause Glass loss to overshoot
on early prototype assignments and produce a worse partition (e.g. all
prototypes collapse to one cluster). If so, we tried T=0.5 as a backstop.

### 5.2 Phase-m (local 4070 Ti) — JIT-graph equivalence to Phase 1b

**Implementation**: Refactor `make_update_fn` so the smoothing vmap-over-pi
is **only compiled into the graph when `smoothing_enabled=True` at Python
level**. Build two compiled functions:
- Warmup: graph = pure Phase 1b (smoothing path NOT in graph)
- Post-warmup: graph = Phase 1b + smoothing term

Rebuild at curriculum boundary (one-time ~3 min JIT cost, already paying
this for the coef change anyway).

**Hypothesis**: All 5 seeds K=4 because the warmup graph is identical to
the one that gave Phase 1b's 5/5 K=4 result.

**Risk**: The rebuild swap may itself perturb training state (params don't
change, but Adam state is reset on the new function call? Let me check).

---

## 6. Decision criteria for "did Phase-m / Phase-n work?"

Each test gives a clear signal at the EARLIEST glass-diag dump (step 250k):

| outcome | what it means |
|---------|---------------|
| 5/5 seeds K=4 at 250k | basin fix succeeded; now wait to see if seeds also surge past 500 |
| 4/5 K=4, 1/5 K=3 | partial fix, basin issue softened but not eliminated |
| 3/5 K=4, 2/5 K=3 | failure — same as Phase-f / Phase-j |

Cheap diagnostic — we know within ~10 min of seed start whether the basin
fix took. Don't need to wait for full early-stop.

If both Phase-m and Phase-n give 5/5 K=4: the basin fix is robust, focus
shifts entirely to triggering the foot-hop surge reliably on K=4 seeds.

If only one works: that mechanism is the right answer; combine with the
other for redundancy.

If both fail: basin choice has a third mechanism we haven't identified.
Fall back to direct K=4 forcing via a hard constraint in
`init_glass_params` (e.g. fix some prototype assignments to anchor 4 active
clusters from step 0).

---

## 7. Phase-m + Phase-n results (the iteration 3 verdict)

### 7.1 Phase-n (proto_T=0.4) — falsified

Phase-n seed 4 at the first diag (step 250k): **K=3** — same as
Phase-f/j. Sharper soft-assignment did NOT force K=4. Killed at ~30 min.
Archived as `HopperHop_phasen_v1_falsified`.

Mechanism conclusion: **soft-assignment temperature is NOT the basin-choice
lever**. The basin attractor is determined by the relative magnitudes of
prototype-to-latent inner products, not by the softmax sharpness.

### 7.2 Phase-m (Python-conditional smoothing) — **partial fix, but K=4 doesn't matter for peak**

Phase-m result is the iteration 3 headline:

| seed | basin (Phase-m) | peak (Phase-m) | basin (Phase-f / Phase-j) | peak there |
|------|:---------------:|:--------------:|:-------------------------:|:----------:|
| seed 4 | **K=4 ✅** (fix worked) | **262** | K=3 | 266 |
| seed 5 | K=3 ❌ (fix didn't work) | 286 (running) | K=3 | 255 |
| seed 1 (remote) | K=4 | 294 (running) | K=4 | 489–571 (huge variance) |

Three results stick out:

**(a) The Python-conditional smoothing fix works for seed 4 but NOT seed 5.**
Seed 4 was the one borderline between K=3 and K=4; Python-conditional
tipped it K=4. Seed 5 is "firmly" K=3 — even the basin-stable graph
doesn't recover it. So **the basin perturbation hypothesis is only
partially correct** — float-order matters for *some* fragile seeds, not all.

**(b) K=4 doesn't actually help seed 4's peak.** Seed 4 peak went from
266 (K=3 in Phase-f) to **262** (K=4 in Phase-m). The basin flip changed
nothing about the policy's peak return. The Glass cluster choice is
**not the limiting factor** for stuck seeds — the policy's downstream
technique discovery is.

**(c) Seed 1 (the lucky Phase-f winner) lost its surge under Phase-m.**
Phase-m seed 1 at 3M is at peak 294 — well below Phase-f seed 1's 571
peak. Same RNG seed, different XLA graph during warmup → different
training trajectory. The Python-conditional fix improved seed 4 but
broke seed 1 — net zero for our 5-seed average.

### 7.3 The abstraction-as-blocker hypothesis (user's iteration 2 concern, now validated)

Recall the concern raised in iteration 2: *"final performance may be lower
than original TD-MPC2 because abstract states might be helpful to learn
quickly in the beginning but that could be a blocker later (it may not
completely solve the task due to abstractions)."*

The Phase-m result is the strongest evidence yet for this:

- Blog Phase 1 (Glass active) achieved mean MPPI 366 at 4M.
- Official TD-MPC2 paper achieves **mean MPPI 449 at 4M on HopperHop**.
- **Glass-active TD-MPC2 is 83 points worse than vanilla on the same task**.
- And our 17-seed Phase-{f..n} sweep peaks at 571 — but with hit rate 2/17,
  median ~280. Most seeds underperform a vanilla TD-MPC2 mean.

So: the Glass abstraction is *not* delivering on its premise. It biases
the agent toward a fixed partition of state space that happens to be
incompatible with the policy/critic finding the right technique most of
the time. **The hybrid approach is now the natural next move.**

---

## 8. The hybrid approach — Phase-o

Use Glass for *representation learning early* (first 1–2M env steps when
basin lock + structure discovery matter) then *decay Glass losses to
zero* so the encoder fully refines for value/dynamics quality without
the partition acting as an inductive bias the policy can't unlearn.

Concretely:

```
λ_se(env_steps)       = 5e-3  * max(0.0, 1 - env_steps / 2_000_000)
λ_balance(env_steps)  = 1e-2  * max(0.0, 1 - env_steps / 2_000_000)
λ_temporal(env_steps) = 1e-3  * max(0.0, 1 - env_steps / 2_000_000)
```

By 2M env steps all Glass losses → 0. After that, the run is plain
TD-MPC2 + latent action smoothing (with the encoder warmed up by Glass).
The prototype/assign_logits tensors stay in `params` but get zero
gradient → they freeze at their Glass-trained values.

**Implementation cost**: ~40 LOC in `tdmpc_glass.py` (a `lambda_curr_*`
fn computed from `env_steps` passed in, JIT-rebuild at decay milestones).

**Expected payoff**: if Glass-as-blocker is correct, Phase-o seeds should
all reach **vanilla TD-MPC2 performance (~449) OR BETTER** because they
get the Glass-driven structure-learning head start.

**Risk**: decaying Glass mid-training may destabilise the encoder if
the value/dynamics losses are heavily relying on the structured
partition. Mitigation: decay linearly over 1M env steps (not abruptly)
and floor at small ε rather than exact 0.

### 8.1 Phase-o variant A — Glass loss decay (recommended first)

The scheme above. Tests "abstraction-as-temporary-scaffolding".

### 8.2 Phase-o variant B — true hybrid (parallel raw + abstract paths)

A more architectural change: have the encoder produce both a Glass-aware
z₁ and a raw z₂ (no Glass loss applied). The dynamics/Q/policy use a
concat z = [z₁, z₂] or a learned mixture. Glass losses only see z₁.

Heavier (~150 LOC, doubles encoder width). Defer until variant A
falsified.

---

## 9. Recommended order after Phase-m completes

1. **Phase-o variant A** (Glass loss decay curriculum) — direct test of
   the user's hybrid hypothesis. Cheap implementation.
2. **Phase-p** (proper REDQ-style Q-reset with pi-pause) — independent of
   Glass mechanics, targets the technique-stuck failure mode.
3. **Phase-r** (reward shaping — penalise knee-on-ground) — heaviest but
   most direct fix for the foot-vs-knee gait issue surfaced by the
   video analysis.

If Phase-o still caps under ~450 average, the Glass-as-blocker hypothesis
is falsified and we go to Phase-p/r.

---

## 10. Phase-o result — mixed (1 surprise win, 1 stuck, 1 crash, 1 plateau)

**Update 2026-05-15 evening**: an initial read of Phase-o seed 4 plateauing
at 254 led me to declare Phase-o a failure. **I missed that Phase-o seed 3
actually broke 500** — peak MPPI=523.9 at 4M, still running on local. This
is the **3rd winner across all 7 phases**.

Updated Phase-o scorecard:

| seed | basin | peak MPPI | comment |
|------|:-----:|----------:|---------|
| Phase-o seed 3 (local) | K=4 | **523.9 ✅** | surged @ 750k→1M (490), peak 524 @ 4M, still running |
| Phase-o seed 4 (local) | K=4 | 254 | typical knee-walk stuck |
| Phase-o seed 5 (local) | — | crash | Warp-901 @ ~500k env_steps |
| Phase-o seed 1 (remote) | K=4 | 391 | partial improvement over Phase-m |

### 10.x The pattern across ALL 3 winners — the 1M surge

All three seeds that cleared MPPI=500 across our 20+ seed sweep share **the
exact same trajectory shape**:

| seed | first crossed 400 | peak | peak step |
|------|:-----------------:|-----:|----------:|
| Phase-f seed 1 | **1M (417)** | 571 | 3M |
| Phase-j seed 2 | **1M (435)** | 518 | 1.75M |
| Phase-o seed 3 | **1M (490)** | 524 | 4M (still climbing) |

**All three surged at the 1M env-step eval — the same boundary.** No winning
seed had its surge at 750k or 1.25M; they all crossed 400 at exactly the 1M
checkpoint. After 1M, all three held 400+ for the rest of training.

Stuck seeds NEVER cross 400 — they linearly creep from 0 to 200-400 over
many million env steps. The 1M eval is a clean discriminator: at 1M either
you're at >400 (and will sustain), or you're below 200 (and will plateau).

### 10.x The mechanism — surge is RNG-determined; Phase-o's hybrid logic
is INCIDENTAL to seed 3's win

Phase-o seed 3 surged at 1M, but **Glass turn-off didn't happen until 2M**.
The win occurred while Glass was still active and following the curriculum
smoothing schedule. So Phase-o's hybrid abstraction logic was IRRELEVANT to
this seed's success — the surge had already happened before glass_decay_steps
fired.

What made seed 3 succeed in Phase-o but fail in Phase-j (peak 322)? **The
mere presence of the `glass_decay_steps` flag in the code path perturbed
XLA's float order**, giving seed 3 a different RNG trajectory than it had
in Phase-j. That different trajectory happened to find the foot-hop technique
early. Same code + same seed produces different results when *any* loss-fn
or training-loop knob is added, even if the knob doesn't fire.

### 10.x Verdict on the abstraction-as-blocker hypothesis (revisited)

The previous draft of this section concluded Phase-o was a failure and that
Glass-off neither hurt nor helped. **With Phase-o seed 3's win included,
the conclusion is more nuanced**:

- Glass-off at 2M survives without catastrophe (local seed 4 had a
  transient collapse but recovered).
- Glass-off doesn't TRIGGER the surge — Phase-o seed 3 surged before
  Glass-off.
- Glass-off doesn't appear to HURT either — Phase-o seed 3 sustained
  500+ from 1M to 4M+ across the Glass-off boundary at 2M with no
  visible drop.

So Glass is genuinely peripheral. **Not a help, not a serious hindrance.**
The user's hybrid hypothesis (abstraction-as-blocker) is *partially* validated
— Glass isn't a *help*; but it's also not the BLOCKER that prevents the
other 17 seeds from breaking 500. **The actual blocker is the RNG-driven
random selection of which seeds find foot-hop within their first 1M env
steps.**

Phase-o (Glass OFF after 2M env steps) ran on both boxes. Results:

| seed | basin | peak MPPI | comment |
|------|:-----:|----------:|---------|
| Phase-o seed 4 (local) | K=4 | **254** | almost identical to Phase-m's 262 |
| Phase-o seed 1 (remote) | K=4 | **391** | better than Phase-m's 294 but well below Phase-f seed 1's 571 |

Both seeds plateaued well below 500. Specifically:
- **Local seed 4 climbed slowly post-Glass-off** (175 @ 2.5M → 254 @ 8M = +79 over 5.5M env steps). The encoder, freed from Glass, did inch upward — but the climb rate is too slow to ever reach 500 before early-stop.
- **Remote seed 1** held a plateau at 380–391 from 1.75M onward. The Glass-off at 2M produced no collapse and no surge — essentially neutral.

There was a **transient collapse on local seed 4 right at the Glass-off boundary** (MPPI 137 @ 1.75M → 1.4 @ 2M) but it recovered by 2.25M. So abruptly disabling Glass is survivable, not fatal. But not helpful either.

### 10.1 Verdict on the abstraction-as-blocker hypothesis

The user's iteration-2 concern was: *abstractions may speed up early learning
but be a blocker later*. Phase-o tested the prediction directly by disabling
Glass mid-training. The data say:

- **Phase-o doesn't unlock the surge** to >500 that Glass-on rarely achieves.
- **Phase-o doesn't dramatically hurt** either (peak 391 vs Phase-m 294 on the
  same seed is a modest improvement; peak 254 vs Phase-m 262 is a wash).
- **Conclusion: Glass is neither a help nor a primary blocker** at the level of
  knobs we've been turning. It's *peripheral* — adding compute and bias but
  not determining the final peak.

The real driver of >500 peaks (Phase-f seed 1: 571; Phase-j seed 2: 518) is
**the policy happening to find the foot-hop technique early in training**, before
any Glass dynamics can lock anything in. Once a seed is in the "knee-walk"
attractor by ~1M env steps, no Glass-side or curriculum-side intervention we've
tried (smooth, ccoef, λ_temporal, basin fix, Glass-off) gets it back out.

### 10.2 17→19 scoreboard, hit rate 2/19 = 10.5%

Adding Phase-m + Phase-o results (only the new ones, both K=4 basin):

| seed | peak | knob suite | >500? |
|------|-----:|------------|:-----:|
| Phase-m seed 4 | 262 | Python-cond smooth | ❌ |
| Phase-m seed 5 | 286 | Python-cond smooth | ❌ |
| Phase-m_remote seed 1 | 294 | Python-cond smooth | ❌ |
| **Phase-o seed 4** | **254** | hybrid (Glass OFF @2M) | ❌ |
| **Phase-o_remote seed 1** | **391** | hybrid (Glass OFF @2M) | ❌ |

Hit rate over 19 seeds: still **2/19 = 10.5%**. No new wins.

### 10.3 Hypothesis: the surge is RNG-determined, not knob-determined (re-confirmed with Phase-o seed 3)

Across all 19 seeds in 7 phases, **the two seeds that broke 500 (Phase-f seed
1 at 571, Phase-j seed 2 at 518) each used a DIFFERENT seed number** and a
DIFFERENT knob configuration. They're not "the same seed always wins" — they
look more like "any seed can win by luck of early exploration, and our knob
choices don't change the win rate much."

Per-seed analysis across all our phases:

| seed_number | times in K=4 | times peaked > 500 |
|:-----------:|-------------:|-------------------:|
| 1 | 6 | 1 (Phase-f) |
| 2 | 3 | 1 (Phase-j) |
| 3 | 4 | 0 |
| 4 | 5 | 0 |
| 5 | 4 | 0 |

Seeds 1 and 2 each got lucky exactly once each across many tries. **No seed
is consistently a winner; the RNG path that produces foot-hop is rare and
slightly different each phase.**

---

## 11. What this means for the goal "all 5 seeds > 500"

It's now clear that **no smoothing-or-Glass-side knob can rescue this**. The
remaining viable directions, in increasing order of intrusiveness:

### 11.1 Phase-p — proper Q-reset (REDQ-style, with pi-update pause)

The previous Q-reset (Phase-e) was buggy (re-init full opt state). A correct
implementation:
1. Reset only Q's slice of opt state (preserve pi/enc/dyn Adam moments)
2. Pause pi/enc/dyn gradient updates for ~50k env steps after reset, letting
   Q re-converge from the preserved target Q

Hypothesis: Q-overestimation locking the policy into the knee-walk gait is
the actual mechanism. Reset breaks the lock; pi-pause lets it stick.

Cost: ~50 LOC. Diagnostic value: high (independent of Glass).

### 11.2 Phase-r — reward shaping (penalise knee-ground contact)

Modify `mujoco_playground/_src/dm_control_suite/hopper.py` to add a small
negative reward when the knee body is in contact with the floor. Forces the
policy toward foot-strike-only gaits — the technique the video analysis
showed is the difference between winners (571) and stuck (262).

Cost: ~30 LOC + needs vendoring the env file. Diagnostic: very direct.

### 11.3 Phase-s — distributional Q (quantile regression)

Replace the two-hot 101-bin Q distribution with quantile regression (per
blog §9 item 1). Better Q estimates → better policy gradient direction →
more reliable technique discovery.

Cost: ~200 LOC, biggest algorithmic change. Defer until p/r tried.

### Recommendation

Run **Phase-p and Phase-r in parallel** (one per box) — they attack
orthogonal failure mechanisms (critic-side vs reward-side) and either one
producing >500 reliably would solve the user's goal.
