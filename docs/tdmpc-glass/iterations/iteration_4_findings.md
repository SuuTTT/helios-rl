# TD-MPC-Glass HopperHop — Iteration 4 plan: paths to reliably hit all 5 > 500

Date: 2026-05-15. Status: **3 of 20 seeds across Phases d→o cleared peak
MPPI=500 (15% hit rate)**, but no knob we've tried reliably triggers the
surge across multiple seeds. This document summarises **what worked**, **what
didn't**, and **9 candidate paths to break the RNG-luck bottleneck**.

Supersedes [iteration_3_findings.md](./iteration_3_findings.md) for the
forward-looking roadmap; that doc remains authoritative for the Phase-d → o
result table.

---

## 1. Summary of iteration-3 results — what worked, what didn't

### 1.1 The three winners (the only seeds to clear MPPI=500)

All three surge at **exactly the 1M env-step eval**:

| seed | knobs | peak | first >400 | basin | gait |
|------|-------|-----:|:----------:|:-----:|------|
| **Phase-f seed 1** | smooth=1e-3 (constant from 0) | 571 | **1M (417)** | K=4 | foot-hop |
| **Phase-j seed 2** | curriculum smooth | 518 | **1M (435)** | K=4 | foot-hop |
| **Phase-o seed 3** | curriculum smooth + glass_decay=2M | 524 | **1M (490)** | K=4 | foot-hop (presumed) |

The 1M checkpoint is a clean discriminator: seeds with MPPI>400 at 1M
sustain through full training; seeds with MPPI<200 at 1M plateau through
to 4-10M and never recover.

### 1.2 What worked

| intervention | effect | mechanism |
|--------------|--------|-----------|
| Latent action smoothing 1e-3 | +133 on lucky K=4 seed (438→571) | policy regulariser, encourages smoother gait |
| Curriculum smoothing (off pre-250k) | preserved K=4 basin slightly (one more win out of 5) | basin lock first, then policy regulariser |
| Glass loss decay at 2M | no harm to surging seeds | confirms Glass is peripheral |
| `--early_stop_patience` flag | saves ~6h per stuck seed | engineering win |
| Python-conditional smoothing graph (Phase-m) | seed 4 K=3 → K=4, but peak unchanged (262 vs 266) | basin geometry isn't the cap |

### 1.3 What didn't work

| intervention | result | why |
|--------------|--------|-----|
| Act-noise anneal 0.30→0.10 (Phase 1c) | -160 on winners | wider noise + decay HURTS late-stage learning |
| Act-noise = 0.40 (Phase-d v1) | mjx-warp-901 crash @1M | extreme actions drive hopper into non-converging solver configs |
| MPPI horizon H=5 alone (Phase-d v2) | plateau 199 by 1.5M | longer planning doesn't fix downstream stuck |
| Naive Q-reset (Phase-e) | corrupts pi (peak 228→3) | reset opt state destroys pi's Adam moments |
| consistency_coef=1.0 (Phase-g) | peak 482 best, no >500 | too aggressive on dynamics regularisation |
| smooth + ccoef combo (Phase-h) | peak 490, plateau | additive but still no surge |
| smooth=1e-4 (Phase-i) | peak 308 (stuck) | too weak to trigger surge |
| proto_T=0.4 (Phase-n) | seed 4 still K=3 | sharper softmax isn't the basin lever |
| λ_temporal=0.05 (Phase-k) | plateau 292 | over-regularising kills the surge |
| Glass-zeroed + smoothing (Phase-l) | peak 289 @4.75M | Glass IS contributing, smoothing alone insufficient |
| Glass-off at 2M (Phase-o) | mixed — 1 win (seed 3), 2 stuck, 1 crash | hybrid abstraction doesn't trigger surges; surges happen during Glass-on phase anyway |

### 1.4 The core insight from iteration 3

**The surge to >500 happens within the first 1M env steps OR not at all**,
and is RNG-determined (whether the policy stumbles on foot-hop technique
during early exploration). Our knob choices act as float-order perturbations
of which RNG path each seed takes — but we don't *control* the surge rate
above ~15%. The Glass cluster basin (K=3 vs K=4) doesn't determine peak.

To break the 15% ceiling we need an intervention that **causes** the
foot-hop discovery rather than waits for luck.

---

## 2. The 9 paths in iteration-4 roadmap

Ranked by `(credibility × expected payoff) / implementation cost`:

| # | path | mechanism (1-line) | LOC | credibility |
|:-:|------|----|----:|:-----------:|
| 1 | **Larger EXPL_UNTIL (25k→500k)** | random actions fill replay with foot-strike data; policy distills | **1** | ⭐⭐⭐ |
| 2 | RND/ICM curiosity bonus | intrinsic reward for novel states; pushes policy out of knee-walk attractor | ~100 | ⭐⭐⭐ |
| 3 | Parallel policies w/ keep-best | K=5 pi heads share enc/dyn; periodically prune to top-3 | ~150 | ⭐⭐⭐ |
| 4 | BC from winner trajectory | replay foot-hop rollouts from Phase-f seed 1, weighted BC loss | ~80 | ⭐⭐⭐ |
| 5 | Direct reward shaping (knee penalty) | `r' = r - 0.1 * knee_in_contact` in env | env edit | ⭐⭐⭐ |
| 6 | Periodic policy reset | re-init pi every 500k env steps | ~30 | ⭐⭐ |
| 7 | Feed cluster id to policy | pi(z, one_hot(argmax(S))) — Glass→control | ~40 | ⭐⭐ |
| 8 | Multi-task (HopperHop + Stand) | HopperStand teaches foot-balance | new config | ⭐⭐ |
| 9 | Larger MPPI NS=512→2048 | 4× planner samples | 1 | ⭐ |

### 2.1 Execution order

Start with the cheapest highest-credibility ones (1, 2, 3) sequentially. If
any single one delivers ≥3/5 seeds > 500 reliably, the iteration is done.
If all three falter, move to 4 (BC) and 5 (reward shaping) which are heavier
but more direct.

Paths 6-9 are tertiary — only run if 1-5 produce inconclusive results.

---

## 3. Why path 1 first

Path 1 (larger EXPL_UNTIL) is the **highest-EV-per-LOC**. The argument:

- The current 25k env steps of random actions = roughly 1 random episode
  per env (1000 steps × 25 batches = 25k). That's barely enough for random
  Hopper to ever produce a foot-strike pattern.
- 500k random env steps = 20× the buffer of random rollouts before policy
  takes over. Statistically, foot-strike events occur in random Hopper
  rollouts at some non-zero rate; with 20× more random data, ~20× more
  foot-strike transitions land in the replay buffer.
- The Q-function trains on the replay buffer, *not* on the current policy's
  distribution. So Q learns foot-strike values from random data.
- Policy gradient direction = `∇ E[Q(a)]`. If Q says foot-strikes are
  high-reward, the policy is *pulled* toward foot-strikes regardless of
  what its current rollouts look like.

This bypasses the "policy stuck in own gait distribution" trap.

**Risk**: Q might not generalise from random rollouts to policy rollouts.
i.e. Q learns foot-strike value but the policy needs specific action
sequences that random sampling rarely produces. We'll see.

---

## 4. Definition of "done" for iteration 4

We stop iterating when ANY of these holds:

- All 5 standard seeds (1-5) achieve peak MPPI ≥ 500 in a single phase
- Mean peak MPPI ≥ 600 over 5 seeds (substantially above official 449)
- 4/5 seeds achieve ≥ 500 AND mean ≥ 550

We accept partial wins (3-4/5) if the new intervention is independently
useful (e.g. a new algorithmic feature with research value beyond this
specific HopperHop sweep).

---

## 5. Phase-p (path 1) just launched

Code change: add `--expl_until N` CLI flag, default = DEFAULTS["EXPL_UNTIL"]
= 25,000. Set 500,000 for Phase-p. 1-LOC change to `train_tdmpc2`'s
EXPL_UNTIL local var, plumb through main().

Running on:
- Local 4070 Ti: SEEDS="4 5 3" (the historically-stuck seeds)
- Remote 4060: SEEDS="1 2" (control — confirm Phase-p doesn't BREAK lucky seeds)

First eval at 250k env steps still happens but with random actions. Real
signal at 500k env steps when policy takes over.

---

## 6. After-paths roadmap (if path 1 falters)

The 9 paths run sequentially, but if path 1 is partially successful
(2-3/5 surges), we may combine it with path 4 (BC from winner) or path 7
(cluster-conditional policy) for cumulative gains. The "stop-iteration"
criteria in §4 still hold.

If paths 1-5 all fail to break 3/5 reliably, the conclusion is that the
HopperHop foot-hop discovery is fundamentally exploration-hard at this
model scale, and we should either:
- Scale up the encoder/dynamics network (capacity argument)
- Move to easier DMC tasks (Cheetah, Walker) and treat HopperHop as a hard
  edge case
- Accept the field's norm of mean-over-seeds reporting rather than
  all-seeds-above-threshold

---

## 7. Hypothesis from Phase-p preliminary data: "abstract + slow exploration = gradual mastery of hard tasks"

User insight after seeing Phase-p local seed 4 reach MPPI 497 at 8.5M from a
K=3 basin (which we had previously believed was structurally capped at ~310):

> *"Is the 10M-step slow-burn run indicating that with abstraction + large
> EXPL_UNTIL we can learn hard tasks gradually — just like humans solve
> complex tasks with abstraction? To prove this hypothesis we need to
> (a) ablate the Glass mode and (b) implement a hierarchical abstraction
> module to tackle even higher complexity."*

### 7.1 What the Phase-p seed 4 trajectory actually shows

Seed 4 in K=3 basin under prior phases (Phase-f, Phase-j, Phase-m) all
plateaued at MPPI 254-266 by 4-6M env-steps. Same seed under Phase-p
(curriculum smoothing + Glass active + 500k random exploration) climbed
slowly but **monotonically** from 0 to 497 over 8.5M env-steps. No 1M-surge
event; just `+30 MPPI per 1M env-steps` of slow-burn ascent. That's a
qualitatively different learning shape from the lucky-surge winners of
Phase-f/j/o.

The seed 4 trajectory **didn't stop at the K=3 cap because nothing forced
it to** once the data + abstraction combination was strong enough. The
old "K=3 caps at 310" finding looks like an artifact of insufficient
exploration data, not a structural limit of the partition.

### 7.2 The user's hypothesis, sharpened

There are two distinct claims:

**(A) Abstraction is a *helpful prior* for slow-burn learning of hard tasks.**
The Glass partition gives the value/dynamics functions a structured latent
to predict over. With enough exploration data (path 1's EXPL_UNTIL=500k),
the slow accumulation of foot-strike transitions in the replay buffer
gradually re-anchors the policy onto the high-reward partition, even from a
"wrong" basin (K=3) start.

**(B) Hierarchical abstractions extend this to harder tasks.**
HopperHop has only ~4 gait phases (foot-strike, push-off, flight, landing).
A K=4 (or K=3-collapsed-to-2-active) partition is enough representational
capacity. For tasks with many more behavioural primitives — e.g.
QuadrupedRun (4 legs × 4 phases = 16 primitives), Acrobot (multi-joint
swing patterns), or full humanoid locomotion — a flat K=8 partition runs
out of granularity. A *hierarchical* abstraction (e.g. K_super=4 super-
clusters, each containing K_sub=8 sub-clusters → 32 effective categories
in a 2-level tree) maintains tractable structural-entropy computation
*and* sufficient capacity for richer behavioural vocabularies.

### 7.3 Tests to validate the hypothesis

To turn this from "interesting result on one seed" into "robust finding"
we need three concrete experiments:

**Test 1 — Glass ablation under Phase-p config.** Run Phase-p with
`--glass_lambda_se 0 --glass_lambda_balance 0 --glass_lambda_temporal 0`
(Glass loss zeroed — same trick as Phase-l). Same EXPL_UNTIL=500k +
curriculum smoothing. If seed 4 still slow-burns to ~500, the abstraction
isn't load-bearing — Path 1 is sufficient on its own. If it caps at ~260
again like Phase-l did, Glass IS load-bearing for slow-burn learning.

This is a 1-line CLI change; can run on 3060Ti as soon as Phase-p
seeds 6-8 finish (~4h each).

**Test 2 — Larger EXPL_UNTIL on a known-stuck seed.** Take Phase-p remote
4060 seed 1 (which failed at peak 151 with EXPL_UNTIL=500k) and rerun with
EXPL_UNTIL=1M or 2M. If higher EXPL_UNTIL rescues it, Path 1 just needs
*more* random data to be reliable. If it still fails, there's some
seed-specific RNG path that no amount of data can rescue.

**Test 3 — Hierarchical Glass on a harder task.** Implement the K_super
layer (~50 LOC in `tdmpc_glass.py`). Don't even test on HopperHop (which
only needs K=4); pick QuadrupedRun or HumanoidWalk where ~16-32 effective
categories is the right capacity. If hierarchical Glass produces faster
or more reliable convergence than flat-K=8 Glass on the harder task,
the hypothesis is validated.

### 7.4 Implementation sketch — hierarchical Glass

Currently:
```
prototypes      μ ∈ ℝ^{N×d}        (N=16 anchors)
assign_logits   L ∈ ℝ^{N×K}        (K=8 clusters)
S = softmax(L, axis=1)              (N×K)
2D-SE on (A=½(P+Pᵀ), S)             where P_kl = c_src^T c_next
```

Hierarchical:
```
prototypes      μ ∈ ℝ^{N×d}                 (N=16 anchors, unchanged)
assign_sub      L_sub ∈ ℝ^{N×K_sub}         (K_sub=8 fine clusters)
assign_super    L_super ∈ ℝ^{K_sub×K_super} (K_super=4 coarse clusters)
S_sub   = softmax(L_sub, axis=1)             (N×K_sub)
S_super = softmax(L_super, axis=1)           (K_sub×K_super)
S_combined = S_sub @ S_super                 (N×K_super, the effective fine→coarse map)
2D-SE on (A, S_combined) gives the coarse partition;
optionally a second 2D-SE on (P_sub, S_sub) for the fine partition.
```

Combined loss = `λ_se_super * SE_2D(A, S_combined) + λ_se_sub * SE_2D(P_sub, S_sub)`.

The two SE losses encourage (a) the coarse partition to capture broad
behavioural categories and (b) the fine partition to capture sub-phases
within each coarse category. Like phonemes within phrases.

Cost: ~50 LOC in `tdmpc_glass.py` + 10 LOC in `init_glass_params`. ~30 min.

### 7.5 Caveats

- The Phase-p seed 4 result is a single seed. Need at least 3-5 seeds
  showing the same slow-burn-to-500 pattern before we generalise.
- We haven't run Test 1 yet (Glass ablation under Phase-p config). Until
  we do, we don't know if Glass actually contributes to the slow-burn,
  or if it's just EXPL_UNTIL doing the work alone.
- Hierarchical Glass on HopperHop is overkill (4 phases, K=8 already
  generous). The right test target is a task where K=8 visibly bottlenecks.

### 7.5b Implemented design — hierarchical Glass module

The hierarchical-Glass implementation per §7.4 sketch landed in
`src/helios/algorithms/tdmpc_glass.py` on 2026-05-16. Three small changes:

#### Code changes

1. **`init_glass_params`** — new optional kwarg `num_super_clusters: int = 0`.
   When > 0, allocates `super_assign_logits ∈ ℝ^{K_sub × K_super}` with the
   same init scale as the existing `assign_logits`. Backward-compatible:
   `num_super_clusters=0` produces the exact same params dict as before.

2. **`glass_transition_graph`** — when `glass_params["super_assign_logits"]`
   exists, additionally computes:
   ```python
   S_super    = softmax(L_super, axis=1)        # (K_sub, K_super)
   S_combined = S @ S_super                     # (N, K_super) — coarse map
   super_se   = 2D-SE(A, S_combined)            # SE on the same prototype graph
   super_balance = sum(relu(mean(S_combined) - 2/K_super)^2)
   super_entropy, super_active, super_cut       # diagnostics, mirror flat-K versions
   ```

3. **`glass_loss_and_aux`** — new kwargs `lambda_super_se`,
   `lambda_super_balance`. When > 0 AND `super_se` is in the diag,
   `total += λ_super_se * super_se + λ_super_balance * super_balance`. Aux
   dict gains `glass_super_*` keys. The flat-Glass numerical behaviour is
   *bit-exact* preserved when both lambdas are 0.

4. **`make_update_fn`** in `tdmpc_glass.py` — new pass-through args
   `glass_lambda_super_se`, `glass_lambda_super_balance`. The
   `enabled_glass`/`disabled_glass` jax.lax.cond branches both gain the
   `glass_super_*` aux keys when the lambdas are > 0 (so PyTree shape
   matches across the cond).

5. **CLI** in `run_benchmark.py` — three new flags:
   `--glass_num_super_clusters`, `--glass_lambda_super_se`,
   `--glass_lambda_super_balance`, plumbed through `glass_overrides` →
   `glass_cfg` → `init_glass_params` and `_build_multi_step`.

#### Math, fully written out

Let `μ ∈ ℝ^{N×d}` (N=16 prototypes), `L_sub ∈ ℝ^{N×K_sub}` (K_sub=8 fine
clusters), and the new `L_super ∈ ℝ^{K_sub×K_super}` (K_super=4 coarse
clusters). For a batch of (z_src, z_next) latent pairs:

1. Soft assign each latent to prototypes:
   `c_src[t] = softmax( ẑ_src[t] @ μ̂ᵀ / T_proto )`  (shape (N,))
2. Build prototype transition matrix:
   `P[k,l] = Σ_t c_src[t,k] · c_next[t,l]`, row-normalise; A = ½(P + Pᵀ).
3. Compute fine partition: `S_sub = softmax(L_sub, axis=1)` (N×K_sub).
4. Compute coarse partition (NEW): `S_super = softmax(L_super, axis=1)`
   (K_sub×K_super), then `S_combined = S_sub @ S_super` (N×K_super).
   `S_combined[n,k]` = "probability prototype n belongs to coarse cluster k".
5. SE loss:
   ```
   L_glass = λ_se * SE_2D(A, S_sub)              # fine-level SE (existing)
           + λ_super_se * SE_2D(A, S_combined)   # coarse-level SE (NEW)
           + λ_balance * (balance_sub + proto_balance)
           + λ_super_balance * balance_super     # NEW
           + λ_temporal * temporal               # existing
   ```

Both SEs operate on the same A; only the partition differs. Minimising
`SE_2D(A, S_sub)` encourages the K_sub=8 partition to find communities;
minimising `SE_2D(A, S_combined)` encourages those communities to themselves
group into K_super=4 super-communities. Like phonemes within phrases.

Compute cost: `S_combined = S_sub @ S_super` is one (N×K_sub)·(K_sub×K_super)
matmul = 16·8·4 = 512 FLOPs, negligible. Second SE call is also O(N·K_super)
= 16·4 = 64 work. The per-step overhead vs flat Glass is ~0.1%.

Parameter count: `super_assign_logits` adds `K_sub × K_super = 8 × 4 = 32`
free parameters. Negligible vs the 2.5M-param TD-MPC2 backbone.

#### Where it doesn't help (and why we run it on HopperHop first anyway)

HopperHop has 4 gait phases. K=8 is already 2× generous. Adding K_super=4
on top of K_sub=8 might just produce S_super ≈ I (each fine cluster
identifies as its own coarse cluster) — equivalent to flat K=8.

But running it on HopperHop first is cheap (~3-5h on the 3060Ti) and
serves as a **smoke test** for the implementation. If the hierarchical
training doesn't break HopperHop, we know the code is sound and can move
to QuadrupedRun (16 effective primitives, where the hierarchy should
actually pay off).

#### Launch plan — Phase-y on 3060Ti after Path 1 seeds 6-8 finish

Config (HopperHop smoke test):
```
--glass_num_super_clusters    4    (K_super=4 super-clusters)
--glass_lambda_super_se       5e-3 (same as λ_se — keeps loss balance)
--glass_lambda_super_balance  1e-2 (same as λ_balance)
+ Phase-m baseline knobs (curriculum smooth, Python-conditional graph)
+ EXPL_UNTIL=500k (Path 1, since it gave seed-4 the slow-burn climb)
```

Run on 3060Ti seeds 1, 2, 3 (overlap with previously-tested seeds for
direct comparison) once Path-1 seeds 6-8 finish (~12h from now).
Launcher: `scripts/run_phasey_hierarchical.sh` (to be written).

If Phase-y on HopperHop produces:
- ~similar peaks to Path 1 → implementation is sound but the task doesn't
  need hierarchy (move to QuadrupedRun next).
- Higher peaks → hierarchy helps even on HopperHop (unexpected; would be
  a positive surprise).
- Crashes / NaNs → debug the hierarchical SE math.

### 7.6 Recommended next moves (updated from §2)

If Phase-p seeds 6-8 (3060Ti) confirm the slow-burn pattern (≥1 of 3
clears 500), promote the user's hypothesis to a tested result and:

1. Run Test 1 (Glass ablation under Phase-p config) on 3060Ti — single
   seed, ~4h.
2. Implement hierarchical Glass per §7.4 sketch — code change in
   tdmpc_glass.py.
3. Pick a harder task (QuadrupedRun) and run hierarchical-vs-flat Glass
   side by side. This is the proper test of claim (B).

The 9-path roadmap from §2 is still valid as a fallback if the
hierarchical-abstraction direction doesn't pan out — but it's now
clearly a "research" direction worth pursuing alongside the engineering
fixes.

---

## 8. Open todos at end of iteration 3

- ✅ Phase-m / Phase-o code + sweeps complete; data preserved.
- ⏳ Render Phase-o seed 3's `best_mppi.pkl` — the new winner, to confirm
  foot-hop gait via video (cluster_to_gait_mapping checklist).
- ⏳ Phase-h seed 1 video rendering deferred (peak 490 K=4 near-winner;
  good comparison point against the 500-clearing winners).
- ⏳ Run a tiny pure-vanilla TD-MPC2 control (no Glass, no smoothing,
  --algos tdmpc2 with the Phase-m smoothing port) for 5 seeds — to know
  the actual TD-MPC2 baseline hit rate on our codebase.

## §11. Path P — Cluster-entropy intrinsic reward (FALSIFIED)

Goal: benchmark-fair alternative to Path 5 (knee penalty). Use Glass clusters
as exploration prior — reward gait diversity via `coef * entropy(last_W cluster_ids)`.
Algorithm-internal, no env modification.

### §11.1 Phase-P (static coef=0.1)
Seed 1: pi=42 → 78 → **MPPI peak 91 @ 1.25M** → 9 → 75 → **collapse to 2.4 @ 2M**.
Killed at 2M. Hypothesis at kill: static intrinsic creates non-stationarity —
when policy converges on one gait, cluster window homogenizes → intrinsic drops
sharply → policy abandons gait to maintain diversity.

### §11.2 Phase-Pa (linear decay coef=0.1 → 0 over [500k, 3M])
Designed as "exploration curriculum" — by 3M run as pure extrinsic.
Seed 1: pi=3.5 → 3.8 → **MPPI peak 24.9 @ 1.25M** → never recovers. Early-stopped
4.25M. **3.6× WORSE than static Phase-P**, not better. Both peaked at exactly 1.25M
then collapsed at 1.5M — implicating coef=0.1 magnitude, not the decay schedule.

### §11.3 Diagnosis
Max entropy bonus per step = `0.1 * log(8) ≈ 0.21`. Over 1000-step episode
≈ 210 reward — comparable to or larger than HopperHop target ~600. Intrinsic
dominates the signal, not nudges. Even at 50% strength (1.75M, Phase-Pa), it's
already corrupting Q estimates. Path P falsified in both static and annealed
forms. Did not retry with smaller coef — moved to Path 7 (cluster as observation,
not reward) as a cleaner architectural alternative.

